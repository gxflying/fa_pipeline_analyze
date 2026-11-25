"""按四行（或配置的 G 行）编译两遍 FP16 softmax。"""
import numpy as np

REGIONS = [['SCALE','MASK','ROWMAX','MAX'], ['SUB_S','EXP_P','ROWSUM'],
           ['DELTA','ALPHA','SCALE_L','ADD_L'], ['SCALE_A','ADD_A']]

def compile_packed(config,p,templates):
    shape=config['shape']
    rows=shape['query_rows']; group=p.detail['group_size']
    if not (1<=rows<=64 and shape['kv_cols']==shape['head_dim_v']==p.detail['register']['elements']==64):
        raise ValueError('紧凑后端支持 1..64 行，KV 列、dv 和 vreg 元素数均为 64')
    if p.detail['fusion_regions']!=REGIONS:
        raise ValueError('紧凑 FP16 路径使用两遍 score 和独立 DELTA/ALPHA 宏')
    def state(op,tile,initial=False):
        r=p.value(1,op,tile,[rows])[0]
        p.values[r[0]].update(allocation=1,small_state=True,initial=initial)
        return r
    def calc(op,unit,inputs,tile,owner=None,wide=False,scalar=False,**kw):
        out=p.value(len(inputs),op,tile,[rows if scalar else 64]*len(inputs))
        p.values[out[0][0]].update(allocation=(1 if scalar else group)*(2 if wide else 1),
                                  words=2 if wide else 1,small_state=scalar)
        p.emit(unit,op,inputs,out,tile=tile,owner_op=owner or op,**kw)
        return out
    def multiply(op,inputs,tile,scalar=False,**kw):
        wide=calc(op,'MUL',inputs,tile,wide=True,scalar=scalar,**kw)
        return calc('CAST '+op,'CVT',[[r] for r in wide],tile,owner=op,scalar=scalar)
    def load(label,owner,count,tile,deps):
        out=p.value(count,'LOAD',tile,[64]*count)
        p.emit('LD',label,destinations=out,deps=deps,tile=tile,width=2,owner_op=owner)
        return out
    old_m=state('STATE',-1,True); old_l=state('STATE',-1,True)
    templates={o.name:o for o in templates}
    delay=p.detail['tensor_store_cast_cycles']
    for name in ('QK','PV'):
        op=templates[name]
        for resource in ('TENSOR_ST','L1_W','L1_W_BUS'):
            op.profiles[resource]=np.pad(op.profiles[resource],(delay,0))
        op.ready_offset+=delay
    previous_a=[]
    for t in range(p.detail['tiles']):
        def macro(label,unit,template,deps=()):
            return p.emit(unit,label,tile=t,macro=templates[template],deps=deps)
        k=macro('load K','DMA_RD','K_DMA'); v=macro('load V','DMA_RD','V_DMA')
        mask=macro('load mask','DMA_RD','MASK_DMA'); qk=macro('QK','TENSOR','QK',[k])
        r=state('ROWMAX',t); z=state('ROWSUM',t)
        for ref in (r,z):
            p.values[ref[0]]['lane_count']=rows
        max_writers=[]; sum_writers=[]; scores=[]; probs=[]
        for row in range(0,rows,group):
            p.current_row=row; n=min(group,rows-row)
            x=load('load score','SCALE',n,t,[qk]); b=load('load mask','MASK',n,t,[mask])
            h=multiply('SCALE',[[a] for a in x],t)
            s=calc('MASK','ADD',list(zip(h,b)),t)
            scores.append(p.emit('ST','store scaled score',[[a] for a in s],tile=t,width=2,owner_op='MASK'))
            max_writers.append(p.emit('REDMAX','ROWMAX',[[a] for a in s],[r]*n,tile=t,reduction=True,lane_start=row))
        p.current_row=-1
        mj=calc('MAX','MAX',[[old_m,r]],t,scalar=True,deps=max_writers)[0]
        for row in range(0,rows,group):
            p.current_row=row; n=min(group,rows-row)
            s=load('load scaled score','SUB_S',n,t,scores)
            u=calc('SUB_S','SUB',[[a,mj] for a in s],t,broadcast_start=row)
            prob=calc('EXP_P','EXP',[[a] for a in u],t)
            probs.append(p.emit('ST','store P',[[a] for a in prob],tile=t,width=2,owner_op='EXP_P'))
            sum_writers.append(p.emit('REDSUM','ROWSUM',[[a] for a in prob],[z]*n,tile=t,reduction=True,lane_start=row))
        p.current_row=-1
        pv=macro('PV','TENSOR','PV',[v,*probs])
        delta=calc('DELTA','SUB',[[old_m,mj]],t,scalar=True)[0]
        alpha=calc('ALPHA','EXP',[[delta]],t,scalar=True)[0]
        sl=multiply('SCALE_L',[[alpha,old_l]],t,scalar=True)[0]
        old_l=calc('ADD_L','ADD',[[sl,z]],t,scalar=True,deps=sum_writers)[0]; old_m=mj
        stores=[]
        for row in range(0,rows,group):
            p.current_row=row; n=min(group,rows-row)
            a=load('load A','SCALE_A',n,t,previous_a); y=load('load Y','ADD_A',n,t,[pv])
            scaled=multiply('SCALE_A',[[ref,alpha] for ref in a],t,broadcast_start=row)
            result=calc('ADD_A','ADD',list(zip(scaled,y)),t)
            stores.append(p.emit('ST','store A',[[ref] for ref in result],tile=t,width=2,owner_op='ADD_A'))
        previous_a=stores
    for ref in (old_m,old_l): p.values[ref[0]]['retain_final']=True
    return p
