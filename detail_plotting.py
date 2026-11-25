"""单个重复 Vector 周期及带公式的执行轨迹绘图。"""
from collections import defaultdict
import numpy as np


def select_steady_window(report):
    anchors=[]
    for tile in sorted({m['tile'] for m in report.get('macros',[])}):
        macros=[m for m in report['macros'] if m['engine']=='Vector' and m['tile']==tile]
        if macros:
            anchors.append((tile,min(m['start'] for m in macros)))
    events=[e for e in report['events'] if e['unit'] not in ('TENSOR','DMA_RD','DMA_WR')]
    if not anchors:
        for tile in sorted({e['tile'] for e in events}):
            anchors.append((tile,min(e['start'] for e in events if e['tile']==tile)))

    def signature(index,period):
        tile,begin=anchors[index]
        end=anchors[index+period][1]
        return sorted((e['unit'],e['label'],e['tile']-tile,
                       max(begin,e['start'])-begin,min(end,e['end'])-begin)
                      for e in events if e['start']<end and e['end']>begin)

    for period in range(1,len(anchors)//2+1):
        candidates=[]
        for index in range(len(anchors)-2*period):
            begin=anchors[index][1]
            end=anchors[index+period][1]
            next_end=anchors[index+2*period][1]
            if end-begin==next_end-end and signature(index,period)==signature(index+period,period):
                candidates.append(index)
        if candidates:
            index=candidates[len(candidates)//2]
            return dict(start=anchors[index][1],end=anchors[index+period][1],
                        tiles=period,verified_repeat=True,
                        comparison_end=anchors[index+2*period][1],
                        criterion='相邻窗口的 Vector 执行事件按相对 tile 和时间归一化后完全一致')
    # 运行太短时如实显示完整观测区间，不强称稳态。
    begin=anchors[0][1] if anchors else 0
    return dict(start=begin,end=report['cycles'],tiles=len(anchors),verified_repeat=False,
                criterion='当前运行未发现两个重复的完整 Vector 周期')


def operation_formula(label,equations):
    if label in equations:
        return '$'+equations[label]+'$'
    symbols={'SCALE':'H_j','SCALE_L':r'\widetilde{\ell}_j','SCALE_A':r'\widetilde{A}_j'}
    if label.startswith('CAST '):
        symbol=symbols.get(label[5:],'x')
        return '$'+symbol+r'^{16}=\mathrm{cast}_{16}('+symbol+r'^{32})$'
    return label


def draw_vector_period(report,output,equations):
    import matplotlib.pyplot as plt
    window=report.get('steady_window') or select_steady_window(report)
    begin,end=window['start'],window['end']
    events=[e for e in report['events'] if e['unit'] not in ('TENSOR','DMA_RD','DMA_WR')
            and e['start']<end and e['end']>begin]
    lanes=list(dict.fromkeys((e['unit'],e['label']) for e in events))
    positions={lane:i for i,lane in enumerate(lanes)}
    plot=report['config']['plot']; dpi=plot['dpi']
    width=max(plot['width_px'],12*(end-begin)+500)
    height=max(plot['height_px']/dpi,len(lanes)*.32+6)
    fig,axes=plt.subplots(6,1,figsize=(width/dpi,height),dpi=dpi,sharex=True,
                          gridspec_kw={'height_ratios':[max(5,len(lanes)*.3),1,1,1,1,1]})
    groups=defaultdict(list)
    for event in events:
        row=positions[event['unit'],event['label']]
        axes[0].barh(row,event['end']-event['start'],left=event['start'],height=.65,
                     color=plt.cm.tab20(event['tile']%20),edgecolor='black',linewidth=.35)
        axes[0].plot(event['result'],row,'v',ms=3,color='green')
        if event['writeback'] is not None:
            axes[0].plot(event['writeback'],row,'|',ms=6,color='red')
        groups[event['instruction']].append(event)
    for group in groups.values():
        registers=[e['register'] for e in group if e.get('register') is not None]
        if not registers: continue
        start=max(begin,min(e['start'] for e in group))
        stop=min(end,max(e['end'] for e in group))
        lanes_written=[e['destination_lane'] for e in group if e.get('destination_lane') is not None]
        text=f'v{min(registers)}'
        if lanes_written: text+=f'[{min(lanes_written)}:{max(lanes_written)+1}]'
        elif max(registers)!=min(registers): text+=f'–v{max(registers)}'
        axes[0].text((start+stop)/2,positions[group[0]['unit'],group[0]['label']],text,
                     ha='center',va='center',fontsize=6.5,clip_on=True)
    axes[0].set_yticks(range(len(lanes)),[u+' '+operation_formula(label,equations) for u,label in lanes],fontsize=9)
    axes[0].set_ylim(len(lanes)-.4,-.6)
    name='完整稳态周期' if window['verified_repeat'] else '完整观测区间（尚未确认重复周期）'
    axes[0].set_title(f"Vector {name} [{begin}, {end})，{end-begin} cycle / {window['tiles']} tile\n"
                      '颜色区分 tile，组内标注物理 vreg；绿三角为结果就绪，红线为 RF 写回')
    sources=[report['allocated_registers'],report['source_operand_requests'],
             report['resources'].get('RF_R',{}),report['resources'].get('RF_W',{}),report['forwarding']]
    labels=['分配 vreg','源操作数需求','RF 读口','RF 写口','转发次数']
    x=np.arange(begin,end)
    for ax,values,label in zip(axes[1:],sources,labels):
        values=[values[t] if isinstance(values,list) else values.get(t,0) for t in x]
        ax.step(x,values,where='post'); ax.set_ylabel(label); ax.grid(alpha=.25)
    if report.get('cross_macro_registers'):
        axes[1].step(x,report['cross_macro_registers'][begin:end],where='post',label='跨宏值的已分配容量',color='orange')
        axes[1].legend(loc='upper right',fontsize=8)
    reg=report['config']['vector_detail']['register']
    size=reg['elements']*(2 if reg['dtype'] in ('fp16','bf16') else 4)
    axes[1].set_title(f"单 vreg {size} B；全程峰值 {report['peak_registers']} vreg；宏间复用已释放的物理编号",fontsize=10)
    for ax,limit in ((axes[1],reg['capacity_registers']),(axes[3],reg['read_ports']),(axes[4],reg['write_ports'])):
        if limit is not None: ax.axhline(limit,color='red',linestyle='--')
    axes[-1].set_xlim(begin,end); axes[-1].set_xlabel('cycle')
    fig.tight_layout(); fig.savefig(output/'vector_detail.png',dpi=dpi); plt.close(fig)


def draw_execution_timeline(report,output,equations,units):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plot=report['config']['plot']; dpi=plot['dpi']; width=plot['width_px']
    end=report['cycles']
    blocks=defaultdict(list)
    for event in report['events']:
        for start,stop in event.get('activity_intervals',[(event['start'],event['end'])]):
            blocks[event['unit']].append(dict(start=start,end=stop,label=event['label'],tile=event['tile'],
                                               macro=event.get('macro_id')))
    lanes=[u for u in ['DMA_RD','DMA_WR','TENSOR','LD',*units,'ST','BR'] if blocks[u]]
    # 同一宏内同名操作归集为一框；框内仍逐段绘制实际活动，保留等待空隙。
    for unit in lanes:
        grouped={}
        for block in blocks[unit]:
            key=block['label'],block['tile'],block['macro']
            if key not in grouped: grouped[key]=dict(block,activity=[])
            group=grouped[key]
            group['start']=min(group['start'],block['start'])
            group['end']=max(group['end'],block['end'])
            group['activity'].append((block['start'],block['end']-block['start']))
        blocks[unit]=sorted(grouped.values(),key=lambda b:(b['start'],b['end']))
    fig=plt.figure(figsize=(width/dpi,12),dpi=dpi)
    renderer=fig.canvas.get_renderer(); font=FontProperties(size=7)
    cursor=0; centers={}
    for unit in lanes:
        right_edges=[]
        for block in blocks[unit]:
            label=operation_formula(block['label'],equations)
            pixels=renderer.get_text_width_height_descent(label,font,ismath=label.startswith('$'))[0]+8
            span=pixels*end/(width*.89)
            center=min(end-span/2,max(span/2,(block['start']+block['end'])/2))
            level=next((i for i,right in enumerate(right_edges) if right<center-span/2),len(right_edges))
            if level==len(right_edges): right_edges.append(0)
            right_edges[level]=center+span/2
            block.update(text=label,text_x=center,level=level)
        levels=max(1,len(right_edges))
        base=cursor+levels*.42+.45
        centers[unit]=base
        for block in blocks[unit]: block['text_y']=base-.55-block['level']*.42
        cursor=base+.8
    fig.set_size_inches(width/dpi,max(plot['height_px']/dpi,cursor*.27+6))
    axes=fig.subplots(5,1,sharex=True,gridspec_kw={'height_ratios':[max(8,cursor*.27),1,1,1,1]})
    for unit in lanes:
        base=centers[unit]
        for block in blocks[unit]:
            center=(block['start']+block['end'])/2
            axes[0].barh(base,block['end']-block['start'],left=block['start'],height=.4,
                         facecolor='none',edgecolor='#666666',linewidth=.4,linestyle=':')
            axes[0].broken_barh(block['activity'],(base-.2,.4),
                                facecolors=plt.cm.tab20(block['tile']%20),linewidth=0)
            axes[0].annotate(block['text'],xy=(center,base-.2),xytext=(block['text_x'],block['text_y']),
                             ha='center',va='bottom',fontsize=7,
                             arrowprops=dict(arrowstyle='-',linewidth=.3,color='#555555'))
        axes[0].axhline(base+.5,color='#dddddd',linewidth=.5)
    axes[0].set_yticks([centers[u] for u in lanes],lanes)
    axes[0].set_ylim(cursor,-.6)
    axes[0].set_title('完整执行归集图；每个宏内同名操作归集为一框，公式标于框上方\n框内实色为实际活动，空白为等待，颜色区分 tile')
    for ax,resource in zip(axes[1:],['L1_R','L1_W','L1_R_BUS','L1_W_BUS']):
        cap=report['config']['l1']['read_gbps' if '_R' in resource else 'write_gbps']/report['config']['clock_ghz']
        scale=100/cap if resource.endswith('_BUS') else 1
        values=report['resources'].get(resource,{})
        ax.plot([values.get(t,0)*scale for t in range(end)],linewidth=.5)
        ax.set_ylabel(resource+(' %' if resource.endswith('_BUS') else ' B/c'))
        ax.grid(alpha=.25)
    axes[-1].set_xlim(0,end); axes[-1].set_xlabel('cycle')
    fig.subplots_adjust(left=.08,right=.99,top=.96,bottom=.04,hspace=.35)
    fig.savefig(output/'pipeline.png',dpi=dpi); plt.close(fig)
