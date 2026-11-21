"""宏层边界回归：队列、依赖、串行引擎和真实跨引擎并行。"""
import copy
import unittest
from collections import Counter
from test_vector_detail import M, config


class MacroTests(unittest.TestCase):
    def run_attention(self, policy='ooo', **settings):
        c=config()
        c['macro_scheduler']=copy.deepcopy(M['MACRO_DEFAULTS'])
        for q in c['macro_scheduler']['queues'].values(): q['policy']=policy
        c['vector_detail'].update(settings)
        c=M['validate_detail'](c)
        p=M['macro_program'](c)
        sim=M['MacroSimulator'](p)
        return p,sim,sim.run()

    def test_macro_contracts_and_scalar_residency(self):
        p,s,r=self.run_attention()
        for tile in range(4):
            macros=[m for m in p.macros if m['tile']==tile]
            pv=next(m for m in macros if m['label']=='PV')
            owner=next(m for m in macros if 'EXP_P' in m['operations'])
            self.assertIn('ROWSUM',owner['operations'])
            self.assertIn(owner['id'],pv['dependencies'])
        forbidden=('spill ADD_L','reload ADD_L','spill ROWSUM','spill ALPHA')
        self.assertFalse(any(i['label'] in forbidden for i in p.instructions))
        ell=[v for v in p.values.values() if v['op']=='ADD_L']
        self.assertTrue(all(v['elements']==[1] for v in ell))
        self.assertEqual(len(ell),2*4)
        self.assertFalse(any(i['label']=='spill EXP_P' for i in p.instructions))
        stores=[e for e in r['events'] if e['label']=='store P']
        self.assertEqual(sum(e['elements']*2 for e in stores),4*2*64*2)
        self.assertTrue(all(p.instructions[e['instruction']]['width']==2 for e in stores))

    def test_serial_engines_and_output_publication(self):
        p,s,r=self.run_attention()
        by_id={m['id']:m for m in r['macros']}
        for engine in ('Tensor','Vector','DMA'):
            records=sorted((m for m in r['macros'] if m['engine']==engine),key=lambda m:m['start'])
            for a,b in zip(records,records[1:]): self.assertLessEqual(a['end'],b['start'])
        for m in r['macros']:
            for dep in m['dependencies']: self.assertLessEqual(by_id[dep]['end'],m['start'])
            ev=[e for e in r['events'] if e['macro_id']==m['id']]
            self.assertTrue(all(max(e['result'],e['writeback'] or 0)<=m['end'] for e in ev))
        self.assertGreater(r['overlap_cycles']['Tensor_Vector_compute'],0)

    def test_fifo_order_and_four_pending_plus_one_active(self):
        p,s,r=self.run_attention('fifo')
        self.assertEqual(max(Counter(m['enqueue'] for m in r['macros']).values()),1)
        for engine in ('Tensor','Vector','DMA'):
            records=[m for m in r['macros'] if m['engine']==engine]
            self.assertEqual([m['id'] for m in records], [m['id'] for m in sorted(records,key=lambda m:m['start'])])
            history=r['macro_queue_history'][engine]
            self.assertLessEqual(max(x['queued'] for x in history),4)
            self.assertLessEqual(max(x['active'] for x in history),1)
            self.assertLessEqual(max(x['queued']+x['active'] for x in history),5)

    def test_ooo_can_bypass_blocked_tensor_head(self):
        c=config(); c['shape']['query_rows']=16
        c['macro_scheduler']=copy.deepcopy(M['MACRO_DEFAULTS'])
        for q in c['macro_scheduler']['queues'].values(): q['policy']='ooo'
        p=M['macro_program'](c); s=M['MacroSimulator'](p); r=s.run()
        records=[m for m in r['macros'] if m['engine']=='Tensor']
        self.assertNotEqual([m['id'] for m in records],[m['id'] for m in sorted(records,key=lambda m:m['start'])])

    def test_single_op_macros_materialize_large_values(self):
        p,s,r=self.run_attention(fusion_regions=[])
        self.assertTrue(any(i['label']=='spill SCALE' for i in p.instructions))
        self.assertTrue(any(i['label']=='reload SCALE' for i in p.instructions))
        self.assertFalse(any('ADD_L' in i['label'] and i['unit']=='ST' for i in p.instructions))
        self.assertGreater(len(p.macros),32)

    def test_whole_vector_fusion_is_split_at_tensor_boundary(self):
        ops=['SCALE','MASK','ROWMAX','MAX','SUB_S','DELTA','EXP_P','ALPHA','ROWSUM','SCALE_L','ADD_L','SCALE_A','ADD_A']
        p,s,r=self.run_attention(fusion_regions=[ops])
        self.assertFalse(any('EXP_P' in m['operations'] and 'ADD_A' in m['operations'] for m in p.macros))

    def test_limits_apply_to_new_backend(self):
        c=config(); c['macro_scheduler']=copy.deepcopy(M['MACRO_DEFAULTS'])
        c['vector_detail']['register'].update(capacity_registers=128,read_ports=1,write_ports=1)
        r=M['MacroSimulator'](M['macro_program'](M['validate_detail'](c))).run()
        self.assertLessEqual(r['peak_registers'],128)
        self.assertLessEqual(r['peak_RF_reads'],1)
        self.assertLessEqual(r['peak_RF_writes'],1)

    def test_small_queues_and_invalid_nonconvex_fusion(self):
        c=config(); c['macro_scheduler']=copy.deepcopy(M['MACRO_DEFAULTS'])
        for q in c['macro_scheduler']['queues'].values(): q['entries']=1
        r=M['MacroSimulator'](M['macro_program'](c)).run()
        self.assertGreater(r['stalls']['macro_frontend_queue_full'],0)
        for history in r['macro_queue_history'].values():
            self.assertLessEqual(max(h['queued'] for h in history),1)
        c['vector_detail']['fusion_regions']=[['SCALE','EXP_P']]
        with self.assertRaisesRegex(ValueError,'依赖环'): M['macro_program'](c)

    def test_row_reduction_and_partial_register_bytes(self):
        c=config(); c['macro_scheduler']=copy.deepcopy(M['MACRO_DEFAULTS'])
        c['shape'].update(kv_cols=65,head_dim_v=97)
        c['vector_detail']['register']['elements']=32
        c['vector_detail']['tiles']=2
        p=M['macro_program'](c); r=M['MacroSimulator'](p).run()
        self.assertEqual(sum(e['elements']*2 for e in r['events'] if e['label']=='store P'),2*2*65*2)
        self.assertEqual(sum(e['elements']*4 for e in r['events'] if e['label']=='store A'),2*2*97*4)
        self.assertGreaterEqual(r['engine_busy_cycles']['Tensor'],r['unit_busy_cycles']['TENSOR'])

    def test_macro_admission_preserves_other_macro_state(self):
        c=config(); c['shape']['query_rows']=64
        c['vector_detail']['register']['capacity_registers']=300
        c['macro_scheduler']=copy.deepcopy(M['MACRO_DEFAULTS'])
        for q in c['macro_scheduler']['queues'].values(): q['policy']='ooo'
        r=M['MacroSimulator'](M['macro_program'](c)).run()
        self.assertLessEqual(r['peak_registers'],300)
        self.assertGreater(r['macro_wait_reasons']['Vector']['RF_macro_working_set'],0)

if __name__=='__main__': unittest.main()
