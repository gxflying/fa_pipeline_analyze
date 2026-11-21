"""新后端行为回归；WSL python3 -B -m unittest -v test_vector_detail。"""
import copy
import json
import runpy
import unittest
from pathlib import Path
from collections import Counter

M=runpy.run_path(str(Path(__file__).with_name('flashattention_pipeline.vector.detail.py')))

def config():
    c=json.loads(Path(__file__).with_name('flashattention_pipeline.vector.detail.json').read_text())
    c['shape']['query_rows']=2
    c['vector_detail']['tiles']=4
    return c

class DetailTests(unittest.TestCase):
    def toy(self,forward=True,ports=None,latency=1):
        c=config(); d=c['vector_detail']; d['group_size']=4; d['tiles']=1
        d['forwarding']=forward; d['register']['read_ports']=ports
        c['vector']['units']['ADD']['latency_cycles']=latency
        p=M['DetailProgram'](c)
        a=p.value(4); b=p.value(4); z=p.value(4)
        p.emit('LD','A',destinations=a,width=4)
        p.emit('LD','B',destinations=b,width=4)
        p.emit('ADD','sum',[list(x) for x in zip(a,b)],z)
        p.emit('ST','output',[[x] for x in z],width=4)
        return M['DetailSimulator'](p).run()

    def test_screenshot_chaining_and_writeback(self):
        r=self.toy()
        adds=[e for e in r['events'] if e['unit']=='ADD']
        self.assertEqual([e['start'] for e in adds],[5,6,7,8])
        self.assertEqual([e['writeback'] for e in adds],[7,8,9,10])
        self.assertEqual(r['peak_RF_writes'],2)

    def test_no_forward_requires_writeback(self):
        r=self.toy(False)
        self.assertEqual([e['start'] for e in r['events'] if e['unit']=='ADD'],[6,7,8,9])
        self.assertEqual([e['start'] for e in r['events'] if e['unit']=='ST'],[8,9,10,11])

    def test_long_latency_accepts_each_cycle(self):
        r=self.toy(latency=3)
        a=[e for e in r['events'] if e['unit']=='ADD']
        self.assertEqual([e['start'] for e in a],[5,6,7,8])
        self.assertTrue(all(e['result']-e['start']==3 for e in a))

    def test_limited_read_port_collects_operands(self):
        r=self.toy(False,1)
        self.assertLessEqual(r['peak_RF_reads'],1)
        self.assertGreater(max(e['result'] for e in r['events']),12)

    def test_full_attention_independent_event_audit(self):
        c=config(); c['vector_detail']['register']['write_ports']=1
        p=M['detail_program'](c); sim=M['DetailSimulator'](p); r=sim.run()
        producers={tuple(e['destination']):e for e in r['events'] if e.get('destination')}
        writes=Counter(); reads=Counter()
        for e in r['events']:
            if e['writeback'] is not None:
                writes[e['writeback']]+=1
                self.assertEqual(e['writeback'],e['result']+1)
            for src in e.get('sources',[]):
                if sim.p.values[src['ref'][0]].get('initial'):
                    self.assertEqual(src['path'],'RF'); reads[src['cycle']]+=1; continue
                prod=producers[tuple(src['ref'])]
                boundary=prod['result'] if src['path']=='forward' else prod['writeback']
                self.assertGreaterEqual(src['cycle'],boundary)
                self.assertLessEqual(src['cycle'],e['start'])
                if src['path']=='RF': reads[src['cycle']]+=1
        self.assertLessEqual(max(writes.values()),1)
        self.assertEqual(dict(reads),dict(sim.used['RF_R']))
        self.assertLessEqual(r['l1_pool_reserved_bytes'],c['l1']['size_kib']*1024)
        for i in p.instructions:
            if i['unit']=='BR' and i['id']+1 in sim.dispatch:
                self.assertGreaterEqual(sim.dispatch[i['id']+1],sim.done[i['id']])

    def test_fusion_boundary_creates_real_traffic(self):
        c=config(); c['vector_detail']['tiles']=1
        full=M['DetailSimulator'](M['detail_program'](c)).run()
        c['vector_detail']['fusion_regions']=[]
        split=M['DetailSimulator'](M['detail_program'](c)).run()
        self.assertGreater(sum(split['resources']['L1_W'].values()),sum(full['resources']['L1_W'].values()))
        self.assertTrue(any(e['label'].startswith('spill') for e in split['events']))

    def test_group_alignment_and_capacity(self):
        c=config(); c['vector_detail']['register']['capacity_registers']=256
        sim=M['DetailSimulator'](M['detail_program'](c)); r=sim.run()
        self.assertLessEqual(r['peak_registers'],256)
        for e in r['events']:
            if e.get('destination') and e['destination'][1]==0:
                size=sim.p.values[e['destination'][0]]['allocation']
                self.assertEqual(e['register']%size,0)
        c['vector_detail']['group_size']=3
        with self.assertRaises(ValueError): M['validate_detail'](c)

if __name__=='__main__': unittest.main()
