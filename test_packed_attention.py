import json
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = runpy.run_path(str(ROOT / 'flashattention_pipeline.vector.detail.py'))


class PackedTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = json.loads((ROOT / 'flashattention_pipeline.vector.detail.fp16.json').read_text())
        cfg['vector_detail']['tiles'] = 2
        cls.program = MODEL['macro_program'](MODEL['validate_detail'](cfg))
        cls.report = MODEL['MacroSimulator'](cls.program).run()

    def events(self, label):
        return [e for e in self.report['events'] if e['tile']==0 and e['label']==label]

    def test_packed_reductions(self):
        for label in ('ROWMAX','ROWSUM'):
            events = self.events(label)
            self.assertEqual(sorted(e['destination_lane'] for e in events),list(range(64)))
            self.assertEqual(len({e['register'] for e in events}),1)
            self.assertEqual(len({e['instruction'] for e in events}),16)
        self.assertEqual(len(self.events('MAX')),1)
        self.assertGreaterEqual(self.events('MAX')[0]['start'],max(e['writeback'] for e in self.events('ROWMAX')))

    def test_independent_branches_and_score_store(self):
        for label in ('SUB_S','SCALE_A'):
            self.assertEqual(sorted(e['broadcast_lane'] for e in self.events(label)),list(range(64)))
        p=self.events('EXP_P'); alpha=self.events('ALPHA')
        self.assertEqual(sum(e['end']-e['start'] for e in p),256)
        self.assertEqual(len(alpha),1)
        self.assertEqual(alpha[0]['end']-alpha[0]['start'],4)
        self.assertNotEqual(alpha[0]['macro_id'],p[0]['macro_id'])
        self.assertNotEqual(self.events('MAX')[0]['macro_id'],p[0]['macro_id'])
        for label in ('store scaled score','load scaled score','store P'):
            self.assertEqual(sum(e['elements']*2 for e in self.events(label)),8192)

    def test_wide_results_cast_before_consumer(self):
        for name,n in [('SCALE',64),('SCALE_L',1),('SCALE_A',64)]:
            mul=self.events(name); casts=self.events('CAST '+name)
            self.assertEqual(len(casts),n)
            for event in mul:
                self.assertEqual(self.program.values[event['destination'][0]]['words'],2)
            self.assertEqual({e['sources'][0]['ref'] for e in casts},{e['destination'] for e in mul})
        self.assertEqual(self.program.values[self.events('ADD_L')[0]['destination'][0]]['allocation'],1)

    def test_macro_boundaries_keep_only_cross_macro_states(self):
        for boundary in self.report['register_macro_boundaries']:
            for value in boundary['live_values']:
                spec=self.program.values[value['value']]
                self.assertTrue(spec.get('small_state') or spec.get('initial'))
        physical={}
        for life in self.report['register_lifetimes'].values():
            for number in range(life['base'],life['base']+life['count']):
                physical.setdefault(number,[]).append(life)
        reused=False
        for lives in physical.values():
            lives.sort(key=lambda life:life['allocate'])
            for previous,current in zip(lives,lives[1:]):
                self.assertIsNotNone(previous['release'])
                self.assertLessEqual(previous['release'],current['allocate'])
                reused |= previous['tile']!=current['tile']
        self.assertTrue(reused)

    def test_steady_window_excludes_drain_and_covers_a_period(self):
        from detail_plotting import select_steady_window
        macros=[]; events=[]
        for tile,start in enumerate([0,10,20,30,38]):
            macros.append(dict(engine='Vector',tile=tile,start=start))
            events.append(dict(unit='MUL',label='SCALE',tile=tile,start=start,end=start+2))
        report=dict(macros=macros,events=events,cycles=45)
        window=select_steady_window(report)
        self.assertTrue(window['verified_repeat'])
        self.assertEqual(window['end']-window['start'],10)
        self.assertLessEqual(window['comparison_end'],30)
