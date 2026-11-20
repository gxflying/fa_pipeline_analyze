import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flashattention_pipeline as p

c = json.loads(p.DEFAULT_CONFIG.with_name('flashattention_pipeline.fp16_inline.json').read_text())
results = []
for name, rd, wr in [('base',256,256),('store2',256,512),('load2',512,256),('both2',512,512)]:
    cfg = copy.deepcopy(c)
    cfg['vector']['read_gbps'] = rd
    cfg['vector']['write_gbps'] = wr
    ops,caps,ii,lower,total,pools = p.solve(cfg)
    named = {o.name:o for o in ops}
    def phases(o,r):
        return {t%ii for a,b,_ in o.runs(r) for t in range(a,b)}
    exp = phases(named['EXP_P'],'EXP')
    result = dict(case=name,read_gbps=rd,write_gbps=wr,ii=ii,lower=lower,operations={})
    for n in ['EXP_P','PV','SCALE_A','ADD_A']:
        o = named[n]
        result['operations'][n] = dict(start=o.start,ready=o.ready,
            occupancy=sum(b-a for a,b,_ in o.runs(o.unit)),overlap_exp=len(exp & phases(o,o.unit)))
    results.append(result)
    print(json.dumps(result),flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2))
