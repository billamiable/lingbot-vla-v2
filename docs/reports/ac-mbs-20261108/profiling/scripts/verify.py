"""Verify profiler captures, fixed-batch controls and input equality."""
import hashlib
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'evidence'
results=[]
for code in ['main','opt']:
    for recipe,mb,ga,ac,steps in [('noac16',16,2,False,60),('ac16',16,2,True,60),('ac32',32,1,True,60),('validate_ac16',16,2,True,80)]:
        name=f'validate_{code}_ac16' if recipe=='validate_ac16' else f'{code}_{recipe}'
        run=ROOT/'runs'/name
        assert json.loads((ROOT/f'{name}.receipt.json').read_text())['passed'],name
        recorded=(run/'source.txt').read_text().splitlines()
        expected={'main':'be969b8fd117fb70550c5d4bf4bc328211b5b1b6','opt':'71a41bca6be05b15a170b5358e3f6b43c4523898'}[code]
        assert len(recorded)==1 and recorded[0].split()[0]==expected,(name,recorded)
        rows=[list(map(json.loads,(run/f'rank{r}.jsonl').read_text().splitlines())) for r in range(8)]
        assert list(map(len,rows))==[steps]*8,name
        assert all(math.isfinite(row[k]) for rank in rows for row in rank for k in ['loss','vla_loss','depth_loss','future_depth_loss','future_video_loss','grad_norm','elapsed_s'])
        cfg=json.loads((run/'effective_args.json').read_text())['train']
        assert (cfg['micro_batch_size'],cfg['gradient_accumulation_steps'],cfg['enable_gradient_checkpointing'])==(mb,ga,ac)
        assert cfg['global_batch_size']==256 and cfg['seed']==42 and cfg['lr']==1e-4 and cfg['max_steps']==50000
        assert not cfg['enable_profiling'] and not cfg['enable_resume'] and cfg['save_steps']==0
        if steps==60:
            assert len(list(run.glob('trace_rank*.json.gz')))==8
        results.append(dict(run=name,steps_per_rank=steps,passed=True))

pairs=[]
for code in ['main','opt']:
    for recipe in ['ac16','ac32','validate_ac16']:
        matches=0
        for rank in range(8):
            a=list(map(json.loads,(ROOT/'runs'/f'{code}_noac16'/f'inputs_rank{rank}.jsonl').read_text().splitlines()))
            other=f'validate_{code}_ac16' if recipe=='validate_ac16' else f'{code}_{recipe}'
            b=list(map(json.loads,(ROOT/'runs'/other/f'inputs_rank{rank}.jsonl').read_text().splitlines()))
            assert len(a)==len(b)==3
            for x,y in zip(a,b):
                assert x['step']==y['step'] and x['tensors']==y['tensors'],(code,recipe,rank,x['step'])
                matches+=len(x['tensors'])
        pairs.append(dict(code=code,a='noac16',b=recipe,tensor_field_matches=matches,updates_times_ranks=24))

hashes=[]
for path in sorted((ROOT/'runs').glob('*/trace_rank*.json.gz')):
    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(ROOT)))
assert len(hashes)==48
(ROOT/'TRACE_SHA256SUMS').write_text('\n'.join(hashes)+'\n')
thermal={}
for result in results:
    path=ROOT/'runs'/result['run']/'smi.csv'
    records=[r for r in csv.reader(path.read_text().splitlines()) if len(r)>=9]
    thermal[result['run']]=sum(any(v.strip()=='Active' for v in r[7:9]) for r in records)
(ROOT/'profile_verification.json').write_text(json.dumps(dict(passed=True,runs=results,input_pairs=pairs,trace_count=48,thermal_slowdown_samples=thermal),indent=2))
print('PASS: six captures, two unprofiled controls, 48 traces and six input comparisons.')
