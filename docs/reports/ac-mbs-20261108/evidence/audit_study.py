import json
from pathlib import Path

ROOT=Path('/root/experiments/lingbot-ac-mbs-20261108')
runs=ROOT/'runs'
checks=[]
for path in sorted(runs.iterdir()):
    cfg=path/'effective_args.json'
    if not cfg.exists():
        continue
    t=json.loads(cfg.read_text())['train']
    check=dict(run=path.name, micro=t['micro_batch_size'], ga=t['gradient_accumulation_steps'],
               global_batch=t['global_batch_size'], checkpointing=t['enable_gradient_checkpointing'],
               optimizer=t['optimizer'], lr=t['lr'], max_steps=t['max_steps'])
    check['passed']=check['micro']*check['ga']*8==256 and check['global_batch']==256 and check['optimizer']=='dist_muon'
    checks.append(check)

pairs=[]
for code in ['main','opt','alloc_main','alloc_opt']:
    for repeat in [1,2]:
        a=runs/f'{code}_ac32_r{repeat}'
        b=runs/f'{code}_noac16_r{repeat}'
        if not (b/'inputs_rank0.jsonl').exists():
            b=runs/f'smoke_{code}_noac16'
        if not (a/'inputs_rank0.jsonl').exists() or not (b/'inputs_rank0.jsonl').exists():
            continue
        pair=dict(a=a.name,b=b.name,compared_updates=0,compared_tensor_fields=0,differences=[])
        for rank in range(8):
            ap=a/f'inputs_rank{rank}.jsonl';bp=b/f'inputs_rank{rank}.jsonl'
            if not ap.exists() or not bp.exists():
                continue
            ar=list(map(json.loads,ap.read_text().splitlines()));br=list(map(json.loads,bp.read_text().splitlines()))
            for x,y in zip(ar,br):
                pair['compared_updates']+=1
                for k in sorted(set(x['tensors'])&set(y['tensors'])):
                    pair['compared_tensor_fields']+=1
                    if x['tensors'][k]!=y['tensors'][k]:
                        pair['differences'].append(dict(rank=rank,step=x['step'],key=k,a=x['tensors'][k],b=y['tensors'][k]))
        pairs.append(pair)
result=dict(config_checks=checks,input_pairs=pairs)
(ROOT/'consistency_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
