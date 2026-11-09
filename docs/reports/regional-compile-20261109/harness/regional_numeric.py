"""Single-update gradient comparison with deterministic explicit flow inputs; smoke only."""
import hashlib
import json
import os
from pathlib import Path
import torch


def flow_fixture(micro_batches,step,rank):
    if os.environ.get('REGIONAL_NUMERIC','0')!='1':return
    for i,batch in enumerate(micro_batches):
        g=torch.Generator(device='cpu').manual_seed(420000+10000*rank+100*step+i)
        batch['noise']=torch.randn(batch['actions'].shape,generator=g,dtype=batch['state'].dtype)
        # A fixed broad set of times is a numerical fixture, not a training-distribution evaluation.
        batch['time']=torch.linspace(.05,.95,batch['state'].shape[0],dtype=batch['state'].dtype)


def audit_gradients(model,step):
    if step!=1 or os.environ.get('REGIONAL_NUMERIC','0')!='1':return
    rank=int(os.environ['RANK']);root=Path(os.environ['NUMERIC_REFERENCE'])
    root.mkdir(parents=True,exist_ok=True)
    path=root/f'grad_rank{rank}.pt';reference=Path(os.environ['BENCH_OUT']).name=='smoke_A'
    previous={} if reference else torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    saved={};rows=[];sum_diff2=sum_ref2=0.0;max_abs=0.0;finite=True
    for name,p in model.named_parameters():
        if p.grad is None:continue
        grad=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
        value=grad.detach().cpu().contiguous()
        digest=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        row=dict(name=name,shape=list(value.shape),dtype=str(value.dtype),sha256=digest)
        finite=finite and bool(torch.isfinite(value).all())
        if reference:saved[name]=value
        else:
            assert name in previous and previous[name].shape==value.shape,name
            ref=previous[name].float();difference=value.float()-ref
            d2=float(difference.double().square().sum());r2=float(ref.double().square().sum())
            delta=float(difference.abs().max()) if value.numel() else 0.0
            row.update(max_abs=delta,relative_l2=(d2/max(r2,1e-300))**.5,bit_equal=torch.equal(value,previous[name]))
            sum_diff2+=d2;sum_ref2+=r2;max_abs=max(max_abs,delta)
        rows.append(row)
    assert finite,'Nonfinite gradients'
    if reference:torch.save(saved,path)
    out=Path(os.environ['BENCH_OUT'])/f'numeric_rank{rank}.json'
    out.write_text(json.dumps(dict(reference=reference,finite=finite,parameters=len(rows),
        relative_l2=(sum_diff2/max(sum_ref2,1e-300))**.5,max_abs=max_abs,sum_diff2=sum_diff2,sum_ref2=sum_ref2,rows=rows),indent=2))
