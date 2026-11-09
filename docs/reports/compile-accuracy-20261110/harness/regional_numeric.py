"""Aligned original-distribution flow inputs, first-update gradients/parameters, sampled layer boundaries."""
import hashlib,json,os
from pathlib import Path
import torch
STEP=0

def set_step(step):
    global STEP
    STEP=step

def flow_fixture(batches,step,rank):
    if os.environ.get('STUDY_ALIGNED_FLOW')!='1':return
    for i,b in enumerate(batches):
        seed=202611100000+rank*1000000+step*16+i
        # Isolate CPU RNG; GPU/global model RNG state is untouched.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            b['noise']=torch.randn(b['actions'].shape,dtype=b['state'].dtype)
            t=torch.distributions.Beta(torch.tensor(1.5),torch.tensor(1.0)).sample((b['state'].shape[0],))
            b['time']=(t*.999+.001).to(b['state'].dtype)

def local(t):return t.to_local() if hasattr(t,'to_local') else t

def compare(value,ref):
    assert value.shape==ref.shape and value.dtype==ref.dtype,(value.shape,ref.shape,value.dtype,ref.dtype)
    d=value.float()-ref.float();d2=float(d.double().square().sum());r2=float(ref.double().square().sum())
    return dict(bit_equal=torch.equal(value,ref),max_abs=float(d.abs().max()) if d.numel() else 0.,relative_l2=(d2/max(r2,1e-300))**.5,sum_diff2=d2,sum_ref2=r2)

def audit(model,step,kind):
    if step!=1 or os.environ.get('STUDY_NUMERIC')!='1':return
    rank=int(os.environ['RANK']);root=Path(os.environ['NUMERIC_REFERENCE']);root.mkdir(parents=True,exist_ok=True)
    reference=os.environ.get('STUDY_REFERENCE')=='1'
    fn=root/f'{kind}_rank{rank}.pt'
    refs={} if reference else torch.load(fn,map_location='cpu',mmap=True,weights_only=True)
    saved={};rows=[];finite=True
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        t=p.grad if kind=='grad' else p
        if t is None:continue
        v=local(t).detach().cpu().contiguous()
        finite=finite and bool(torch.isfinite(v).all())
        row=dict(name=name,shape=list(v.shape),dtype=str(v.dtype))
        if reference:saved[name]=v
        else:
            assert name in refs,name
            row.update(compare(v,refs[name]))
        rows.append(row)
    assert finite,(kind,'Nonfinite')
    if reference:torch.save(saved,fn)
    else:assert set(refs)==set(x['name'] for x in rows),kind
    out=dict(reference=reference,finite=finite,parameters=len(rows),rows=rows,
             sum_diff2=sum(x.get('sum_diff2',0) for x in rows),sum_ref2=sum(x.get('sum_ref2',0) for x in rows))
    Path(os.environ['BENCH_OUT'],f'{kind}_rank{rank}.json').write_text(json.dumps(out,indent=2))

def audit_gradients(model,step):audit(model,step,'grad')
def audit_parameters(model,step):audit(model,step,'parameter')

def boundary(label,values,register_grad=False):
    if STEP!=1 or os.environ.get('STUDY_NUMERIC')!='1' or os.environ.get('RANK')!='0':return
    if int(label.split('.')[1]) not in [0,1,17,35]:return
    reference=os.environ.get('STUDY_REFERENCE')=='1'
    root=Path(os.environ['NUMERIC_REFERENCE'])/'boundaries';root.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['BENCH_OUT'])/'boundaries.jsonl'
    def save(name,value):
        v=local(value).detach().cpu().contiguous();p=root/(name+'.pt')
        row=dict(name=name,shape=list(v.shape),dtype=str(v.dtype),finite=bool(torch.isfinite(v).all()))
        if reference:torch.save(v,p)
        else:
            assert p.exists(),name
            row.update(compare(v,torch.load(p,map_location='cpu',mmap=True,weights_only=True)))
        with out.open('a') as f:f.write(json.dumps(row)+'\n')
    def visit(x,name):
        if isinstance(x,torch.Tensor):
            save(name,x)
            if register_grad and x.requires_grad:
                x.register_hook(lambda grad,n=name:save(n+'.grad',grad))
        elif isinstance(x,(tuple,list)):
            for i,v in enumerate(x):visit(v,name+'.'+str(i))
    visit(values,label)
