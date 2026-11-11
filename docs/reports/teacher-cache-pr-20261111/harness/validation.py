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


def install_subset(module):
    """Exercise the real builder on complete episodes; keep the production dataset unchanged."""
    if os.environ.get('CACHE_BUILD_SUBSET')!='1':return
    from types import SimpleNamespace
    original=module.build_vla_dataset
    class FirstEpisodes(torch.utils.data.Dataset):
        def __init__(self,base):
            self.base=base;sub=base._datasets[0];episodes=sub.dataset.meta.episodes
            ends=[int(x) for x in episodes['dataset_to_index']]
            k=next((i+1 for i,end in enumerate(ends) if end>=1024),len(ends))
            self.n=ends[k-1];self.return_sample_index=False;self.dataset_start_index=[0]
            selected={key:[int(x) for x in episodes[key][:k]] for key in ['dataset_from_index','dataset_to_index']}
            self._datasets=[SimpleNamespace(chunk_size=sub.chunk_size,dataset=SimpleNamespace(repo_id=sub.dataset.repo_id,num_frames=self.n,meta=SimpleNamespace(episodes=selected)))]
        def __len__(self):return self.n
        def __getitem__(self,index):
            self.base.return_sample_index=self.return_sample_index
            return self.base[index]
    def build(*args,**kwargs):return FirstEpisodes(original(*args,**kwargs))
    module.build_vla_dataset=build


def install_layout_probe(module):
    if os.environ.get('CAPTURE_TARGET_LAYOUT')!='1':return
    original=module.log_teacher_cache_diff
    def wrapped(micro_batch,online,step):
        if step==1:
            def info(x):return dict(shape=list(x.shape),stride=list(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype))
            data={k:dict(online=info(v),cached=info(micro_batch[k])) for k,v in online.items()}
            Path(os.environ['BENCH_OUT'],f'target_layout_rank{os.environ["RANK"]}.json').write_text(json.dumps(data,indent=2))
        return original(micro_batch,online,step)
    module.log_teacher_cache_diff=wrapped
