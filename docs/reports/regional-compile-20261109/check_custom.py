from pathlib import Path
import sys,json,torch
r=Path('/root/experiments/lingbot-regional-20261109');sys.path.insert(0,str(r/'src'))
from lingbotvla.ops.swiglu import swiglu,_SwiGLU
from torch.utils.checkpoint import checkpoint
assert not __import__('subprocess').check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
torch.manual_seed(42)
g=torch.randn(4,9728,device='cuda',dtype=torch.bfloat16,requires_grad=True)
u=torch.randn_like(g,requires_grad=True);d=torch.randn_like(g)
y=_SwiGLU.apply(g,u,True);ref=torch.autograd.grad(y,(g,u),d)
checks=[]
for name,f in [('custom',swiglu),('compiled_checkpoint',torch.compile(lambda a,b:checkpoint(swiglu,a,b,use_reentrant=False),fullgraph=True))]:
 z=f(g,u);grads=torch.autograd.grad(z,(g,u),d)
 row=dict(name=name,output_equal=torch.equal(y,z),grad_gate_equal=torch.equal(ref[0],grads[0]),grad_up_equal=torch.equal(ref[1],grads[1]))
 assert all(v for k,v in row.items() if k!='name'),row
 checks.append(row)
(r/'custom_swiglu_check.json').write_text(json.dumps(checks,indent=2));print(checks)
