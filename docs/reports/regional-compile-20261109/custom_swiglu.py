# Graph-compatible, opaque boundary for the existing exact Triton kernels.
# Forward/backward kernels remain unchanged; Inductor does not reserialize nested JIT helpers.
@torch.library.custom_op("lingbot_regional::swiglu_fwd", mutates_args=())
def _regional_swiglu_fwd(g: torch.Tensor, u: torch.Tensor, exact: bool) -> torch.Tensor:
    return (_fwd_triton if _use_triton(g) else _fwd_torch)(g.contiguous(),u.contiguous(),exact)

@_regional_swiglu_fwd.register_fake
def _regional_swiglu_fake(g,u,exact):
    return torch.empty_like(g)

@torch.library.custom_op("lingbot_regional::swiglu_bwd", mutates_args=())
def _regional_swiglu_bwd(do: torch.Tensor, g: torch.Tensor, u: torch.Tensor, exact: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return (_bwd_triton if _use_triton(g) else _bwd_torch)(do.contiguous(),g.contiguous(),u.contiguous(),exact)

@_regional_swiglu_bwd.register_fake
def _regional_swiglu_bwd_fake(do,g,u,exact):
    return torch.empty_like(g),torch.empty_like(u)

def _regional_setup(ctx,inputs,output):
    g,u,exact=inputs
    ctx.save_for_backward(g,u);ctx.exact=exact

def _regional_backward(ctx,do):
    g,u=ctx.saved_tensors
    dg,du=_regional_swiglu_bwd(do,g,u,ctx.exact)
    return dg,du,None

_regional_swiglu_fwd.register_autograd(_regional_backward,setup_context=_regional_setup)

def swiglu(gate: torch.Tensor, up: torch.Tensor, exact: bool = True) -> torch.Tensor:
    return _regional_swiglu_fwd(gate,up,exact)
