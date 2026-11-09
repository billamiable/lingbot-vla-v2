
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = 'aea99a3a-f9a9-4741-9a21-1504d118bbbc'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_yxtyoq69/aea99a3a-f9a9-4741-9a21-1504d118bbbc_dfd7q7gd/attempt_0/0/error.json'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.accumulated_recompile_limit = 4096
torch._inductor.config.fx_graph_cache = False
torch._inductor.config.comprehensive_padding = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_B/codegen'
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = False



isolate_fails_code_str = None




# torch version: 2.8.0+cu128
# torch cuda version: 12.8
# torch git version: a1cb3cc05d46d198467bebbb6e8fba50a325d4e7


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA Graphics Device : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_12, tangents_1, tangents_2, tangents_3):
        view_9 = torch.ops.aten.view.default(tangents_3, [32, 51, 1024]);  tangents_3 = None
        view_10 = torch.ops.aten.view.default(view_9, [1632, 1024]);  view_9 = None
        permute_4 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_5 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm = torch.ops.aten.mm.default(view_10, permute_5);  permute_5 = None
        permute_6 = torch.ops.aten.permute.default(view_10, [1, 0])
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(primals_1, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(primals_1, rsqrt)
        mul_1 = torch.ops.aten.mul.Tensor(primals_4, mul)
        permute = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm = torch.ops.aten.addmm.default(primals_6, primals_2, permute);  primals_6 = permute = None
        unsqueeze = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_8, primals_2, permute_1);  primals_8 = permute_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        add_1 = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2 = torch.ops.aten.mul.Tensor(add_1, mul_1)
        add_2 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        view = torch.ops.aten.view.default(add_2, [1632, 768]);  add_2 = None
        mm_1 = torch.ops.aten.mm.default(permute_6, view);  permute_6 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(view_10, [0], True);  view_10 = None
        view_11 = torch.ops.aten.view.default(sum_1, [1024]);  sum_1 = None
        view_12 = torch.ops.aten.view.default(mm, [32, 51, 768]);  mm = None
        view_13 = torch.ops.aten.view.default(tangents_2, [32, 51, 1024]);  tangents_2 = None
        view_14 = torch.ops.aten.view.default(view_13, [1632, 1024]);  view_13 = None
        permute_3 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_9 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_2 = torch.ops.aten.mm.default(view_14, permute_9);  permute_9 = None
        permute_10 = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_10, view);  permute_10 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(view_14, [0], True);  view_14 = None
        view_15 = torch.ops.aten.view.default(sum_2, [1024]);  sum_2 = None
        view_16 = torch.ops.aten.view.default(mm_2, [32, 51, 768]);  mm_2 = None
        add_3 = torch.ops.aten.add.Tensor(view_12, view_16);  view_12 = view_16 = None
        view_17 = torch.ops.aten.view.default(tangents_1, [32, 51, 4096]);  tangents_1 = None
        view_18 = torch.ops.aten.view.default(view_17, [1632, 4096]);  view_17 = None
        permute_2 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        permute_13 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_4 = torch.ops.aten.mm.default(view_18, permute_13);  permute_13 = None
        permute_14 = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_14, view);  permute_14 = view = None
        sum_3 = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_19 = torch.ops.aten.view.default(sum_3, [4096]);  sum_3 = None
        view_20 = torch.ops.aten.view.default(mm_4, [32, 51, 768]);  mm_4 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, view_20);  add_3 = view_20 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(add_4, [1], True)
        mul_3 = torch.ops.aten.mul.Tensor(add_4, add_1);  add_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(add_4, mul_1);  add_4 = mul_1 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        squeeze = torch.ops.aten.squeeze.dim(sum_4, 1);  sum_4 = None
        permute_17 = torch.ops.aten.permute.default(squeeze, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_17, primals_2);  permute_17 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(squeeze, [0], True);  squeeze = None
        view_21 = torch.ops.aten.view.default(sum_6, [768]);  sum_6 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_5, 1);  sum_5 = None
        permute_20 = torch.ops.aten.permute.default(squeeze_1, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_20, primals_2);  permute_20 = primals_2 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(squeeze_1, [0], True);  squeeze_1 = None
        view_22 = torch.ops.aten.view.default(sum_7, [768]);  sum_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_3, primals_4);  primals_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_3, mul);  mul_3 = mul = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1], True);  mul_6 = None
        view_23 = torch.ops.aten.view.default(sum_8, [768]);  sum_8 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_5, primals_1)
        mul_8 = torch.ops.aten.mul.Tensor(mul_5, rsqrt);  mul_5 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_9 = torch.ops.aten.mul.Scalar(sum_9, -0.5);  sum_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, pow_2);  mul_9 = pow_2 = None
        expand = torch.ops.aten.expand.default(mul_10, [32, 51, 768]);  mul_10 = None
        div = torch.ops.aten.div.Scalar(expand, 768);  expand = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(primals_1, 1.0);  primals_1 = None
        mul_11 = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_12 = torch.ops.aten.mul.Tensor(div, mul_11);  div = mul_11 = None
        add_5 = torch.ops.aten.add.Tensor(mul_8, mul_12);  mul_8 = mul_12 = None
        return (add_5, None, mm_5, view_23, mm_7, view_22, mm_6, view_21, view_19, mm_3, view_15, mm_1, view_11)
        
def load_args(reader):
    buf0 = reader.storage(None, 5013504, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 51, 768), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32, 768), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf2, (4096, 768), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768,), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768, 768), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768,), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768, 768), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1024, 768), is_leaf=True)  # primals_10
    buf9 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024, 768), is_leaf=True)  # primals_12
    buf10 = reader.storage(None, 26738688, device=device(type='cuda', index=0))
    reader.tensor(buf10, (32, 51, 32, 128), is_leaf=True)  # tangents_1
    buf11 = reader.storage(None, 6684672, device=device(type='cuda', index=0))
    reader.tensor(buf11, (32, 51, 8, 128), is_leaf=True)  # tangents_2
    buf12 = reader.storage(None, 6684672, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32, 51, 8, 128), is_leaf=True)  # tangents_3
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)