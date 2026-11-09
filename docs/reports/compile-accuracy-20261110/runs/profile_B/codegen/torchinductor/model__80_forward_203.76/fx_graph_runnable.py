
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
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.fx_graph_cache = False
torch._inductor.config.comprehensive_padding = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_B/codegen'
torch._inductor.config.test_configs.runtime_triton_dtype_assert = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True



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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13):
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(primals_1, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(primals_1, rsqrt);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(primals_4, mul);  mul = None
        permute = torch.ops.aten.permute.default(primals_5, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_6, primals_2, permute);  permute = None
        unsqueeze = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_8, primals_2, permute_1);  permute_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        add_1 = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2 = torch.ops.aten.mul.Tensor(add_1, mul_1);  add_1 = mul_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        view = torch.ops.aten.view.default(add_2, [1632, 768]);  add_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_3, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_9, view, permute_2);  primals_9 = permute_2 = None
        view_1 = torch.ops.aten.view.default(addmm_2, [32, 51, 4096]);  addmm_2 = None
        view_2 = torch.ops.aten.view.default(view_1, [32, 51, -1, 128]);  view_1 = None
        permute_3 = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_11, view, permute_3);  primals_11 = permute_3 = None
        view_4 = torch.ops.aten.view.default(addmm_3, [32, 51, 1024]);  addmm_3 = None
        view_5 = torch.ops.aten.view.default(view_4, [32, 51, -1, 128]);  view_4 = None
        permute_4 = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_4 = torch.ops.aten.addmm.default(primals_13, view, permute_4);  primals_13 = view = permute_4 = None
        view_7 = torch.ops.aten.view.default(addmm_4, [32, 51, 1024]);  addmm_4 = None
        view_8 = torch.ops.aten.view.default(view_7, [32, 51, -1, 128]);  view_7 = None
        return (view_2, view_5, view_8, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_10, primals_12)
        
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
    buf8 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf8, (4096,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024, 768), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1024,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1024, 768), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1024,), is_leaf=True)  # primals_13
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)