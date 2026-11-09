
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = 'f0a6b136-6d66-493e-b30c-6ea62e8f3f48'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_9r_mlamj/f0a6b136-6d66-493e-b30c-6ea62e8f3f48_jbm358k5/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_SWI/codegen'
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
        mul = torch.ops.aten.mul.Tensor(primals_1, primals_7);  primals_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(primals_7, [1, 2])
        clamp_min = torch.ops.aten.clamp_min.default(sum_1, 1);  sum_1 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul, [1, 2])
        div = torch.ops.aten.div.Tensor(sum_2, clamp_min);  sum_2 = clamp_min = None
        sum_3 = torch.ops.aten.sum.default(mul);  mul = None
        sum_4 = torch.ops.aten.sum.default(primals_7)
        clamp_min_1 = torch.ops.aten.clamp_min.default(sum_4, 1);  sum_4 = None
        div_1 = torch.ops.aten.div.Tensor(sum_3, clamp_min_1);  sum_3 = None
        add = torch.ops.aten.add.Tensor(div_1, primals_2);  primals_2 = None
        add_1 = torch.ops.aten.add.Tensor(add, primals_3);  add = primals_3 = None
        add_2 = torch.ops.aten.add.Tensor(add_1, primals_4);  add_1 = primals_4 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, primals_5);  add_2 = primals_5 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, primals_6);  add_3 = None
        return (add_4, div_1, div, primals_6, primals_7, clamp_min_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 176000, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 50, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf1, (), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf2, (), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf3, (), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf4, (), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf5, (), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 88000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf6, (32, 50, 55), dtype=torch.bool, is_leaf=True)  # primals_7
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)