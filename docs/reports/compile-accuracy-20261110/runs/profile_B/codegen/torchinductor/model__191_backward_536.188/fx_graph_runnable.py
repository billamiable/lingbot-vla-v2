
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

    
    
    def forward(self, primals_7, clamp_min_1, tangents_1, tangents_2):
        convert_element_type = torch.ops.prims.convert_element_type.default(tangents_1, torch.bfloat16)
        add_5 = torch.ops.aten.add.Tensor(tangents_2, convert_element_type);  tangents_2 = convert_element_type = None
        div_2 = torch.ops.aten.div.Tensor(add_5, clamp_min_1);  add_5 = clamp_min_1 = None
        expand = torch.ops.aten.expand.default(div_2, [32, 50, 55]);  div_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(expand, primals_7);  expand = primals_7 = None
        return (mul_1, tangents_1, tangents_1, tangents_1, tangents_1, tangents_1, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 88000, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf0, (32, 50, 55), dtype=torch.bool, is_leaf=True)  # primals_7
    buf1 = reader.storage(None, 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (), dtype=torch.int64, is_leaf=True)  # clamp_min_1
    buf2 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf2, (), is_leaf=True)  # tangents_1
    buf3 = reader.storage(None, 2, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (), dtype=torch.bfloat16, is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)