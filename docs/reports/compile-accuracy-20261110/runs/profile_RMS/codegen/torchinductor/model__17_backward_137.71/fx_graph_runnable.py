
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = '0f6e7cb8-c1f1-41a7-9e08-194e874e5462'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_liomiyi6/0f6e7cb8-c1f1-41a7-9e08-194e874e5462_11eht_uw/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_RMS/codegen'
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

    
    
    def forward(self, tangents_1, tangents_2):
        slice_1 = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, 2560)
        slice_2 = torch.ops.aten.slice.Tensor(tangents_1, 1, 2560, 5120);  tangents_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_2, 1);  slice_2 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [8, 32, 2560]);  unsqueeze = None
        div = torch.ops.aten.div.Scalar(expand, 32);  expand = None
        view_2 = torch.ops.aten.view.default(div, [256, 2560]);  div = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(slice_1, 1);  slice_1 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [8, 32, 2560]);  unsqueeze_1 = None
        div_1 = torch.ops.aten.div.Scalar(expand_1, 32);  expand_1 = None
        view_3 = torch.ops.aten.view.default(div_1, [256, 2560]);  div_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_2, [0]);  tangents_2 = None
        return (sum_1, view_3, view_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 81920, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (8, 5120), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
    buf1 = reader.storage(None, 1310720, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 8, 2560), dtype=torch.bfloat16, is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)