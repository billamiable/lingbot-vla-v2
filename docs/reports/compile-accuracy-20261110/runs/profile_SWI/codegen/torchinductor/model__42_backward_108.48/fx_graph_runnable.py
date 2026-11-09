
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

    
    
    def forward(self, primals_4, view, convert_element_type_5, tangents_1, tangents_2):
        mul_3 = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        neg = torch.ops.aten.neg.default(mul_3);  mul_3 = None
        view_2 = torch.ops.aten.view.default(neg, [1600, 55]);  neg = None
        permute = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_1 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm = torch.ops.aten.mm.default(view_2, permute_1);  permute_1 = None
        permute_2 = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1 = torch.ops.aten.sum.dim_IntList(view_2, [0], True);  view_2 = None
        view_3 = torch.ops.aten.view.default(sum_1, [55]);  sum_1 = None
        view_4 = torch.ops.aten.view.default(mm, [32, 50, 768]);  mm = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        full_default = torch.ops.aten.full.default([32, 51, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(full_default, convert_element_type_10, 1, -50, 9223372036854775807);  full_default = convert_element_type_10 = None
        mul_4 = torch.ops.aten.mul.Tensor(tangents_2, 0.004);  tangents_2 = None
        return (mul_4, mul_4, slice_scatter, mm_1, view_3, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 84480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (55, 768), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 2457600, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1600, 768), dtype=torch.bfloat16, is_leaf=True)  # view
    buf2 = reader.storage(None, 176124, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32, 50, 55), (2752, 55, 1), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_5
    buf3 = reader.storage(None, 176000, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (32, 50, 55), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
    buf4 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf4, (), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)