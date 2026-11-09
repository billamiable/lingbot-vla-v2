
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = 'e120077a-9255-47ec-9d26-5d0c985de943'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_wz74sf15/e120077a-9255-47ec-9d26-5d0c985de943_iswpxo4r/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_A/codegen'
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

    
    
    def forward(self, tangents_1):
        slice_1 = torch.ops.aten.slice.Tensor(tangents_1, 0, 0, 1632)
        slice_2 = torch.ops.aten.slice.Tensor(tangents_1, 0, 1632, 3264)
        slice_3 = torch.ops.aten.slice.Tensor(tangents_1, 0, 3264, 4896)
        slice_4 = torch.ops.aten.slice.Tensor(tangents_1, 0, 4896, 6528)
        slice_5 = torch.ops.aten.slice.Tensor(tangents_1, 0, 6528, 8160)
        slice_6 = torch.ops.aten.slice.Tensor(tangents_1, 0, 8160, 9792)
        slice_7 = torch.ops.aten.slice.Tensor(tangents_1, 0, 9792, 11424)
        slice_8 = torch.ops.aten.slice.Tensor(tangents_1, 0, 11424, 13056)
        slice_9 = torch.ops.aten.slice.Tensor(tangents_1, 0, 13056, 14688)
        slice_10 = torch.ops.aten.slice.Tensor(tangents_1, 0, 14688, 16320)
        slice_11 = torch.ops.aten.slice.Tensor(tangents_1, 0, 16320, 17952)
        slice_12 = torch.ops.aten.slice.Tensor(tangents_1, 0, 17952, 19584)
        slice_13 = torch.ops.aten.slice.Tensor(tangents_1, 0, 19584, 21216)
        slice_14 = torch.ops.aten.slice.Tensor(tangents_1, 0, 21216, 22848)
        slice_15 = torch.ops.aten.slice.Tensor(tangents_1, 0, 22848, 24480)
        slice_16 = torch.ops.aten.slice.Tensor(tangents_1, 0, 24480, 26112)
        slice_17 = torch.ops.aten.slice.Tensor(tangents_1, 0, 26112, 27744)
        slice_18 = torch.ops.aten.slice.Tensor(tangents_1, 0, 27744, 29376)
        slice_19 = torch.ops.aten.slice.Tensor(tangents_1, 0, 29376, 31008)
        slice_20 = torch.ops.aten.slice.Tensor(tangents_1, 0, 31008, 32640)
        slice_21 = torch.ops.aten.slice.Tensor(tangents_1, 0, 32640, 34272)
        slice_22 = torch.ops.aten.slice.Tensor(tangents_1, 0, 34272, 35904)
        slice_23 = torch.ops.aten.slice.Tensor(tangents_1, 0, 35904, 37536)
        slice_24 = torch.ops.aten.slice.Tensor(tangents_1, 0, 37536, 39168)
        slice_25 = torch.ops.aten.slice.Tensor(tangents_1, 0, 39168, 40800)
        slice_26 = torch.ops.aten.slice.Tensor(tangents_1, 0, 40800, 42432)
        slice_27 = torch.ops.aten.slice.Tensor(tangents_1, 0, 42432, 44064)
        slice_28 = torch.ops.aten.slice.Tensor(tangents_1, 0, 44064, 45696)
        slice_29 = torch.ops.aten.slice.Tensor(tangents_1, 0, 45696, 47328)
        slice_30 = torch.ops.aten.slice.Tensor(tangents_1, 0, 47328, 48960)
        slice_31 = torch.ops.aten.slice.Tensor(tangents_1, 0, 48960, 50592)
        slice_32 = torch.ops.aten.slice.Tensor(tangents_1, 0, 50592, 52224)
        slice_33 = torch.ops.aten.slice.Tensor(tangents_1, 0, 52224, 53856)
        slice_34 = torch.ops.aten.slice.Tensor(tangents_1, 0, 53856, 55488)
        slice_35 = torch.ops.aten.slice.Tensor(tangents_1, 0, 55488, 57120)
        slice_36 = torch.ops.aten.slice.Tensor(tangents_1, 0, 57120, 58752);  tangents_1 = None
        return (slice_1, slice_2, slice_3, slice_4, slice_5, slice_6, slice_7, slice_8, slice_9, slice_10, slice_11, slice_12, slice_13, slice_14, slice_15, slice_16, slice_17, slice_18, slice_19, slice_20, slice_21, slice_22, slice_23, slice_24, slice_25, slice_26, slice_27, slice_28, slice_29, slice_30, slice_31, slice_32, slice_33, slice_34, slice_35, slice_36)
        
def load_args(reader):
    buf0 = reader.storage(None, 7520256, device=device(type='cuda', index=0))
    reader.tensor(buf0, (58752, 32), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)