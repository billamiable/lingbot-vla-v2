
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36):
        cat = torch.ops.aten.cat.default([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36]);  primals_1 = primals_2 = primals_3 = primals_4 = primals_5 = primals_6 = primals_7 = primals_8 = primals_9 = primals_10 = primals_11 = primals_12 = primals_13 = primals_14 = primals_15 = primals_16 = primals_17 = primals_18 = primals_19 = primals_20 = primals_21 = primals_22 = primals_23 = primals_24 = primals_25 = primals_26 = primals_27 = primals_28 = primals_29 = primals_30 = primals_31 = primals_32 = primals_33 = primals_34 = primals_35 = primals_36 = None
        return (cat,)
        
def load_args(reader):
    buf0 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1632, 32), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1632, 32), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1632, 32), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1632, 32), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1632, 32), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1632, 32), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1632, 32), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1632, 32), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1632, 32), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1632, 32), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1632, 32), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1632, 32), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1632, 32), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1632, 32), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1632, 32), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1632, 32), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1632, 32), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1632, 32), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1632, 32), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1632, 32), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1632, 32), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1632, 32), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1632, 32), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1632, 32), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1632, 32), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1632, 32), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1632, 32), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1632, 32), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1632, 32), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1632, 32), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1632, 32), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1632, 32), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1632, 32), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1632, 32), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1632, 32), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1632, 32), is_leaf=True)  # primals_36
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)