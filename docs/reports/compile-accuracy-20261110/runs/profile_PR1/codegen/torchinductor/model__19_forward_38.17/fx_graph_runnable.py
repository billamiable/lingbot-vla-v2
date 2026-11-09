
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = '7e822c46-d4c0-43c7-89eb-eaf111a6d924'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_87kt43ql/7e822c46-d4c0-43c7-89eb-eaf111a6d924_utzpe_1w/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_PR1/codegen'
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12):
        repeat = torch.ops.aten.repeat.default(primals_1, [32, 1, 1]);  primals_1 = None
        full_default = torch.ops.aten.full.default([32, 72], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([32, 8], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2 = torch.ops.aten.full.default([32, 8], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat = torch.ops.aten.cat.default([primals_2, primals_4, primals_6, repeat], 1);  primals_2 = primals_4 = primals_6 = None
        cat_1 = torch.ops.aten.cat.default([primals_8, primals_5, primals_7, primals_7], 1);  primals_8 = primals_5 = primals_7 = None
        cat_2 = torch.ops.aten.cat.default([primals_10, primals_3, primals_9, primals_9], 1);  primals_10 = primals_9 = None
        cat_3 = torch.ops.aten.cat.default([primals_11, full_default, full_default_1, full_default_2], 1);  primals_11 = None
        full_default_3 = torch.ops.aten.full.default([32, 286], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view = torch.ops.aten.view.default(primals_12, [96]);  primals_12 = None
        return (view, cat, cat_1, cat_2, cat_3, full_default_3, repeat, primals_3, full_default, full_default_1, full_default_2)
        
def load_args(reader):
    buf0 = reader.storage(None, 40960, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (8, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 32440320, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 198, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 18432, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (32, 72), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 11796480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (32, 72, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 2304, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf4, (32, 72), dtype=torch.bool, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1310720, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (32, 8, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf6, (32, 8), dtype=torch.bool, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 6336, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf7, (32, 198), dtype=torch.bool, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf8, (32, 8), dtype=torch.int64, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 50688, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf9, (32, 198), dtype=torch.int64, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 6336, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf10, (32, 198), dtype=torch.bool, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 96, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf11, (32, 3), dtype=torch.bool, is_leaf=True)  # primals_12
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)