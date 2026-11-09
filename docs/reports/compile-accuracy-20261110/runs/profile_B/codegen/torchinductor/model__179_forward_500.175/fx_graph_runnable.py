
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_2, torch.float32)
        sub = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1);  convert_element_type_1 = None
        abs_1 = torch.ops.aten.abs.default(sub);  sub = None
        lt = torch.ops.aten.lt.Scalar(abs_1, 1.0)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(abs_1, 2)
        mul = torch.ops.aten.mul.Tensor(pow_1, 0.5);  pow_1 = None
        div = torch.ops.aten.div.Tensor(mul, 1.0);  mul = None
        sub_1 = torch.ops.aten.sub.Tensor(abs_1, 0.5);  abs_1 = None
        where = torch.ops.aten.where.self(lt, div, sub_1);  lt = div = sub_1 = None
        mean = torch.ops.aten.mean.default(where);  where = None
        slice_2 = torch.ops.aten.slice.Tensor(primals_3, 1, -8, 9223372036854775807);  primals_3 = None
        cat = torch.ops.aten.cat.default([primals_4, slice_2], 1);  primals_4 = slice_2 = None
        repeat = torch.ops.aten.repeat.default(primals_5, [32, 1, 1]);  primals_5 = None
        return (cat, repeat, convert_element_type, mean, primals_2, convert_element_type)
        
def load_args(reader):
    buf0 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 256, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 256, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    reader.tensor(buf2, (32, 64, 2560), (732160, 2560, 1), dtype=torch.bfloat16, storage_offset=2560, is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 1310720, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (256, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_5
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)