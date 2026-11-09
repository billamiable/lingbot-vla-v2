
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
        add = torch.ops.aten.add.Tensor(primals_2, primals_1)
        mul = torch.ops.aten.mul.Tensor(add, 0.004)
        mul_1 = torch.ops.aten.mul.Tensor(primals_1, 0.004)
        mul_2 = torch.ops.aten.mul.Tensor(primals_2, 0.004)
        add_1 = primals_3 + 1;  primals_3 = None
        slice_2 = torch.ops.aten.slice.Tensor(primals_4, 1, -50, 9223372036854775807);  primals_4 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None
        view = torch.ops.aten.view.default(convert_element_type, [1600, 768]);  convert_element_type = None
        permute = torch.ops.aten.permute.default(primals_5, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [32, 50, 55]);  addmm = None
        sub = torch.ops.aten.sub.Tensor(primals_7, view_1);  primals_7 = view_1 = None
        abs_1 = torch.ops.aten.abs.default(sub)
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        sign = torch.ops.aten.sign.default(convert_element_type_4);  convert_element_type_4 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(sign, torch.bfloat16);  sign = None
        return (abs_1, mul, primals_1, add_1, mul_1, primals_2, mul_2, add, mul, primals_5, view, convert_element_type_5)
        
def load_args(reader):
    buf0 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf0, (), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf1, (), is_leaf=True)  # primals_2
    reader.symint(1)  # primals_3
    buf2 = reader.storage(None, 5013504, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 51, 768), is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 84480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (55, 768), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf4 = reader.storage(None, 110, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (55,), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf5 = reader.storage(None, 176000, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (32, 50, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_7
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)