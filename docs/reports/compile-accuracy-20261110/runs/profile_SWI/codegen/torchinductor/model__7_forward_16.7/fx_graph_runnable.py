
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_3, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [1], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_1);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view = torch.ops.aten.view.default(convert_element_type_1, [-1, 4096]);  convert_element_type_1 = None
        permute = torch.ops.aten.permute.default(primals_5, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_6, view, permute);  primals_6 = permute = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(addmm, torch.float32)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.5)
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476);  convert_element_type_5 = None
        erf = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_2, add_2);  mul_2 = add_2 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_8, convert_element_type_6, permute_1);  primals_8 = permute_1 = None
        return (addmm_1, primals_1, primals_3, primals_5, primals_7, getitem_1, rsqrt, view, addmm, convert_element_type_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    reader.symint(1024)  # primals_4
    buf3 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (4096, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf4 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (4096,), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf5 = reader.storage(None, 20971520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (2560, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf6 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_8
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)