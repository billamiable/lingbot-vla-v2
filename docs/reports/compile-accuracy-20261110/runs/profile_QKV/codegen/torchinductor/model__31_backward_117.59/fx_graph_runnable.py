
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = '6ae81076-c770-4920-a34f-4682d36b85bc'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_tojjk6_z/6ae81076-c770-4920-a34f-4682d36b85bc_dnf90thj/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_QKV/codegen'
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

    
    
    def forward(self, primals_1, primals_2, rsqrt, tangents_1):
        mul_2 = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  primals_2 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_3 = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_1);  tangents_1 = convert_element_type_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_3, [0, 1], True);  mul_3 = None
        view = torch.ops.aten.view.default(sum_1, [2560]);  sum_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul_2, torch.float32);  mul_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_2, convert_element_type)
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_6 = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, pow_2);  mul_6 = pow_2 = None
        expand = torch.ops.aten.expand.default(mul_7, [32, 286, 2560]);  mul_7 = None
        div = torch.ops.aten.div.Scalar(expand, 2560);  expand = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 1.0);  convert_element_type = None
        mul_8 = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_9 = torch.ops.aten.mul.Tensor(div, mul_8);  div = mul_8 = None
        add_1 = torch.ops.aten.add.Tensor(mul_5, mul_9);  mul_5 = mul_9 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        return (convert_element_type_3, view)
        
def load_args(reader):
    buf0 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 36608, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 286, 1), is_leaf=True)  # rsqrt
    buf3 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)