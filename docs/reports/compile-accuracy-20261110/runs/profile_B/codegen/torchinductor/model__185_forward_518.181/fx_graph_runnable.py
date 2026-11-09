
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

    
    
    def forward(self, primals_1, primals_2):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_2, torch.float32)
        sub = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        mean = torch.ops.aten.mean.default(pow_1);  pow_1 = None
        mul = torch.ops.aten.mul.Tensor(mean, 1.0)
        mul_1 = torch.ops.aten.mul.Tensor(mean, 1.0)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2.0)
        sum_1 = torch.ops.aten.sum.dim_IntList(pow_2, [-1], True);  pow_2 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(pow_3, 1e-06)
        expand = torch.ops.aten.expand.default(clamp_min, [32, 256, 1024]);  clamp_min = None
        div = torch.ops.aten.div.Tensor(convert_element_type, expand);  expand = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2.0)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_4, [-1], True);  pow_4 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(pow_5, 1e-06);  pow_5 = None
        expand_1 = torch.ops.aten.expand.default(clamp_min_1, [32, 256, 1024])
        div_1 = torch.ops.aten.div.Tensor(convert_element_type_1, expand_1);  convert_element_type_1 = expand_1 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(div, 2)
        sum_3 = torch.ops.aten.sum.dim_IntList(pow_6, [-1], True);  pow_6 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(sum_3, 0.5);  sum_3 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(div_1, 2)
        sum_4 = torch.ops.aten.sum.dim_IntList(pow_8, [-1], True);  pow_8 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(pow_7, 1e-06)
        clamp_min_3 = torch.ops.aten.clamp_min.default(pow_9, 1e-06);  pow_9 = None
        div_2 = torch.ops.aten.div.Tensor(div_1, clamp_min_3);  div_1 = None
        div_3 = torch.ops.aten.div.Tensor(div, clamp_min_2);  div = clamp_min_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(div_3, div_2);  div_3 = div_2 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_2, [-1]);  mul_2 = None
        mean_1 = torch.ops.aten.mean.default(sum_5);  sum_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(1.0, mean_1);  mean_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_1, 0.2)
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, 0.2)
        add = torch.ops.aten.add.Tensor(mul_1, mul_4);  mul_1 = mul_4 = None
        return (add, convert_element_type, mean, mul, sub_1, mul_3, primals_2, convert_element_type, pow_3, clamp_min_1, pow_7, clamp_min_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 256, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 256, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)