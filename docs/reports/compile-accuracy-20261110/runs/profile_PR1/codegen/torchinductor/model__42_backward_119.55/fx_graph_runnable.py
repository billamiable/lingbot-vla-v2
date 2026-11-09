
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

    
    
    def forward(self, primals_2, convert_element_type, pow_3, clamp_min_1, pow_7, clamp_min_3, tangents_1, tangents_2):
        mul_5 = torch.ops.aten.mul.Tensor(tangents_1, 0.2)
        neg = torch.ops.aten.neg.default(mul_5);  mul_5 = None
        expand_2 = torch.ops.aten.expand.default(neg, [32, 256]);  neg = None
        div_4 = torch.ops.aten.div.Scalar(expand_2, 8192);  expand_2 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(div_4, -1);  div_4 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze, [32, 256, 1024]);  unsqueeze = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(primals_2, torch.float32);  primals_2 = None
        expand_1 = torch.ops.aten.expand.default(clamp_min_1, [32, 256, 1024]);  clamp_min_1 = None
        div_1 = torch.ops.aten.div.Tensor(convert_element_type_1, expand_1);  expand_1 = None
        div_2 = torch.ops.aten.div.Tensor(div_1, clamp_min_3);  div_1 = clamp_min_3 = None
        mul_6 = torch.ops.aten.mul.Tensor(expand_3, div_2);  expand_3 = div_2 = None
        clamp_min = torch.ops.aten.clamp_min.default(pow_3, 1e-06)
        expand = torch.ops.aten.expand.default(clamp_min, [32, 256, 1024]);  clamp_min = None
        div = torch.ops.aten.div.Tensor(convert_element_type, expand)
        clamp_min_2 = torch.ops.aten.clamp_min.default(pow_7, 1e-06)
        div_3 = torch.ops.aten.div.Tensor(div, clamp_min_2)
        div_6 = torch.ops.aten.div.Tensor(div_3, clamp_min_2);  div_3 = None
        neg_1 = torch.ops.aten.neg.default(mul_6)
        mul_7 = torch.ops.aten.mul.Tensor(neg_1, div_6);  neg_1 = div_6 = None
        div_7 = torch.ops.aten.div.Tensor(mul_6, clamp_min_2);  mul_6 = clamp_min_2 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        div_8 = torch.ops.aten.div.Tensor(div, pow_7)
        eq = torch.ops.aten.eq.Scalar(pow_7, 0);  pow_7 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(eq, full_default, div_8);  eq = div_8 = None
        mul_8 = torch.ops.aten.mul.Tensor(sum_6, where);  sum_6 = where = None
        add_1 = torch.ops.aten.add.Tensor(div_7, mul_8);  div_7 = mul_8 = None
        div_10 = torch.ops.aten.div.Tensor(div, expand);  div = None
        neg_2 = torch.ops.aten.neg.default(add_1)
        mul_9 = torch.ops.aten.mul.Tensor(neg_2, div_10);  neg_2 = div_10 = None
        div_11 = torch.ops.aten.div.Tensor(add_1, expand);  add_1 = expand = None
        add_2 = torch.ops.aten.add.Tensor(tangents_2, div_11);  tangents_2 = div_11 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_9, [2], True);  mul_9 = None
        ge = torch.ops.aten.ge.Scalar(pow_3, 1e-06)
        where_1 = torch.ops.aten.where.self(ge, sum_7, full_default);  ge = sum_7 = None
        div_12 = torch.ops.aten.div.Tensor(convert_element_type, pow_3)
        eq_1 = torch.ops.aten.eq.Scalar(pow_3, 0);  pow_3 = None
        where_2 = torch.ops.aten.where.self(eq_1, full_default, div_12);  eq_1 = full_default = div_12 = None
        mul_10 = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, mul_10);  add_2 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(tangents_1, 1.0);  tangents_1 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub, 2.384185791015625e-07);  sub = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, mul_11);  mul_12 = mul_11 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, mul_13);  add_3 = mul_13 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        return (convert_element_type_3, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 256, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32, 256, 1024), is_leaf=True)  # convert_element_type
    buf2 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 256, 1), is_leaf=True)  # pow_3
    buf3 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf3, (32, 256, 1), is_leaf=True)  # clamp_min_1
    buf4 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf4, (32, 256, 1), is_leaf=True)  # pow_7
    buf5 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf5, (32, 256, 1), is_leaf=True)  # clamp_min_3
    buf6 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf6, (), is_leaf=True)  # tangents_1
    buf7 = reader.storage(None, 33554432, device=device(type='cuda', index=0))
    reader.tensor(buf7, (32, 256, 1024), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)