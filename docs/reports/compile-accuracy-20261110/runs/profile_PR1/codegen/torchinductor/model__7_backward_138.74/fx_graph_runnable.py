
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

    
    
    def forward(self, primals_1, primals_3, primals_5, primals_7, getitem_1, rsqrt, view, addmm, convert_element_type_6, tangents_1):
        permute_1 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_2 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_3, convert_element_type_6);  permute_3 = convert_element_type_6 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1 = torch.ops.aten.view.default(sum_1, [2560]);  sum_1 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_5, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_3);  mul_3 = None
        add_2 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(add_2, 0.5);  add_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_5, convert_element_type_5)
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, -0.5);  mul_7 = None
        exp = torch.ops.aten.exp.default(mul_8);  mul_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_10 = torch.ops.aten.mul.Tensor(convert_element_type_5, mul_9);  convert_element_type_5 = mul_9 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, mul_10);  mul_6 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_14, add_4);  convert_element_type_14 = add_4 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        permute = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_6 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_2 = torch.ops.aten.mm.default(convert_element_type_16, permute_6);  permute_6 = None
        permute_7 = torch.ops.aten.permute.default(convert_element_type_16, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_16, [0], True);  convert_element_type_16 = None
        view_2 = torch.ops.aten.view.default(sum_2, [4096]);  sum_2 = None
        view_3 = torch.ops.aten.view.default(mm_2, [24576, 1024]);  mm_2 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_21, convert_element_type_23);  convert_element_type_23 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, 1024)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_13, [1], True)
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_3, torch.float32);  primals_3 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_13, mul);  mul_13 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_15, [1], True);  mul_15 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul, sum_4);  sum_4 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul_14, sum_3);  mul_14 = sum_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(sub_2, mul_16);  sub_2 = mul_16 = None
        div = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_17 = torch.ops.aten.mul.Tensor(div, sub_3);  div = sub_3 = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_21, mul);  mul = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_18, [0]);  mul_18 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(convert_element_type_21, [0]);  convert_element_type_21 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16);  mul_17 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        return (convert_element_type_26, convert_element_type_27, convert_element_type_25, None, mm_3, view_2, mm_1, view_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 33554432, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (4096, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf3 = reader.storage(None, 20971520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (2560, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf4 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf4, (24576, 1), is_leaf=True)  # getitem_1
    buf5 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf5, (24576, 1), is_leaf=True)  # rsqrt
    buf6 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (6144, 4096), dtype=torch.bfloat16, is_leaf=True)  # view
    buf7 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (6144, 4096), dtype=torch.bfloat16, is_leaf=True)  # addmm
    buf8 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (6144, 4096), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_6
    buf9 = reader.storage(None, 31457280, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (6144, 2560), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)