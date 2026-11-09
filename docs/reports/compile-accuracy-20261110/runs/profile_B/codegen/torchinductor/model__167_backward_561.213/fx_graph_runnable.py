
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, tangents_1, tangents_2, tangents_3):
        view_9 = torch.ops.aten.view.default(tangents_3, [32, 286, 1024]);  tangents_3 = None
        view_10 = torch.ops.aten.view.default(view_9, [9152, 1024]);  view_9 = None
        permute_3 = torch.ops.aten.permute.default(view_10, [1, 0])
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(primals_3, convert_element_type_1)
        view = torch.ops.aten.view.default(mul_1, [9152, 2560]);  mul_1 = None
        mm_3 = torch.ops.aten.mm.default(permute_3, view);  permute_3 = None
        permute_2 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_5 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_4 = torch.ops.aten.mm.default(view_10, permute_5);  view_10 = permute_5 = None
        view_11 = torch.ops.aten.view.default(mm_4, [32, 286, 2560]);  mm_4 = None
        mul_6 = torch.ops.aten.mul.Tensor(tangents_2, primals_6);  primals_6 = None
        permute_1 = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        mm_1 = torch.ops.aten.mm.default(view, permute_1)
        view_4 = torch.ops.aten.view.default(mm_1, [32, 286, 1024]);  mm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [32, 286, -1, 128]);  view_4 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_2 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_8, rsqrt_2)
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_7 = torch.ops.aten.mul.Tensor(tangents_2, convert_element_type_9);  tangents_2 = convert_element_type_9 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1, 2], True);  mul_7 = None
        view_12 = torch.ops.aten.view.default(sum_1, [128]);  sum_1 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(mul_6, torch.float32);  mul_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_16, convert_element_type_8)
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_16, rsqrt_2);  convert_element_type_16 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_8, [3], True);  mul_8 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_10 = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, pow_4);  mul_10 = pow_4 = None
        expand = torch.ops.aten.expand.default(mul_11, [32, 286, 8, 128]);  mul_11 = None
        div = torch.ops.aten.div.Scalar(expand, 128);  expand = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_8, 1.0);  convert_element_type_8 = None
        mul_12 = torch.ops.aten.mul.Scalar(pow_5, 2.0);  pow_5 = None
        mul_13 = torch.ops.aten.mul.Tensor(div, mul_12);  div = mul_12 = None
        add_3 = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view_13 = torch.ops.aten.view.default(convert_element_type_17, [32, 286, 1024]);  convert_element_type_17 = None
        view_14 = torch.ops.aten.view.default(view_13, [9152, 1024]);  view_13 = None
        permute_7 = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_7, view);  permute_7 = None
        permute_9 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_6 = torch.ops.aten.mm.default(view_14, permute_9);  view_14 = permute_9 = None
        view_15 = torch.ops.aten.view.default(mm_6, [32, 286, 2560]);  mm_6 = None
        add_4 = torch.ops.aten.add.Tensor(view_11, view_15);  view_11 = view_15 = None
        mul_14 = torch.ops.aten.mul.Tensor(tangents_1, primals_4);  primals_4 = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        mm = torch.ops.aten.mm.default(view, permute)
        view_1 = torch.ops.aten.view.default(mm, [32, 286, 4096]);  mm = None
        view_2 = torch.ops.aten.view.default(view_1, [32, 286, -1, 128]);  view_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_1 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_4, rsqrt_1)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_15, [0, 1, 2], True);  mul_15 = None
        view_16 = torch.ops.aten.view.default(sum_3, [128]);  sum_3 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(mul_14, torch.float32);  mul_14 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_22, convert_element_type_4)
        mul_17 = torch.ops.aten.mul.Tensor(convert_element_type_22, rsqrt_1);  convert_element_type_22 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_16, [3], True);  mul_16 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_18 = torch.ops.aten.mul.Scalar(sum_4, -0.5);  sum_4 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, pow_6);  mul_18 = pow_6 = None
        expand_1 = torch.ops.aten.expand.default(mul_19, [32, 286, 32, 128]);  mul_19 = None
        div_1 = torch.ops.aten.div.Scalar(expand_1, 128);  expand_1 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_4, 1.0);  convert_element_type_4 = None
        mul_20 = torch.ops.aten.mul.Scalar(pow_7, 2.0);  pow_7 = None
        mul_21 = torch.ops.aten.mul.Tensor(div_1, mul_20);  div_1 = mul_20 = None
        add_5 = torch.ops.aten.add.Tensor(mul_17, mul_21);  mul_17 = mul_21 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        view_17 = torch.ops.aten.view.default(convert_element_type_23, [32, 286, 4096]);  convert_element_type_23 = None
        view_18 = torch.ops.aten.view.default(view_17, [9152, 4096]);  view_17 = None
        permute_11 = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_11, view);  permute_11 = view = None
        permute_13 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_8 = torch.ops.aten.mm.default(view_18, permute_13);  view_18 = permute_13 = None
        view_19 = torch.ops.aten.view.default(mm_8, [32, 286, 2560]);  mm_8 = None
        add_6 = torch.ops.aten.add.Tensor(add_4, view_19);  add_4 = view_19 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_6, primals_3);  primals_3 = None
        mul_23 = torch.ops.aten.mul.Tensor(add_6, convert_element_type_1);  add_6 = convert_element_type_1 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_23, [0, 1], True);  mul_23 = None
        view_20 = torch.ops.aten.view.default(sum_5, [2560]);  sum_5 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(mul_22, torch.float32);  mul_22 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_28, convert_element_type)
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_28, rsqrt);  convert_element_type_28 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_24, [2], True);  mul_24 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_26 = torch.ops.aten.mul.Scalar(sum_6, -0.5);  sum_6 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, pow_8);  mul_26 = pow_8 = None
        expand_2 = torch.ops.aten.expand.default(mul_27, [32, 286, 2560]);  mul_27 = None
        div_2 = torch.ops.aten.div.Scalar(expand_2, 2560);  expand_2 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 1.0);  convert_element_type = None
        mul_28 = torch.ops.aten.mul.Scalar(pow_9, 2.0);  pow_9 = None
        mul_29 = torch.ops.aten.mul.Tensor(div_2, mul_28);  div_2 = mul_28 = None
        add_7 = torch.ops.aten.add.Tensor(mul_25, mul_29);  mul_25 = mul_29 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        return (convert_element_type_29, mm_7, view_20, view_16, mm_5, view_12, mm_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 20971520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (4096, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (128,), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 5242880, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (1024, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (128,), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 5242880, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (1024, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 74973184, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (32, 286, 32, 128), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
    buf8 = reader.storage(None, 18743296, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (32, 286, 8, 128), dtype=torch.bfloat16, is_leaf=True)  # tangents_2
    buf9 = reader.storage(None, 18743296, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (32, 286, 8, 128), dtype=torch.bfloat16, is_leaf=True)  # tangents_3
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)