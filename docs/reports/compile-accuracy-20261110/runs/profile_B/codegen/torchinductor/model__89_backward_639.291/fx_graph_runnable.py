
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, tangents_1):
        view_12 = torch.ops.aten.view.default(tangents_1, [9152, 2560])
        permute_4 = torch.ops.aten.permute.default(view_12, [1, 0])
        slice_2 = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 286);  primals_2 = None
        permute = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        clone = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view = torch.ops.aten.view.default(clone, [9152, 4096]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute)
        view_1 = torch.ops.aten.view.default(mm, [32, 286, 2560]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(primals_4, convert_element_type_3)
        permute_1 = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        view_4 = torch.ops.aten.view.default(mul_1, [9152, 2560]);  mul_1 = None
        mm_1 = torch.ops.aten.mm.default(view_4, permute_1)
        view_5 = torch.ops.aten.view.default(mm_1, [32, 286, 9728]);  mm_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_5, torch.float32)
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_6)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_6, sigmoid);  convert_element_type_6 = sigmoid = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        mm_2 = torch.ops.aten.mm.default(view_4, permute_2)
        view_7 = torch.ops.aten.view.default(mm_2, [32, 286, 9728]);  mm_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_7, view_7)
        view_8 = torch.ops.aten.view.default(mul_3, [9152, 9728]);  mul_3 = None
        mm_4 = torch.ops.aten.mm.default(permute_4, view_8);  permute_4 = view_8 = None
        permute_3 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_6 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_5 = torch.ops.aten.mm.default(view_12, permute_6);  view_12 = permute_6 = None
        view_13 = torch.ops.aten.view.default(mm_5, [32, 286, 9728]);  mm_5 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_13, convert_element_type_7);  convert_element_type_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_13, view_7);  view_13 = view_7 = None
        view_14 = torch.ops.aten.view.default(mul_4, [9152, 9728]);  mul_4 = None
        permute_8 = torch.ops.aten.permute.default(view_14, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_8, view_4);  permute_8 = None
        permute_10 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_7 = torch.ops.aten.mm.default(view_14, permute_10);  view_14 = permute_10 = None
        view_15 = torch.ops.aten.view.default(mm_7, [32, 286, 2560]);  mm_7 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(view_5)
        full_default = torch.ops.aten.full.default([32, 286, 9728], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub = torch.ops.aten.sub.Tensor(full_default, sigmoid_1);  full_default = None
        mul_6 = torch.ops.aten.mul.Tensor(view_5, sub);  view_5 = sub = None
        add_3 = torch.ops.aten.add.Scalar(mul_6, 1);  mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(sigmoid_1, add_3);  sigmoid_1 = add_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_5, mul_7);  mul_5 = mul_7 = None
        view_16 = torch.ops.aten.view.default(mul_8, [9152, 9728]);  mul_8 = None
        permute_13 = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_8 = torch.ops.aten.mm.default(permute_13, view_4);  permute_13 = view_4 = None
        permute_15 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_9 = torch.ops.aten.mm.default(view_16, permute_15);  view_16 = permute_15 = None
        view_17 = torch.ops.aten.view.default(mm_9, [32, 286, 2560]);  mm_9 = None
        add_4 = torch.ops.aten.add.Tensor(view_15, view_17);  view_15 = view_17 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_4, primals_4);  primals_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(add_4, convert_element_type_3);  add_4 = convert_element_type_3 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_10, [0, 1], True);  mul_10 = None
        view_18 = torch.ops.aten.view.default(sum_1, [2560]);  sum_1 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(mul_9, torch.float32);  mul_9 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_24, convert_element_type_2)
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_24, rsqrt);  convert_element_type_24 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_11, [2], True);  mul_11 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_13 = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, pow_2);  mul_13 = pow_2 = None
        expand = torch.ops.aten.expand.default(mul_14, [32, 286, 2560]);  mul_14 = None
        div = torch.ops.aten.div.Scalar(expand, 2560);  expand = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 1.0);  convert_element_type_2 = None
        mul_15 = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_16 = torch.ops.aten.mul.Tensor(div, mul_15);  div = mul_15 = None
        add_5 = torch.ops.aten.add.Tensor(mul_12, mul_16);  mul_12 = mul_16 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16);  add_5 = None
        add_6 = torch.ops.aten.add.Tensor(convert_element_type_25, tangents_1);  convert_element_type_25 = tangents_1 = None
        view_19 = torch.ops.aten.view.default(add_6, [9152, 2560])
        permute_17 = torch.ops.aten.permute.default(view_19, [1, 0])
        mm_10 = torch.ops.aten.mm.default(permute_17, view);  permute_17 = view = None
        permute_19 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_11 = torch.ops.aten.mm.default(view_19, permute_19);  view_19 = permute_19 = None
        view_20 = torch.ops.aten.view.default(mm_11, [32, 286, 4096]);  mm_11 = None
        full_default_1 = torch.ops.aten.full.default([32, 337, 4096], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(full_default_1, view_20, 1, 0, 286);  full_default_1 = view_20 = None
        return (add_6, slice_scatter, mm_10, view_18, mm_8, mm_6, mm_4)
        
def load_args(reader):
    buf0 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 88342528, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 337, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 20971520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (2560, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 49807360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (9728, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 49807360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (9728, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 49807360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (2560, 9728), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)