
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14):
        permute = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        addmm = torch.ops.aten.addmm.default(primals_6, primals_4, permute);  primals_6 = permute = None
        iota = torch.ops.prims.iota.default(384, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        lt = torch.ops.aten.lt.Scalar(iota, 192.0)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(iota, torch.float32)
        mul = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.0026109660574412533);  convert_element_type_3 = None
        add = torch.ops.aten.add.Tensor(mul, 0.0);  mul = None
        sub = torch.ops.aten.sub.Tensor(383, iota);  iota = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_4, 0.0026109660574412533);  convert_element_type_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(1.0, mul_1);  mul_1 = None
        where = torch.ops.aten.where.self(lt, add, sub_1);  lt = add = sub_1 = None
        pow_1 = torch.ops.aten.pow.Scalar(1000.0, where);  where = None
        mul_2 = torch.ops.aten.mul.Tensor(pow_1, 0.004);  pow_1 = None
        reciprocal = torch.ops.aten.reciprocal.default(mul_2);  mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(reciprocal, 1.0);  reciprocal = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 2);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, 3.141592653589793);  mul_4 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(mul_5, 0);  mul_5 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_7, 1);  primals_7 = None
        mul_6 = torch.ops.aten.mul.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        sin = torch.ops.aten.sin.default(mul_6)
        cos = torch.ops.aten.cos.default(mul_6);  mul_6 = None
        cat = torch.ops.aten.cat.default([sin, cos], 1);  sin = cos = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        view = torch.ops.aten.view.default(primals_10, [1600, 55]);  primals_10 = None
        permute_1 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_9, view, permute_1);  primals_9 = permute_1 = None
        view_1 = torch.ops.aten.view.default(addmm_1, [32, 50, 768]);  addmm_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1)
        expand = torch.ops.aten.expand.default(unsqueeze_2, [-1, 50, -1]);  unsqueeze_2 = None
        cat_1 = torch.ops.aten.cat.default([view_1, expand], -1);  view_1 = expand = None
        view_2 = torch.ops.aten.view.default(cat_1, [1600, 1536]);  cat_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_11, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_12, view_2, permute_2);  primals_12 = permute_2 = None
        view_3 = torch.ops.aten.view.default(addmm_2, [32, 50, 768])
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_12)
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_12, sigmoid);  convert_element_type_12 = sigmoid = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(mul_7, torch.bfloat16);  mul_7 = None
        view_4 = torch.ops.aten.view.default(convert_element_type_13, [1600, 768]);  convert_element_type_13 = None
        permute_3 = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_14, view_4, permute_3);  primals_14 = permute_3 = None
        view_5 = torch.ops.aten.view.default(addmm_3, [32, 50, 768]);  addmm_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        cat_2 = torch.ops.aten.cat.default([unsqueeze_3, view_5], 1);  unsqueeze_3 = view_5 = None
        full_default = torch.ops.aten.full.default([32, 51], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1 = torch.ops.aten.full.default([32, 51], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        slice_5 = torch.ops.aten.slice.Tensor(full_1, 1, 0, 2)
        copy = torch.ops.aten.copy.default(slice_5, full_default_1);  slice_5 = full_default_1 = None
        slice_scatter = torch.ops.aten.slice_scatter.default(full_1, copy, 1, 0, 2);  copy = None
        cat_3 = torch.ops.aten.cat.default([primals_1, full_default], 1);  full_default = None
        cat_4 = torch.ops.aten.cat.default([primals_2, slice_scatter], 1);  primals_2 = slice_scatter = None
        cumsum = torch.ops.aten.cumsum.default(cat_4, 1);  cat_4 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(cumsum, 1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(cumsum, 2);  cumsum = None
        le = torch.ops.aten.le.Tensor(unsqueeze_4, unsqueeze_5);  unsqueeze_4 = unsqueeze_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(cat_3, 1)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(cat_3, 2);  cat_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(unsqueeze_6, unsqueeze_7);  unsqueeze_6 = unsqueeze_7 = None
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(le, mul_8);  le = mul_8 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(primals_1, 0);  primals_1 = None
        bitwise_not = torch.ops.aten.bitwise_not.default(unsqueeze_8);  unsqueeze_8 = None
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(bitwise_not, full_default_2, primals_3);  bitwise_not = full_default_2 = None
        amax = torch.ops.aten.amax.default(where_1, [0, 2]);  where_1 = None
        add_1 = torch.ops.aten.add.Tensor(amax, 1);  amax = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(add_1, 1);  add_1 = None
        full_default_3 = torch.ops.aten.full.default([32, 51], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cumsum_1 = torch.ops.aten.cumsum.default(full_default_3, 1);  full_default_3 = None
        add_2 = torch.ops.aten.add.Tensor(unsqueeze_9, cumsum_1);  unsqueeze_9 = cumsum_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_2, 1);  add_2 = None
        full_default_5 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(full_1, full_default_5, sub_2);  full_1 = full_default_5 = sub_2 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(where_2, 0);  where_2 = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_10, [3, -1, -1]);  unsqueeze_10 = None
        cat_5 = torch.ops.aten.cat.default([primals_3, expand_1], -1);  primals_3 = expand_1 = None
        return (bitwise_and, cat_5, cat_2, convert_element_type_5, primals_4, primals_11, primals_13, view, view_2, addmm_2, view_4)
        
def load_args(reader):
    buf0 = reader.storage(None, 9152, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf0, (32, 286), dtype=torch.bool, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 9152, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf1, (32, 286), dtype=torch.bool, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 219648, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (3, 32, 286), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 3520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (32, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 84480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (768, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (768,), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 64, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (32,), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 84480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (768, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (768,), dtype=torch.bfloat16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 176000, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (32, 50, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 2359296, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (768, 1536), dtype=torch.bfloat16, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (768,), dtype=torch.bfloat16, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 1536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (768,), dtype=torch.bfloat16, is_leaf=True)  # primals_14
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)