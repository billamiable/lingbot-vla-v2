
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16):
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
        permute = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_5, convert_element_type_1, permute);  primals_5 = permute = None
        view = torch.ops.aten.view.default(addmm, [24576, 3, 16, -1]);  addmm = None
        permute_1 = torch.ops.aten.permute.default(view, [1, 0, 2, 3]);  view = None
        unbind = torch.ops.aten.unbind.int(permute_1);  permute_1 = None
        getitem_2 = unbind[0]
        getitem_3 = unbind[1]
        getitem_4 = unbind[2];  unbind = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_6, -2)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_7, -2)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze)
        slice_1 = torch.ops.aten.slice.Tensor(convert_element_type_5, 2, 0, 32)
        slice_2 = torch.ops.aten.slice.Tensor(convert_element_type_5, 2, 32, 9223372036854775807);  convert_element_type_5 = None
        neg = torch.ops.aten.neg.default(slice_2);  slice_2 = None
        cat = torch.ops.aten.cat.default([neg, slice_1], -1);  neg = slice_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze);  unsqueeze = None
        slice_3 = torch.ops.aten.slice.Tensor(convert_element_type_6, 2, 0, 32)
        slice_4 = torch.ops.aten.slice.Tensor(convert_element_type_6, 2, 32, 9223372036854775807);  convert_element_type_6 = None
        neg_1 = torch.ops.aten.neg.default(slice_4);  slice_4 = None
        cat_1 = torch.ops.aten.cat.default([neg_1, slice_3], -1);  neg_1 = slice_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = unsqueeze_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        permute_2 = torch.ops.aten.permute.default(convert_element_type_7, [1, 0, 2]);  convert_element_type_7 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(permute_2, 0);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_8, [1, 0, 2]);  convert_element_type_8 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(permute_3, 0);  permute_3 = None
        permute_4 = torch.ops.aten.permute.default(getitem_4, [1, 0, 2]);  getitem_4 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(permute_4, 0);  permute_4 = None
        permute_5 = torch.ops.aten.permute.default(unsqueeze_2, [0, 2, 1, 3]);  unsqueeze_2 = None
        permute_6 = torch.ops.aten.permute.default(unsqueeze_3, [0, 2, 1, 3]);  unsqueeze_3 = None
        permute_7 = torch.ops.aten.permute.default(unsqueeze_4, [0, 2, 1, 3]);  unsqueeze_4 = None
        view_1 = torch.ops.aten.view.default(permute_5, [-1, 16, 64]);  permute_5 = None
        view_2 = torch.ops.aten.view.default(permute_6, [-1, 16, 64]);  permute_6 = None
        view_3 = torch.ops.aten.view.default(permute_7, [24576, 16, 64]);  permute_7 = None
        _flash_attn_varlen_forward = torch.ops.flash_attn._flash_attn_varlen_forward.default(view_1, view_2, view_3, primals_8, primals_8, 256, 256, 0.0, 0.125, False)
        getitem_5 = _flash_attn_varlen_forward[0]
        getitem_6 = _flash_attn_varlen_forward[1]
        getitem_8 = _flash_attn_varlen_forward[3];  _flash_attn_varlen_forward = None
        view_4 = torch.ops.aten.view.default(getitem_5, [1, -1, 16, 64])
        view_5 = torch.ops.aten.view.default(view_4, [24576, -1]);  view_4 = None
        permute_8 = torch.ops.aten.permute.default(primals_9, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_10, view_5, permute_8);  primals_10 = view_5 = permute_8 = None
        add_4 = torch.ops.aten.add.Tensor(primals_3, addmm_1)
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_12, [1], correction = 0, keepdim = True)
        getitem_9 = var_mean_1[0]
        getitem_10 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_10);  convert_element_type_12 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, primals_11);  mul_6 = None
        add_6 = torch.ops.aten.add.Tensor(mul_7, primals_12);  mul_7 = primals_12 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        permute_9 = torch.ops.aten.permute.default(primals_13, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_14, convert_element_type_13, permute_9);  primals_14 = permute_9 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32)
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_17)
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, convert_element_type_17);  mul_8 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, 0.044715);  mul_9 = None
        add_7 = torch.ops.aten.add.Tensor(convert_element_type_17, mul_10);  mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_17, 0.5);  convert_element_type_17 = None
        tanh = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_12, add_8);  mul_12 = add_8 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        permute_10 = torch.ops.aten.permute.default(primals_15, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_16, convert_element_type_18, permute_10);  primals_16 = permute_10 = None
        add_9 = torch.ops.aten.add.Tensor(add_4, addmm_3);  add_4 = addmm_3 = None
        return (add_9, primals_1, primals_3, primals_4, primals_6, primals_7, primals_8, primals_9, primals_11, primals_13, primals_15, getitem_1, rsqrt, convert_element_type_1, view_1, view_2, view_3, getitem_5, getitem_6, getitem_8, addmm_1, getitem_10, rsqrt_1, convert_element_type_13, addmm_2, convert_element_type_18)
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 6291456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (3072, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 6144, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (3072,), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf5, (24576, 64), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf6, (24576, 64), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 388, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf7, (97,), dtype=torch.int32, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (4096, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 8192, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (4096,), dtype=torch.bfloat16, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_16
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)