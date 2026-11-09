
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21):
        view = torch.ops.aten.view.default(primals_3, [8192, 2560]);  primals_3 = None
        permute = torch.ops.aten.permute.default(primals_1, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_2, view, permute);  primals_2 = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [32, 256, 2560])
        view_2 = torch.ops.aten.view.default(primals_6, [2304, 2560]);  primals_6 = None
        permute_1 = torch.ops.aten.permute.default(primals_4, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_5, view_2, permute_1);  primals_5 = permute_1 = None
        view_3 = torch.ops.aten.view.default(addmm_1, [32, 72, 2560])
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_6, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_1);  convert_element_type_6 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_7);  mul = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_2 = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_3);  convert_element_type_8 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_9);  mul_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, primals_10);  mul_3 = primals_10 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        permute_2 = torch.ops.aten.permute.default(primals_11, [1, 0])
        view_4 = torch.ops.aten.view.default(convert_element_type_9, [8192, 2560])
        mm = torch.ops.aten.mm.default(view_4, permute_2);  permute_2 = None
        view_5 = torch.ops.aten.view.default(mm, [32, 256, 128]);  mm = None
        cat = torch.ops.aten.cat.default([convert_element_type_7, convert_element_type_9], -2);  convert_element_type_7 = convert_element_type_9 = None
        permute_3 = torch.ops.aten.permute.default(primals_12, [1, 0])
        view_6 = torch.ops.aten.view.default(cat, [10496, 2560]);  cat = None
        mm_1 = torch.ops.aten.mm.default(view_6, permute_3);  permute_3 = None
        view_7 = torch.ops.aten.view.default(mm_1, [32, 328, 256]);  mm_1 = None
        split = torch.ops.aten.split.Tensor(view_7, 128, -1);  view_7 = None
        getitem_4 = split[0]
        getitem_5 = split[1];  split = None
        view_8 = torch.ops.aten.view.default(view_5, [32, 256, 4, -1]);  view_5 = None
        permute_4 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        view_10 = torch.ops.aten.view.default(getitem_4, [32, 328, 4, -1]);  getitem_4 = None
        permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        view_12 = torch.ops.aten.view.default(getitem_5, [32, 328, 4, -1]);  getitem_5 = None
        permute_6 = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        mul_4 = torch.ops.aten.mul.Tensor(permute_4, 0.4204482076268573);  permute_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(permute_5, 0.4204482076268573);  permute_5 = None
        permute_7 = torch.ops.aten.permute.default(mul_5, [0, 1, 3, 2]);  mul_5 = None
        expand = torch.ops.aten.expand.default(mul_4, [32, 4, 256, 32]);  mul_4 = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_14 = torch.ops.aten.view.default(clone, [128, 256, 32]);  clone = None
        expand_1 = torch.ops.aten.expand.default(permute_7, [32, 4, 32, 328]);  permute_7 = None
        clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_15 = torch.ops.aten.view.default(clone_1, [128, 32, 328]);  clone_1 = None
        bmm = torch.ops.aten.bmm.default(view_14, view_15)
        view_16 = torch.ops.aten.view.default(bmm, [32, 4, 256, 328])
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        amax = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        expand_2 = torch.ops.aten.expand.default(convert_element_type_17, [32, 4, 256, 328]);  convert_element_type_17 = None
        view_17 = torch.ops.aten.view.default(expand_2, [128, 256, 328]);  expand_2 = None
        expand_3 = torch.ops.aten.expand.default(permute_6, [32, 4, 328, 32]);  permute_6 = None
        clone_2 = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_18 = torch.ops.aten.view.default(clone_2, [128, 328, 32]);  clone_2 = None
        bmm_1 = torch.ops.aten.bmm.default(view_17, view_18)
        view_19 = torch.ops.aten.view.default(bmm_1, [32, 4, 256, 32]);  bmm_1 = None
        permute_8 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        clone_3 = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_20 = torch.ops.aten.view.default(clone_3, [32, 256, 128]);  clone_3 = None
        permute_9 = torch.ops.aten.permute.default(primals_13, [1, 0])
        view_21 = torch.ops.aten.view.default(view_20, [8192, 128]);  view_20 = None
        mm_2 = torch.ops.aten.mm.default(view_21, permute_9);  permute_9 = None
        view_22 = torch.ops.aten.view.default(mm_2, [32, 256, 2560]);  mm_2 = None
        add_4 = torch.ops.aten.add.Tensor(view_22, view_1);  view_22 = view_1 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(add_4, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_22, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_2[0]
        getitem_7 = var_mean_2[1];  var_mean_2 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_7);  convert_element_type_22 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_7 = torch.ops.aten.mul.Tensor(mul_6, primals_14);  mul_6 = None
        add_6 = torch.ops.aten.add.Tensor(mul_7, primals_15);  mul_7 = primals_15 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        permute_10 = torch.ops.aten.permute.default(primals_16, [1, 0])
        view_23 = torch.ops.aten.view.default(convert_element_type_23, [8192, 2560]);  convert_element_type_23 = None
        mm_3 = torch.ops.aten.mm.default(view_23, permute_10);  permute_10 = None
        view_24 = torch.ops.aten.view.default(mm_3, [32, 256, 2560])
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.5)
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476);  convert_element_type_26 = None
        erf = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_8, add_7);  mul_8 = add_7 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        permute_11 = torch.ops.aten.permute.default(primals_17, [1, 0])
        view_25 = torch.ops.aten.view.default(convert_element_type_27, [8192, 2560]);  convert_element_type_27 = None
        mm_4 = torch.ops.aten.mm.default(view_25, permute_11);  permute_11 = None
        view_26 = torch.ops.aten.view.default(mm_4, [32, 256, 2560]);  mm_4 = None
        add_8 = torch.ops.aten.add.Tensor(view_26, add_4);  view_26 = None
        view_27 = torch.ops.aten.view.default(add_8, [8192, 2560]);  add_8 = None
        permute_12 = torch.ops.aten.permute.default(primals_18, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_19, view_27, permute_12);  primals_19 = permute_12 = None
        view_28 = torch.ops.aten.view.default(addmm_2, [32, 256, 1024])
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_33, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_3[0]
        getitem_9 = var_mean_3[1];  var_mean_3 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_9);  convert_element_type_33 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_12 = torch.ops.aten.mul.Tensor(mul_11, primals_20);  mul_11 = None
        add_10 = torch.ops.aten.add.Tensor(mul_12, primals_21);  mul_12 = primals_21 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        permute_30 = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
        permute_31 = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        permute_32 = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_33 = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        return (convert_element_type_34, primals_1, primals_4, primals_7, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, view, addmm, view_2, addmm_1, getitem_1, rsqrt, getitem_3, rsqrt_1, view_4, view_6, bmm, amax, sum_1, view_21, add_4, getitem_7, rsqrt_2, view_23, mm_3, view_25, view_27, addmm_2, getitem_9, rsqrt_3, permute_30, permute_31, permute_32, permute_33)
        
def load_args(reader):
    buf0 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32, 256, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 11796480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (32, 72, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (128, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 1310720, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (256, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (2560, 128), dtype=torch.bfloat16, is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf16, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 5242880, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf17, (1024, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf18, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf19, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf20, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_21
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)