
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
torch._inductor.config.comprehensive_padding = True
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_SWI/codegen'
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

    
    
    def forward(self, primals_1, primals_4, primals_7, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, view, addmm, view_2, addmm_1, getitem_1, rsqrt, getitem_3, rsqrt_1, view_4, view_6, bmm, amax, sum_1, view_21, add_4, getitem_7, rsqrt_2, view_23, mm_3, view_25, view_27, addmm_2, getitem_9, rsqrt_3, permute_30, permute_31, permute_32, permute_33, tangents_1):
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(primals_20, torch.float32);  primals_20 = None
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_37);  convert_element_type_37 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, 1024)
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_14, [2], True)
        view_28 = torch.ops.aten.view.default(addmm_2, [32, 256, 1024]);  addmm_2 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        sub_4 = torch.ops.aten.sub.Tensor(convert_element_type_33, getitem_9);  convert_element_type_33 = getitem_9 = None
        mul_11 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_14, mul_11);  mul_14 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_16, [2], True);  mul_16 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_11, sum_3);  sum_3 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_15, sum_2);  mul_15 = sum_2 = None
        sub_7 = torch.ops.aten.sub.Tensor(sub_6, mul_17);  sub_6 = mul_17 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
        mul_18 = torch.ops.aten.mul.Tensor(div_1, sub_7);  div_1 = sub_7 = None
        mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_35, mul_11);  mul_11 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_19, [0, 1]);  mul_19 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(convert_element_type_35, [0, 1]);  convert_element_type_35 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(sum_4, torch.bfloat16);  sum_4 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        view_29 = torch.ops.aten.view.default(convert_element_type_39, [8192, 1024]);  convert_element_type_39 = None
        permute_12 = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_13 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_5 = torch.ops.aten.mm.default(view_29, permute_13);  permute_13 = None
        permute_14 = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_6 = torch.ops.aten.mm.default(permute_14, view_27);  permute_14 = view_27 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(view_29, [0], True);  view_29 = None
        view_30 = torch.ops.aten.view.default(sum_6, [1024]);  sum_6 = None
        view_31 = torch.ops.aten.view.default(mm_5, [32, 256, 2560]);  mm_5 = None
        view_32 = torch.ops.aten.view.default(view_31, [8192, 2560])
        permute_17 = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_17, view_25);  permute_17 = view_25 = None
        permute_11 = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_19 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_8 = torch.ops.aten.mm.default(view_32, permute_19);  view_32 = permute_19 = None
        view_33 = torch.ops.aten.view.default(mm_8, [32, 256, 2560]);  mm_8 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        view_24 = torch.ops.aten.view.default(mm_3, [32, 256, 2560]);  mm_3 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_21 = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, -0.5);  mul_22 = None
        exp_1 = torch.ops.aten.exp.default(mul_23);  mul_23 = None
        mul_24 = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_24);  convert_element_type_26 = mul_24 = None
        add_12 = torch.ops.aten.add.Tensor(mul_21, mul_25);  mul_21 = mul_25 = None
        mul_26 = torch.ops.aten.mul.Tensor(convert_element_type_50, add_12);  convert_element_type_50 = add_12 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None
        view_34 = torch.ops.aten.view.default(convert_element_type_52, [8192, 2560]);  convert_element_type_52 = None
        permute_21 = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_21, view_23);  permute_21 = view_23 = None
        permute_10 = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_23 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_10 = torch.ops.aten.mm.default(view_34, permute_23);  view_34 = permute_23 = None
        view_35 = torch.ops.aten.view.default(mm_10, [32, 256, 2560]);  mm_10 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        convert_element_type_59 = torch.ops.prims.convert_element_type.default(primals_14, torch.float32);  primals_14 = None
        mul_28 = torch.ops.aten.mul.Tensor(convert_element_type_57, convert_element_type_59);  convert_element_type_59 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, 2560)
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_28, [2], True)
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_22, getitem_7);  convert_element_type_22 = getitem_7 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_28, mul_6);  mul_28 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_30, [2], True);  mul_30 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_6, sum_8);  sum_8 = None
        sub_9 = torch.ops.aten.sub.Tensor(mul_29, sum_7);  mul_29 = sum_7 = None
        sub_10 = torch.ops.aten.sub.Tensor(sub_9, mul_31);  sub_9 = mul_31 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_2, 2560);  rsqrt_2 = None
        mul_32 = torch.ops.aten.mul.Tensor(div_2, sub_10);  div_2 = sub_10 = None
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_57, mul_6);  mul_6 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_33, [0, 1]);  mul_33 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(convert_element_type_57, [0, 1]);  convert_element_type_57 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(sum_9, torch.bfloat16);  sum_9 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(sum_10, torch.bfloat16);  sum_10 = None
        add_13 = torch.ops.aten.add.Tensor(view_31, convert_element_type_61);  view_31 = convert_element_type_61 = None
        view_36 = torch.ops.aten.view.default(add_13, [8192, 2560])
        permute_25 = torch.ops.aten.permute.default(view_36, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_25, view_21);  permute_25 = view_21 = None
        permute_9 = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_27 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_12 = torch.ops.aten.mm.default(view_36, permute_27);  view_36 = permute_27 = None
        view_37 = torch.ops.aten.view.default(mm_12, [32, 256, 128]);  mm_12 = None
        view_38 = torch.ops.aten.view.default(view_37, [32, 256, 4, 32]);  view_37 = None
        permute_29 = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_12 = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_39 = torch.ops.aten.view.default(clone_12, [128, 256, 32]);  clone_12 = None
        bmm_2 = torch.ops.aten.bmm.default(permute_30, view_39);  permute_30 = None
        bmm_3 = torch.ops.aten.bmm.default(view_39, permute_31);  view_39 = permute_31 = None
        view_40 = torch.ops.aten.view.default(bmm_2, [32, 4, 328, 32]);  bmm_2 = None
        view_41 = torch.ops.aten.view.default(bmm_3, [32, 4, 256, 328]);  bmm_3 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        view_16 = torch.ops.aten.view.default(bmm, [32, 4, 256, 328]);  bmm = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(view_16, torch.float32);  view_16 = None
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_16, amax);  convert_element_type_16 = amax = None
        exp = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_34 = torch.ops.aten.mul.Tensor(convert_element_type_72, div);  convert_element_type_72 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_34, [-1], True)
        neg = torch.ops.aten.neg.default(div);  div = None
        fma = torch.ops.prims.fma.default(neg, sum_11, mul_34);  neg = sum_11 = mul_34 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_42 = torch.ops.aten.view.default(convert_element_type_73, [128, 256, 328]);  convert_element_type_73 = None
        bmm_4 = torch.ops.aten.bmm.default(permute_32, view_42);  permute_32 = None
        bmm_5 = torch.ops.aten.bmm.default(view_42, permute_33);  view_42 = permute_33 = None
        view_43 = torch.ops.aten.view.default(bmm_4, [32, 4, 32, 328]);  bmm_4 = None
        view_44 = torch.ops.aten.view.default(bmm_5, [32, 4, 256, 32]);  bmm_5 = None
        permute_34 = torch.ops.aten.permute.default(view_43, [0, 1, 3, 2]);  view_43 = None
        mul_35 = torch.ops.aten.mul.Tensor(permute_34, 0.4204482076268573);  permute_34 = None
        mul_36 = torch.ops.aten.mul.Tensor(view_44, 0.4204482076268573);  view_44 = None
        permute_35 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_13 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_46 = torch.ops.aten.view.default(clone_13, [32, 328, 128]);  clone_13 = None
        permute_36 = torch.ops.aten.permute.default(mul_35, [0, 2, 1, 3]);  mul_35 = None
        view_48 = torch.ops.aten.view.default(permute_36, [32, 328, 128]);  permute_36 = None
        permute_37 = torch.ops.aten.permute.default(mul_36, [0, 2, 1, 3]);  mul_36 = None
        clone_14 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_50 = torch.ops.aten.view.default(clone_14, [32, 256, 128]);  clone_14 = None
        cat_1 = torch.ops.aten.cat.default([view_48, view_46], 2);  view_48 = view_46 = None
        view_51 = torch.ops.aten.view.default(cat_1, [10496, 256]);  cat_1 = None
        permute_38 = torch.ops.aten.permute.default(view_51, [1, 0])
        mm_13 = torch.ops.aten.mm.default(permute_38, view_6);  permute_38 = view_6 = None
        permute_3 = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_40 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_14 = torch.ops.aten.mm.default(view_51, permute_40);  view_51 = permute_40 = None
        view_52 = torch.ops.aten.view.default(mm_14, [32, 328, 2560]);  mm_14 = None
        slice_1 = torch.ops.aten.slice.Tensor(view_52, 1, 0, 72)
        slice_2 = torch.ops.aten.slice.Tensor(view_52, 1, 72, 328);  view_52 = None
        view_53 = torch.ops.aten.view.default(view_50, [8192, 128]);  view_50 = None
        permute_42 = torch.ops.aten.permute.default(view_53, [1, 0])
        mm_15 = torch.ops.aten.mm.default(permute_42, view_4);  permute_42 = view_4 = None
        permute_2 = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_44 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_16 = torch.ops.aten.mm.default(view_53, permute_44);  view_53 = permute_44 = None
        view_54 = torch.ops.aten.view.default(mm_16, [32, 256, 2560]);  mm_16 = None
        add_14 = torch.ops.aten.add.Tensor(slice_2, view_54);  slice_2 = view_54 = None
        convert_element_type_86 = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(primals_9, torch.float32);  primals_9 = None
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_86, convert_element_type_88);  convert_element_type_88 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 2560)
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_38, [2], True)
        view_1 = torch.ops.aten.view.default(addmm, [32, 256, 2560]);  addmm = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_3);  convert_element_type_8 = getitem_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, mul_2);  mul_38 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(mul_40, [2], True);  mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_2, sum_13);  sum_13 = None
        sub_12 = torch.ops.aten.sub.Tensor(mul_39, sum_12);  mul_39 = sum_12 = None
        sub_13 = torch.ops.aten.sub.Tensor(sub_12, mul_41);  sub_12 = mul_41 = None
        div_3 = torch.ops.aten.div.Tensor(rsqrt_1, 2560);  rsqrt_1 = None
        mul_42 = torch.ops.aten.mul.Tensor(div_3, sub_13);  div_3 = sub_13 = None
        mul_43 = torch.ops.aten.mul.Tensor(convert_element_type_86, mul_2);  mul_2 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(mul_43, [0, 1]);  mul_43 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(convert_element_type_86, [0, 1]);  convert_element_type_86 = None
        convert_element_type_90 = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        convert_element_type_91 = torch.ops.prims.convert_element_type.default(sum_14, torch.bfloat16);  sum_14 = None
        convert_element_type_92 = torch.ops.prims.convert_element_type.default(sum_15, torch.bfloat16);  sum_15 = None
        add_15 = torch.ops.aten.add.Tensor(add_13, convert_element_type_90);  add_13 = convert_element_type_90 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_95 = torch.ops.prims.convert_element_type.default(primals_7, torch.float32);  primals_7 = None
        mul_45 = torch.ops.aten.mul.Tensor(convert_element_type_93, convert_element_type_95);  convert_element_type_95 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 2560)
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_45, [2], True)
        view_3 = torch.ops.aten.view.default(addmm_1, [32, 72, 2560]);  addmm_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        sub = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_1);  convert_element_type_6 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_45, mul);  mul_45 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_47, [2], True);  mul_47 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul, sum_17);  sum_17 = None
        sub_15 = torch.ops.aten.sub.Tensor(mul_46, sum_16);  mul_46 = sum_16 = None
        sub_16 = torch.ops.aten.sub.Tensor(sub_15, mul_48);  sub_15 = mul_48 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt, 2560);  rsqrt = None
        mul_49 = torch.ops.aten.mul.Tensor(div_4, sub_16);  div_4 = sub_16 = None
        mul_50 = torch.ops.aten.mul.Tensor(convert_element_type_93, mul);  mul = None
        sum_18 = torch.ops.aten.sum.dim_IntList(mul_50, [0, 1]);  mul_50 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(convert_element_type_93, [0, 1]);  convert_element_type_93 = None
        convert_element_type_97 = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        convert_element_type_98 = torch.ops.prims.convert_element_type.default(sum_18, torch.bfloat16);  sum_18 = None
        convert_element_type_99 = torch.ops.prims.convert_element_type.default(sum_19, torch.bfloat16);  sum_19 = None
        view_55 = torch.ops.aten.view.default(convert_element_type_97, [2304, 2560]);  convert_element_type_97 = None
        permute_1 = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_46 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_17 = torch.ops.aten.mm.default(view_55, permute_46);  permute_46 = None
        permute_47 = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_18 = torch.ops.aten.mm.default(permute_47, view_2);  permute_47 = view_2 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        view_56 = torch.ops.aten.view.default(sum_20, [2560]);  sum_20 = None
        view_57 = torch.ops.aten.view.default(mm_17, [32, 72, 2560]);  mm_17 = None
        view_58 = torch.ops.aten.view.default(add_15, [8192, 2560]);  add_15 = None
        permute = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        permute_50 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_19 = torch.ops.aten.mm.default(view_58, permute_50);  permute_50 = None
        permute_51 = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_20 = torch.ops.aten.mm.default(permute_51, view);  permute_51 = view = None
        sum_21 = torch.ops.aten.sum.dim_IntList(view_58, [0], True);  view_58 = None
        view_59 = torch.ops.aten.view.default(sum_21, [2560]);  sum_21 = None
        view_60 = torch.ops.aten.view.default(mm_19, [32, 256, 2560]);  mm_19 = None
        return (mm_20, view_59, view_60, mm_18, view_56, view_57, convert_element_type_98, convert_element_type_99, convert_element_type_91, convert_element_type_92, mm_15, mm_13, mm_11, convert_element_type_62, convert_element_type_63, mm_9, mm_7, mm_6, view_30, convert_element_type_40, convert_element_type_41)
        
def load_args(reader):
    buf0 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf3 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_9
    buf4 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (128, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_11
    buf5 = reader.storage(None, 1310720, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (256, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_12
    buf6 = reader.storage(None, 655360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (2560, 128), dtype=torch.bfloat16, is_leaf=True)  # primals_13
    buf7 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_14
    buf8 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_16
    buf9 = reader.storage(None, 13107200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (2560, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_17
    buf10 = reader.storage(None, 5242880, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf10, (1024, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_18
    buf11 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf11, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_20
    buf12 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # view
    buf13 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # addmm
    buf14 = reader.storage(None, 11796480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (2304, 2560), dtype=torch.bfloat16, is_leaf=True)  # view_2
    buf15 = reader.storage(None, 11796480, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (2304, 2560), dtype=torch.bfloat16, is_leaf=True)  # addmm_1
    buf16 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf16, (32, 72, 1), is_leaf=True)  # getitem_1
    buf17 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf17, (32, 72, 1), is_leaf=True)  # rsqrt
    buf18 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf18, (32, 256, 1), is_leaf=True)  # getitem_3
    buf19 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf19, (32, 256, 1), is_leaf=True)  # rsqrt_1
    buf20 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf20, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # view_4
    buf21 = reader.storage(None, 53739520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf21, (10496, 2560), dtype=torch.bfloat16, is_leaf=True)  # view_6
    buf22 = reader.storage(None, 21495808, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf22, (128, 256, 328), dtype=torch.bfloat16, is_leaf=True)  # bmm
    buf23 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf23, (32, 4, 256, 1), is_leaf=True)  # amax
    buf24 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf24, (32, 4, 256, 1), is_leaf=True)  # sum_1
    buf25 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf25, (8192, 128), dtype=torch.bfloat16, is_leaf=True)  # view_21
    buf26 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf26, (32, 256, 2560), dtype=torch.bfloat16, is_leaf=True)  # add_4
    buf27 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32, 256, 1), is_leaf=True)  # getitem_7
    buf28 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32, 256, 1), is_leaf=True)  # rsqrt_2
    buf29 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf29, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # view_23
    buf30 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf30, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # mm_3
    buf31 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf31, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # view_25
    buf32 = reader.storage(None, 41943040, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf32, (8192, 2560), dtype=torch.bfloat16, is_leaf=True)  # view_27
    buf33 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf33, (8192, 1024), dtype=torch.bfloat16, is_leaf=True)  # addmm_2
    buf34 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf34, (32, 256, 1), is_leaf=True)  # getitem_9
    buf35 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf35, (32, 256, 1), is_leaf=True)  # rsqrt_3
    buf36 = reader.storage(None, 21495808, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf36, (128, 328, 256), (83968, 1, 328), dtype=torch.bfloat16, is_leaf=True)  # permute_30
    buf37 = reader.storage(None, 2686976, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf37, (128, 32, 328), (10496, 1, 32), dtype=torch.bfloat16, is_leaf=True)  # permute_31
    buf38 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf38, (128, 32, 256), (8192, 1, 32), dtype=torch.bfloat16, is_leaf=True)  # permute_32
    buf39 = reader.storage(None, 2686976, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf39, (128, 328, 32), (10496, 1, 328), dtype=torch.bfloat16, is_leaf=True)  # permute_33
    buf40 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf40, (32, 256, 1024), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)