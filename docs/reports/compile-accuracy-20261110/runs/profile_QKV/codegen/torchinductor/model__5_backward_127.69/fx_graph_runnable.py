
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

    
    
    def forward(self, primals_1, primals_3, primals_4, primals_6, primals_7, primals_8, primals_9, primals_11, primals_13, primals_15, getitem_1, rsqrt, convert_element_type_1, view_1, view_2, view_3, getitem_5, getitem_6, getitem_8, addmm_1, getitem_10, rsqrt_1, convert_element_type_13, addmm_2, convert_element_type_18, tangents_1):
        permute_10 = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_11 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm = torch.ops.aten.mm.default(tangents_1, permute_11);  permute_11 = None
        permute_12 = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_12, convert_element_type_18);  permute_12 = convert_element_type_18 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view_6 = torch.ops.aten.view.default(sum_1, [1024]);  sum_1 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(addmm_2, torch.float32);  addmm_2 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_17, convert_element_type_17)
        mul_9 = torch.ops.aten.mul.Tensor(mul_8, convert_element_type_17)
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, 0.044715);  mul_9 = None
        add_7 = torch.ops.aten.add.Tensor(convert_element_type_17, mul_10);  mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(add_7, 0.7978845608028654);  add_7 = None
        tanh = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_8 = torch.ops.aten.add.Tensor(tanh, 1)
        mul_19 = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_20 = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_2 = torch.ops.aten.sub.Tensor(1, mul_20);  mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_8, 0.134145);  mul_8 = None
        add_12 = torch.ops.aten.add.Tensor(mul_21, 1);  mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_12, 0.7978845608028654);  add_12 = None
        mul_12 = torch.ops.aten.mul.Tensor(convert_element_type_17, 0.5);  convert_element_type_17 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_12, sub_2);  mul_12 = sub_2 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, mul_22);  mul_23 = mul_22 = None
        add_13 = torch.ops.aten.add.Tensor(mul_19, mul_24);  mul_19 = mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_26, add_13);  convert_element_type_26 = add_13 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        permute_9 = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_15 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_2 = torch.ops.aten.mm.default(convert_element_type_28, permute_15);  permute_15 = None
        permute_16 = torch.ops.aten.permute.default(convert_element_type_28, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_16, convert_element_type_13);  permute_16 = convert_element_type_13 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_28, [0], True);  convert_element_type_28 = None
        view_7 = torch.ops.aten.view.default(sum_2, [4096]);  sum_2 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(primals_11, torch.float32);  primals_11 = None
        mul_27 = torch.ops.aten.mul.Tensor(convert_element_type_33, convert_element_type_35);  convert_element_type_35 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_27, 1024)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_27, [1], True)
        add_4 = torch.ops.aten.add.Tensor(primals_3, addmm_1);  addmm_1 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_10);  convert_element_type_12 = getitem_10 = None
        mul_6 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_27, mul_6);  mul_27 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_29, [1], True);  mul_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_6, sum_4);  sum_4 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_28, sum_3);  mul_28 = sum_3 = None
        sub_5 = torch.ops.aten.sub.Tensor(sub_4, mul_30);  sub_4 = mul_30 = None
        div = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        mul_31 = torch.ops.aten.mul.Tensor(div, sub_5);  div = sub_5 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_33, mul_6);  mul_6 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_32, [0]);  mul_32 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(convert_element_type_33, [0]);  convert_element_type_33 = None
        convert_element_type_37 = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None
        convert_element_type_38 = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        add_14 = torch.ops.aten.add.Tensor(tangents_1, convert_element_type_37);  tangents_1 = convert_element_type_37 = None
        permute_8 = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_19 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_4 = torch.ops.aten.mm.default(add_14, permute_19);  permute_19 = None
        permute_20 = torch.ops.aten.permute.default(add_14, [1, 0])
        view_4 = torch.ops.aten.view.default(getitem_5, [1, -1, 16, 64])
        view_5 = torch.ops.aten.view.default(view_4, [24576, -1]);  view_4 = None
        mm_5 = torch.ops.aten.mm.default(permute_20, view_5);  permute_20 = view_5 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(add_14, [0], True)
        view_8 = torch.ops.aten.view.default(sum_7, [1024]);  sum_7 = None
        view_9 = torch.ops.aten.view.default(mm_4, [1, 24576, 16, 64]);  mm_4 = None
        view_10 = torch.ops.aten.view.default(view_9, [24576, 16, 64]);  view_9 = None
        empty = torch.ops.aten.empty.memory_format([24576, 16, 64], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_23 = torch.ops.aten.permute.default(empty, [0, 1, 2]);  empty = None
        empty_1 = torch.ops.aten.empty.memory_format([24576, 16, 64], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_24 = torch.ops.aten.permute.default(empty_1, [0, 1, 2]);  empty_1 = None
        empty_2 = torch.ops.aten.empty.memory_format([24576, 16, 64], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25 = torch.ops.aten.permute.default(empty_2, [0, 1, 2]);  empty_2 = None
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.flash_attn._flash_attn_varlen_backward.default, dout = view_10, q = view_1, k = view_2, v = view_3, out = getitem_5, softmax_lse = getitem_6, cu_seqlens_q = primals_8, cu_seqlens_k = primals_8, max_seqlen_q = 256, max_seqlen_k = 256, dropout_p = 0.0, softmax_scale = 0.125, causal = False, window_size_left = -1, window_size_right = -1, softcap = 0.0, alibi_slopes = None, deterministic = False, rng_state = getitem_8, zero_tensors = False, _dq_base_index = 0, _dk_base_index = 1, _dv_base_index = 2, _all_bases = [permute_23, permute_24, permute_25]);  view_10 = view_1 = view_2 = view_3 = getitem_5 = getitem_6 = primals_8 = getitem_8 = permute_23 = permute_24 = permute_25 = None
        getitem_12 = auto_functionalized_v2[1]
        getitem_13 = auto_functionalized_v2[2]
        getitem_14 = auto_functionalized_v2[3];  auto_functionalized_v2 = None
        view_14 = torch.ops.aten.view.default(getitem_13, [1, 24576, 16, 64]);  getitem_13 = None
        permute_32 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(permute_32, 0);  permute_32 = None
        permute_33 = torch.ops.aten.permute.default(squeeze_3, [1, 0, 2]);  squeeze_3 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(permute_33, torch.float32);  permute_33 = None
        view_15 = torch.ops.aten.view.default(getitem_12, [1, 24576, 16, 64]);  getitem_12 = None
        permute_34 = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        squeeze_4 = torch.ops.aten.squeeze.dim(permute_34, 0);  permute_34 = None
        permute_35 = torch.ops.aten.permute.default(squeeze_4, [1, 0, 2]);  squeeze_4 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(permute_35, torch.float32);  permute_35 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_7, -2);  primals_7 = None
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_44, unsqueeze_1)
        slice_5 = torch.ops.aten.slice.Tensor(mul_33, 2, 0, 32)
        slice_6 = torch.ops.aten.slice.Tensor(mul_33, 2, 32, 64);  mul_33 = None
        neg_2 = torch.ops.aten.neg.default(slice_5);  slice_5 = None
        full_default = torch.ops.aten.full.default([24576, 16, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(full_default, neg_2, 2, 32, 9223372036854775807);  neg_2 = None
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(full_default, slice_6, 2, 0, 32);  slice_6 = None
        add_15 = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_6, -2);  primals_6 = None
        mul_34 = torch.ops.aten.mul.Tensor(convert_element_type_44, unsqueeze);  convert_element_type_44 = None
        add_16 = torch.ops.aten.add.Tensor(add_15, mul_34);  add_15 = mul_34 = None
        mul_35 = torch.ops.aten.mul.Tensor(convert_element_type_45, unsqueeze_1);  unsqueeze_1 = None
        slice_7 = torch.ops.aten.slice.Tensor(mul_35, 2, 0, 32)
        slice_8 = torch.ops.aten.slice.Tensor(mul_35, 2, 32, 64);  mul_35 = None
        neg_3 = torch.ops.aten.neg.default(slice_7);  slice_7 = None
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(full_default, neg_3, 2, 32, 9223372036854775807);  neg_3 = None
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(full_default, slice_8, 2, 0, 32);  full_default = slice_8 = None
        add_17 = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        mul_36 = torch.ops.aten.mul.Tensor(convert_element_type_45, unsqueeze);  convert_element_type_45 = unsqueeze = None
        add_18 = torch.ops.aten.add.Tensor(add_17, mul_36);  add_17 = mul_36 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None
        convert_element_type_47 = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None
        view_16 = torch.ops.aten.view.default(getitem_14, [1, 24576, 16, 64]);  getitem_14 = None
        permute_36 = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        squeeze_5 = torch.ops.aten.squeeze.dim(permute_36, 0);  permute_36 = None
        permute_37 = torch.ops.aten.permute.default(squeeze_5, [1, 0, 2]);  squeeze_5 = None
        cat_2 = torch.ops.aten.cat.default([convert_element_type_47, convert_element_type_46, permute_37]);  convert_element_type_47 = convert_element_type_46 = permute_37 = None
        view_17 = torch.ops.aten.view.default(cat_2, [3, 24576, 16, 64]);  cat_2 = None
        permute_38 = torch.ops.aten.permute.default(view_17, [1, 0, 2, 3]);  view_17 = None
        clone_4 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_18 = torch.ops.aten.view.default(clone_4, [24576, 3072]);  clone_4 = None
        permute = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_39 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_6 = torch.ops.aten.mm.default(view_18, permute_39);  permute_39 = None
        permute_40 = torch.ops.aten.permute.default(view_18, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_40, convert_element_type_1);  permute_40 = convert_element_type_1 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        view_19 = torch.ops.aten.view.default(sum_8, [3072]);  sum_8 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_3, torch.float32);  primals_3 = None
        convert_element_type_54 = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        sub_6 = torch.ops.aten.sub.Tensor(convert_element_type_53, getitem_1);  convert_element_type_53 = getitem_1 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_6, rsqrt);  sub_6 = None
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_52, convert_element_type_54);  convert_element_type_54 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 1024)
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_38, [1], True)
        mul_40 = torch.ops.aten.mul.Tensor(mul_38, mul_37);  mul_38 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_40, [1], True);  mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_37, sum_10);  sum_10 = None
        sub_7 = torch.ops.aten.sub.Tensor(mul_39, sum_9);  mul_39 = sum_9 = None
        sub_8 = torch.ops.aten.sub.Tensor(sub_7, mul_41);  sub_7 = mul_41 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        mul_42 = torch.ops.aten.mul.Tensor(div_1, sub_8);  div_1 = sub_8 = None
        mul_43 = torch.ops.aten.mul.Tensor(convert_element_type_52, mul_37);  mul_37 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_43, [0]);  mul_43 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(convert_element_type_52, [0]);  convert_element_type_52 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(mul_42, torch.bfloat16);  mul_42 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(sum_11, torch.bfloat16);  sum_11 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(sum_12, torch.bfloat16);  sum_12 = None
        add_19 = torch.ops.aten.add.Tensor(add_14, convert_element_type_56);  add_14 = convert_element_type_56 = None
        return (convert_element_type_57, convert_element_type_58, add_19, mm_7, view_19, None, None, None, mm_5, view_8, convert_element_type_38, convert_element_type_39, mm_3, view_7, mm_1, view_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 6291456, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (3072, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf3, (24576, 64), is_leaf=True)  # primals_6
    buf4 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf4, (24576, 64), is_leaf=True)  # primals_7
    buf5 = reader.storage(None, 388, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf5, (97,), dtype=torch.int32, is_leaf=True)  # primals_8
    buf6 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (1024, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_9
    buf7 = reader.storage(None, 2048, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (1024,), dtype=torch.bfloat16, is_leaf=True)  # primals_11
    buf8 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (4096, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_13
    buf9 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (1024, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_15
    buf10 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf10, (24576, 1), is_leaf=True)  # getitem_1
    buf11 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf11, (24576, 1), is_leaf=True)  # rsqrt
    buf12 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf12, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_1
    buf13 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf13, (24576, 16, 64), dtype=torch.bfloat16, is_leaf=True)  # view_1
    buf14 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf14, (24576, 16, 64), dtype=torch.bfloat16, is_leaf=True)  # view_2
    buf15 = reader.storage(None, 150994944, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf15, (24576, 16, 64), (3072, 64, 1), dtype=torch.bfloat16, storage_offset=2048, is_leaf=True)  # view_3
    buf16 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf16, (24576, 16, 64), dtype=torch.bfloat16, is_leaf=True)  # getitem_5
    buf17 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf17, (16, 24576), is_leaf=True)  # getitem_6
    buf18 = reader.storage(None, 16, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf18, (2,), dtype=torch.int64, is_leaf=True)  # getitem_8
    buf19 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf19, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # addmm_1
    buf20 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf20, (24576, 1), is_leaf=True)  # getitem_10
    buf21 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf21, (24576, 1), is_leaf=True)  # rsqrt_1
    buf22 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf22, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_13
    buf23 = reader.storage(None, 201326592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf23, (24576, 4096), dtype=torch.bfloat16, is_leaf=True)  # addmm_2
    buf24 = reader.storage(None, 201326592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf24, (24576, 4096), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_18
    buf25 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf25, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)