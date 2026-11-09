
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, tangents_1, tangents_2):
        view_18 = torch.ops.aten.view.default(tangents_1, [1632, 768]);  tangents_1 = None
        view_20 = torch.ops.aten.view.default(view_18, [32, 51, 768]);  view_18 = None
        view_21 = torch.ops.aten.view.default(view_20, [1632, 768])
        permute_9 = torch.ops.aten.permute.default(view_21, [1, 0])
        slice_2 = torch.ops.aten.slice.Tensor(primals_2, 1, 286, 337);  primals_2 = None
        permute = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        clone = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view = torch.ops.aten.view.default(clone, [1632, 4096]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute)
        view_1 = torch.ops.aten.view.default(mm, [32, 51, 768]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = primals_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul = torch.ops.aten.mul.Tensor(add, rsqrt)
        mul_1 = torch.ops.aten.mul.Tensor(primals_5, mul)
        permute_1 = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        addmm = torch.ops.aten.addmm.default(primals_7, primals_3, permute_1);  primals_7 = permute_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        permute_2 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        addmm_1 = torch.ops.aten.addmm.default(primals_9, primals_3, permute_2);  primals_9 = permute_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        add_2 = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2 = torch.ops.aten.mul.Tensor(add_2, mul_1)
        add_3 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        view_4 = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        permute_5 = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        mm_2 = torch.ops.aten.mm.default(view_4, permute_5)
        sigmoid_2 = torch.ops.aten.sigmoid.default(mm_2)
        mul_7 = torch.ops.aten.mul.Tensor(mm_2, sigmoid_2)
        permute_6 = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        mm_3 = torch.ops.aten.mm.default(view_4, permute_6)
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, mm_3)
        mm_5 = torch.ops.aten.mm.default(permute_9, mul_8);  permute_9 = mul_8 = None
        permute_7 = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_11 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_6 = torch.ops.aten.mm.default(view_21, permute_11);  permute_11 = None
        mul_9 = torch.ops.aten.mul.Tensor(mm_6, mul_7);  mul_7 = None
        mul_10 = torch.ops.aten.mul.Tensor(mm_6, mm_3);  mm_6 = mm_3 = None
        permute_13 = torch.ops.aten.permute.default(mul_9, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_13, view_4);  permute_13 = None
        permute_15 = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_8 = torch.ops.aten.mm.default(mul_9, permute_15);  mul_9 = permute_15 = None
        full_default_1 = torch.ops.aten.full.default([1632, 704], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub = torch.ops.aten.sub.Tensor(full_default_1, sigmoid_2);  full_default_1 = None
        mul_11 = torch.ops.aten.mul.Tensor(mm_2, sub);  mm_2 = sub = None
        add_9 = torch.ops.aten.add.Scalar(mul_11, 1);  mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(sigmoid_2, add_9);  sigmoid_2 = add_9 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_10, mul_12);  mul_10 = mul_12 = None
        permute_18 = torch.ops.aten.permute.default(mul_13, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_18, view_4);  permute_18 = None
        permute_20 = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_10 = torch.ops.aten.mm.default(mul_13, permute_20);  mul_13 = permute_20 = None
        add_10 = torch.ops.aten.add.Tensor(mm_8, mm_10);  mm_8 = mm_10 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(view_21, torch.bfloat16);  view_21 = None
        empty_7 = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        permute_3 = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        mm_1 = torch.ops.aten.mm.default(view_4, permute_3)
        sigmoid = torch.ops.aten.sigmoid.default(mm_1);  mm_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_11, 0);  primals_11 = None
        add_4 = torch.ops.aten.add.Tensor(sigmoid, unsqueeze_2);  unsqueeze_2 = None
        topk = torch.ops.aten.topk.default(add_4, 4);  add_4 = None
        getitem_1 = topk[1];  topk = None
        view_7 = torch.ops.aten.view.default(getitem_1, [6528])
        sort = torch.ops.aten.sort.stable(view_7, stable = True)
        getitem_4 = sort[1];  sort = None
        sort_1 = torch.ops.aten.sort.default(getitem_4);  getitem_4 = None
        getitem_6 = sort_1[1];  sort_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(getitem_6, torch.int32);  getitem_6 = None
        view_9 = torch.ops.aten.view.default(convert_element_type_7, [1632, 4]);  convert_element_type_7 = None
        triton_kernel_wrapper_functional_proxy_6 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 1, constant_args_idx = 48, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': convert_element_type_11, 'O': empty_7, 'index': view_9}, tensors_to_clone = ['O']);  convert_element_type_11 = empty_7 = None
        getitem_12 = triton_kernel_wrapper_functional_proxy_6['O'];  triton_kernel_wrapper_functional_proxy_6 = None
        empty_8 = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        full_default = torch.ops.aten.full.default([32], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 42, grid = [(7, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'out_ptr': full_default, 'x_ptr': view_7}, tensors_to_clone = ['out_ptr']);  full_default = view_7 = None
        getitem_2 = triton_kernel_wrapper_functional_proxy['out_ptr'];  triton_kernel_wrapper_functional_proxy = None
        cumsum = torch.ops.aten.cumsum.default(getitem_2, 0);  getitem_2 = None
        triton_kernel_wrapper_functional_proxy_7 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 49, grid = [(52, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem_12, 'b_ptr': convert_element_type_6, 'c_ptr': empty_8, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  convert_element_type_6 = empty_8 = None
        getitem_13 = triton_kernel_wrapper_functional_proxy_7['c_ptr'];  triton_kernel_wrapper_functional_proxy_7 = None
        empty_9 = torch.ops.aten.empty.memory_format([32, 768, 512], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_22 = torch.ops.aten.permute.default(empty_9, [0, 1, 2]);  empty_9 = None
        gather = torch.ops.aten.gather.default(sigmoid, 1, getitem_1)
        sum_2 = torch.ops.aten.sum.dim_IntList(gather, [-1], True)
        add_6 = torch.ops.aten.add.Tensor(sum_2, 1e-20);  sum_2 = None
        div = torch.ops.aten.div.Tensor(gather, add_6);  gather = None
        mul_3 = torch.ops.aten.mul.Tensor(div, 4.0)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        empty = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_1 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 1, constant_args_idx = 43, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': convert_element_type_2, 'O': empty, 'index': view_9}, tensors_to_clone = ['O']);  convert_element_type_2 = empty = None
        getitem_7 = triton_kernel_wrapper_functional_proxy_1['O'];  triton_kernel_wrapper_functional_proxy_1 = None
        empty_1 = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_2 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 44, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem_7, 'b_ptr': convert_element_type_4, 'c_ptr': empty_1, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  empty_1 = None
        getitem_8 = triton_kernel_wrapper_functional_proxy_2['c_ptr'];  triton_kernel_wrapper_functional_proxy_2 = None
        empty_2 = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_3 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 45, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem_7, 'b_ptr': convert_element_type_5, 'c_ptr': empty_2, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  empty_2 = None
        getitem_9 = triton_kernel_wrapper_functional_proxy_3['c_ptr'];  triton_kernel_wrapper_functional_proxy_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(getitem_8, torch.float32)
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_8)
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_8, sigmoid_1);  convert_element_type_8 = sigmoid_1 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_9, getitem_9)
        view_10 = torch.ops.aten.view.default(convert_element_type_3, [-1, 1]);  convert_element_type_3 = None
        empty_3 = torch.ops.aten.empty.memory_format([6528, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_4 = torch.ops.aten.permute.default(empty_3, [0, 1]);  empty_3 = None
        view_11 = torch.ops.aten.view.default(view_9, [6528])
        index_put = torch.ops.aten.index_put.default(permute_4, [view_11], view_10);  permute_4 = view_10 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, index_put)
        triton_kernel_wrapper_functional_proxy_8 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 4, constant_args_idx = 50, grid = [(24, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem_12, 'b_ptr': mul_6, 'c_ptr': permute_22, 'cumsum_K': cumsum}, tensors_to_clone = ['c_ptr']);  getitem_12 = mul_6 = permute_22 = None
        getitem_14 = triton_kernel_wrapper_functional_proxy_8['c_ptr'];  triton_kernel_wrapper_functional_proxy_8 = None
        mul_14 = torch.ops.aten.mul.Tensor(getitem_13, index_put);  index_put = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_5, getitem_13);  mul_5 = getitem_13 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_15, [-1]);  mul_15 = None
        index = torch.ops.aten.index.Tensor(sum_3, [view_11]);  sum_3 = view_11 = None
        view_25 = torch.ops.aten.view.default(index, [1632, 4]);  index = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_14, getitem_9);  getitem_9 = None
        mul_18 = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_14);  convert_element_type_9 = mul_14 = None
        empty_10 = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_9 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 51, grid = [(78, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_18, 'b_ptr': convert_element_type_5, 'c_ptr': empty_10, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  convert_element_type_5 = empty_10 = None
        getitem_15 = triton_kernel_wrapper_functional_proxy_9['c_ptr'];  triton_kernel_wrapper_functional_proxy_9 = None
        empty_11 = torch.ops.aten.empty.memory_format([32, 512, 768], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_23 = torch.ops.aten.permute.default(empty_11, [0, 1, 2]);  empty_11 = None
        triton_kernel_wrapper_functional_proxy_10 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 4, constant_args_idx = 52, grid = [(24, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_18, 'b_ptr': getitem_7, 'c_ptr': permute_23, 'cumsum_K': cumsum}, tensors_to_clone = ['c_ptr']);  mul_18 = permute_23 = None
        getitem_16 = triton_kernel_wrapper_functional_proxy_10['c_ptr'];  triton_kernel_wrapper_functional_proxy_10 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(getitem_8)
        full_default_2 = torch.ops.aten.full.default([6528, 512], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_1 = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_5);  full_default_2 = None
        mul_19 = torch.ops.aten.mul.Tensor(getitem_8, sub_1);  getitem_8 = sub_1 = None
        add_11 = torch.ops.aten.add.Scalar(mul_19, 1);  mul_19 = None
        mul_20 = torch.ops.aten.mul.Tensor(sigmoid_5, add_11);  sigmoid_5 = add_11 = None
        mul_21 = torch.ops.aten.mul.Tensor(mul_17, mul_20);  mul_17 = mul_20 = None
        empty_13 = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_11 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 53, grid = [(78, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_21, 'b_ptr': convert_element_type_4, 'c_ptr': empty_13, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  convert_element_type_4 = empty_13 = None
        getitem_17 = triton_kernel_wrapper_functional_proxy_11['c_ptr'];  triton_kernel_wrapper_functional_proxy_11 = None
        empty_14 = torch.ops.aten.empty.memory_format([32, 512, 768], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_25 = torch.ops.aten.permute.default(empty_14, [0, 1, 2]);  empty_14 = None
        triton_kernel_wrapper_functional_proxy_12 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 4, constant_args_idx = 54, grid = [(24, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_21, 'b_ptr': getitem_7, 'c_ptr': permute_25, 'cumsum_K': cumsum}, tensors_to_clone = ['c_ptr']);  mul_21 = getitem_7 = permute_25 = cumsum = None
        getitem_18 = triton_kernel_wrapper_functional_proxy_12['c_ptr'];  triton_kernel_wrapper_functional_proxy_12 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_17, getitem_15);  getitem_17 = getitem_15 = None
        empty_15 = torch.ops.aten.empty.memory_format([1632, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_13 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 3, constant_args_idx = 55, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': add_12, 'Y': empty_15, 'index': view_9}, tensors_to_clone = ['Y']);  add_12 = empty_15 = view_9 = None
        getitem_19 = triton_kernel_wrapper_functional_proxy_13['Y'];  triton_kernel_wrapper_functional_proxy_13 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(getitem_14, torch.float32);  getitem_14 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(getitem_18, torch.float32);  getitem_18 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(getitem_19, torch.float32);  getitem_19 = None
        add_13 = torch.ops.aten.add.Tensor(add_10, convert_element_type_18);  add_10 = convert_element_type_18 = None
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_17, 4.0);  convert_element_type_17 = None
        div_2 = torch.ops.aten.div.Tensor(div, add_6);  div = None
        neg = torch.ops.aten.neg.default(mul_22)
        mul_23 = torch.ops.aten.mul.Tensor(neg, div_2);  neg = div_2 = None
        div_3 = torch.ops.aten.div.Tensor(mul_22, add_6);  mul_22 = add_6 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(mul_23, [1], True);  mul_23 = None
        expand = torch.ops.aten.expand.default(sum_4, [1632, 4]);  sum_4 = None
        add_14 = torch.ops.aten.add.Tensor(div_3, expand);  div_3 = expand = None
        full_default_3 = torch.ops.aten.full.default([1632, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default_3, 1, getitem_1, add_14);  full_default_3 = add_14 = None
        sub_2 = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_24 = torch.ops.aten.mul.Tensor(sigmoid, sub_2);  sigmoid = sub_2 = None
        mul_25 = torch.ops.aten.mul.Tensor(scatter_add, mul_24);  scatter_add = mul_24 = None
        add_15 = torch.ops.aten.add.Tensor(tangents_2, mul_25);  tangents_2 = mul_25 = None
        permute_26 = torch.ops.aten.permute.default(add_15, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_26, view_4);  permute_26 = view_4 = None
        permute_28 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_12 = torch.ops.aten.mm.default(add_15, permute_28);  add_15 = permute_28 = None
        add_16 = torch.ops.aten.add.Tensor(add_13, mm_12);  add_13 = mm_12 = None
        view_28 = torch.ops.aten.view.default(add_16, [32, 51, 768]);  add_16 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(view_28, [1], True)
        mul_26 = torch.ops.aten.mul.Tensor(view_28, add_2);  add_2 = None
        mul_27 = torch.ops.aten.mul.Tensor(view_28, mul_1);  view_28 = mul_1 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_27, [1], True);  mul_27 = None
        squeeze = torch.ops.aten.squeeze.dim(sum_5, 1);  sum_5 = None
        permute_30 = torch.ops.aten.permute.default(squeeze, [1, 0])
        mm_13 = torch.ops.aten.mm.default(permute_30, primals_3);  permute_30 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(squeeze, [0], True);  squeeze = None
        view_29 = torch.ops.aten.view.default(sum_7, [768]);  sum_7 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_6, 1);  sum_6 = None
        permute_33 = torch.ops.aten.permute.default(squeeze_1, [1, 0])
        mm_14 = torch.ops.aten.mm.default(permute_33, primals_3);  permute_33 = primals_3 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(squeeze_1, [0], True);  squeeze_1 = None
        view_30 = torch.ops.aten.view.default(sum_8, [768]);  sum_8 = None
        mul_28 = torch.ops.aten.mul.Tensor(mul_26, primals_5);  primals_5 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_26, mul);  mul_26 = mul = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_29, [0, 1], True);  mul_29 = None
        view_31 = torch.ops.aten.view.default(sum_9, [768]);  sum_9 = None
        mul_30 = torch.ops.aten.mul.Tensor(mul_28, add)
        mul_31 = torch.ops.aten.mul.Tensor(mul_28, rsqrt);  mul_28 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_30, [2], True);  mul_30 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_32 = torch.ops.aten.mul.Scalar(sum_10, -0.5);  sum_10 = None
        mul_33 = torch.ops.aten.mul.Tensor(mul_32, pow_2);  mul_32 = pow_2 = None
        expand_1 = torch.ops.aten.expand.default(mul_33, [32, 51, 768]);  mul_33 = None
        div_4 = torch.ops.aten.div.Scalar(expand_1, 768);  expand_1 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(add, 1.0);  add = None
        mul_34 = torch.ops.aten.mul.Scalar(pow_3, 2.0);  pow_3 = None
        mul_35 = torch.ops.aten.mul.Tensor(div_4, mul_34);  div_4 = mul_34 = None
        add_17 = torch.ops.aten.add.Tensor(mul_31, mul_35);  mul_31 = mul_35 = None
        add_18 = torch.ops.aten.add.Tensor(add_17, view_20);  add_17 = view_20 = None
        view_32 = torch.ops.aten.view.default(add_18, [1632, 768])
        permute_36 = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_15 = torch.ops.aten.mm.default(permute_36, view);  permute_36 = view = None
        permute_38 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_16 = torch.ops.aten.mm.default(view_32, permute_38);  view_32 = permute_38 = None
        view_33 = torch.ops.aten.view.default(mm_16, [32, 51, 4096]);  mm_16 = None
        full_default_4 = torch.ops.aten.full.default([32, 337, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter = torch.ops.aten.slice_scatter.default(full_default_4, view_33, 1, 286, 337);  full_default_4 = view_33 = None
        view_6 = torch.ops.aten.view.default(getitem_1, [-1]);  getitem_1 = None
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_6, -1);  view_6 = None
        eq = torch.ops.aten.eq.Tensor(unsqueeze_3, iota);  unsqueeze_3 = iota = None
        convert_element_type = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
        sum_1 = torch.ops.aten.sum.dim_IntList(convert_element_type, [0]);  convert_element_type = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        add_5 = torch.ops.aten.add.Tensor(primals_13, convert_element_type_1);  convert_element_type_1 = None
        copy__1 = torch.ops.aten.copy_.default(primals_13, add_5);  primals_13 = add_5 = copy__1 = None
        return (add_18, slice_scatter, None, mm_15, view_31, mm_14, view_30, mm_13, view_29, mm_11, None, None, None, convert_element_type_16, convert_element_type_15, convert_element_type_14, mm_9, mm_7, mm_5)
        
def load_args(reader):
    buf0 = reader.storage(None, 5013504, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 51, 768), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 176685056, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32, 337, 4096), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 768), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 12582912, device=device(type='cuda', index=0))
    reader.tensor(buf3, (768, 4096), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf4, (768,), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf5, (768, 768), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf6, (768,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf7, (768, 768), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf8, (768,), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf9, (32, 768), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf10, (32,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf11, (32,), is_leaf=True)  # primals_13
    buf12 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32, 512, 768), is_leaf=True)  # primals_14
    buf13 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32, 512, 768), is_leaf=True)  # primals_15
    buf14 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32, 768, 512), is_leaf=True)  # primals_16
    buf15 = reader.storage(None, 2162688, device=device(type='cuda', index=0))
    reader.tensor(buf15, (704, 768), is_leaf=True)  # primals_17
    buf16 = reader.storage(None, 2162688, device=device(type='cuda', index=0))
    reader.tensor(buf16, (704, 768), is_leaf=True)  # primals_18
    buf17 = reader.storage(None, 2162688, device=device(type='cuda', index=0))
    reader.tensor(buf17, (768, 704), is_leaf=True)  # primals_19
    buf18 = reader.storage(None, 5013504, device=device(type='cuda', index=0))
    reader.tensor(buf18, (32, 51, 768), is_leaf=True)  # tangents_1
    buf19 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1632, 32), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)