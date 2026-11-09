
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
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19):
        slice_2 = torch.ops.aten.slice.Tensor(primals_2, 1, 286, 337)
        permute = torch.ops.aten.permute.default(primals_4, [1, 0])
        clone = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view = torch.ops.aten.view.default(clone, [1632, 4096]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1 = torch.ops.aten.view.default(mm, [32, 51, 768]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul = torch.ops.aten.mul.Tensor(add, rsqrt);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(primals_5, mul);  mul = None
        permute_1 = torch.ops.aten.permute.default(primals_6, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_7, primals_3, permute_1);  permute_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(addmm, 1);  addmm = None
        permute_2 = torch.ops.aten.permute.default(primals_8, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_9, primals_3, permute_2);  permute_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(addmm_1, 1);  addmm_1 = None
        add_2 = torch.ops.aten.add.Tensor(unsqueeze, 1);  unsqueeze = None
        mul_2 = torch.ops.aten.mul.Tensor(add_2, mul_1);  add_2 = mul_1 = None
        add_3 = torch.ops.aten.add.Tensor(mul_2, unsqueeze_1);  mul_2 = unsqueeze_1 = None
        view_4 = torch.ops.aten.view.default(add_3, [-1, 768]);  add_3 = None
        permute_3 = torch.ops.aten.permute.default(primals_10, [1, 0])
        mm_1 = torch.ops.aten.mm.default(view_4, permute_3);  permute_3 = None
        sigmoid = torch.ops.aten.sigmoid.default(mm_1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_11, 0)
        add_4 = torch.ops.aten.add.Tensor(sigmoid, unsqueeze_2);  unsqueeze_2 = None
        topk = torch.ops.aten.topk.default(add_4, 4);  add_4 = None
        getitem_1 = topk[1];  topk = None
        gather = torch.ops.aten.gather.default(sigmoid, 1, getitem_1);  sigmoid = None
        mean_1 = torch.ops.aten.mean.default(gather)
        view_5 = torch.ops.aten.view.default(mean_1, [1]);  mean_1 = None
        view_6 = torch.ops.aten.view.default(getitem_1, [-1])
        iota = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_6, -1);  view_6 = None
        eq = torch.ops.aten.eq.Tensor(unsqueeze_3, iota);  unsqueeze_3 = iota = None
        convert_element_type = torch.ops.prims.convert_element_type.default(eq, torch.int64);  eq = None
        sum_1 = torch.ops.aten.sum.dim_IntList(convert_element_type, [0]);  convert_element_type = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        add_5 = torch.ops.aten.add.Tensor(primals_13, convert_element_type_1);  convert_element_type_1 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(gather, [-1], True)
        add_6 = torch.ops.aten.add.Tensor(sum_2, 1e-20);  sum_2 = None
        div = torch.ops.aten.div.Tensor(gather, add_6);  gather = add_6 = None
        mul_3 = torch.ops.aten.mul.Tensor(div, 4.0);  div = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16)
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16);  mul_3 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16)
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16)
        view_7 = torch.ops.aten.view.default(getitem_1, [6528]);  getitem_1 = None
        full_default = torch.ops.aten.full.default([32], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 406, grid = [(7, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'out_ptr': full_default, 'x_ptr': view_7}, tensors_to_clone = ['out_ptr']);  full_default = None
        getitem_2 = triton_kernel_wrapper_functional_proxy['out_ptr'];  triton_kernel_wrapper_functional_proxy = None
        sort = torch.ops.aten.sort.stable(view_7, stable = True);  view_7 = None
        getitem_4 = sort[1];  sort = None
        sort_1 = torch.ops.aten.sort.default(getitem_4);  getitem_4 = None
        getitem_6 = sort_1[1];  sort_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(getitem_6, torch.int32);  getitem_6 = None
        view_9 = torch.ops.aten.view.default(convert_element_type_7, [1632, 4]);  convert_element_type_7 = None
        empty = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_1 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 1, constant_args_idx = 407, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': convert_element_type_2, 'O': empty, 'index': view_9}, tensors_to_clone = ['O']);  convert_element_type_2 = empty = None
        getitem_7 = triton_kernel_wrapper_functional_proxy_1['O'];  triton_kernel_wrapper_functional_proxy_1 = None
        cumsum = torch.ops.aten.cumsum.default(getitem_2, 0);  getitem_2 = None
        empty_1 = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_2 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 408, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem_7, 'b_ptr': convert_element_type_4, 'c_ptr': empty_1, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  convert_element_type_4 = empty_1 = None
        getitem_8 = triton_kernel_wrapper_functional_proxy_2['c_ptr'];  triton_kernel_wrapper_functional_proxy_2 = None
        empty_2 = torch.ops.aten.empty.memory_format([6528, 512], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_3 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 409, grid = [(204, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': getitem_7, 'b_ptr': convert_element_type_5, 'c_ptr': empty_2, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  getitem_7 = convert_element_type_5 = empty_2 = None
        getitem_9 = triton_kernel_wrapper_functional_proxy_3['c_ptr'];  triton_kernel_wrapper_functional_proxy_3 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(getitem_8, torch.float32);  getitem_8 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_8)
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_8, sigmoid_1);  convert_element_type_8 = sigmoid_1 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_9, getitem_9);  convert_element_type_9 = getitem_9 = None
        view_10 = torch.ops.aten.view.default(convert_element_type_3, [-1, 1]);  convert_element_type_3 = None
        empty_3 = torch.ops.aten.empty.memory_format([6528, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        permute_4 = torch.ops.aten.permute.default(empty_3, [0, 1]);  empty_3 = None
        view_11 = torch.ops.aten.view.default(view_9, [6528])
        index_put = torch.ops.aten.index_put.default(permute_4, [view_11], view_10);  permute_4 = view_11 = view_10 = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_5, index_put);  mul_5 = index_put = None
        empty_4 = torch.ops.aten.empty.memory_format([6528, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_4 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 2, constant_args_idx = 410, grid = [(306, 32, 1)], tma_descriptor_metadata = {}, kwargs = {'a_ptr': mul_6, 'b_ptr': convert_element_type_6, 'c_ptr': empty_4, 'cumsum_M': cumsum}, tensors_to_clone = ['c_ptr']);  mul_6 = convert_element_type_6 = empty_4 = cumsum = None
        getitem_10 = triton_kernel_wrapper_functional_proxy_4['c_ptr'];  triton_kernel_wrapper_functional_proxy_4 = None
        empty_5 = torch.ops.aten.empty.memory_format([1632, 768], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        triton_kernel_wrapper_functional_proxy_5 = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 3, constant_args_idx = 411, grid = [(1632, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'X': getitem_10, 'Y': empty_5, 'index': view_9}, tensors_to_clone = ['Y']);  getitem_10 = empty_5 = view_9 = None
        getitem_11 = triton_kernel_wrapper_functional_proxy_5['Y'];  triton_kernel_wrapper_functional_proxy_5 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
        permute_5 = torch.ops.aten.permute.default(primals_17, [1, 0])
        mm_2 = torch.ops.aten.mm.default(view_4, permute_5);  permute_5 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(mm_2)
        mul_7 = torch.ops.aten.mul.Tensor(mm_2, sigmoid_2);  mm_2 = sigmoid_2 = None
        permute_6 = torch.ops.aten.permute.default(primals_18, [1, 0])
        mm_3 = torch.ops.aten.mm.default(view_4, permute_6);  view_4 = permute_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, mm_3);  mul_7 = mm_3 = None
        permute_7 = torch.ops.aten.permute.default(primals_19, [1, 0])
        mm_4 = torch.ops.aten.mm.default(mul_8, permute_7);  mul_8 = permute_7 = None
        add_7 = torch.ops.aten.add.Tensor(convert_element_type_10, mm_4);  convert_element_type_10 = mm_4 = None
        view_14 = torch.ops.aten.view.default(add_7, [32, 51, 768]);  add_7 = None
        add_8 = torch.ops.aten.add.Tensor(view_14, add);  view_14 = add = None
        copy_ = torch.ops.aten.copy_.default(primals_12, view_5);  primals_12 = view_5 = copy_ = None
        copy__1 = torch.ops.aten.copy_.default(primals_13, add_5);  add_5 = copy__1 = None
        return (add_8, mm_1, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19)
        
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
    buf11 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1,), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32, 512, 768), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32, 512, 768), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 50331648, device=device(type='cuda', index=0))
    reader.tensor(buf15, (32, 768, 512), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 2162688, device=device(type='cuda', index=0))
    reader.tensor(buf16, (704, 768), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 2162688, device=device(type='cuda', index=0))
    reader.tensor(buf17, (704, 768), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 2162688, device=device(type='cuda', index=0))
    reader.tensor(buf18, (768, 704), is_leaf=True)  # primals_19
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)