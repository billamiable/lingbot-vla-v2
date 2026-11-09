
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        sigmoid = torch.ops.aten.sigmoid.default(primals_4)
        sum_1 = torch.ops.aten.sum.dim_IntList(sigmoid, [-1], True)
        div = torch.ops.aten.div.Tensor(sigmoid, sum_1);  sigmoid = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_1, 1)
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, 32]);  unsqueeze = None
        full_default = torch.ops.aten.full.default([1152, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add = torch.ops.aten.scatter_add.default(full_default, 0, expand, div);  expand = div = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_3, 1)
        convert_element_type = torch.ops.prims.convert_element_type.default(unsqueeze_1, torch.float32);  unsqueeze_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(convert_element_type, 1);  convert_element_type = None
        div_1 = torch.ops.aten.div.Tensor(scatter_add, clamp_min);  scatter_add = None
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 0, grid = [(1152, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'logits_ptr': primals_4, 'seg_starts_ptr': primals_2, 'seg_lengths_ptr': primals_3, 'f_out_ptr': full_default}, tensors_to_clone = ['f_out_ptr']);  primals_2 = primals_3 = full_default = None
        getitem = triton_kernel_wrapper_functional_proxy['f_out_ptr'];  triton_kernel_wrapper_functional_proxy = None
        reciprocal = torch.ops.aten.reciprocal.default(clamp_min)
        mul = torch.ops.aten.mul.Tensor(reciprocal, 8.0);  reciprocal = None
        mul_1 = torch.ops.aten.mul.Tensor(getitem, mul);  mul = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, div_1);  mul_1 = div_1 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_2, [-1]);  mul_2 = None
        view = torch.ops.aten.view.default(sum_2, [36, 32]);  sum_2 = None
        mean = torch.ops.aten.mean.dim(view, [1]);  view = None
        unbind = torch.ops.aten.unbind.int(mean);  mean = None
        getitem_1 = unbind[0]
        getitem_2 = unbind[1]
        getitem_3 = unbind[2]
        getitem_4 = unbind[3]
        getitem_5 = unbind[4]
        getitem_6 = unbind[5]
        getitem_7 = unbind[6]
        getitem_8 = unbind[7]
        getitem_9 = unbind[8]
        getitem_10 = unbind[9]
        getitem_11 = unbind[10]
        getitem_12 = unbind[11]
        getitem_13 = unbind[12]
        getitem_14 = unbind[13]
        getitem_15 = unbind[14]
        getitem_16 = unbind[15]
        getitem_17 = unbind[16]
        getitem_18 = unbind[17]
        getitem_19 = unbind[18]
        getitem_20 = unbind[19]
        getitem_21 = unbind[20]
        getitem_22 = unbind[21]
        getitem_23 = unbind[22]
        getitem_24 = unbind[23]
        getitem_25 = unbind[24]
        getitem_26 = unbind[25]
        getitem_27 = unbind[26]
        getitem_28 = unbind[27]
        getitem_29 = unbind[28]
        getitem_30 = unbind[29]
        getitem_31 = unbind[30]
        getitem_32 = unbind[31]
        getitem_33 = unbind[32]
        getitem_34 = unbind[33]
        getitem_35 = unbind[34]
        getitem_36 = unbind[35];  unbind = None
        return (getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, primals_1, primals_4, sum_1, clamp_min, getitem)
        
def load_args(reader):
    buf0 = reader.storage(None, 470016, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (58752,), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 9216, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (1152,), dtype=torch.int64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 9216, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (1152,), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 7520256, device=device(type='cuda', index=0))
    reader.tensor(buf3, (58752, 32), is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)