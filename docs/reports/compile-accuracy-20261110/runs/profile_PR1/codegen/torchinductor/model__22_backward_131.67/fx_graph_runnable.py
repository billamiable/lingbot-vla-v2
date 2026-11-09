
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = '7e822c46-d4c0-43c7-89eb-eaf111a6d924'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_87kt43ql/7e822c46-d4c0-43c7-89eb-eaf111a6d924_utzpe_1w/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_PR1/codegen'
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

    
    
    def forward(self, primals_4, primals_11, primals_13, view, view_2, addmm_2, view_4, tangents_1):
        slice_18 = torch.ops.aten.slice.Tensor(tangents_1, 1, 0, 1)
        slice_19 = torch.ops.aten.slice.Tensor(tangents_1, 1, 1, 51);  tangents_1 = None
        squeeze = torch.ops.aten.squeeze.dim(slice_18, 1);  slice_18 = None
        full_default_6 = torch.ops.aten.full.default([32, 768], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(full_default_6, squeeze, 0, 0, 9223372036854775807);  full_default_6 = squeeze = None
        clone = torch.ops.aten.clone.default(slice_19, memory_format = torch.contiguous_format);  slice_19 = None
        view_6 = torch.ops.aten.view.default(clone, [1600, 768]);  clone = None
        permute_3 = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_4 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm = torch.ops.aten.mm.default(view_6, permute_4);  permute_4 = None
        permute_5 = torch.ops.aten.permute.default(view_6, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_5, view_4);  permute_5 = view_4 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(view_6, [0], True);  view_6 = None
        view_7 = torch.ops.aten.view.default(sum_1, [768]);  sum_1 = None
        view_8 = torch.ops.aten.view.default(mm, [32, 50, 768]);  mm = None
        view_3 = torch.ops.aten.view.default(addmm_2, [32, 50, 768]);  addmm_2 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(view_3)
        full_default_7 = torch.ops.aten.full.default([32, 50, 768], 1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        sub_3 = torch.ops.aten.sub.Tensor(full_default_7, sigmoid_1);  full_default_7 = None
        mul_9 = torch.ops.aten.mul.Tensor(view_3, sub_3);  view_3 = sub_3 = None
        add_3 = torch.ops.aten.add.Scalar(mul_9, 1);  mul_9 = None
        mul_10 = torch.ops.aten.mul.Tensor(sigmoid_1, add_3);  sigmoid_1 = add_3 = None
        mul_11 = torch.ops.aten.mul.Tensor(view_8, mul_10);  view_8 = mul_10 = None
        view_9 = torch.ops.aten.view.default(mul_11, [1600, 768]);  mul_11 = None
        permute_2 = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_9 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_2 = torch.ops.aten.mm.default(view_9, permute_9);  permute_9 = None
        permute_10 = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_10, view_2);  permute_10 = view_2 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(view_9, [0], True);  view_9 = None
        view_10 = torch.ops.aten.view.default(sum_2, [768]);  sum_2 = None
        view_11 = torch.ops.aten.view.default(mm_2, [32, 50, 1536]);  mm_2 = None
        slice_20 = torch.ops.aten.slice.Tensor(view_11, 2, 0, 768);  view_11 = None
        view_12 = torch.ops.aten.view.default(slice_20, [1600, 768]);  slice_20 = None
        permute_13 = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_4 = torch.ops.aten.mm.default(permute_13, view);  permute_13 = view = None
        sum_3 = torch.ops.aten.sum.dim_IntList(view_12, [0], True);  view_12 = None
        view_13 = torch.ops.aten.view.default(sum_3, [768]);  sum_3 = None
        permute_16 = torch.ops.aten.permute.default(slice_scatter_2, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_16, primals_4);  permute_16 = primals_4 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(slice_scatter_2, [0], True);  slice_scatter_2 = None
        view_14 = torch.ops.aten.view.default(sum_4, [768]);  sum_4 = None
        return (None, None, None, None, mm_5, view_14, None, mm_4, view_13, None, mm_3, view_10, mm_1, view_7)
        
def load_args(reader):
    buf0 = reader.storage(None, 3520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf1 = reader.storage(None, 2359296, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (768, 1536), dtype=torch.bfloat16, is_leaf=True)  # primals_11
    buf2 = reader.storage(None, 1179648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (768, 768), dtype=torch.bfloat16, is_leaf=True)  # primals_13
    buf3 = reader.storage(None, 176000, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (1600, 55), dtype=torch.bfloat16, is_leaf=True)  # view
    buf4 = reader.storage(None, 4915200, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (1600, 1536), dtype=torch.bfloat16, is_leaf=True)  # view_2
    buf5 = reader.storage(None, 2457600, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (1600, 768), dtype=torch.bfloat16, is_leaf=True)  # addmm_2
    buf6 = reader.storage(None, 2457600, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (1600, 768), dtype=torch.bfloat16, is_leaf=True)  # view_4
    buf7 = reader.storage(None, 2506752, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (32, 51, 768), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)