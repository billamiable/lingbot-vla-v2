
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

    
    
    def forward(self, unsqueeze_4, tangents_1, tangents_2):
        expand_3 = torch.ops.aten.expand.default(unsqueeze_4, [32, 337, 2, 64]);  unsqueeze_4 = None
        clone = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12 = torch.ops.aten.view.default(clone, [32, 337, 128]);  clone = None
        sin = torch.ops.aten.sin.default(view_12)
        mul_1 = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(mul_1, 2);  mul_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(tangents_2, unsqueeze_6)
        slice_17 = torch.ops.aten.slice.Tensor(mul_6, 3, 0, 64)
        slice_18 = torch.ops.aten.slice.Tensor(mul_6, 3, 64, 128);  mul_6 = None
        neg_2 = torch.ops.aten.neg.default(slice_17);  slice_17 = None
        full_default = torch.ops.aten.full.default([32, 337, 8, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(full_default, neg_2, 3, 64, 9223372036854775807);  neg_2 = None
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(full_default, slice_18, 3, 0, 64);  full_default = slice_18 = None
        add_2 = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        cos = torch.ops.aten.cos.default(view_12);  view_12 = None
        mul = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(mul, 2);  mul = None
        mul_7 = torch.ops.aten.mul.Tensor(tangents_2, unsqueeze_5);  tangents_2 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, mul_7);  add_2 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(tangents_1, unsqueeze_6);  unsqueeze_6 = None
        slice_19 = torch.ops.aten.slice.Tensor(mul_8, 3, 0, 64)
        slice_20 = torch.ops.aten.slice.Tensor(mul_8, 3, 64, 128);  mul_8 = None
        neg_3 = torch.ops.aten.neg.default(slice_19);  slice_19 = None
        full_default_2 = torch.ops.aten.full.default([32, 337, 32, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(full_default_2, neg_3, 3, 64, 9223372036854775807);  neg_3 = None
        slice_scatter_5 = torch.ops.aten.slice_scatter.default(full_default_2, slice_20, 3, 0, 64);  full_default_2 = slice_20 = None
        add_4 = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(tangents_1, unsqueeze_5);  tangents_1 = unsqueeze_5 = None
        add_5 = torch.ops.aten.add.Tensor(add_4, mul_9);  add_4 = mul_9 = None
        return (None, None, add_5, add_3)
        
def load_args(reader):
    buf0 = reader.storage(None, 8282112, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 337, 1, 64), (21568, 1, 21568, 337), is_leaf=True)  # unsqueeze_4
    buf1 = reader.storage(None, 176685056, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32, 337, 32, 128), is_leaf=True)  # tangents_1
    buf2 = reader.storage(None, 44171264, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 337, 8, 128), is_leaf=True)  # tangents_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)