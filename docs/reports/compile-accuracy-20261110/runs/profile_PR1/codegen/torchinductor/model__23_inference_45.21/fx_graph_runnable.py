
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

    
    
    def forward(self, arg0_1, arg1_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        expand = torch.ops.aten.expand.default(unsqueeze_2, [3, 32, -1, 1]);  unsqueeze_2 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(unsqueeze_3, torch.float32);  unsqueeze_3 = None
        expand_1 = torch.ops.aten.expand.default(expand, [3, 32, 64, 1]);  expand = None
        view = torch.ops.aten.view.default(expand_1, [96, 64, 1]);  expand_1 = None
        expand_2 = torch.ops.aten.expand.default(convert_element_type, [3, 32, 1, 337]);  convert_element_type = None
        view_1 = torch.ops.aten.view.default(expand_2, [96, 1, 337]);  expand_2 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2 = torch.ops.aten.view.default(bmm, [3, 32, 64, 337])
        permute = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
        select = torch.ops.aten.select.int(permute, 0, 0)
        select_1 = torch.ops.aten.select.int(permute, 0, 1);  permute = None
        slice_5 = torch.ops.aten.slice.Tensor(select_1, 2, 1, 60, 3);  select_1 = None
        slice_6 = torch.ops.aten.slice.Tensor(select, 2, 1, 60, 3);  select = slice_6 = None
        view_3 = torch.ops.aten.view.default(bmm, [3, 32, 64, 337]);  bmm = None
        permute_1 = torch.ops.aten.permute.default(view_3, [0, 1, 3, 2]);  view_3 = None
        select_2 = torch.ops.aten.select.int(permute_1, 0, 0)
        slice_scatter = torch.ops.aten.slice_scatter.default(select_2, slice_5, 2, 1, 60, 3);  select_2 = slice_5 = None
        select_scatter = torch.ops.aten.select_scatter.default(permute_1, slice_scatter, 0, 0);  permute_1 = slice_scatter = None
        permute_2 = torch.ops.aten.permute.default(select_scatter, [0, 1, 3, 2]);  select_scatter = None
        view_4 = torch.ops.aten.view.default(permute_2, [96, 64, 337]);  permute_2 = None
        view_6 = torch.ops.aten.view.default(view_4, [3, 32, 64, 337])
        permute_4 = torch.ops.aten.permute.default(view_6, [0, 1, 3, 2]);  view_6 = None
        select_5 = torch.ops.aten.select.int(permute_4, 0, 0);  permute_4 = None
        slice_10 = torch.ops.aten.slice.Tensor(select_5, 2, 2, 60, 3);  select_5 = slice_10 = None
        view_7 = torch.ops.aten.view.default(view_4, [3, 32, 64, 337])
        permute_5 = torch.ops.aten.permute.default(view_7, [0, 1, 3, 2]);  view_7 = None
        select_6 = torch.ops.aten.select.int(permute_5, 0, 2);  permute_5 = None
        slice_11 = torch.ops.aten.slice.Tensor(select_6, 2, 2, 60, 3);  select_6 = None
        view_8 = torch.ops.aten.view.default(view_4, [3, 32, 64, 337]);  view_4 = None
        permute_6 = torch.ops.aten.permute.default(view_8, [0, 1, 3, 2]);  view_8 = None
        select_7 = torch.ops.aten.select.int(permute_6, 0, 0)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(select_7, slice_11, 2, 2, 60, 3);  select_7 = slice_11 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(permute_6, slice_scatter_1, 0, 0);  permute_6 = slice_scatter_1 = None
        permute_7 = torch.ops.aten.permute.default(select_scatter_1, [0, 1, 3, 2]);  select_scatter_1 = None
        permute_9 = torch.ops.aten.permute.default(permute_7, [0, 1, 3, 2]);  permute_7 = None
        select_9 = torch.ops.aten.select.int(permute_9, 0, 0);  permute_9 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(select_9, 2);  select_9 = None
        expand_3 = torch.ops.aten.expand.default(unsqueeze_4, [32, 337, 2, 64]);  unsqueeze_4 = None
        clone = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_12 = torch.ops.aten.view.default(clone, [32, 337, 128]);  clone = None
        cos = torch.ops.aten.cos.default(view_12)
        mul = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        sin = torch.ops.aten.sin.default(view_12);  view_12 = None
        mul_1 = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        return (mul, mul_1)
        
def load_args(reader):
    buf0 = reader.storage(None, 258816, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (3, 32, 337), dtype=torch.int64, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (64,), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)