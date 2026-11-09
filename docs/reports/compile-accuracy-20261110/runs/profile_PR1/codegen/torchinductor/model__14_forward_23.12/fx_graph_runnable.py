
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4):
        cat = torch.ops.aten.cat.default([primals_3, primals_2, primals_1], 2);  primals_3 = primals_2 = primals_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_4, 2);  primals_4 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 66])
        full = torch.ops.aten.full.default([32, 3, 66], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        expand_1 = torch.ops.aten.expand.default(unsqueeze, [-1, -1, 64]);  unsqueeze = None
        slice_3 = torch.ops.aten.slice.Tensor(full, 2, 1, 65)
        copy = torch.ops.aten.copy.default(slice_3, expand_1);  slice_3 = expand_1 = None
        slice_scatter = torch.ops.aten.slice_scatter.default(full, copy, 2, 1, 65);  full = copy = None
        full_default = torch.ops.aten.full.default([32, 3, 66], 151655, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([], 151652, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select = torch.ops.aten.select.int(full_default, 2, 0)
        copy_1 = torch.ops.aten.copy.default(select, full_default_1);  select = full_default_1 = None
        select_scatter = torch.ops.aten.select_scatter.default(full_default, copy_1, 2, 0);  full_default = copy_1 = None
        full_default_2 = torch.ops.aten.full.default([], 151653, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        select_3 = torch.ops.aten.select.int(select_scatter, 2, -1)
        copy_2 = torch.ops.aten.copy.default(select_3, full_default_2);  select_3 = full_default_2 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(select_scatter, copy_2, 2, -1);  select_scatter = copy_2 = None
        view = torch.ops.aten.view.default(cat, [32, 198, 2560]);  cat = None
        clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_1 = torch.ops.aten.view.default(clone, [32, 198]);  clone = None
        view_5 = torch.ops.aten.view.default(select_scatter_1, [32, 198]);  select_scatter_1 = None
        view_6 = torch.ops.aten.view.default(slice_scatter, [32, 198]);  slice_scatter = None
        return (view, view_1, view_5, view_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 3, 1, 2560), (0, 0, 2560, 1), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 31457280, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 3, 64, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32, 3, 1, 2560), (0, 0, 2560, 1), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 96, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf3, (32, 3), dtype=torch.bool, is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)