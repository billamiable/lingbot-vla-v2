
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

    
    
    def forward(self, primals_2, primals_3, primals_4, tangents_1):
        full_default = torch.ops.aten.full.default([24576, 1024], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_add = torch.ops.aten.index_add.default(full_default, 0, primals_4, tangents_1);  full_default = primals_4 = tangents_1 = None
        full_default_1 = torch.ops.aten.full.default([4, 24576, 1024], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 3)
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 2)
        add_3 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 1)
        add_4 = torch.ops.aten.add.Tensor(add_3, select_scatter_2);  add_3 = select_scatter_2 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(full_default_1, index_add, 0, 0);  full_default_1 = index_add = None
        add_5 = torch.ops.aten.add.Tensor(add_4, select_scatter_3);  add_4 = select_scatter_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_3, 2);  primals_3 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_5, unsqueeze);  add_5 = unsqueeze = None
        convert_element_type = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        eq = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_5 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(unsqueeze_1, full_default_5, convert_element_type);  unsqueeze_1 = full_default_5 = convert_element_type = None
        full_default_6 = torch.ops.aten.full.default([2304, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put = torch.ops.aten.index_put.default(full_default_6, [primals_2], where, True);  full_default_6 = primals_2 = where = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(index_put, torch.bfloat16);  index_put = None
        return (convert_element_type_1, None, None, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (4, 24576), dtype=torch.int64, is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 196608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (4, 24576), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf2 = reader.storage(None, 196608, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf2, (24576,), dtype=torch.int64, is_leaf=True)  # primals_4
    buf3 = reader.storage(None, 50331648, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (24576, 1024), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)