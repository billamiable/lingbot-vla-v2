
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
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_PR1/codegen'
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
        embedding = torch.ops.aten.embedding.default(primals_1, primals_2);  primals_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_3, 2)
        mul = torch.ops.aten.mul.Tensor(embedding, unsqueeze);  embedding = unsqueeze = None
        select = torch.ops.aten.select.int(mul, 0, 0)
        select_1 = torch.ops.aten.select.int(mul, 0, 1)
        add = torch.ops.aten.add.Tensor(select, select_1);  select = select_1 = None
        select_2 = torch.ops.aten.select.int(mul, 0, 2)
        add_1 = torch.ops.aten.add.Tensor(add, select_2);  add = select_2 = None
        select_3 = torch.ops.aten.select.int(mul, 0, 3);  mul = None
        add_2 = torch.ops.aten.add.Tensor(add_1, select_3);  add_1 = select_3 = None
        index = torch.ops.aten.index.Tensor(add_2, [primals_4]);  add_2 = None
        return (index, primals_2, primals_3, primals_4)
        
def load_args(reader):
    buf0 = reader.storage(None, 4718592, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (2304, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 786432, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (4, 24576), dtype=torch.int64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 196608, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (4, 24576), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 196608, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf3, (24576,), dtype=torch.int64, is_leaf=True)  # primals_4
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)