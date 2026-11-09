
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
        self.fw_graph0 = GraphModule()
        self.joint_graph0 = GraphModule()
        self.mask_graph0 = GraphModule()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, getitem, getitem_1, tangents_1):
        view_1 = torch.ops.aten.view.default(tangents_1, [32, 337, 32, 128]);  tangents_1 = None
        permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        full_default = torch.ops.aten.full.default([32, 32, 337], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_1, primals_2, primals_3, getitem, getitem_1, permute_1, full_default, fw_graph0, joint_graph0, (337, 337, primals_5, primals_4, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, 128, 128, mask_graph0), 0.08838834764831845, {'BLOCK_M': 32, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2, 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True}, (), (primals_6,));  primals_1 = primals_2 = primals_3 = getitem = getitem_1 = permute_1 = full_default = fw_graph0 = joint_graph0 = primals_5 = primals_4 = primals_7 = primals_8 = primals_9 = primals_10 = primals_11 = primals_12 = mask_graph0 = primals_6 = None
        getitem_2 = flex_attention_backward[0]
        getitem_3 = flex_attention_backward[1]
        getitem_4 = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_2, getitem_3, getitem_4, None, None, None, None, None, None, None, None, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 176685056, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 32, 337, 128), is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 44171264, device=device(type='cuda', index=0))
    reader.tensor(buf1, (32, 8, 337, 128), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 44171264, device=device(type='cuda', index=0))
    reader.tensor(buf2, (32, 8, 337, 128), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf3, (32, 32, 3, 3), dtype=torch.int32, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 12288, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf4, (32, 32, 3), dtype=torch.int32, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 116294656, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf5, (32, 32, 337, 337), dtype=torch.bool, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 12288, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf6, (32, 32, 3), dtype=torch.int32, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf7, (32, 32, 3, 3), dtype=torch.int32, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 12288, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf8, (32, 32, 3), dtype=torch.int32, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf9, (32, 32, 3, 3), dtype=torch.int32, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 12288, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf10, (32, 32, 3), dtype=torch.int32, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 36864, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf11, (32, 32, 3, 3), dtype=torch.int32, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 176685056, device=device(type='cuda', index=0))
    reader.tensor(buf12, (32, 32, 337, 128), is_leaf=True)  # getitem
    buf13 = reader.storage(None, 1380352, device=device(type='cuda', index=0))
    reader.tensor(buf13, (32, 32, 337), is_leaf=True)  # getitem_1
    buf14 = reader.storage(None, 176685056, device=device(type='cuda', index=0))
    reader.tensor(buf14, (32, 337, 4096), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)