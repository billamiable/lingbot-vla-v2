
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7):
        slice_2 = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 286)
        permute = torch.ops.aten.permute.default(primals_3, [1, 0])
        clone = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
        view = torch.ops.aten.view.default(clone, [9152, 4096]);  clone = None
        mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1 = torch.ops.aten.view.default(mm, [32, 286, 2560]);  mm = None
        add = torch.ops.aten.add.Tensor(view_1, primals_1);  view_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add, torch.float32)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_2, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type_2, rsqrt);  convert_element_type_2 = rsqrt = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(primals_4, convert_element_type_3);  convert_element_type_3 = None
        permute_1 = torch.ops.aten.permute.default(primals_5, [1, 0])
        view_4 = torch.ops.aten.view.default(mul_1, [9152, 2560]);  mul_1 = None
        mm_1 = torch.ops.aten.mm.default(view_4, permute_1);  permute_1 = None
        view_5 = torch.ops.aten.view.default(mm_1, [32, 286, 9728]);  mm_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_6)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_6, sigmoid);  convert_element_type_6 = sigmoid = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_6, [1, 0])
        mm_2 = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_7 = torch.ops.aten.view.default(mm_2, [32, 286, 9728]);  mm_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_7, view_7);  convert_element_type_7 = view_7 = None
        permute_3 = torch.ops.aten.permute.default(primals_7, [1, 0])
        view_8 = torch.ops.aten.view.default(mul_3, [9152, 9728]);  mul_3 = None
        mm_3 = torch.ops.aten.mm.default(view_8, permute_3);  view_8 = permute_3 = None
        view_9 = torch.ops.aten.view.default(mm_3, [32, 286, 2560]);  mm_3 = None
        add_2 = torch.ops.aten.add.Tensor(view_9, add);  view_9 = add = None
        return (add_2, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7)
        
def load_args(reader):
    buf0 = reader.storage(None, 46858240, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 286, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 88342528, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32, 337, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 20971520, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (2560, 4096), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 5120, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (2560,), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 49807360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (9728, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 49807360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (9728, 2560), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 49807360, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (2560, 9728), dtype=torch.bfloat16, is_leaf=True)  # primals_7
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)