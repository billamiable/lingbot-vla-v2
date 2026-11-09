
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

    
    
    def forward(self, primals_1, primals_4, sum_1, clamp_min, getitem, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36):
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(tangents_1, 0);  tangents_1 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(tangents_2, 0);  tangents_2 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(tangents_3, 0);  tangents_3 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(tangents_4, 0);  tangents_4 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(tangents_5, 0);  tangents_5 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(tangents_6, 0);  tangents_6 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(tangents_7, 0);  tangents_7 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(tangents_8, 0);  tangents_8 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(tangents_9, 0);  tangents_9 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(tangents_10, 0);  tangents_10 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(tangents_11, 0);  tangents_11 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(tangents_12, 0);  tangents_12 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(tangents_13, 0);  tangents_13 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(tangents_14, 0);  tangents_14 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(tangents_15, 0);  tangents_15 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(tangents_16, 0);  tangents_16 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(tangents_17, 0);  tangents_17 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(tangents_18, 0);  tangents_18 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(tangents_19, 0);  tangents_19 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(tangents_20, 0);  tangents_20 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(tangents_21, 0);  tangents_21 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(tangents_22, 0);  tangents_22 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(tangents_23, 0);  tangents_23 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(tangents_24, 0);  tangents_24 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(tangents_25, 0);  tangents_25 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(tangents_26, 0);  tangents_26 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(tangents_27, 0);  tangents_27 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(tangents_28, 0);  tangents_28 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(tangents_29, 0);  tangents_29 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(tangents_30, 0);  tangents_30 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(tangents_31, 0);  tangents_31 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(tangents_32, 0);  tangents_32 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(tangents_33, 0);  tangents_33 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(tangents_34, 0);  tangents_34 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(tangents_35, 0);  tangents_35 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(tangents_36, 0);  tangents_36 = None
        cat = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, unsqueeze_8, unsqueeze_9, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38]);  unsqueeze_3 = unsqueeze_4 = unsqueeze_5 = unsqueeze_6 = unsqueeze_7 = unsqueeze_8 = unsqueeze_9 = unsqueeze_10 = unsqueeze_11 = unsqueeze_12 = unsqueeze_13 = unsqueeze_14 = unsqueeze_15 = unsqueeze_16 = unsqueeze_17 = unsqueeze_18 = unsqueeze_19 = unsqueeze_20 = unsqueeze_21 = unsqueeze_22 = unsqueeze_23 = unsqueeze_24 = unsqueeze_25 = unsqueeze_26 = unsqueeze_27 = unsqueeze_28 = unsqueeze_29 = unsqueeze_30 = unsqueeze_31 = unsqueeze_32 = unsqueeze_33 = unsqueeze_34 = unsqueeze_35 = unsqueeze_36 = unsqueeze_37 = unsqueeze_38 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_39, [36, 32]);  unsqueeze_39 = None
        div_2 = torch.ops.aten.div.Scalar(expand_1, 32);  expand_1 = None
        view_1 = torch.ops.aten.view.default(div_2, [1152]);  div_2 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
        expand_2 = torch.ops.aten.expand.default(unsqueeze_40, [1152, 32]);  unsqueeze_40 = None
        reciprocal = torch.ops.aten.reciprocal.default(clamp_min)
        mul = torch.ops.aten.mul.Tensor(reciprocal, 8.0);  reciprocal = None
        mul_1 = torch.ops.aten.mul.Tensor(getitem, mul);  getitem = mul = None
        mul_3 = torch.ops.aten.mul.Tensor(expand_2, mul_1);  expand_2 = mul_1 = None
        div_3 = torch.ops.aten.div.Tensor(mul_3, clamp_min);  mul_3 = clamp_min = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_1, 1);  primals_1 = None
        expand = torch.ops.aten.expand.default(unsqueeze, [-1, 32]);  unsqueeze = None
        gather = torch.ops.aten.gather.default(div_3, 0, expand);  div_3 = expand = None
        sigmoid = torch.ops.aten.sigmoid.default(primals_4);  primals_4 = None
        div = torch.ops.aten.div.Tensor(sigmoid, sum_1)
        div_5 = torch.ops.aten.div.Tensor(div, sum_1);  div = None
        neg = torch.ops.aten.neg.default(gather)
        mul_4 = torch.ops.aten.mul.Tensor(neg, div_5);  neg = div_5 = None
        div_6 = torch.ops.aten.div.Tensor(gather, sum_1);  gather = sum_1 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_4, [1], True);  mul_4 = None
        expand_3 = torch.ops.aten.expand.default(sum_3, [58752, 32]);  sum_3 = None
        add = torch.ops.aten.add.Tensor(div_6, expand_3);  div_6 = expand_3 = None
        sub = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_5 = torch.ops.aten.mul.Tensor(sigmoid, sub);  sigmoid = sub = None
        mul_6 = torch.ops.aten.mul.Tensor(add, mul_5);  add = mul_5 = None
        return (None, None, None, mul_6)
        
def load_args(reader):
    buf0 = reader.storage(None, 470016, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (58752,), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 7520256, device=device(type='cuda', index=0))
    reader.tensor(buf1, (58752, 32), is_leaf=True)  # primals_4
    buf2 = reader.storage(None, 235008, device=device(type='cuda', index=0))
    reader.tensor(buf2, (58752, 1), is_leaf=True)  # sum_1
    buf3 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1152, 1), is_leaf=True)  # clamp_min
    buf4 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1152, 32), is_leaf=True)  # getitem
    buf5 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf5, (), is_leaf=True)  # tangents_1
    reader.tensor(buf5, (), storage_offset=1, is_leaf=True)  # tangents_2
    reader.tensor(buf5, (), storage_offset=2, is_leaf=True)  # tangents_3
    reader.tensor(buf5, (), storage_offset=3, is_leaf=True)  # tangents_4
    reader.tensor(buf5, (), storage_offset=4, is_leaf=True)  # tangents_5
    reader.tensor(buf5, (), storage_offset=5, is_leaf=True)  # tangents_6
    reader.tensor(buf5, (), storage_offset=6, is_leaf=True)  # tangents_7
    reader.tensor(buf5, (), storage_offset=7, is_leaf=True)  # tangents_8
    reader.tensor(buf5, (), storage_offset=8, is_leaf=True)  # tangents_9
    reader.tensor(buf5, (), storage_offset=9, is_leaf=True)  # tangents_10
    reader.tensor(buf5, (), storage_offset=10, is_leaf=True)  # tangents_11
    reader.tensor(buf5, (), storage_offset=11, is_leaf=True)  # tangents_12
    reader.tensor(buf5, (), storage_offset=12, is_leaf=True)  # tangents_13
    reader.tensor(buf5, (), storage_offset=13, is_leaf=True)  # tangents_14
    reader.tensor(buf5, (), storage_offset=14, is_leaf=True)  # tangents_15
    reader.tensor(buf5, (), storage_offset=15, is_leaf=True)  # tangents_16
    reader.tensor(buf5, (), storage_offset=16, is_leaf=True)  # tangents_17
    reader.tensor(buf5, (), storage_offset=17, is_leaf=True)  # tangents_18
    reader.tensor(buf5, (), storage_offset=18, is_leaf=True)  # tangents_19
    reader.tensor(buf5, (), storage_offset=19, is_leaf=True)  # tangents_20
    reader.tensor(buf5, (), storage_offset=20, is_leaf=True)  # tangents_21
    reader.tensor(buf5, (), storage_offset=21, is_leaf=True)  # tangents_22
    reader.tensor(buf5, (), storage_offset=22, is_leaf=True)  # tangents_23
    reader.tensor(buf5, (), storage_offset=23, is_leaf=True)  # tangents_24
    reader.tensor(buf5, (), storage_offset=24, is_leaf=True)  # tangents_25
    reader.tensor(buf5, (), storage_offset=25, is_leaf=True)  # tangents_26
    reader.tensor(buf5, (), storage_offset=26, is_leaf=True)  # tangents_27
    reader.tensor(buf5, (), storage_offset=27, is_leaf=True)  # tangents_28
    reader.tensor(buf5, (), storage_offset=28, is_leaf=True)  # tangents_29
    reader.tensor(buf5, (), storage_offset=29, is_leaf=True)  # tangents_30
    reader.tensor(buf5, (), storage_offset=30, is_leaf=True)  # tangents_31
    reader.tensor(buf5, (), storage_offset=31, is_leaf=True)  # tangents_32
    reader.tensor(buf5, (), storage_offset=32, is_leaf=True)  # tangents_33
    reader.tensor(buf5, (), storage_offset=33, is_leaf=True)  # tangents_34
    reader.tensor(buf5, (), storage_offset=34, is_leaf=True)  # tangents_35
    reader.tensor(buf5, (), storage_offset=35, is_leaf=True)  # tangents_36
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)