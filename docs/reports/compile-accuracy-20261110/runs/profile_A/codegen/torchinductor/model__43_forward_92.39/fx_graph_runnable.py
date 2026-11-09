
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/inductor'
os.environ['TRITON_CACHE_DIR'] = '/root/qihank/.cache/sync_bench/triton'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHELASTIC_RESTART_COUNT'] = '0'
os.environ['TORCHELASTIC_MAX_RESTARTS'] = '0'
os.environ['TORCHELASTIC_RUN_ID'] = 'e120077a-9255-47ec-9d26-5d0c985de943'
os.environ['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCHELASTIC_ERROR_FILE'] = '/tmp/torchelastic_wz74sf15/e120077a-9255-47ec-9d26-5d0c985de943_iswpxo4r/attempt_0/0/error.json'

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
torch._inductor.config.trace.debug_dir = '/root/experiments/lingbot-compile-accuracy-20261110/runs/profile_A/codegen'
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

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37):
        amax = torch.ops.aten.amax.default(primals_2, [-1], True)
        abs_1 = torch.ops.aten.abs.default(amax)
        eq = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(eq, full_default, amax);  eq = amax = None
        squeeze = torch.ops.aten.squeeze.dims(where, [-1])
        sub = torch.ops.aten.sub.Tensor(primals_2, where);  where = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [-1]);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        add = torch.ops.aten.add.Tensor(log, squeeze);  log = squeeze = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(add, 2)
        mean = torch.ops.aten.mean.default(pow_1);  pow_1 = None
        amax_1 = torch.ops.aten.amax.default(primals_3, [-1], True)
        abs_2 = torch.ops.aten.abs.default(amax_1)
        eq_1 = torch.ops.aten.eq.Scalar(abs_2, inf);  abs_2 = None
        where_1 = torch.ops.aten.where.self(eq_1, full_default, amax_1);  eq_1 = amax_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(where_1, [-1])
        sub_1 = torch.ops.aten.sub.Tensor(primals_3, where_1);  where_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp_1, [-1]);  exp_1 = None
        log_1 = torch.ops.aten.log.default(sum_2);  sum_2 = None
        add_1 = torch.ops.aten.add.Tensor(log_1, squeeze_1);  log_1 = squeeze_1 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_1, 2)
        mean_1 = torch.ops.aten.mean.default(pow_2);  pow_2 = None
        amax_2 = torch.ops.aten.amax.default(primals_4, [-1], True)
        abs_3 = torch.ops.aten.abs.default(amax_2)
        eq_2 = torch.ops.aten.eq.Scalar(abs_3, inf);  abs_3 = None
        where_2 = torch.ops.aten.where.self(eq_2, full_default, amax_2);  eq_2 = amax_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(where_2, [-1])
        sub_2 = torch.ops.aten.sub.Tensor(primals_4, where_2);  where_2 = None
        exp_2 = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(exp_2, [-1]);  exp_2 = None
        log_2 = torch.ops.aten.log.default(sum_3);  sum_3 = None
        add_2 = torch.ops.aten.add.Tensor(log_2, squeeze_2);  log_2 = squeeze_2 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(add_2, 2)
        mean_2 = torch.ops.aten.mean.default(pow_3);  pow_3 = None
        amax_3 = torch.ops.aten.amax.default(primals_5, [-1], True)
        abs_4 = torch.ops.aten.abs.default(amax_3)
        eq_3 = torch.ops.aten.eq.Scalar(abs_4, inf);  abs_4 = None
        where_3 = torch.ops.aten.where.self(eq_3, full_default, amax_3);  eq_3 = amax_3 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(where_3, [-1])
        sub_3 = torch.ops.aten.sub.Tensor(primals_5, where_3);  where_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1]);  exp_3 = None
        log_3 = torch.ops.aten.log.default(sum_4);  sum_4 = None
        add_3 = torch.ops.aten.add.Tensor(log_3, squeeze_3);  log_3 = squeeze_3 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_3, 2)
        mean_3 = torch.ops.aten.mean.default(pow_4);  pow_4 = None
        amax_4 = torch.ops.aten.amax.default(primals_6, [-1], True)
        abs_5 = torch.ops.aten.abs.default(amax_4)
        eq_4 = torch.ops.aten.eq.Scalar(abs_5, inf);  abs_5 = None
        where_4 = torch.ops.aten.where.self(eq_4, full_default, amax_4);  eq_4 = amax_4 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(where_4, [-1])
        sub_4 = torch.ops.aten.sub.Tensor(primals_6, where_4);  where_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1]);  exp_4 = None
        log_4 = torch.ops.aten.log.default(sum_5);  sum_5 = None
        add_4 = torch.ops.aten.add.Tensor(log_4, squeeze_4);  log_4 = squeeze_4 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(add_4, 2)
        mean_4 = torch.ops.aten.mean.default(pow_5);  pow_5 = None
        amax_5 = torch.ops.aten.amax.default(primals_7, [-1], True)
        abs_6 = torch.ops.aten.abs.default(amax_5)
        eq_5 = torch.ops.aten.eq.Scalar(abs_6, inf);  abs_6 = None
        where_5 = torch.ops.aten.where.self(eq_5, full_default, amax_5);  eq_5 = amax_5 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(where_5, [-1])
        sub_5 = torch.ops.aten.sub.Tensor(primals_7, where_5);  where_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1]);  exp_5 = None
        log_5 = torch.ops.aten.log.default(sum_6);  sum_6 = None
        add_5 = torch.ops.aten.add.Tensor(log_5, squeeze_5);  log_5 = squeeze_5 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(add_5, 2)
        mean_5 = torch.ops.aten.mean.default(pow_6);  pow_6 = None
        amax_6 = torch.ops.aten.amax.default(primals_8, [-1], True)
        abs_7 = torch.ops.aten.abs.default(amax_6)
        eq_6 = torch.ops.aten.eq.Scalar(abs_7, inf);  abs_7 = None
        where_6 = torch.ops.aten.where.self(eq_6, full_default, amax_6);  eq_6 = amax_6 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(where_6, [-1])
        sub_6 = torch.ops.aten.sub.Tensor(primals_8, where_6);  where_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1]);  exp_6 = None
        log_6 = torch.ops.aten.log.default(sum_7);  sum_7 = None
        add_6 = torch.ops.aten.add.Tensor(log_6, squeeze_6);  log_6 = squeeze_6 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
        mean_6 = torch.ops.aten.mean.default(pow_7);  pow_7 = None
        amax_7 = torch.ops.aten.amax.default(primals_9, [-1], True)
        abs_8 = torch.ops.aten.abs.default(amax_7)
        eq_7 = torch.ops.aten.eq.Scalar(abs_8, inf);  abs_8 = None
        where_7 = torch.ops.aten.where.self(eq_7, full_default, amax_7);  eq_7 = amax_7 = None
        squeeze_7 = torch.ops.aten.squeeze.dims(where_7, [-1])
        sub_7 = torch.ops.aten.sub.Tensor(primals_9, where_7);  where_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1]);  exp_7 = None
        log_7 = torch.ops.aten.log.default(sum_8);  sum_8 = None
        add_7 = torch.ops.aten.add.Tensor(log_7, squeeze_7);  log_7 = squeeze_7 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(add_7, 2)
        mean_7 = torch.ops.aten.mean.default(pow_8);  pow_8 = None
        amax_8 = torch.ops.aten.amax.default(primals_10, [-1], True)
        abs_9 = torch.ops.aten.abs.default(amax_8)
        eq_8 = torch.ops.aten.eq.Scalar(abs_9, inf);  abs_9 = None
        where_8 = torch.ops.aten.where.self(eq_8, full_default, amax_8);  eq_8 = amax_8 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(where_8, [-1])
        sub_8 = torch.ops.aten.sub.Tensor(primals_10, where_8);  where_8 = None
        exp_8 = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1]);  exp_8 = None
        log_8 = torch.ops.aten.log.default(sum_9);  sum_9 = None
        add_8 = torch.ops.aten.add.Tensor(log_8, squeeze_8);  log_8 = squeeze_8 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(add_8, 2)
        mean_8 = torch.ops.aten.mean.default(pow_9);  pow_9 = None
        amax_9 = torch.ops.aten.amax.default(primals_11, [-1], True)
        abs_10 = torch.ops.aten.abs.default(amax_9)
        eq_9 = torch.ops.aten.eq.Scalar(abs_10, inf);  abs_10 = None
        where_9 = torch.ops.aten.where.self(eq_9, full_default, amax_9);  eq_9 = amax_9 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(where_9, [-1])
        sub_9 = torch.ops.aten.sub.Tensor(primals_11, where_9);  where_9 = None
        exp_9 = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1]);  exp_9 = None
        log_9 = torch.ops.aten.log.default(sum_10);  sum_10 = None
        add_9 = torch.ops.aten.add.Tensor(log_9, squeeze_9);  log_9 = squeeze_9 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(add_9, 2)
        mean_9 = torch.ops.aten.mean.default(pow_10);  pow_10 = None
        amax_10 = torch.ops.aten.amax.default(primals_12, [-1], True)
        abs_11 = torch.ops.aten.abs.default(amax_10)
        eq_10 = torch.ops.aten.eq.Scalar(abs_11, inf);  abs_11 = None
        where_10 = torch.ops.aten.where.self(eq_10, full_default, amax_10);  eq_10 = amax_10 = None
        squeeze_10 = torch.ops.aten.squeeze.dims(where_10, [-1])
        sub_10 = torch.ops.aten.sub.Tensor(primals_12, where_10);  where_10 = None
        exp_10 = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1]);  exp_10 = None
        log_10 = torch.ops.aten.log.default(sum_11);  sum_11 = None
        add_10 = torch.ops.aten.add.Tensor(log_10, squeeze_10);  log_10 = squeeze_10 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(add_10, 2)
        mean_10 = torch.ops.aten.mean.default(pow_11);  pow_11 = None
        amax_11 = torch.ops.aten.amax.default(primals_13, [-1], True)
        abs_12 = torch.ops.aten.abs.default(amax_11)
        eq_11 = torch.ops.aten.eq.Scalar(abs_12, inf);  abs_12 = None
        where_11 = torch.ops.aten.where.self(eq_11, full_default, amax_11);  eq_11 = amax_11 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(where_11, [-1])
        sub_11 = torch.ops.aten.sub.Tensor(primals_13, where_11);  where_11 = None
        exp_11 = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1]);  exp_11 = None
        log_11 = torch.ops.aten.log.default(sum_12);  sum_12 = None
        add_11 = torch.ops.aten.add.Tensor(log_11, squeeze_11);  log_11 = squeeze_11 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(add_11, 2)
        mean_11 = torch.ops.aten.mean.default(pow_12);  pow_12 = None
        amax_12 = torch.ops.aten.amax.default(primals_14, [-1], True)
        abs_13 = torch.ops.aten.abs.default(amax_12)
        eq_12 = torch.ops.aten.eq.Scalar(abs_13, inf);  abs_13 = None
        where_12 = torch.ops.aten.where.self(eq_12, full_default, amax_12);  eq_12 = amax_12 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(where_12, [-1])
        sub_12 = torch.ops.aten.sub.Tensor(primals_14, where_12);  where_12 = None
        exp_12 = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1]);  exp_12 = None
        log_12 = torch.ops.aten.log.default(sum_13);  sum_13 = None
        add_12 = torch.ops.aten.add.Tensor(log_12, squeeze_12);  log_12 = squeeze_12 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(add_12, 2)
        mean_12 = torch.ops.aten.mean.default(pow_13);  pow_13 = None
        amax_13 = torch.ops.aten.amax.default(primals_15, [-1], True)
        abs_14 = torch.ops.aten.abs.default(amax_13)
        eq_13 = torch.ops.aten.eq.Scalar(abs_14, inf);  abs_14 = None
        where_13 = torch.ops.aten.where.self(eq_13, full_default, amax_13);  eq_13 = amax_13 = None
        squeeze_13 = torch.ops.aten.squeeze.dims(where_13, [-1])
        sub_13 = torch.ops.aten.sub.Tensor(primals_15, where_13);  where_13 = None
        exp_13 = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1]);  exp_13 = None
        log_13 = torch.ops.aten.log.default(sum_14);  sum_14 = None
        add_13 = torch.ops.aten.add.Tensor(log_13, squeeze_13);  log_13 = squeeze_13 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
        mean_13 = torch.ops.aten.mean.default(pow_14);  pow_14 = None
        amax_14 = torch.ops.aten.amax.default(primals_16, [-1], True)
        abs_15 = torch.ops.aten.abs.default(amax_14)
        eq_14 = torch.ops.aten.eq.Scalar(abs_15, inf);  abs_15 = None
        where_14 = torch.ops.aten.where.self(eq_14, full_default, amax_14);  eq_14 = amax_14 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(where_14, [-1])
        sub_14 = torch.ops.aten.sub.Tensor(primals_16, where_14);  where_14 = None
        exp_14 = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1]);  exp_14 = None
        log_14 = torch.ops.aten.log.default(sum_15);  sum_15 = None
        add_14 = torch.ops.aten.add.Tensor(log_14, squeeze_14);  log_14 = squeeze_14 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(add_14, 2)
        mean_14 = torch.ops.aten.mean.default(pow_15);  pow_15 = None
        amax_15 = torch.ops.aten.amax.default(primals_17, [-1], True)
        abs_16 = torch.ops.aten.abs.default(amax_15)
        eq_15 = torch.ops.aten.eq.Scalar(abs_16, inf);  abs_16 = None
        where_15 = torch.ops.aten.where.self(eq_15, full_default, amax_15);  eq_15 = amax_15 = None
        squeeze_15 = torch.ops.aten.squeeze.dims(where_15, [-1])
        sub_15 = torch.ops.aten.sub.Tensor(primals_17, where_15);  where_15 = None
        exp_15 = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1]);  exp_15 = None
        log_15 = torch.ops.aten.log.default(sum_16);  sum_16 = None
        add_15 = torch.ops.aten.add.Tensor(log_15, squeeze_15);  log_15 = squeeze_15 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(add_15, 2)
        mean_15 = torch.ops.aten.mean.default(pow_16);  pow_16 = None
        amax_16 = torch.ops.aten.amax.default(primals_18, [-1], True)
        abs_17 = torch.ops.aten.abs.default(amax_16)
        eq_16 = torch.ops.aten.eq.Scalar(abs_17, inf);  abs_17 = None
        where_16 = torch.ops.aten.where.self(eq_16, full_default, amax_16);  eq_16 = amax_16 = None
        squeeze_16 = torch.ops.aten.squeeze.dims(where_16, [-1])
        sub_16 = torch.ops.aten.sub.Tensor(primals_18, where_16);  where_16 = None
        exp_16 = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(exp_16, [-1]);  exp_16 = None
        log_16 = torch.ops.aten.log.default(sum_17);  sum_17 = None
        add_16 = torch.ops.aten.add.Tensor(log_16, squeeze_16);  log_16 = squeeze_16 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(add_16, 2)
        mean_16 = torch.ops.aten.mean.default(pow_17);  pow_17 = None
        amax_17 = torch.ops.aten.amax.default(primals_19, [-1], True)
        abs_18 = torch.ops.aten.abs.default(amax_17)
        eq_17 = torch.ops.aten.eq.Scalar(abs_18, inf);  abs_18 = None
        where_17 = torch.ops.aten.where.self(eq_17, full_default, amax_17);  eq_17 = amax_17 = None
        squeeze_17 = torch.ops.aten.squeeze.dims(where_17, [-1])
        sub_17 = torch.ops.aten.sub.Tensor(primals_19, where_17);  where_17 = None
        exp_17 = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(exp_17, [-1]);  exp_17 = None
        log_17 = torch.ops.aten.log.default(sum_18);  sum_18 = None
        add_17 = torch.ops.aten.add.Tensor(log_17, squeeze_17);  log_17 = squeeze_17 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(add_17, 2)
        mean_17 = torch.ops.aten.mean.default(pow_18);  pow_18 = None
        amax_18 = torch.ops.aten.amax.default(primals_20, [-1], True)
        abs_19 = torch.ops.aten.abs.default(amax_18)
        eq_18 = torch.ops.aten.eq.Scalar(abs_19, inf);  abs_19 = None
        where_18 = torch.ops.aten.where.self(eq_18, full_default, amax_18);  eq_18 = amax_18 = None
        squeeze_18 = torch.ops.aten.squeeze.dims(where_18, [-1])
        sub_18 = torch.ops.aten.sub.Tensor(primals_20, where_18);  where_18 = None
        exp_18 = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(exp_18, [-1]);  exp_18 = None
        log_18 = torch.ops.aten.log.default(sum_19);  sum_19 = None
        add_18 = torch.ops.aten.add.Tensor(log_18, squeeze_18);  log_18 = squeeze_18 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(add_18, 2)
        mean_18 = torch.ops.aten.mean.default(pow_19);  pow_19 = None
        amax_19 = torch.ops.aten.amax.default(primals_21, [-1], True)
        abs_20 = torch.ops.aten.abs.default(amax_19)
        eq_19 = torch.ops.aten.eq.Scalar(abs_20, inf);  abs_20 = None
        where_19 = torch.ops.aten.where.self(eq_19, full_default, amax_19);  eq_19 = amax_19 = None
        squeeze_19 = torch.ops.aten.squeeze.dims(where_19, [-1])
        sub_19 = torch.ops.aten.sub.Tensor(primals_21, where_19);  where_19 = None
        exp_19 = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(exp_19, [-1]);  exp_19 = None
        log_19 = torch.ops.aten.log.default(sum_20);  sum_20 = None
        add_19 = torch.ops.aten.add.Tensor(log_19, squeeze_19);  log_19 = squeeze_19 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(add_19, 2)
        mean_19 = torch.ops.aten.mean.default(pow_20);  pow_20 = None
        amax_20 = torch.ops.aten.amax.default(primals_22, [-1], True)
        abs_21 = torch.ops.aten.abs.default(amax_20)
        eq_20 = torch.ops.aten.eq.Scalar(abs_21, inf);  abs_21 = None
        where_20 = torch.ops.aten.where.self(eq_20, full_default, amax_20);  eq_20 = amax_20 = None
        squeeze_20 = torch.ops.aten.squeeze.dims(where_20, [-1])
        sub_20 = torch.ops.aten.sub.Tensor(primals_22, where_20);  where_20 = None
        exp_20 = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_21 = torch.ops.aten.sum.dim_IntList(exp_20, [-1]);  exp_20 = None
        log_20 = torch.ops.aten.log.default(sum_21);  sum_21 = None
        add_20 = torch.ops.aten.add.Tensor(log_20, squeeze_20);  log_20 = squeeze_20 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(add_20, 2)
        mean_20 = torch.ops.aten.mean.default(pow_21);  pow_21 = None
        amax_21 = torch.ops.aten.amax.default(primals_23, [-1], True)
        abs_22 = torch.ops.aten.abs.default(amax_21)
        eq_21 = torch.ops.aten.eq.Scalar(abs_22, inf);  abs_22 = None
        where_21 = torch.ops.aten.where.self(eq_21, full_default, amax_21);  eq_21 = amax_21 = None
        squeeze_21 = torch.ops.aten.squeeze.dims(where_21, [-1])
        sub_21 = torch.ops.aten.sub.Tensor(primals_23, where_21);  where_21 = None
        exp_21 = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(exp_21, [-1]);  exp_21 = None
        log_21 = torch.ops.aten.log.default(sum_22);  sum_22 = None
        add_21 = torch.ops.aten.add.Tensor(log_21, squeeze_21);  log_21 = squeeze_21 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(add_21, 2)
        mean_21 = torch.ops.aten.mean.default(pow_22);  pow_22 = None
        amax_22 = torch.ops.aten.amax.default(primals_24, [-1], True)
        abs_23 = torch.ops.aten.abs.default(amax_22)
        eq_22 = torch.ops.aten.eq.Scalar(abs_23, inf);  abs_23 = None
        where_22 = torch.ops.aten.where.self(eq_22, full_default, amax_22);  eq_22 = amax_22 = None
        squeeze_22 = torch.ops.aten.squeeze.dims(where_22, [-1])
        sub_22 = torch.ops.aten.sub.Tensor(primals_24, where_22);  where_22 = None
        exp_22 = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(exp_22, [-1]);  exp_22 = None
        log_22 = torch.ops.aten.log.default(sum_23);  sum_23 = None
        add_22 = torch.ops.aten.add.Tensor(log_22, squeeze_22);  log_22 = squeeze_22 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(add_22, 2)
        mean_22 = torch.ops.aten.mean.default(pow_23);  pow_23 = None
        amax_23 = torch.ops.aten.amax.default(primals_25, [-1], True)
        abs_24 = torch.ops.aten.abs.default(amax_23)
        eq_23 = torch.ops.aten.eq.Scalar(abs_24, inf);  abs_24 = None
        where_23 = torch.ops.aten.where.self(eq_23, full_default, amax_23);  eq_23 = amax_23 = None
        squeeze_23 = torch.ops.aten.squeeze.dims(where_23, [-1])
        sub_23 = torch.ops.aten.sub.Tensor(primals_25, where_23);  where_23 = None
        exp_23 = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(exp_23, [-1]);  exp_23 = None
        log_23 = torch.ops.aten.log.default(sum_24);  sum_24 = None
        add_23 = torch.ops.aten.add.Tensor(log_23, squeeze_23);  log_23 = squeeze_23 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(add_23, 2)
        mean_23 = torch.ops.aten.mean.default(pow_24);  pow_24 = None
        amax_24 = torch.ops.aten.amax.default(primals_26, [-1], True)
        abs_25 = torch.ops.aten.abs.default(amax_24)
        eq_24 = torch.ops.aten.eq.Scalar(abs_25, inf);  abs_25 = None
        where_24 = torch.ops.aten.where.self(eq_24, full_default, amax_24);  eq_24 = amax_24 = None
        squeeze_24 = torch.ops.aten.squeeze.dims(where_24, [-1])
        sub_24 = torch.ops.aten.sub.Tensor(primals_26, where_24);  where_24 = None
        exp_24 = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_25 = torch.ops.aten.sum.dim_IntList(exp_24, [-1]);  exp_24 = None
        log_24 = torch.ops.aten.log.default(sum_25);  sum_25 = None
        add_24 = torch.ops.aten.add.Tensor(log_24, squeeze_24);  log_24 = squeeze_24 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
        mean_24 = torch.ops.aten.mean.default(pow_25);  pow_25 = None
        amax_25 = torch.ops.aten.amax.default(primals_27, [-1], True)
        abs_26 = torch.ops.aten.abs.default(amax_25)
        eq_25 = torch.ops.aten.eq.Scalar(abs_26, inf);  abs_26 = None
        where_25 = torch.ops.aten.where.self(eq_25, full_default, amax_25);  eq_25 = amax_25 = None
        squeeze_25 = torch.ops.aten.squeeze.dims(where_25, [-1])
        sub_25 = torch.ops.aten.sub.Tensor(primals_27, where_25);  where_25 = None
        exp_25 = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_26 = torch.ops.aten.sum.dim_IntList(exp_25, [-1]);  exp_25 = None
        log_25 = torch.ops.aten.log.default(sum_26);  sum_26 = None
        add_25 = torch.ops.aten.add.Tensor(log_25, squeeze_25);  log_25 = squeeze_25 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(add_25, 2)
        mean_25 = torch.ops.aten.mean.default(pow_26);  pow_26 = None
        amax_26 = torch.ops.aten.amax.default(primals_28, [-1], True)
        abs_27 = torch.ops.aten.abs.default(amax_26)
        eq_26 = torch.ops.aten.eq.Scalar(abs_27, inf);  abs_27 = None
        where_26 = torch.ops.aten.where.self(eq_26, full_default, amax_26);  eq_26 = amax_26 = None
        squeeze_26 = torch.ops.aten.squeeze.dims(where_26, [-1])
        sub_26 = torch.ops.aten.sub.Tensor(primals_28, where_26);  where_26 = None
        exp_26 = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        sum_27 = torch.ops.aten.sum.dim_IntList(exp_26, [-1]);  exp_26 = None
        log_26 = torch.ops.aten.log.default(sum_27);  sum_27 = None
        add_26 = torch.ops.aten.add.Tensor(log_26, squeeze_26);  log_26 = squeeze_26 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(add_26, 2)
        mean_26 = torch.ops.aten.mean.default(pow_27);  pow_27 = None
        amax_27 = torch.ops.aten.amax.default(primals_29, [-1], True)
        abs_28 = torch.ops.aten.abs.default(amax_27)
        eq_27 = torch.ops.aten.eq.Scalar(abs_28, inf);  abs_28 = None
        where_27 = torch.ops.aten.where.self(eq_27, full_default, amax_27);  eq_27 = amax_27 = None
        squeeze_27 = torch.ops.aten.squeeze.dims(where_27, [-1])
        sub_27 = torch.ops.aten.sub.Tensor(primals_29, where_27);  where_27 = None
        exp_27 = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_28 = torch.ops.aten.sum.dim_IntList(exp_27, [-1]);  exp_27 = None
        log_27 = torch.ops.aten.log.default(sum_28);  sum_28 = None
        add_27 = torch.ops.aten.add.Tensor(log_27, squeeze_27);  log_27 = squeeze_27 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
        mean_27 = torch.ops.aten.mean.default(pow_28);  pow_28 = None
        amax_28 = torch.ops.aten.amax.default(primals_30, [-1], True)
        abs_29 = torch.ops.aten.abs.default(amax_28)
        eq_28 = torch.ops.aten.eq.Scalar(abs_29, inf);  abs_29 = None
        where_28 = torch.ops.aten.where.self(eq_28, full_default, amax_28);  eq_28 = amax_28 = None
        squeeze_28 = torch.ops.aten.squeeze.dims(where_28, [-1])
        sub_28 = torch.ops.aten.sub.Tensor(primals_30, where_28);  where_28 = None
        exp_28 = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_29 = torch.ops.aten.sum.dim_IntList(exp_28, [-1]);  exp_28 = None
        log_28 = torch.ops.aten.log.default(sum_29);  sum_29 = None
        add_28 = torch.ops.aten.add.Tensor(log_28, squeeze_28);  log_28 = squeeze_28 = None
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(add_28, 2)
        mean_28 = torch.ops.aten.mean.default(pow_29);  pow_29 = None
        amax_29 = torch.ops.aten.amax.default(primals_31, [-1], True)
        abs_30 = torch.ops.aten.abs.default(amax_29)
        eq_29 = torch.ops.aten.eq.Scalar(abs_30, inf);  abs_30 = None
        where_29 = torch.ops.aten.where.self(eq_29, full_default, amax_29);  eq_29 = amax_29 = None
        squeeze_29 = torch.ops.aten.squeeze.dims(where_29, [-1])
        sub_29 = torch.ops.aten.sub.Tensor(primals_31, where_29);  where_29 = None
        exp_29 = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_30 = torch.ops.aten.sum.dim_IntList(exp_29, [-1]);  exp_29 = None
        log_29 = torch.ops.aten.log.default(sum_30);  sum_30 = None
        add_29 = torch.ops.aten.add.Tensor(log_29, squeeze_29);  log_29 = squeeze_29 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(add_29, 2)
        mean_29 = torch.ops.aten.mean.default(pow_30);  pow_30 = None
        amax_30 = torch.ops.aten.amax.default(primals_32, [-1], True)
        abs_31 = torch.ops.aten.abs.default(amax_30)
        eq_30 = torch.ops.aten.eq.Scalar(abs_31, inf);  abs_31 = None
        where_30 = torch.ops.aten.where.self(eq_30, full_default, amax_30);  eq_30 = amax_30 = None
        squeeze_30 = torch.ops.aten.squeeze.dims(where_30, [-1])
        sub_30 = torch.ops.aten.sub.Tensor(primals_32, where_30);  where_30 = None
        exp_30 = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_31 = torch.ops.aten.sum.dim_IntList(exp_30, [-1]);  exp_30 = None
        log_30 = torch.ops.aten.log.default(sum_31);  sum_31 = None
        add_30 = torch.ops.aten.add.Tensor(log_30, squeeze_30);  log_30 = squeeze_30 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(add_30, 2)
        mean_30 = torch.ops.aten.mean.default(pow_31);  pow_31 = None
        amax_31 = torch.ops.aten.amax.default(primals_33, [-1], True)
        abs_32 = torch.ops.aten.abs.default(amax_31)
        eq_31 = torch.ops.aten.eq.Scalar(abs_32, inf);  abs_32 = None
        where_31 = torch.ops.aten.where.self(eq_31, full_default, amax_31);  eq_31 = amax_31 = None
        squeeze_31 = torch.ops.aten.squeeze.dims(where_31, [-1])
        sub_31 = torch.ops.aten.sub.Tensor(primals_33, where_31);  where_31 = None
        exp_31 = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_32 = torch.ops.aten.sum.dim_IntList(exp_31, [-1]);  exp_31 = None
        log_31 = torch.ops.aten.log.default(sum_32);  sum_32 = None
        add_31 = torch.ops.aten.add.Tensor(log_31, squeeze_31);  log_31 = squeeze_31 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
        mean_31 = torch.ops.aten.mean.default(pow_32);  pow_32 = None
        amax_32 = torch.ops.aten.amax.default(primals_34, [-1], True)
        abs_33 = torch.ops.aten.abs.default(amax_32)
        eq_32 = torch.ops.aten.eq.Scalar(abs_33, inf);  abs_33 = None
        where_32 = torch.ops.aten.where.self(eq_32, full_default, amax_32);  eq_32 = amax_32 = None
        squeeze_32 = torch.ops.aten.squeeze.dims(where_32, [-1])
        sub_32 = torch.ops.aten.sub.Tensor(primals_34, where_32);  where_32 = None
        exp_32 = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        sum_33 = torch.ops.aten.sum.dim_IntList(exp_32, [-1]);  exp_32 = None
        log_32 = torch.ops.aten.log.default(sum_33);  sum_33 = None
        add_32 = torch.ops.aten.add.Tensor(log_32, squeeze_32);  log_32 = squeeze_32 = None
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(add_32, 2)
        mean_32 = torch.ops.aten.mean.default(pow_33);  pow_33 = None
        amax_33 = torch.ops.aten.amax.default(primals_35, [-1], True)
        abs_34 = torch.ops.aten.abs.default(amax_33)
        eq_33 = torch.ops.aten.eq.Scalar(abs_34, inf);  abs_34 = None
        where_33 = torch.ops.aten.where.self(eq_33, full_default, amax_33);  eq_33 = amax_33 = None
        squeeze_33 = torch.ops.aten.squeeze.dims(where_33, [-1])
        sub_33 = torch.ops.aten.sub.Tensor(primals_35, where_33);  where_33 = None
        exp_33 = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_34 = torch.ops.aten.sum.dim_IntList(exp_33, [-1]);  exp_33 = None
        log_33 = torch.ops.aten.log.default(sum_34);  sum_34 = None
        add_33 = torch.ops.aten.add.Tensor(log_33, squeeze_33);  log_33 = squeeze_33 = None
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(add_33, 2)
        mean_33 = torch.ops.aten.mean.default(pow_34);  pow_34 = None
        amax_34 = torch.ops.aten.amax.default(primals_36, [-1], True)
        abs_35 = torch.ops.aten.abs.default(amax_34)
        eq_34 = torch.ops.aten.eq.Scalar(abs_35, inf);  abs_35 = None
        where_34 = torch.ops.aten.where.self(eq_34, full_default, amax_34);  eq_34 = amax_34 = None
        squeeze_34 = torch.ops.aten.squeeze.dims(where_34, [-1])
        sub_34 = torch.ops.aten.sub.Tensor(primals_36, where_34);  where_34 = None
        exp_34 = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_35 = torch.ops.aten.sum.dim_IntList(exp_34, [-1]);  exp_34 = None
        log_34 = torch.ops.aten.log.default(sum_35);  sum_35 = None
        add_34 = torch.ops.aten.add.Tensor(log_34, squeeze_34);  log_34 = squeeze_34 = None
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
        mean_34 = torch.ops.aten.mean.default(pow_35);  pow_35 = None
        amax_35 = torch.ops.aten.amax.default(primals_37, [-1], True)
        abs_36 = torch.ops.aten.abs.default(amax_35)
        eq_35 = torch.ops.aten.eq.Scalar(abs_36, inf);  abs_36 = None
        where_35 = torch.ops.aten.where.self(eq_35, full_default, amax_35);  eq_35 = full_default = amax_35 = None
        squeeze_35 = torch.ops.aten.squeeze.dims(where_35, [-1])
        sub_35 = torch.ops.aten.sub.Tensor(primals_37, where_35);  where_35 = None
        exp_35 = torch.ops.aten.exp.default(sub_35);  sub_35 = None
        sum_36 = torch.ops.aten.sum.dim_IntList(exp_35, [-1]);  exp_35 = None
        log_35 = torch.ops.aten.log.default(sum_36);  sum_36 = None
        add_35 = torch.ops.aten.add.Tensor(log_35, squeeze_35);  log_35 = squeeze_35 = None
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(add_35, 2)
        mean_35 = torch.ops.aten.mean.default(pow_36);  pow_36 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(mean, 0)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(mean_1, 0)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(mean_2, 0)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(mean_3, 0)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(mean_4, 0)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(mean_5, 0)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(mean_6, 0)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(mean_7, 0)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(mean_8, 0)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(mean_9, 0)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(mean_10, 0)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(mean_11, 0)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(mean_12, 0)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(mean_13, 0)
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(mean_14, 0)
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(mean_15, 0)
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(mean_16, 0)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(mean_17, 0)
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(mean_18, 0)
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(mean_19, 0)
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(mean_20, 0)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(mean_21, 0)
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(mean_22, 0)
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(mean_23, 0)
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(mean_24, 0)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(mean_25, 0)
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(mean_26, 0)
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(mean_27, 0)
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(mean_28, 0)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(mean_29, 0)
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(mean_30, 0)
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(mean_31, 0)
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(mean_32, 0)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(mean_33, 0)
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(mean_34, 0)
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(mean_35, 0)
        cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7, unsqueeze_8, unsqueeze_9, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23, unsqueeze_24, unsqueeze_25, unsqueeze_26, unsqueeze_27, unsqueeze_28, unsqueeze_29, unsqueeze_30, unsqueeze_31, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35]);  unsqueeze = unsqueeze_1 = unsqueeze_2 = unsqueeze_3 = unsqueeze_4 = unsqueeze_5 = unsqueeze_6 = unsqueeze_7 = unsqueeze_8 = unsqueeze_9 = unsqueeze_10 = unsqueeze_11 = unsqueeze_12 = unsqueeze_13 = unsqueeze_14 = unsqueeze_15 = unsqueeze_16 = unsqueeze_17 = unsqueeze_18 = unsqueeze_19 = unsqueeze_20 = unsqueeze_21 = unsqueeze_22 = unsqueeze_23 = unsqueeze_24 = unsqueeze_25 = unsqueeze_26 = unsqueeze_27 = unsqueeze_28 = unsqueeze_29 = unsqueeze_30 = unsqueeze_31 = unsqueeze_32 = unsqueeze_33 = unsqueeze_34 = unsqueeze_35 = None
        mean_36 = torch.ops.aten.mean.default(cat);  cat = None
        mul = torch.ops.aten.mul.Tensor(mean_36, 0.0001);  mean_36 = None
        return (mul, mean, mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, mean_8, mean_9, mean_10, mean_11, mean_12, mean_13, mean_14, mean_15, mean_16, mean_17, mean_18, mean_19, mean_20, mean_21, mean_22, mean_23, mean_24, mean_25, mean_26, mean_27, mean_28, mean_29, mean_30, mean_31, mean_32, mean_33, mean_34, mean_35, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_31, add_32, add_33, add_34, add_35)
        
def load_args(reader):
    buf0 = reader.storage(None, 176000, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32, 50, 55), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1632, 32), is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1632, 32), is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf3, (1632, 32), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf4, (1632, 32), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf5, (1632, 32), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf6, (1632, 32), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf7, (1632, 32), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf8, (1632, 32), is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1632, 32), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1632, 32), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1632, 32), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1632, 32), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf13, (1632, 32), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf14, (1632, 32), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf15, (1632, 32), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1632, 32), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf17, (1632, 32), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1632, 32), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1632, 32), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1632, 32), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf21, (1632, 32), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf22, (1632, 32), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf23, (1632, 32), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1632, 32), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf25, (1632, 32), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1632, 32), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1632, 32), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1632, 32), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1632, 32), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf30, (1632, 32), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf31, (1632, 32), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1632, 32), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf33, (1632, 32), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf34, (1632, 32), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf35, (1632, 32), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 208896, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1632, 32), is_leaf=True)  # primals_37
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)