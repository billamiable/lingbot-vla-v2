# AOT ID: ['44_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /root/qihank/.cache/sync_bench/inductor/e2/ce26ekrlemwqxn4xjvnndhlmkp7hr64djfccriuzs3dnun2hlb6m.py
# Topologically Sorted Source Nodes: [preds, target, mse_loss], Original ATen: [aten._to_copy, aten.mse_loss]
# Source node to ATen node mapping:
#   mse_loss => mean, pow_1, sub
#   preds => convert_element_type
#   target => convert_element_type_1
# Graph fragment:
#   %convert_element_type : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=3] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
triton_red_fused__to_copy_mse_loss_0 = async_compile.triton('triton_red_fused__to_copy_mse_loss_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mse_loss_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4096, 'r0_': 100663296}}
)
@triton.jit
def triton_red_fused__to_copy_mse_loss_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 16384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + 16384*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 - tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tl.store(out_ptr0 + (r0_1 + 16384*x0), tmp1, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/zt/cztxgrq4w57jhciwvdrigvehl4lzmrqlpq5uwovueeuuhfgw4zgg.py
# Topologically Sorted Source Nodes: [target, pred_norm, target_norm, cosine_similarity], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.div, aten.clamp_min, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   cosine_similarity => clamp_min_2, clamp_min_3, div_2, div_3, mul_2, pow_6, pow_7, pow_8, pow_9, sum_3, sum_4, sum_5
#   pred_norm => div, pow_2, pow_3, sum_1
#   target => convert_element_type_1
#   target_norm => clamp_min_1, div_1, pow_4, pow_5, sum_2
# Graph fragment:
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type, 2.0), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_2, [-1], True), kwargs = {})
#   %pow_3 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %expand), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1, 2.0), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [-1], True), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_5, 1e-06), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_1, %expand_1), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_6, [-1], True), kwargs = {})
#   %pow_7 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %pow_8 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_1, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_8, [-1], True), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_7, 1e-06), kwargs = {})
#   %clamp_min_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_9, 1e-06), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, %clamp_min_3), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %clamp_min_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %div_2), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [-1]), kwargs = {})
triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_1 = async_compile.triton('triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 5, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 327680, 'r0_': 50331648}}
)
@triton.jit
def triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_1(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None)
    tmp14 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp2, 0))
    tmp5 = libdevice.sqrt(tmp4)
    tmp6 = 1e-06
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp13 = libdevice.sqrt(tmp12)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [R0_BLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = libdevice.sqrt(tmp19)
    tmp21 = triton_helpers.maximum(tmp20, tmp6)
    tmp22 = (tmp15 / tmp21)
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [R0_BLOCK])
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp27 = libdevice.sqrt(tmp26)
    tmp28 = triton_helpers.maximum(tmp27, tmp6)
    tmp29 = triton_helpers.maximum(tmp13, tmp6)
    tmp30 = (tmp8 / tmp29)
    tmp31 = (tmp22 / tmp28)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [R0_BLOCK])
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp13, None)
    tl.debug_barrier()
    tl.store(in_out_ptr2 + (x0), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr3 + (x0), tmp28, None)
    tl.store(out_ptr0 + (x0), tmp35, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/hz/chz5v5headagq4qjsdgli3uq4xnitpporyurtppkknpygdxhqvt4.py
# Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_5,), kwargs = {})
triton_red_fused_mean_2 = async_compile.triton('triton_red_fused_mean_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 32768}}
)
@triton.jit
def triton_red_fused_mean_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/in/cinb6d2rq433gjfgpqmggqaub2gx2g7q5bnpxs4tgkb36rpp47oh.py
# Topologically Sorted Source Nodes: [target, mse_loss, mul, weighted_mse_loss, mean, cosine_loss, mul_2, weighted_cosine_loss, loss], Original ATen: [aten._to_copy, aten.mse_loss, aten.mul, aten.mean, aten.rsub, aten.add]
# Source node to ATen node mapping:
#   cosine_loss => sub_1
#   loss => add
#   mean => mean_1
#   mse_loss => mean, pow_1, sub
#   mul => mul
#   mul_2 => mul_3
#   target => convert_element_type_1
#   weighted_cosine_loss => mul_4
#   weighted_mse_loss => mul_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=3] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 1.0), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_5,), kwargs = {})
#   %sub_1 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %mean_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, 0.2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, 0.2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_4), kwargs = {})
triton_per_fused__to_copy_add_mean_mse_loss_mul_rsub_3 = async_compile.triton('triton_per_fused__to_copy_add_mean_mse_loss_mul_rsub_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mse_loss_mul_rsub_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 2048}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mse_loss_mul_rsub_3(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp6 = tl.load(in_out_ptr1 + (0))
    tmp7 = tl.broadcast_to(tmp6, [1])
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp3 = triton_helpers.promote_to_tensor(tl.sum(tmp1, 0))
    tmp4 = 8388608.0
    tmp5 = (tmp3 / tmp4)
    tmp8 = 8192.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp5 * tmp10
    tmp13 = 0.2
    tmp14 = tmp11 * tmp13
    tmp15 = tmp12 + tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp5, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([1], 0, tl.int32)), tmp11, None)
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp15, None)
    tl.store(out_ptr1 + (tl.full([1], 0, tl.int32)), tmp14, None)
    tl.store(out_ptr2 + (tl.full([1], 0, tl.int32)), tmp12, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (32, 256, 1024), (262144, 1024, 1))
    assert_size_stride(primals_2, (32, 256, 1024), (262144, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 256, 1024), (262144, 1024, 1), torch.float32)
        buf2 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [preds, target, mse_loss], Original ATen: [aten._to_copy, aten.mse_loss]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mse_loss_0.run(primals_1, primals_2, buf0, buf2, 512, 16384, stream=stream0)
        del primals_1
        buf5 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf6 = reinterpret_tensor(buf5, (32, 256, 1), (256, 1, 1), 0); del buf5  # reuse
        buf9 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf10 = reinterpret_tensor(buf9, (32, 256, 1), (256, 1, 1), 0); del buf9  # reuse
        buf7 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf8 = reinterpret_tensor(buf7, (32, 256, 1), (256, 1, 1), 0); del buf7  # reuse
        buf11 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf12 = reinterpret_tensor(buf11, (32, 256, 1), (256, 1, 1), 0); del buf11  # reuse
        buf13 = empty_strided_cuda((32, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [target, pred_norm, target_norm, cosine_similarity], Original ATen: [aten._to_copy, aten.linalg_vector_norm, aten.div, aten.clamp_min, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clamp_min_div_linalg_vector_norm_mul_sum_1.run(buf6, buf10, buf8, buf12, buf0, primals_2, buf13, 8192, 1024, stream=stream0)
        buf14 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mean], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_2.run(buf13, buf14, 1, 8192, stream=stream0)
        del buf13
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        buf15 = buf14; del buf14  # reuse
        buf16 = empty_strided_cuda((), (), torch.float32)
        buf18 = empty_strided_cuda((), (), torch.float32)
        buf17 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [target, mse_loss, mul, weighted_mse_loss, mean, cosine_loss, mul_2, weighted_cosine_loss, loss], Original ATen: [aten._to_copy, aten.mse_loss, aten.mul, aten.mean, aten.rsub, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mse_loss_mul_rsub_3.run(buf4, buf15, buf2, buf16, buf18, buf17, 1, 512, stream=stream0)
        del buf2
    return (buf16, buf0, buf4, buf17, buf15, buf18, primals_2, buf0, buf6, buf8, buf10, buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
