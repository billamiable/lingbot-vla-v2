# AOT ID: ['46_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/75/c75zqnb33lltzco52jhhjpbehcepotaqo2pl72lzxcexut5mv5va.py
# Topologically Sorted Source Nodes: [all_scores, all_probs, scatter_add_], Original ATen: [aten.sigmoid, aten.div, aten.scatter_add]
# Source node to ATen node mapping:
#   all_probs => div
#   all_scores => sigmoid
#   scatter_add_ => scatter_add
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%primals_4,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sigmoid, %sum_1), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %div), kwargs = {})
triton_poi_fused_div_scatter_add_sigmoid_0 = async_compile.triton('triton_poi_fused_div_scatter_add_sigmoid_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_scatter_add_sigmoid_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 294912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_scatter_add_sigmoid_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/n6/cn6le6lfgp6jjip5y3hcr77vebmueekjpjdyy4yw77w5rutws2df.py
# Topologically Sorted Source Nodes: [all_scores, sum_1, all_probs, scatter_add_], Original ATen: [aten.sigmoid, aten.sum, aten.div, aten.scatter_add]
# Source node to ATen node mapping:
#   all_probs => div
#   all_scores => sigmoid
#   scatter_add_ => scatter_add
#   sum_1 => sum_1
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%primals_4,), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sigmoid, %sum_1), kwargs = {})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default, 0, %expand, %div), kwargs = {})
triton_per_fused_div_scatter_add_sigmoid_sum_1 = async_compile.triton('triton_per_fused_div_scatter_add_sigmoid_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_scatter_add_sigmoid_sum_1', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_scatter_add_sigmoid_sum_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 58752
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tl.device_assert(((0 <= tmp6) & (tmp6 < 1152)) | ~(xmask), "index out of bounds: 0 <= tmp6 < 1152")
    tmp8 = (tmp1 / tmp5)
    tl.atomic_add(out_ptr1 + (tl.broadcast_to(r0_1 + 32*tmp6, [XBLOCK, R0_BLOCK])), tmp8, xmask, sem='relaxed')
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/triton_moe_loss.py:63
_topk_segment_count_kernel_0 = async_compile.triton('_topk_segment_count_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_topk_segment_count_kernel_0', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'logits_ptr': '*fp32', 'seg_starts_ptr': '*i64', 'seg_lengths_ptr': '*i64', 'f_out_ptr': '*fp32', 'stride_logits_n': 'i32', 'E': 'constexpr', 'K': 'constexpr', 'BLOCK_E': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'E': 32, 'K': 4, 'BLOCK_E': 32}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _topk_segment_count_kernel(
    logits_ptr,         # [N_total, E]
    seg_starts_ptr,     # [S_total]
    seg_lengths_ptr,    # [S_total]
    f_out_ptr,          # [S_total, E]
    stride_logits_n,    # stride of logits along token dim
    E: tl.constexpr,    # actual num_experts
    K: tl.constexpr,    # top_k
    BLOCK_E: tl.constexpr,  # next power of 2 >= E
):
    """Each program computes f_i (expert counts) for one segment.

    Iterates over tokens in the segment, performs K rounds of argmax to
    find top-K experts, and accumulates per-expert hit counts.
    No gradient needed — f_i is always detached in the loss.
    """
    seg_id = tl.program_id(0)
    seg_start = tl.load(seg_starts_ptr + seg_id)
    seg_len = tl.load(seg_lengths_ptr + seg_id)

    expert_offs = tl.arange(0, BLOCK_E)  # [BLOCK_E]
    mask_e = expert_offs < E
    f_acc = tl.zeros((BLOCK_E,), dtype=tl.float32)

    for t in range(0, seg_len):
        row_ptr = logits_ptr + (seg_start + t) * stride_logits_n
        logits_row = tl.load(row_ptr + expert_offs, mask=mask_e, other=float('-inf'))

        # K rounds of argmax to find top-K indices
        row_copy = logits_row
        for _k in range(K):
            max_val = tl.max(row_copy, axis=0)
            is_max = (row_copy == max_val)
            # Distribute count evenly among ties (rare with float32)
            n_ties = tl.sum(is_max.to(tl.float32), axis=0)
            f_acc += tl.where(is_max, 1.0 / n_ties, 0.0)
            row_copy = tl.where(is_max, float('-inf'), row_copy)

    # Write f_count (unnormalized) — caller normalizes by (E / K) / seg_len
    out_ptr = f_out_ptr + seg_id * E
    tl.store(out_ptr + expert_offs, f_acc, mask=mask_e)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/x3/cx3ogkjsbscj7bahldalf6m5qss4u4c25wxthixx73ubtncmosc7.py
# Topologically Sorted Source Nodes: [float_1, clamp, P_i, inv_lens, f_i, mul_1, loss_per_seg], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.reciprocal, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   P_i => div_1
#   clamp => clamp_min
#   f_i => mul_1
#   float_1 => convert_element_type
#   inv_lens => mul, reciprocal
#   loss_per_seg => sum_2
#   mul_1 => mul_2
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_1, torch.float32), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type, 1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%scatter_add, %clamp_min), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 8.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %mul), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %div_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [-1]), kwargs = {})
triton_per_fused__to_copy_clamp_div_mul_reciprocal_sum_2 = async_compile.triton('triton_per_fused__to_copy_clamp_div_mul_reciprocal_sum_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clamp_div_mul_reciprocal_sum_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 27648, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_clamp_div_mul_reciprocal_sum_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1152
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp10 = tl.load(in_ptr2 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = tl.full([1, 1], 1, tl.int32)
    tmp6 = (tmp5 / tmp3)
    tmp7 = 8.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp4 * tmp8
    tmp11 = (tmp10 / tmp3)
    tmp12 = tmp9 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ur/curhgycuza3rwl3quvtxellc6glcx2gojig2xjl6b24qynm3nzv2.py
# Topologically Sorted Source Nodes: [layer_losses], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   layer_losses => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view, [1]), kwargs = {})
triton_per_fused_mean_3 = async_compile.triton('triton_per_fused_mean_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 288, 'r0_': 4608}}
)
@triton.jit
def triton_per_fused_mean_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 36
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 32.0
    tmp6 = (tmp4 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (58752, ), (1, ))
    assert_size_stride(primals_2, (1152, ), (1, ))
    assert_size_stride(primals_3, (1152, ), (1, ))
    assert_size_stride(primals_4, (58752, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1152, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [all_scores, all_probs, scatter_add_], Original ATen: [aten.sigmoid, aten.div, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_scatter_add_sigmoid_0.run(buf2, 36864, stream=stream0)
        buf0 = empty_strided_cuda((58752, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [all_scores, sum_1, all_probs, scatter_add_], Original ATen: [aten.sigmoid, aten.sum, aten.div, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_scatter_add_sigmoid_sum_1.run(primals_4, primals_1, buf0, buf2, 58752, 32, stream=stream0)
        buf1 = empty_strided_cuda((1152, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [P_sum], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_scatter_add_sigmoid_0.run(buf1, 36864, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _topk_segment_count_kernel_0.run(primals_4, primals_2, primals_3, buf1, 32, 32, 4, 32, 1152, 1, 1, stream=stream0)
        del primals_2
        buf4 = empty_strided_cuda((1152, 1), (1, 1), torch.float32)
        buf6 = empty_strided_cuda((1152, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, clamp, P_i, inv_lens, f_i, mul_1, loss_per_seg], Original ATen: [aten._to_copy, aten.clamp, aten.div, aten.reciprocal, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_clamp_div_mul_reciprocal_sum_2.run(primals_3, buf1, buf2, buf4, buf6, 1152, 32, stream=stream0)
        del buf2
        del primals_3
        buf7 = empty_strided_cuda((36, ), (1, ), torch.float32)
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [layer_losses], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_3.run(buf8, buf6, 36, 32, stream=stream0)
        del buf6
    return (reinterpret_tensor(buf8, (), (), 0), reinterpret_tensor(buf8, (), (), 1), reinterpret_tensor(buf8, (), (), 2), reinterpret_tensor(buf8, (), (), 3), reinterpret_tensor(buf8, (), (), 4), reinterpret_tensor(buf8, (), (), 5), reinterpret_tensor(buf8, (), (), 6), reinterpret_tensor(buf8, (), (), 7), reinterpret_tensor(buf8, (), (), 8), reinterpret_tensor(buf8, (), (), 9), reinterpret_tensor(buf8, (), (), 10), reinterpret_tensor(buf8, (), (), 11), reinterpret_tensor(buf8, (), (), 12), reinterpret_tensor(buf8, (), (), 13), reinterpret_tensor(buf8, (), (), 14), reinterpret_tensor(buf8, (), (), 15), reinterpret_tensor(buf8, (), (), 16), reinterpret_tensor(buf8, (), (), 17), reinterpret_tensor(buf8, (), (), 18), reinterpret_tensor(buf8, (), (), 19), reinterpret_tensor(buf8, (), (), 20), reinterpret_tensor(buf8, (), (), 21), reinterpret_tensor(buf8, (), (), 22), reinterpret_tensor(buf8, (), (), 23), reinterpret_tensor(buf8, (), (), 24), reinterpret_tensor(buf8, (), (), 25), reinterpret_tensor(buf8, (), (), 26), reinterpret_tensor(buf8, (), (), 27), reinterpret_tensor(buf8, (), (), 28), reinterpret_tensor(buf8, (), (), 29), reinterpret_tensor(buf8, (), (), 30), reinterpret_tensor(buf8, (), (), 31), reinterpret_tensor(buf8, (), (), 32), reinterpret_tensor(buf8, (), (), 33), reinterpret_tensor(buf8, (), (), 34), reinterpret_tensor(buf8, (), (), 35), primals_1, primals_4, buf0, buf4, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((58752, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((58752, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
