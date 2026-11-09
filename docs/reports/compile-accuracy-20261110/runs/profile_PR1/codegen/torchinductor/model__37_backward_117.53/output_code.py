# AOT ID: ['37_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/5g/c5gy35zm2j5kai2pltdxyafs6tibype5xdg7sxqgnkyioh25lh3z.py
# Topologically Sorted Source Nodes: [clone_4, queries_4], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_4 => convert_element_type_35
#   queries_4 => convert_element_type_33, mul_11, sub_4
# Graph fragment:
#   %convert_element_type_35 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_28, torch.float32), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_33, %getitem_9), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_35, %mul_11), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [0, 1]), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_35, [0, 1]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_0 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 34603008, 'r0_': 65536}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 1024*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/gi/cgip3aosnkpdie5oxjkfp4gfefjfsbgll6iq4bkny2kyz5npo3gv.py
# Topologically Sorted Source Nodes: [clone_4, queries_4], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_4 => convert_element_type_35
#   queries_4 => convert_element_type_33, mul_11, sub_4
# Graph fragment:
#   %convert_element_type_35 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_28, torch.float32), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_33, %getitem_9), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_35, %mul_11), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_19, [0, 1]), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_4, torch.bfloat16), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_1 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 266240, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_1(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/xu/cxunjasd36fpyzs4cdqugcxtstzdyzarespzk7suah4ekjgeuyfe.py
# Topologically Sorted Source Nodes: [clone_4, clone_6, queries_4], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_4 => convert_element_type_35
#   clone_6 => convert_element_type_37
#   queries_4 => convert_element_type_33, mul_11, sub_4
# Graph fragment:
#   %convert_element_type_35 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.float32), kwargs = {})
#   %convert_element_type_37 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_20, torch.float32), kwargs = {})
#   %mul_14 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_35, %convert_element_type_37), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, 1024), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [2], True), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_28, torch.float32), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_33, %getitem_9), kwargs = {})
#   %mul_11 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %mul_11), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_16, [2], True), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %sum_3), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_15, %sum_2), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_6, %mul_17), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 1024), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_7), kwargs = {})
#   %convert_element_type_39 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.bfloat16), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_2 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536, 'r0_': 67110912}}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_out_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [R0_BLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 0.0009765625
    tmp19 = tmp12 * tmp18
    tmp20 = 1024.0
    tmp21 = tmp4 * tmp20
    tmp22 = tmp21 - tmp7
    tmp23 = tmp13 * tmp17
    tmp24 = tmp22 - tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp26, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/oy/coyygf5ikk7ynditmwsntmbvh4zqmoozgb5lz5y225cuism5kmvo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_29, [0], True), kwargs = {})
triton_red_fused_sum_3 = async_compile.triton('triton_red_fused_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 17301504, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 131072*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/gk/cgkpkl6u7vemwtkybtfkzwf465m7q3rbx4g2ntd5gxqjfwtyojf3.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_29, [0], True), kwargs = {})
triton_per_fused_sum_4 = async_compile.triton('triton_per_fused_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 266240, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/yq/cyq4xb2aedmjjogi6qjqdagziszzpcj32kpatvhzbw4zhdvzbyjs.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.gelu_backward, aten.gelu]
# Source node to ATen node mapping:
#   input_3 => add_7, convert_element_type_26, erf, mul_9
# Graph fragment:
#   %convert_element_type_50 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_33, torch.float32), kwargs = {})
#   %convert_element_type_26 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_24, torch.float32), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_9,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.5), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %convert_element_type_26), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, -0.5), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_23,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, 0.3989422804014327), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %mul_24), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %mul_25), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_50, %add_12), kwargs = {})
#   %convert_element_type_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_gelu_backward_5 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167772160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20971520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/mb/cmbz5jhhfu2bejc4pikzcdvz7t4cmdiz7ggpndn23eciqqxo25gi.py
# Topologically Sorted Source Nodes: [clone_8, clone_10, input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
# Source node to ATen node mapping:
#   clone_10 => convert_element_type_59
#   clone_8 => convert_element_type_57
#   input_1 => convert_element_type_22, mul_6, sub_3
# Graph fragment:
#   %convert_element_type_57 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.float32), kwargs = {})
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.float32), kwargs = {})
#   %mul_28 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_57, %convert_element_type_59), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, 2560), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [2], True), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %getitem_7), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %mul_6), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_30, [2], True), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %sum_8), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_29, %sum_7), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mul_31), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 2560), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sub_10), kwargs = {})
#   %convert_element_type_61 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_32, torch.bfloat16), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_31, %convert_element_type_61), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_6 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536, 'r0_': 209720320}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp4 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask, tmp17, _tmp16)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp18 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = 0.000390625
        tmp20 = tmp12 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp26 = 2560.0
        tmp27 = tmp25 * tmp26
        tmp28 = tmp27 - tmp6
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp10
        tmp32 = tmp31 * tmp12
        tmp33 = tmp32 * tmp16
        tmp34 = tmp28 - tmp33
        tmp35 = tmp20 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp18 + tmp36
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp37, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/c6/cc6onghmkujzgehgor7rka4mtbekkz4euhumzaztbqmaub67x7fx.py
# Topologically Sorted Source Nodes: [clone_8, input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_8 => convert_element_type_57
#   input_1 => convert_element_type_22, mul_6, sub_3
# Graph fragment:
#   %convert_element_type_57 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.float32), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %getitem_7), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_57, %mul_6), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_33, [0, 1]), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_57, [0, 1]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 86507520, 'r0_': 65536}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 163840
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_2 + 327680*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 2560*r0_2 + 327680*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp13, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/hu/chul2jrdiqwpp2jlerubfrzi2b3sfmdgqybgjezsjcfsgjbvnr65.py
# Topologically Sorted Source Nodes: [clone_8, input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_8 => convert_element_type_57
#   input_1 => convert_element_type_22, mul_6, sub_3
# Graph fragment:
#   %convert_element_type_57 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_35, torch.float32), kwargs = {})
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %getitem_7), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_57, %mul_6), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_33, [0, 1]), kwargs = {})
#   %convert_element_type_62 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_9, torch.bfloat16), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 665600, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_8(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/cl/cclse67w4q56k7d5wy6nbhmqyf3bw6cmtxfxtsckx23zbyay4muw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_29,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 256)
    x2 = ((xindex // 8192) % 4)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 128*x1 + 32768*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/h6/ch6f2dtpumybuh2ah4etxjjhdil3jzvknkeheuoahwlc3emcbamy.py
# Topologically Sorted Source Nodes: [float_1, softmax], Original ATen: [aten._to_copy, aten._softmax, aten._softmax_backward_data]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_16
#   softmax => div, exp, sub_2
# Graph fragment:
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_41, torch.float32), kwargs = {})
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_16, %amax), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_34 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_72, %div), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_34, [-1], True), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%div,), kwargs = {})
#   %fma : [num_users=1] = call_function[target=torch.ops.prims.fma.default](args = (%neg, %sum_11, %mul_34), kwargs = {})
#   %convert_element_type_73 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%fma, torch.bfloat16), kwargs = {})
triton_per_fused__softmax__softmax_backward_data__to_copy_10 = async_compile.triton('triton_per_fused__softmax__softmax_backward_data__to_copy_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__softmax_backward_data__to_copy_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 262144, 'r0_': 85983232}}
)
@triton.jit
def triton_per_fused__softmax__softmax_backward_data__to_copy_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    r0_numel = 328
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 328*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (r0_1 + 328*x0), r0_mask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp8 = (tmp6 / tmp7)
    tmp9 = tmp1 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = -tmp8
    tmp15 = libdevice.fma(tmp14, tmp13, tmp9)
    tmp16 = tmp15.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 328*x0), tmp16, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/lg/clgrdslotshxvhdbzhkbtgmmhsxcf4azwowri647dypay3mqcpkq.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_48, %view_46], 2), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5373952, 'x': 16121856}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10496
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 328)
    y1 = yindex // 328
    y3 = yindex
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 328*(((x2) % 128)) + 41984*y1), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = 0.4204482076268573
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1, 1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr1 + (32*y0 + 10496*(((((-128) + x2) // 32) % 4)) + 41984*y1 + ((((-128) + x2) % 32))), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp4, tmp9, tmp13)
    tl.store(out_ptr0 + (x2 + 256*y3), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/u4/cu4n2h4ooqwb4bzbkfu66ksgdrvtemzw535xsd5gohxwirsklxdo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
# Source node to ATen node mapping:
# Graph fragment:
#   %view_53 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%view_50, [8192, 128]), kwargs = {})
triton_poi_fused_view_12 = async_compile.triton('triton_poi_fused_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((x1 % 256)) + 8192*(x0 // 32) + 32768*(x1 // 256) + ((x0 % 32))), None).to(tl.float32)
    tmp1 = 0.4204482076268573
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/g3/cg33citsm75cmu2qyejaepfekef7u7l56b2vvioewc6gtsmrmcki.py
# Topologically Sorted Source Nodes: [clone_15, clone_17, latents], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_15 => convert_element_type_86
#   clone_17 => convert_element_type_88
#   latents => convert_element_type_8, mul_2, sub_1
# Graph fragment:
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_2, %view_54), kwargs = {})
#   %convert_element_type_86 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_14, torch.float32), kwargs = {})
#   %convert_element_type_88 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.float32), kwargs = {})
#   %mul_38 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_86, %convert_element_type_88), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, 2560), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [2], True), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_8, %getitem_3), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %mul_2), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [2], True), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %sum_13), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_39, %sum_12), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_12, %mul_41), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 2560), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sub_13), kwargs = {})
#   %convert_element_type_90 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_42, torch.bfloat16), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %convert_element_type_90), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 65536, 'r0_': 251663360}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (184320 + r0_2 + 2560*x0 + 839680*x1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 2560*x3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_2 + 2560*x3), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 - tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tmp6 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(r0_mask, tmp19, _tmp18)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp20 = tl.load(in_out_ptr0 + (r0_2 + 2560*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr0 + (184320 + r0_2 + 2560*x0 + 839680*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (r0_2 + 2560*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr3 + (r0_2 + 2560*x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = 0.000390625
        tmp22 = tmp14 * tmp21
        tmp25 = tmp23 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp30 = 2560.0
        tmp31 = tmp29 * tmp30
        tmp32 = tmp31 - tmp8
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp34 - tmp12
        tmp36 = tmp35 * tmp14
        tmp37 = tmp36 * tmp18
        tmp38 = tmp32 - tmp37
        tmp39 = tmp22 * tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp41 = tmp20 + tmp40
        tl.store(in_out_ptr0 + (r0_2 + 2560*x3), tmp41, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/u3/cu3iprj5znd6cwfq4jbdinaripywbm65ycj7t4aaktahvpzkzzrk.py
# Topologically Sorted Source Nodes: [clone_15, latents], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_15 => convert_element_type_86
#   latents => convert_element_type_8, mul_2, sub_1
# Graph fragment:
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_2, %view_54), kwargs = {})
#   %convert_element_type_86 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_14, torch.float32), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_8, %getitem_3), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_86, %mul_2), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [0, 1]), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_86, [0, 1]), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_14 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 128450560, 'r0_': 65536}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 163840
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (184320 + x0 + 2560*(((r0_2 + 128*x1) % 256)) + 839680*((r0_2 + 128*x1) // 256)), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 2560*r0_2 + 327680*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + 2560*r0_2 + 327680*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r0_2 + 128*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/55/c55etn5zon6p4ieyxtv43q5yi3mvepf7zbogxztdxonogh5nwcpa.py
# Topologically Sorted Source Nodes: [clone_19, x_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_19 => convert_element_type_93
#   x_1 => convert_element_type_6, mul, sub
# Graph fragment:
#   %convert_element_type_93 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_93, %mul), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_50, [0, 1]), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_93, [0, 1]), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 24330240, 'r0_': 18432}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 46080
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2560*(((r0_2 + 128*x1) % 72)) + 839680*((r0_2 + 128*x1) // 72)), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 2560*r0_2 + 327680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 128*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 - tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp1 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask & xmask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/nb/cnbx3m5gh6iuamzyicokff2v6tilur5jqjqluqlrkbnsepl5phmc.py
# Topologically Sorted Source Nodes: [clone_19, clone_21, x_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_19 => convert_element_type_93
#   clone_21 => convert_element_type_95
#   x_1 => convert_element_type_6, mul, sub
# Graph fragment:
#   %convert_element_type_93 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %convert_element_type_95 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_7, torch.float32), kwargs = {})
#   %mul_45 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_93, %convert_element_type_95), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, 2560), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_45, [2], True), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %mul), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_47, [2], True), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %sum_17), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_46, %sum_16), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %mul_48), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 2560), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sub_16), kwargs = {})
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_49, torch.bfloat16), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_16 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18432, 'r0_': 47191040}}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2304
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 72)
    x1 = xindex // 72
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    tmp10 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 2560*x0 + 839680*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_out_ptr0 + (r0_2 + 2560*x3), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp4 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(r0_mask & xmask, tmp17, _tmp16)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp20 = tl.load(in_ptr0 + (r0_2 + 2560*x0 + 839680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_out_ptr0 + (r0_2 + 2560*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = 0.000390625
        tmp19 = tmp12 * tmp18
        tmp21 = tmp20.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp25 = 2560.0
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 - tmp6
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp10
        tmp31 = tmp30 * tmp12
        tmp32 = tmp31 * tmp16
        tmp33 = tmp27 - tmp32
        tmp34 = tmp19 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_2 + 2560*x3), tmp35, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/2g/c2gqyss7jnxx7etmj5lnp2ku2watd3x7qg7bamizwu35zsbl2y4g.py
# Topologically Sorted Source Nodes: [clone_19, x_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone_19 => convert_element_type_93
#   x_1 => convert_element_type_6, mul, sub
# Graph fragment:
#   %convert_element_type_93 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %getitem_1), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_93, %mul), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_50, [0, 1]), kwargs = {})
#   %convert_element_type_98 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_18, torch.bfloat16), kwargs = {})
triton_per_fused_native_layer_norm_native_layer_norm_backward_17 = async_compile.triton('triton_per_fused_native_layer_norm_native_layer_norm_backward_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 194560, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_native_layer_norm_native_layer_norm_backward_17(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
    r0_numel = 18
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/sl/csl2gmle7xg5mkk6zpwwllnd2g6snoruliuyxcgnkerv6zvkdjyl.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_55, [0], True), kwargs = {})
triton_red_fused_sum_18 = async_compile.triton('triton_red_fused_sum_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 12165120, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_18(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 46080
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_2 + 327680*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/kb/ckbswurymcvpvd5p3zugfeustopbuokm4h6r44gxnzgphzwamq7o.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_55, [0], True), kwargs = {})
triton_per_fused_sum_19 = async_compile.triton('triton_per_fused_sum_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 32},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 194560, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_19(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
    r0_numel = 18
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/jg/cjgv5dqqyp6gx6dkxf66ohw7m7g2cphif5bolncbvj6lh6glfg35.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_58, [0], True), kwargs = {})
triton_red_fused_sum_20 = async_compile.triton('triton_red_fused_sum_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 43253760, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_20(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 163840
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_2 + 327680*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ms/cmsr4wgyfwjt4it4n4pah2krl4ih6uxgn5nr32oxmeboja7ul7sh.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_21 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_58, [0], True), kwargs = {})
triton_per_fused_sum_21 = async_compile.triton('triton_per_fused_sum_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 665600, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_21(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_4, primals_7, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, view, addmm, view_2, addmm_1, getitem_1, rsqrt, getitem_3, rsqrt_1, view_4, view_6, bmm, amax, sum_1, view_21, add_4, getitem_7, rsqrt_2, view_23, mm_3, view_25, view_27, addmm_2, getitem_9, rsqrt_3, permute_30, permute_31, permute_32, permute_33, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (2560, 2560), (2560, 1))
    assert_size_stride(primals_4, (2560, 2560), (2560, 1))
    assert_size_stride(primals_7, (2560, ), (1, ))
    assert_size_stride(primals_9, (2560, ), (1, ))
    assert_size_stride(primals_11, (128, 2560), (2560, 1))
    assert_size_stride(primals_12, (256, 2560), (2560, 1))
    assert_size_stride(primals_13, (2560, 128), (128, 1))
    assert_size_stride(primals_14, (2560, ), (1, ))
    assert_size_stride(primals_16, (2560, 2560), (2560, 1))
    assert_size_stride(primals_17, (2560, 2560), (2560, 1))
    assert_size_stride(primals_18, (1024, 2560), (2560, 1))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(view, (8192, 2560), (2560, 1))
    assert_size_stride(addmm, (8192, 2560), (2560, 1))
    assert_size_stride(view_2, (2304, 2560), (2560, 1))
    assert_size_stride(addmm_1, (2304, 2560), (2560, 1))
    assert_size_stride(getitem_1, (32, 72, 1), (72, 1, 1))
    assert_size_stride(rsqrt, (32, 72, 1), (72, 1, 1))
    assert_size_stride(getitem_3, (32, 256, 1), (256, 1, 1))
    assert_size_stride(rsqrt_1, (32, 256, 1), (256, 1, 1))
    assert_size_stride(view_4, (8192, 2560), (2560, 1))
    assert_size_stride(view_6, (10496, 2560), (2560, 1))
    assert_size_stride(bmm, (128, 256, 328), (83968, 328, 1))
    assert_size_stride(amax, (32, 4, 256, 1), (1024, 256, 1, 1))
    assert_size_stride(sum_1, (32, 4, 256, 1), (1024, 256, 1, 1))
    assert_size_stride(view_21, (8192, 128), (128, 1))
    assert_size_stride(add_4, (32, 256, 2560), (655360, 2560, 1))
    assert_size_stride(getitem_7, (32, 256, 1), (256, 1, 1))
    assert_size_stride(rsqrt_2, (32, 256, 1), (256, 1, 1))
    assert_size_stride(view_23, (8192, 2560), (2560, 1))
    assert_size_stride(mm_3, (8192, 2560), (2560, 1))
    assert_size_stride(view_25, (8192, 2560), (2560, 1))
    assert_size_stride(view_27, (8192, 2560), (2560, 1))
    assert_size_stride(addmm_2, (8192, 1024), (1024, 1))
    assert_size_stride(getitem_9, (32, 256, 1), (256, 1, 1))
    assert_size_stride(rsqrt_3, (32, 256, 1), (256, 1, 1))
    assert_size_stride(permute_30, (128, 328, 256), (83968, 1, 328))
    assert_size_stride(permute_31, (128, 32, 328), (10496, 1, 32))
    assert_size_stride(permute_32, (128, 32, 256), (8192, 1, 32))
    assert_size_stride(permute_33, (128, 328, 32), (10496, 1, 328))
    assert_size_stride(tangents_1, (32, 256, 1024), (262144, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1024, 64), (1, 1024), torch.float32)
        buf4 = empty_strided_cuda((1024, 64), (1, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [clone_4, queries_4], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_0.run(tangents_1, addmm_2, getitem_9, rsqrt_3, buf2, buf4, 65536, 128, stream=stream0)
        buf6 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_4, queries_4], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf2, buf6, 1024, 64, stream=stream0)
        del buf2
        buf7 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_4], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_1.run(buf4, buf7, 1024, 64, stream=stream0)
        buf8 = reinterpret_tensor(addmm_2, (32, 256, 1024), (262144, 1024, 1), 0); del addmm_2  # reuse
        # Topologically Sorted Source Nodes: [clone_4, clone_6, queries_4], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_2.run(buf8, tangents_1, primals_20, getitem_9, rsqrt_3, 8192, 1024, stream=stream0)
        del getitem_9
        del primals_20
        del rsqrt_3
        del tangents_1
        buf10 = empty_strided_cuda((1024, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (1024, 8192), (1, 1024), 0), view_27, out=buf10)
        del view_27
        buf11 = reinterpret_tensor(buf4, (1, 1024, 64), (65536, 1, 1024), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3.run(buf8, buf11, 65536, 128, stream=stream0)
        buf12 = empty_strided_cuda((1, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_4.run(buf11, buf12, 1024, 64, stream=stream0)
        del buf11
        buf9 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf8, (8192, 1024), (1024, 1), 0), primals_18, out=buf9)
        del buf8
        del primals_18
        buf13 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (2560, 8192), (1, 2560), 0), view_25, out=buf13)
        del view_25
        buf14 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf9, primals_17, out=buf14)
        del primals_17
        buf15 = reinterpret_tensor(buf14, (32, 256, 2560), (655360, 2560, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_5.run(buf15, mm_3, 20971520, stream=stream0)
        del mm_3
        buf16 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (2560, 8192), (1, 2560), 0), view_23, out=buf16)
        del view_23
        buf17 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf15, (8192, 2560), (2560, 1), 0), primals_16, out=buf17)
        del buf15
        del primals_16
        buf26 = reinterpret_tensor(buf9, (32, 256, 2560), (655360, 2560, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [clone_8, clone_10, input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_6.run(buf26, buf17, primals_14, add_4, getitem_7, rsqrt_2, 8192, 2560, stream=stream0)
        del primals_14
        buf20 = empty_strided_cuda((2560, 64), (1, 2560), torch.float32)
        buf22 = empty_strided_cuda((2560, 64), (1, 2560), torch.float32)
        # Topologically Sorted Source Nodes: [clone_8, input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_7.run(buf17, add_4, getitem_7, rsqrt_2, buf20, buf22, 163840, 128, stream=stream0)
        del add_4
        del getitem_7
        del rsqrt_2
        buf27 = empty_strided_cuda((2560, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (2560, 8192), (1, 2560), 0), view_21, out=buf27)
        del view_21
        buf24 = empty_strided_cuda((2560, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_8, input_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_8.run(buf20, buf24, 2560, 64, stream=stream0)
        buf25 = empty_strided_cuda((2560, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_8], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_8.run(buf22, buf25, 2560, 64, stream=stream0)
        buf28 = empty_strided_cuda((8192, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (8192, 2560), (2560, 1), 0), primals_13, out=buf28)
        del primals_13
        buf29 = empty_strided_cuda((32, 4, 256, 32), (32768, 8192, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf28, buf29, 1048576, stream=stream0)
        buf30 = empty_strided_cuda((128, 328, 32), (10496, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_30, reinterpret_tensor(buf29, (128, 256, 32), (8192, 32, 1), 0), out=buf30)
        del permute_30
        buf31 = empty_strided_cuda((128, 256, 328), (83968, 328, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (128, 256, 32), (8192, 32, 1), 0), permute_31, out=buf31)
        del permute_31
        buf33 = reinterpret_tensor(bmm, (32, 4, 256, 328), (335872, 83968, 328, 1), 0); del bmm  # reuse
        # Topologically Sorted Source Nodes: [float_1, softmax], Original ATen: [aten._to_copy, aten._softmax, aten._softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__softmax_backward_data__to_copy_10.run(buf33, buf31, amax, sum_1, 32768, 328, stream=stream0)
        del amax
        del buf31
        del sum_1
        buf35 = reinterpret_tensor(buf29, (128, 256, 32), (8192, 32, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (128, 256, 328), (83968, 328, 1), 0), permute_33, out=buf35)
        del permute_33
        buf34 = empty_strided_cuda((128, 32, 328), (10496, 328, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(permute_32, reinterpret_tensor(buf33, (128, 256, 328), (83968, 328, 1), 0), out=buf34)
        del buf33
        del permute_32
        buf36 = empty_strided_cuda((32, 328, 256), (83968, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf34, buf30, buf36, 10496, 256, stream=stream0)
        del buf30
        del buf34
        buf37 = empty_strided_cuda((256, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_41], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (256, 10496), (1, 256), 0), view_6, out=buf37)
        del view_6
        buf39 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_12.run(buf35, buf39, 1048576, stream=stream0)
        del buf35
        buf40 = empty_strided_cuda((128, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_45], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (128, 8192), (1, 128), 0), view_4, out=buf40)
        del view_4
        buf41 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf39, primals_11, out=buf41)
        del buf39
        del primals_11
        buf38 = empty_strided_cuda((10496, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf36, (10496, 256), (256, 1), 0), primals_12, out=buf38)
        del buf36
        del primals_12
        buf50 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [clone_15, clone_17, latents], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13.run(buf50, buf38, buf41, primals_9, addmm, getitem_3, rsqrt_1, 8192, 2560, stream=stream0)
        del primals_9
        buf44 = buf22; del buf22  # reuse
        buf46 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [clone_15, latents], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_14.run(buf38, buf41, addmm, getitem_3, rsqrt_1, buf44, buf46, 163840, 128, stream=stream0)
        del addmm
        del getitem_3
        del rsqrt_1
        buf65 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_53], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (2560, 8192), (1, 2560), 0), view, out=buf65)
        del view
        buf48 = empty_strided_cuda((2560, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_15, latents], Original ATen: [aten.add, aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_8.run(buf44, buf48, 2560, 64, stream=stream0)
        del buf44
        buf49 = empty_strided_cuda((2560, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_15], Original ATen: [aten.add, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_8.run(buf46, buf49, 2560, 64, stream=stream0)
        buf53 = empty_strided_cuda((2560, 18), (1, 2560), torch.float32)
        buf55 = empty_strided_cuda((2560, 18), (1, 2560), torch.float32)
        # Topologically Sorted Source Nodes: [clone_19, x_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_15.run(buf38, addmm_1, getitem_1, rsqrt, buf53, buf55, 46080, 128, stream=stream0)
        buf59 = reinterpret_tensor(addmm_1, (32, 72, 2560), (184320, 2560, 1), 0); del addmm_1  # reuse
        # Topologically Sorted Source Nodes: [clone_19, clone_21, x_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_16.run(buf59, buf38, primals_7, getitem_1, rsqrt, 2304, 2560, stream=stream0)
        del buf38
        del getitem_1
        del primals_7
        del rsqrt
        buf57 = empty_strided_cuda((2560, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_19, x_1], Original ATen: [aten.native_layer_norm_backward, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_17.run(buf53, buf57, 2560, 18, stream=stream0)
        del buf53
        buf58 = empty_strided_cuda((2560, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_19], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_native_layer_norm_backward_17.run(buf55, buf58, 2560, 18, stream=stream0)
        buf62 = reinterpret_tensor(buf55, (1, 2560, 18), (46080, 1, 2560), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_18.run(buf59, buf62, 46080, 128, stream=stream0)
        buf63 = empty_strided_cuda((1, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_19.run(buf62, buf63, 2560, 18, stream=stream0)
        del buf62
        buf66 = reinterpret_tensor(buf46, (1, 2560, 64), (163840, 1, 2560), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_20.run(buf50, buf66, 163840, 128, stream=stream0)
        buf67 = empty_strided_cuda((1, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_21.run(buf66, buf67, 2560, 64, stream=stream0)
        del buf66
        buf64 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (8192, 2560), (2560, 1), 0), primals_1, out=buf64)
        del buf50
        del primals_1
        buf61 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_49], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (2560, 2304), (1, 2560), 0), view_2, out=buf61)
        del view_2
        buf60 = empty_strided_cuda((2304, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (2304, 2560), (2560, 1), 0), primals_4, out=buf60)
        del buf59
        del primals_4
    return (buf65, reinterpret_tensor(buf67, (2560, ), (1, ), 0), reinterpret_tensor(buf64, (32, 256, 2560), (655360, 2560, 1), 0), buf61, reinterpret_tensor(buf63, (2560, ), (1, ), 0), reinterpret_tensor(buf60, (32, 72, 2560), (184320, 2560, 1), 0), buf57, buf58, buf48, buf49, buf40, buf37, buf27, buf24, buf25, buf16, buf13, buf10, reinterpret_tensor(buf12, (1024, ), (1, ), 0), buf6, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((128, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((256, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((2560, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_17 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_18 = rand_strided((1024, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_20 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    view = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_2 = rand_strided((2304, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_1 = rand_strided((2304, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((32, 72, 1), (72, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((32, 72, 1), (72, 1, 1), device='cuda:0', dtype=torch.float32)
    getitem_3 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    view_4 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_6 = rand_strided((10496, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    bmm = rand_strided((128, 256, 328), (83968, 328, 1), device='cuda:0', dtype=torch.bfloat16)
    amax = rand_strided((32, 4, 256, 1), (1024, 256, 1, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((32, 4, 256, 1), (1024, 256, 1, 1), device='cuda:0', dtype=torch.float32)
    view_21 = rand_strided((8192, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    add_4 = rand_strided((32, 256, 2560), (655360, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_7 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_2 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    view_23 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_3 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_25 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    view_27 = rand_strided((8192, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_2 = rand_strided((8192, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_3 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    permute_30 = rand_strided((128, 328, 256), (83968, 1, 328), device='cuda:0', dtype=torch.bfloat16)
    permute_31 = rand_strided((128, 32, 328), (10496, 1, 32), device='cuda:0', dtype=torch.bfloat16)
    permute_32 = rand_strided((128, 32, 256), (8192, 1, 32), device='cuda:0', dtype=torch.bfloat16)
    permute_33 = rand_strided((128, 328, 32), (10496, 1, 328), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_4, primals_7, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, view, addmm, view_2, addmm_1, getitem_1, rsqrt, getitem_3, rsqrt_1, view_4, view_6, bmm, amax, sum_1, view_21, add_4, getitem_7, rsqrt_2, view_23, mm_3, view_25, view_27, addmm_2, getitem_9, rsqrt_3, permute_30, permute_31, permute_32, permute_33, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
