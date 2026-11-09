# AOT ID: ['5_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/4l/c4lnsbltmfk6rxugckuz4uthawllppebsoggsipkc4a2kvyliwtv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
triton_red_fused_sum_0 = async_compile.triton('triton_red_fused_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 51380224, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 192
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
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/je/cjeguewtpxesfixysuvbjghvijz4lkmv2dkkyhbozhfhjifsdeo4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [0], True), kwargs = {})
triton_red_fused_sum_1 = async_compile.triton('triton_red_fused_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 528384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/3d/c3d2lslegb5i7yu4viqilfyfw3yv7ufs3tepmdef3knqzoktkp2t.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu_backward, aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_7, add_8, convert_element_type_17, mul_10, mul_11, mul_12, mul_8, mul_9, tanh
# Graph fragment:
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %convert_element_type_17 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_2, torch.float32), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %convert_element_type_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %convert_element_type_17), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 0.044715), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_17, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_11,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, 0.5), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %tanh), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_20), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 0.134145), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, 1), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, 0.7978845608028654), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, 0.5), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %sub_2), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %mul_22), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %mul_24), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %add_13), kwargs = {})
#   %convert_element_type_28 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_25, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_gelu_backward_2 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 805306368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_gelu_backward_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tmp4 * tmp3
    tmp6 = 0.044715
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = 0.5
    tmp15 = tmp13 * tmp14
    tmp16 = tmp3 * tmp14
    tmp17 = tmp11 * tmp11
    tmp18 = tmp12 - tmp17
    tmp19 = tmp16 * tmp18
    tmp20 = 0.134145
    tmp21 = tmp4 * tmp20
    tmp22 = tmp21 + tmp12
    tmp23 = tmp22 * tmp9
    tmp24 = tmp19 * tmp23
    tmp25 = tmp15 + tmp24
    tmp26 = tmp1 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp27, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/kt/cktpttj6locodvf3btwgcy2pbfiq5yowq5p5welaw4hzbsmb6nu4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_28, [0], True), kwargs = {})
triton_red_fused_sum_3 = async_compile.triton('triton_red_fused_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 202309632, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 114688
    r0_numel = 878
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 4096
    x0 = (xindex % 4096)
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 878*x1
        tmp1 = tl.full([1, 1], 24576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 4096*r0_2 + 3596288*x1), r0_mask & tmp2, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/bm/cbmihntobxaaibzypfff6ijtb2ldwg6x53hqvwnt6peivwfau7ae.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_28, [0], True), kwargs = {})
triton_per_fused_sum_4 = async_compile.triton('triton_per_fused_sum_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 475136, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 28
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*r0_1), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/zz/czzw3ixcbeeiyzg5mdzqsgj4e7lwt75uuvxzuel7prh47xkr2dp6.py
# Topologically Sorted Source Nodes: [clone, clone_2, hidden_states, layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone => convert_element_type_33
#   clone_2 => convert_element_type_35
#   hidden_states => add_4
#   layer_norm_1 => convert_element_type_12, mul_6, sub_1
# Graph fragment:
#   %convert_element_type_33 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_11, torch.float32), kwargs = {})
#   %mul_27 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_33, %convert_element_type_35), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, 1024), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_27, [1], True), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %addmm_1), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %getitem_10), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %mul_6), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_29, [1], True), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %sum_4), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_28, %sum_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_4, %mul_30), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 1024), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_5), kwargs = {})
#   %convert_element_type_37 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_31, torch.bfloat16), kwargs = {})
#   %add_14 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_1, %convert_element_type_37), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608, 'r0_': 301991936}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel):
    xnumel = 24576
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
    tmp8 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp10 = tmp8 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 - tmp12
    tmp15 = tmp13 * tmp14
    tmp16 = tmp4 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [R0_BLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp21 = 0.0009765625
    tmp22 = tmp14 * tmp21
    tmp23 = 1024.0
    tmp24 = tmp4 * tmp23
    tmp25 = tmp24 - tmp7
    tmp26 = tmp15 * tmp19
    tmp27 = tmp25 - tmp26
    tmp28 = tmp22 * tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp20 + tmp29
    tl.store(out_ptr2 + (r0_1 + 1024*x0), tmp30, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/x3/cx3agxbacv7thahkkpdve4peurd6m3r6e3zhsq2u2ihg3gdaypji.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
# Source node to ATen node mapping:
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_47, %convert_element_type_46, %permute_37],), kwargs = {})
triton_poi_fused_stack_6 = async_compile.triton('triton_poi_fused_stack_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 415236096}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75497472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 1024
    x0 = (xindex % 64)
    x3 = (xindex % 1024)
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 24576, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 32, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tmp7 & tmp4
    tmp9 = tl.load(in_ptr0 + ((-32) + x3 + 1024*(x2)), tmp8, other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.load(in_ptr1 + ((-32) + x0 + 64*(x2)), tmp8, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp10 * tmp11
    tmp13 = -tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = 0.0
    tmp17 = tl.where(tmp7, tmp15, tmp16)
    tmp18 = tmp5 < tmp6
    tmp19 = tmp18 & tmp4
    tmp20 = tl.load(in_ptr0 + (32 + x3 + 1024*(x2)), tmp19, other=0.0).to(tl.float32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (32 + x0 + 64*(x2)), tmp19, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp19, tmp23, tmp24)
    tmp26 = tl.where(tmp18, tmp25, tmp16)
    tmp27 = tmp17 + tmp26
    tmp28 = tl.load(in_ptr0 + (x3 + 1024*(x2)), tmp4, other=0.0).to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.load(in_ptr2 + (x0 + 64*(x2)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp27 + tmp31
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp4, tmp33, tmp34)
    tmp36 = tmp0 >= tmp3
    tmp37 = tl.full([1], 49152, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = x0
    tmp41 = tl.full([1], 32, tl.int64)
    tmp42 = tmp40 >= tmp41
    tmp43 = tmp42 & tmp39
    tmp44 = tl.load(in_ptr3 + ((-32) + x3 + 1024*((-24576) + x2)), tmp43, other=0.0).to(tl.float32)
    tmp45 = tmp44.to(tl.float32)
    tmp46 = tl.load(in_ptr1 + ((-32) + x0 + 64*((-24576) + x2)), tmp43, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp45 * tmp46
    tmp48 = -tmp47
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp43, tmp48, tmp49)
    tmp51 = 0.0
    tmp52 = tl.where(tmp42, tmp50, tmp51)
    tmp53 = tmp40 < tmp41
    tmp54 = tmp53 & tmp39
    tmp55 = tl.load(in_ptr3 + (32 + x3 + 1024*((-24576) + x2)), tmp54, other=0.0).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tl.load(in_ptr1 + (32 + x0 + 64*((-24576) + x2)), tmp54, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp56 * tmp57
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp54, tmp58, tmp59)
    tmp61 = tl.where(tmp53, tmp60, tmp51)
    tmp62 = tmp52 + tmp61
    tmp63 = tl.load(in_ptr3 + (x3 + 1024*((-24576) + x2)), tmp39, other=0.0).to(tl.float32)
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tl.load(in_ptr2 + (x0 + 64*((-24576) + x2)), tmp39, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tmp62 + tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp39, tmp68, tmp69)
    tmp71 = tmp0 >= tmp37
    tmp72 = tl.full([1], 73728, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tl.load(in_ptr4 + (x3 + 1024*((-49152) + x2)), tmp71, other=0.0).to(tl.float32)
    tmp75 = tl.where(tmp39, tmp70, tmp74)
    tmp76 = tl.where(tmp4, tmp35, tmp75)
    tl.store(out_ptr0 + (x4), tmp76, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/qt/cqtpcomvxtpchkfkcmrgin7p5pz2rbe22h3rxnvypbbmgziufv77.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_38,), kwargs = {memory_format: torch.contiguous_format})
#   %view_18 : [num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [24576, 3072]), kwargs = {})
triton_poi_fused__unsafe_view_clone_7 = async_compile.triton('triton_poi_fused__unsafe_view_clone_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 452984832}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75497472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1024*x1 + 25165824*(x0 // 1024) + ((x0 % 1024))), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/h2/ch2nuzchlzbr5g2yqbux7phjh6gzlaurivirmxyjt5n6dzjpww4n.py
# Topologically Sorted Source Nodes: [clone, hidden_states, layer_norm_1, clone_5, clone_6], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone => convert_element_type_33
#   clone_5 => convert_element_type_52
#   clone_6 => convert_element_type_53
#   hidden_states => add_4
#   layer_norm_1 => convert_element_type_12, mul_6, sub_1
# Graph fragment:
#   %convert_element_type_33 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %addmm_1), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %getitem_10), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_33, %mul_6), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_32, [0]), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_33, [0]), kwargs = {})
#   %convert_element_type_52 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_53, %getitem_1), kwargs = {})
#   %mul_37 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %mul_37), kwargs = {})
#   %sum_11 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_43, [0]), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_52, [0]), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 205520896, 'r0_': 393216}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 131072
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp29 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0_2 + 192*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r0_2 + 192*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tl.load(in_ptr5 + (x0 + 1024*r0_2 + 196608*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr6 + (r0_2 + 192*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp22 = tl.load(in_ptr7 + (r0_2 + 192*x1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
        tmp14 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp2.to(tl.float32)
        tmp21 = tmp19 - tmp20
        tmp23 = tmp21 * tmp22
        tmp24 = tmp18 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(r0_mask, tmp27, _tmp26)
        tmp28 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(r0_mask, tmp30, _tmp29)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp26, None)
    tl.store(out_ptr3 + (x3), tmp29, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/kw/ckwkavszsj3alzr2lsdmcepufp7bxkg3rnn4ezndkc6lza5gqq76.py
# Topologically Sorted Source Nodes: [clone, hidden_states, layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   clone => convert_element_type_33
#   hidden_states => add_4
#   layer_norm_1 => convert_element_type_12, mul_6, sub_1
# Graph fragment:
#   %convert_element_type_33 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %addmm_1), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %getitem_10), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_33, %mul_6), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_32, [0]), kwargs = {})
#   %convert_element_type_38 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_5, torch.bfloat16), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 528384, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9(in_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1024*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = tmp2.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/2l/c2lfa74abozorrckb72ts5r3igega2lssk6wclzfuxk7zg3x6z24.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_18, [0], True), kwargs = {})
triton_red_fused_sum_10 = async_compile.triton('triton_red_fused_sum_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 152082432, 'r0_': 0}}
)
@triton.jit
def triton_red_fused_sum_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 113664
    r0_numel = 665
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 3072
    x0 = (xindex % 3072)
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 665*x1
        tmp1 = tl.full([1, 1], 24576, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 3072*r0_2 + 2042880*x1), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/4t/c4tl3evqok2ajbsbcwsh5fzptaw2jcn4pljtfkesdyr5aej4vmgz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_18, [0], True), kwargs = {})
triton_per_fused_sum_11 = async_compile.triton('triton_per_fused_sum_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 466944, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_11(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 3072
    r0_numel = 37
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (x0 + 3072*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ah/cahgfmlpa3pex6dxhjshawdsbhwjxuqam66y4s35lw3vuuvbgv2h.py
# Topologically Sorted Source Nodes: [clone_5, clone_6, clone_7], Original ATen: [aten.native_layer_norm_backward, aten.add]
# Source node to ATen node mapping:
#   clone_5 => convert_element_type_52
#   clone_6 => convert_element_type_53
#   clone_7 => convert_element_type_54
# Graph fragment:
#   %convert_element_type_52 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_6, torch.float32), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %convert_element_type_54 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float32), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_53, %getitem_1), kwargs = {})
#   %mul_37 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt), kwargs = {})
#   %mul_38 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %convert_element_type_54), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, 1024), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_38, [1], True), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_38, %mul_37), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_40, [1], True), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %sum_10), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_39, %sum_9), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_7, %mul_41), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 1024), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sub_8), kwargs = {})
#   %convert_element_type_56 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_42, torch.bfloat16), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %convert_element_type_56), kwargs = {})
triton_per_fused_add_native_layer_norm_backward_12 = async_compile.triton('triton_per_fused_add_native_layer_norm_backward_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 196608, 'r0_': 251660288}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel):
    xnumel = 24576
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
    tmp8 = tl.load(in_ptr2 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_out_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
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
    tmp19 = 0.0009765625
    tmp20 = tmp12 * tmp19
    tmp21 = 1024.0
    tmp22 = tmp4 * tmp21
    tmp23 = tmp22 - tmp7
    tmp24 = tmp13 * tmp17
    tmp25 = tmp23 - tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp18 + tmp27
    tl.store(in_out_ptr0 + (r0_1 + 1024*x0), tmp28, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_4, primals_6, primals_7, primals_8, primals_9, primals_11, primals_13, primals_15, getitem_1, rsqrt, convert_element_type_1, view_1, view_2, view_3, getitem_5, getitem_6, getitem_8, addmm_1, getitem_10, rsqrt_1, convert_element_type_13, addmm_2, convert_element_type_18, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_3, (24576, 1024), (1024, 1))
    assert_size_stride(primals_4, (3072, 1024), (1024, 1))
    assert_size_stride(primals_6, (24576, 64), (64, 1))
    assert_size_stride(primals_7, (24576, 64), (64, 1))
    assert_size_stride(primals_8, (97, ), (1, ))
    assert_size_stride(primals_9, (1024, 1024), (1024, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_13, (4096, 1024), (1024, 1))
    assert_size_stride(primals_15, (1024, 4096), (4096, 1))
    assert_size_stride(getitem_1, (24576, 1), (1, 1))
    assert_size_stride(rsqrt, (24576, 1), (1, 1))
    assert_size_stride(convert_element_type_1, (24576, 1024), (1024, 1))
    assert_size_stride(view_1, (24576, 16, 64), (1024, 64, 1))
    assert_size_stride(view_2, (24576, 16, 64), (1024, 64, 1))
    assert_size_stride(view_3, (24576, 16, 64), (3072, 64, 1))
    assert_size_stride(getitem_5, (24576, 16, 64), (1024, 64, 1))
    assert_size_stride(getitem_6, (16, 24576), (24576, 1))
    assert_size_stride(getitem_8, (2, ), (1, ))
    assert_size_stride(addmm_1, (24576, 1024), (1024, 1))
    assert_size_stride(getitem_10, (24576, 1), (1, 1))
    assert_size_stride(rsqrt_1, (24576, 1), (1, 1))
    assert_size_stride(convert_element_type_13, (24576, 1024), (1024, 1))
    assert_size_stride(addmm_2, (24576, 4096), (4096, 1))
    assert_size_stride(convert_element_type_18, (24576, 4096), (4096, 1))
    assert_size_stride(tangents_1, (24576, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24576, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(tangents_1, primals_15, out=buf0)
        del primals_15
        buf1 = empty_strided_cuda((1024, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1024, 24576), (1, 1024), 0), convert_element_type_18, out=buf1)
        del convert_element_type_18
        buf2 = empty_strided_cuda((1, 1024, 128), (131072, 1, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(tangents_1, buf2, 131072, 192, stream=stream0)
        buf3 = empty_strided_cuda((1, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf2, buf3, 1024, 128, stream=stream0)
        buf4 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu_backward, aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_gelu_backward_2.run(buf4, addmm_2, 100663296, stream=stream0)
        del addmm_2
        buf5 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, primals_13, out=buf5)
        del primals_13
        buf6 = empty_strided_cuda((4096, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (4096, 24576), (1, 4096), 0), convert_element_type_13, out=buf6)
        del convert_element_type_13
        buf7 = empty_strided_cuda((1, 4096, 28), (114688, 1, 4096), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_3.run(buf4, buf7, 114688, 878, stream=stream0)
        del buf4
        buf8 = empty_strided_cuda((1, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_4.run(buf7, buf8, 4096, 28, stream=stream0)
        del buf7
        buf17 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone, clone_2, hidden_states, layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf5, primals_11, primals_3, addmm_1, getitem_10, rsqrt_1, tangents_1, buf17, 24576, 1024, stream=stream0)
        del primals_11
        del tangents_1
        buf18 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf17, primals_9, out=buf18)
        del primals_9
        buf22 = empty_strided_cuda((24576, 16, 64), (1024, 64, 1), torch.bfloat16)
        buf23 = empty_strided_cuda((24576, 16, 64), (1024, 64, 1), torch.bfloat16)
        buf24 = empty_strided_cuda((24576, 16, 64), (1024, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf25 = torch.ops.flash_attn._flash_attn_varlen_backward.default(reinterpret_tensor(buf18, (24576, 16, 64), (1024, 64, 1), 0), view_1, view_2, view_3, getitem_5, getitem_6, buf22, buf23, buf24, primals_8, primals_8, 256, 256, 0.0, 0.125, False, -1, -1, 0.0, None, False, getitem_8, zero_tensors=False)
        del getitem_6
        del getitem_8
        del primals_8
        del view_1
        del view_2
        del view_3
        buf30 = empty_strided_cuda((73728, 16, 64), (1024, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_6.run(buf22, primals_7, primals_6, buf23, buf24, buf30, 75497472, stream=stream0)
        del primals_6
        del primals_7
        buf31 = empty_strided_cuda((24576, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_7.run(buf30, buf31, 75497472, stream=stream0)
        del buf30
        buf32 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf31, primals_4, out=buf32)
        del primals_4
        buf11 = reinterpret_tensor(buf2, (1024, 128), (1, 1024), 0); del buf2  # reuse
        buf13 = empty_strided_cuda((1024, 128), (1, 1024), torch.float32)
        buf38 = empty_strided_cuda((1024, 128), (1, 1024), torch.float32)
        buf40 = empty_strided_cuda((1024, 128), (1, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [clone, hidden_states, layer_norm_1, clone_5, clone_6], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8.run(buf5, primals_3, addmm_1, getitem_10, rsqrt_1, buf32, getitem_1, rsqrt, buf11, buf13, buf38, buf40, 131072, 192, stream=stream0)
        del addmm_1
        del buf5
        del getitem_10
        del rsqrt_1
        buf15 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone, hidden_states, layer_norm_1], Original ATen: [aten.native_layer_norm_backward, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf11, buf15, 1024, 128, stream=stream0)
        del buf11
        buf16 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf13, buf16, 1024, 128, stream=stream0)
        buf19 = empty_strided_cuda((1024, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (1024, 24576), (1, 1024), 0), reinterpret_tensor(getitem_5, (24576, 1024), (1024, 1), 0), out=buf19)
        del getitem_5
        buf20 = reinterpret_tensor(buf13, (1, 1024, 128), (131072, 1, 1024), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_0.run(buf17, buf20, 131072, 192, stream=stream0)
        buf21 = empty_strided_cuda((1, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_1.run(buf20, buf21, 1024, 128, stream=stream0)
        del buf20
        buf29 = buf25
        assert_size_stride(buf29, (16, 36864), (36864, 1), 'torch.ops.flash_attn._flash_attn_varlen_backward.default')
        assert_alignment(buf29, 16, 'torch.ops.flash_attn._flash_attn_varlen_backward.default')
        del buf22
        del buf23
        del buf24
        del buf25
        del buf29
        buf33 = empty_strided_cuda((3072, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_42], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (3072, 24576), (1, 3072), 0), convert_element_type_1, out=buf33)
        del convert_element_type_1
        buf34 = empty_strided_cuda((1, 3072, 37), (113664, 1, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_10.run(buf31, buf34, 113664, 665, stream=stream0)
        del buf31
        buf35 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_11.run(buf34, buf35, 3072, 37, stream=stream0)
        del buf34
        buf44 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [clone_5, clone_6, clone_7], Original ATen: [aten.native_layer_norm_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_backward_12.run(buf44, buf32, primals_1, primals_3, getitem_1, rsqrt, 24576, 1024, stream=stream0)
        del buf32
        del getitem_1
        del primals_1
        del primals_3
        del rsqrt
        buf42 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_5, clone_6], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf38, buf42, 1024, 128, stream=stream0)
        del buf38
        buf43 = empty_strided_cuda((1024, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clone_5], Original ATen: [aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_9.run(buf40, buf43, 1024, 128, stream=stream0)
        del buf40
    return (buf42, buf43, buf44, buf33, reinterpret_tensor(buf35, (3072, ), (1, ), 0), None, None, None, buf19, reinterpret_tensor(buf21, (1024, ), (1, ), 0), buf15, buf16, buf6, reinterpret_tensor(buf8, (4096, ), (1, ), 0), buf1, reinterpret_tensor(buf3, (1024, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((24576, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24576, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((97, ), (1, ), device='cuda:0', dtype=torch.int32)
    primals_9 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_15 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((24576, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((24576, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_1 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1 = rand_strided((24576, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_2 = rand_strided((24576, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_3 = rand_strided((24576, 16, 64), (3072, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((24576, 16, 64), (1024, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_6 = rand_strided((16, 24576), (24576, 1), device='cuda:0', dtype=torch.float32)
    getitem_8 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.int64)
    addmm_1 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_10 = rand_strided((24576, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt_1 = rand_strided((24576, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    convert_element_type_13 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    addmm_2 = rand_strided((24576, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type_18 = rand_strided((24576, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_3, primals_4, primals_6, primals_7, primals_8, primals_9, primals_11, primals_13, primals_15, getitem_1, rsqrt, convert_element_type_1, view_1, view_2, view_3, getitem_5, getitem_6, getitem_8, addmm_1, getitem_10, rsqrt_1, convert_element_type_13, addmm_2, convert_element_type_18, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
