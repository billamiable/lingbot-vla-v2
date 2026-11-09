# AOT ID: ['22_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/ai/caibvq3dgakneykabuthecvby2scxbjbbq274mrzcaowq67ccpcj.py
# Topologically Sorted Source Nodes: [invert, valid_prefix_pos, amax], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.amax]
# Source node to ATen node mapping:
#   amax => amax
#   invert => bitwise_not
#   valid_prefix_pos => full_default_2, where_1
# Graph fragment:
#   %bitwise_not : [num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%unsqueeze_8,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not, %full_default_2, %primals_3), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_1, [0, 2]), kwargs = {})
triton_per_fused_amax_bitwise_not_masked_fill_0 = async_compile.triton('triton_per_fused_amax_bitwise_not_masked_fill_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_amax_bitwise_not_masked_fill_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512, 'r0_': 228800}}
)
@triton.jit
def triton_per_fused_amax_bitwise_not_masked_fill_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    r0_numel = 858
    R0_BLOCK: tl.constexpr = 1024
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
    r0_1 = (r0_index % 286)
    x0 = xindex
    r0_2 = r0_index // 286
    tmp0 = tl.load(in_ptr0 + (r0_1 + 286*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 286*x0 + 9152*r0_2), r0_mask, other=0.0)
    tmp1 = tmp0 == 0
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp1, tmp3, tmp2)
    tmp5 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, -9223372036854775808)
    tmp8 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp7, 0))
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/sr/csrhntoaismcrx4vvfynchry3ryoebhxwq2y5lpmuzlm54dnlxhn.py
# Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   position_ids => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %expand_1], -1), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 78336}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 51) % 32)
    x0 = (xindex % 51)
    x3 = xindex // 51
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 + tmp1
    tmp3 = 1 + x0
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 - tmp1
    tmp6 = tl.full([1], False, tl.int1)
    tmp7 = tl.where(tmp6, tmp1, tmp5)
    tl.store(out_ptr0 + (x0 + 337*x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/lz/clznyoxhexnjvupp6lxjgkqd5vkk77tsdigiwislbnwtfl7eae7o.py
# Topologically Sorted Source Nodes: [pos_emb, time_emb], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   pos_emb => cat
#   time_emb => convert_element_type_5
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%sin, %cos], 1), kwargs = {})
#   %convert_element_type_5 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_2 = async_compile.triton('triton_poi_fused__to_copy_cat_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 98304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_cat_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = xindex // 768
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 192.0
    tmp8 = tmp6 < tmp7
    tmp9 = 0.0026109660574412533
    tmp10 = tmp6 * tmp9
    tmp11 = 0.0
    tmp12 = tmp10 + tmp11
    tmp13 = 383 + (-1)*(x0)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp9
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = 1000.0
    tmp20 = libdevice.pow(tmp19, tmp18)
    tmp21 = 0.004
    tmp22 = tmp20 * tmp21
    tmp23 = tl.full([1], 1, tl.int32)
    tmp24 = (tmp23 / tmp22)
    tmp25 = tmp24 * tmp16
    tmp26 = 2.0
    tmp27 = tmp25 * tmp26
    tmp28 = 3.141592653589793
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr0 + (x1), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp29 * tmp31
    tmp33 = tl_math.sin(tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp4, tmp33, tmp34)
    tmp36 = tmp0 >= tmp3
    tmp37 = tl.full([1], 768, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = (-384) + x0
    tmp40 = tmp39.to(tl.float32)
    tmp41 = 192.0
    tmp42 = tmp40 < tmp41
    tmp43 = 0.0026109660574412533
    tmp44 = tmp40 * tmp43
    tmp45 = 0.0
    tmp46 = tmp44 + tmp45
    tmp47 = 383 + (-1)*((-384) + x0)
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp43
    tmp50 = 1.0
    tmp51 = tmp50 - tmp49
    tmp52 = tl.where(tmp42, tmp46, tmp51)
    tmp53 = 1000.0
    tmp54 = libdevice.pow(tmp53, tmp52)
    tmp55 = 0.004
    tmp56 = tmp54 * tmp55
    tmp57 = tl.full([1], 1, tl.int32)
    tmp58 = (tmp57 / tmp56)
    tmp59 = tmp58 * tmp50
    tmp60 = 2.0
    tmp61 = tmp59 * tmp60
    tmp62 = 3.141592653589793
    tmp63 = tmp61 * tmp62
    tmp64 = tl.load(in_ptr0 + (x1), tmp36, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp63 * tmp65
    tmp67 = tl_math.cos(tmp66)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp36, tmp67, tmp68)
    tmp70 = tl.where(tmp4, tmp35, tmp69)
    tmp71 = tmp70.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp71, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/qh/cqh3bsd4vyogsj2j6pplkuvydicvvjjd2xrdvshs4zdpcdl4sghx.py
# Topologically Sorted Source Nodes: [att_masks_1, cumsum], Original ATen: [aten.cat, aten.cumsum]
# Source node to ATen node mapping:
#   att_masks_1 => cat_4
#   cumsum => cumsum
# Graph fragment:
#   %cat_4 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_2, %slice_scatter_default], 1), kwargs = {})
#   %cumsum : [num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%cat_4, 1), kwargs = {})
triton_per_fused_cat_cumsum_3 = async_compile.triton('triton_per_fused_cat_cumsum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_cumsum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 183328}}
)
@triton.jit
def triton_per_fused_cat_cumsum_3(in_ptr0, out_ptr0, xnumel, r0_numel):
    xnumel = 32
    XBLOCK: tl.constexpr = 1
    r0_numel = 337
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
    tmp0 = r0_1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 286, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (286*x0 + (r0_1)), r0_mask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 337, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.broadcast_to((-286) + r0_1, [R0_BLOCK])
    tmp10 = tl.full([1], 2, tl.int64)
    tmp11 = tmp9 < tmp10
    tmp12 = tmp11 & tmp6
    tmp13 = tl.full([1], True, tl.int1)
    tmp14 = tl.full(tmp13.shape, False, tmp13.dtype)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.full([1], False, tl.int1)
    tmp17 = tl.where(tmp11, tmp15, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp5, tmp19)
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21.to(tl.int64)
    tmp23 = tl.broadcast_to(tmp22, [R0_BLOCK])
    tmp24, = tl.associative_scan((tmp23,), 0, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r0_1 + 337*x0), tmp24, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/zm/czmfde7yzznltkwfy2i67tqs4qvdcyaz7m2t43tnv2utmqd3ufuw.py
# Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   position_ids => cat_5
# Graph fragment:
#   %cat_5 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %expand_1], -1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 658944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 286)
    x1 = xindex // 286
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 337*x1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/6v/c6vehzyw4ha6d44azdirr4cvtl3brictboxp266ql7s76phzokso.py
# Topologically Sorted Source Nodes: [action_time_emb], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   action_time_emb => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %expand], -1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 14843904}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2457600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1536)
    x3 = xindex // 1536
    x2 = xindex // 76800
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (768*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1536, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (768*x2 + ((-768) + x0)), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/jp/cjp5qhauabjljk23fnrmn4ftm27npf47k7ej3jc67ixq46nhuxnw.py
# Topologically Sorted Source Nodes: [action_time_emb_2], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   action_time_emb_2 => convert_element_type_12, convert_element_type_13, mul_7, sigmoid
# Graph fragment:
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_12,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %sigmoid), kwargs = {})
#   %convert_element_type_13 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
triton_poi_fused_silu_6 = async_compile.triton('triton_poi_fused_silu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7372800}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_silu_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1228800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/lm/clmj3wlkvxp52d4dyhui2jkggebwnq6ye4qkeuy3kestk44hwuit.py
# Topologically Sorted Source Nodes: [embs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   embs => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_3, %view_5], 1), kwargs = {})
triton_poi_fused_cat_7 = async_compile.triton('triton_poi_fused_cat_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7569408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1253376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 768) % 51)
    x0 = (xindex % 768)
    x2 = xindex // 39168
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 768*x2), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 51, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 768*((-1) + x1) + 38400*x2), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/on/conmqimru6cb73vmjyv2e74eh5bg3y6hjmogotphf6wls3xyfc6j.py
# Topologically Sorted Source Nodes: [att_2d_masks, pad_2d_masks, att_2d_masks_1], Original ATen: [aten.le, aten.mul, aten.bitwise_and]
# Source node to ATen node mapping:
#   att_2d_masks => le
#   att_2d_masks_1 => bitwise_and
#   pad_2d_masks => mul_8
# Graph fragment:
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_4, %unsqueeze_5), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_6, %unsqueeze_7), kwargs = {})
#   %bitwise_and : [num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%le, %mul_8), kwargs = {})
triton_poi_fused_bitwise_and_le_mul_8 = async_compile.triton('triton_poi_fused_bitwise_and_le_mul_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_le_mul_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 7365472}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_and_le_mul_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3634208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 337)
    x2 = xindex // 113569
    x3 = xindex // 337
    x1 = ((xindex // 337) % 337)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 337*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 <= tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 286, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tl.load(in_ptr1 + (286*x2 + (x0)), tmp7 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = tmp3 >= tmp6
    tmp10 = tl.full([1], 337, tl.int64)
    tmp11 = tmp3 < tmp10
    tmp12 = tl.full([1], True, tl.int1)
    tmp13 = tl.full(tmp12.shape, False, tmp12.dtype)
    tmp14 = tl.where(tmp9, tmp12, tmp13)
    tmp15 = tl.where(tmp7, tmp8, tmp14)
    tmp16 = x1
    tmp17 = tmp16 >= tmp4
    tmp18 = tmp16 < tmp6
    tmp19 = tl.load(in_ptr1 + (286*x2 + (x1)), tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp20 = tmp16 >= tmp6
    tmp21 = tmp16 < tmp10
    tmp22 = tl.full([1], True, tl.int1)
    tmp23 = tl.full(tmp22.shape, False, tmp22.dtype)
    tmp24 = tl.where(tmp20, tmp22, tmp23)
    tmp25 = tl.where(tmp18, tmp19, tmp24)
    tmp26 = tmp15 & tmp25
    tmp27 = tmp2 & tmp26
    tl.store(out_ptr0 + (x4), tmp27, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14 = args
    args.clear()
    assert_size_stride(primals_1, (32, 286), (286, 1))
    assert_size_stride(primals_2, (32, 286), (286, 1))
    assert_size_stride(primals_3, (3, 32, 286), (9152, 286, 1))
    assert_size_stride(primals_4, (32, 55), (55, 1))
    assert_size_stride(primals_5, (768, 55), (55, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (768, 55), (55, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (32, 50, 55), (2750, 55, 1))
    assert_size_stride(primals_11, (768, 1536), (1536, 1))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, 768), (768, 1))
    assert_size_stride(primals_14, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf10 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [invert, valid_prefix_pos, amax], Original ATen: [aten.bitwise_not, aten.masked_fill, aten.amax]
        stream0 = get_raw_stream(0)
        triton_per_fused_amax_bitwise_not_masked_fill_0.run(primals_1, primals_3, buf10, 32, 858, stream=stream0)
        buf13 = empty_strided_cuda((3, 32, 337), (10784, 337, 1), torch.int64)
        buf12 = reinterpret_tensor(buf13, (3, 32, 51), (10784, 337, 1), 286)  # alias
        # Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf10, buf12, 4896, stream=stream0)
        del buf10
        buf0 = empty_strided_cuda((32, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [state_emb], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, primals_4, reinterpret_tensor(primals_5, (55, 768), (1, 55), 0), alpha=1, beta=1, out=buf0)
        del primals_5
        del primals_6
        buf1 = empty_strided_cuda((32, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [pos_emb, time_emb], Original ATen: [aten.cat, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_cat_2.run(primals_7, buf1, 24576, stream=stream0)
        del primals_7
        buf8 = empty_strided_cuda((32, 337), (337, 1), torch.int64)
        # Topologically Sorted Source Nodes: [att_masks_1, cumsum], Original ATen: [aten.cat, aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_cumsum_3.run(primals_2, buf8, 32, 337, stream=stream0)
        del primals_2
        buf11 = reinterpret_tensor(buf13, (3, 32, 286), (10784, 337, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [position_ids], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(primals_3, buf11, 27456, stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((1600, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [action_emb], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(primals_10, (1600, 55), (55, 1), 0), reinterpret_tensor(primals_8, (55, 768), (1, 55), 0), alpha=1, beta=1, out=buf2)
        del primals_8
        del primals_9
        buf3 = empty_strided_cuda((32, 50, 1536), (76800, 1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [action_time_emb], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(buf2, buf1, buf3, 2457600, stream=stream0)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [action_time_emb_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf3, (1600, 1536), (1536, 1), 0), reinterpret_tensor(primals_11, (1536, 768), (1, 1536), 0), alpha=1, beta=1, out=buf4)
        del primals_12
        buf5 = empty_strided_cuda((32, 50, 768), (38400, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [action_time_emb_2], Original ATen: [aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_silu_6.run(buf4, buf5, 1228800, stream=stream0)
        buf6 = empty_strided_cuda((1600, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [action_time_emb_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, reinterpret_tensor(buf5, (1600, 768), (768, 1), 0), reinterpret_tensor(primals_13, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf6)
        del primals_14
        buf7 = empty_strided_cuda((32, 51, 768), (39168, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [embs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_7.run(buf0, buf6, buf7, 1253376, stream=stream0)
        del buf0
        del buf6
        buf9 = empty_strided_cuda((32, 337, 337), (113569, 337, 1), torch.bool)
        # Topologically Sorted Source Nodes: [att_2d_masks, pad_2d_masks, att_2d_masks_1], Original ATen: [aten.le, aten.mul, aten.bitwise_and]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bitwise_and_le_mul_8.run(buf8, primals_1, buf9, 3634208, stream=stream0)
        del buf8
        del primals_1
    return (buf9, buf13, buf7, buf1, primals_4, primals_11, primals_13, reinterpret_tensor(primals_10, (1600, 55), (55, 1), 0), reinterpret_tensor(buf3, (1600, 1536), (1536, 1), 0), buf4, reinterpret_tensor(buf5, (1600, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 286), (286, 1), device='cuda:0', dtype=torch.bool)
    primals_2 = rand_strided((32, 286), (286, 1), device='cuda:0', dtype=torch.bool)
    primals_3 = rand_strided((3, 32, 286), (9152, 286, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((32, 55), (55, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((768, 55), (55, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((768, 55), (55, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((32, 50, 55), (2750, 55, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((768, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
