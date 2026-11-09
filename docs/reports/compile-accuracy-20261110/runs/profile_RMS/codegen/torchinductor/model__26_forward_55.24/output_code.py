# AOT ID: ['26_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/o5/co5jm3z7t4ruk7po7rm7ocs6ul4ofjqqrdonawzbacfiza4rr37k.py
# Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   position_ids_expanded => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 517632}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/25/c25vpv7iiya2qocivdpiv7m27g3pno5lutwnfvqppea6lb5kq6nk.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => slice_scatter_default_1
# Graph fragment:
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%select_int_1, %slice_11, 2, 2, 60, 3), kwargs = {})
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 13803520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 337
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = (yindex % 64)
    x2 = xindex
    y1 = yindex // 64
    y3 = yindex
    tmp39 = tl.load(in_ptr0 + (x2 + 337*y3), xmask, eviction_policy='evict_last')
    tmp0 = y0
    tmp1 = tl.full([1, 1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 60, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (((-2) + y0) % 3)
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp7
    tmp10 = tl.full([1, 1], 2, tl.int32)
    tmp11 = tl.full([1, 1], 0, tl.int32)
    tmp12 = tmp10 == tmp11
    tmp13 = tl.broadcast_to(2 + 3*(triton_helpers.div_floor_integer((-2) + y0,  3)), [YBLOCK, XBLOCK])
    tmp14 = tl.full([1, 1], 1, tl.int64)
    tmp15 = tmp13 >= tmp14
    tmp16 = tl.full([1, 1], 60, tl.int64)
    tmp17 = tmp13 < tmp16
    tmp18 = tl.full([1, 1], 0, tl.int64)
    tmp19 = tmp14 == tmp18
    tmp20 = tmp15 & tmp17
    tmp21 = tmp20 & tmp19
    tmp22 = tmp21 & tmp9
    tmp23 = tl.load(in_ptr0 + (690513 + x2 + 1011*(triton_helpers.div_floor_integer((-2) + y0,  3)) + 21568*y1), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.load(in_ptr0 + (674 + x2 + 1011*(triton_helpers.div_floor_integer((-2) + y0,  3)) + 21568*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp21, tmp23, tmp24)
    tmp26 = tl.load(in_ptr0 + (1381026 + x2 + 1011*(triton_helpers.div_floor_integer((-2) + y0,  3)) + 21568*y1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.where(tmp12, tmp25, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp9, tmp27, tmp28)
    tmp30 = tl.full([1, 1], 0, tl.int32)
    tmp31 = tmp30 == tmp30
    tmp32 = tl.full([1, 1], 1, tl.int64)
    tmp33 = tmp0 >= tmp32
    tmp34 = (((-1) + y0) % 3)
    tmp35 = tmp34 == tmp6
    tmp36 = tmp33 & tmp4
    tmp37 = tmp36 & tmp35
    tmp38 = tl.load(in_ptr0 + (690513 + x2 + 1011*(triton_helpers.div_floor_integer((-1) + y0,  3)) + 21568*y1), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp31, tmp40, tmp39)
    tmp42 = tl.where(tmp9, tmp29, tmp41)
    tl.store(out_ptr0 + (y0 + 64*x2 + 21568*y1), tmp42, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/fw/cfw4g64qcol743dhvcsi5dpchwlj7pzgbfaak52czookbq2zu4xy.py
# Topologically Sorted Source Nodes: [_generalized_scatter_3], Original ATen: []
# Source node to ATen node mapping:
#   _generalized_scatter_3 => select_scatter_default_1
# Graph fragment:
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_5, %slice_scatter_default_1, 0, 0), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 19324928, 'x': 13803520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 337
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y2 = yindex // 2048
    x3 = xindex
    y0 = (yindex % 64)
    y1 = ((yindex // 64) % 32)
    y5 = (yindex % 2048)
    y6 = yindex
    y4 = yindex // 64
    tmp3 = tl.load(in_ptr0 + (y0 + 64*x3 + 21568*y1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (x3 + 337*y5), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3 + 337*y6), xmask, eviction_policy='evict_last')
    tmp0 = y2
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = y0
    tmp5 = tl.full([1, 1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1, 1], 60, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = (((-1) + y0) % 3)
    tmp10 = tl.full([1, 1], 0, tl.int64)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp6 & tmp8
    tmp13 = tmp12 & tmp11
    tmp14 = tl.load(in_ptr1 + (690513 + x3 + 1011*(triton_helpers.div_floor_integer((-1) + y0,  3)) + 21568*y1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp18 = tl.where(tmp2, tmp16, tmp17)
    tmp19 = tl.where(tmp2, tmp3, tmp18)
    tl.store(out_ptr0 + (y0 + 64*x3 + 21568*y4), tmp19, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/jn/cjnykqs7vkst4c2mnoacnokdqjgbc6y5chbc27rhbyoc3zysyajr.py
# Topologically Sorted Source Nodes: [emb], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   emb => unsqueeze_4
# Graph fragment:
#   %unsqueeze_4 : [num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%select_9, 2), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 2760704}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10784
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 337)
    y1 = yindex // 337
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 337*x2 + 21568*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/2u/c2up52j7w6fy2qniqy3k2umczk6eetfihmvwy73t3xogj4wf47fq.py
# Topologically Sorted Source Nodes: [mul_2, cat_1, mul_3, q_embed], Original ATen: [aten.mul, aten.cat, aten.add]
# Source node to ATen node mapping:
#   cat_1 => cat
#   mul_2 => mul_2
#   mul_3 => mul_3
#   q_embed => add
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, %unsqueeze_5), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_13], -1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_6), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
triton_poi_fused_add_cat_mul_4 = async_compile.triton('triton_poi_fused_add_cat_mul_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 883425280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10784
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x5 = xindex
    y4 = yindex
    x2 = (xindex % 128)
    y0 = (yindex % 337)
    y1 = yindex // 337
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x5 + 4096*y4), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), ymask, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = x2
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1, 1], 64, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr0 + (64 + 128*x3 + 4096*y4 + (x2)), tmp10 & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = -tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tmp6 >= tmp9
    tmp16 = tl.full([1, 1], 128, tl.int64)
    tmp17 = tmp6 < tmp16
    tmp18 = tl.load(in_ptr0 + (128*x3 + 4096*y4 + ((-64) + x2)), tmp15 & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp10, tmp14, tmp18)
    tmp20 = tl_math.sin(tmp1)
    tmp21 = tmp20 * tmp3
    tmp22 = tmp19 * tmp21
    tmp23 = tmp5 + tmp22
    tl.store(out_ptr0 + (x5 + 4096*y4), tmp23, ymask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/mz/cmzeqvvqlj52kwyyjto7p3eonmejjoek6vic3ep6t55vwbikovhm.py
# Topologically Sorted Source Nodes: [mul_4, cat_2, mul_5, k_embed], Original ATen: [aten.mul, aten.cat, aten.add]
# Source node to ATen node mapping:
#   cat_2 => cat_1
#   k_embed => add_1
#   mul_4 => mul_4
#   mul_5 => mul_5
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %unsqueeze_5), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_15], -1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_6), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
triton_poi_fused_add_cat_mul_5 = async_compile.triton('triton_poi_fused_add_cat_mul_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 1024}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 220856320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_mul_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10784
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    x2 = (xindex % 128)
    y0 = (yindex % 337)
    y1 = yindex // 337
    x3 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x5 + 1024*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl_math.cos(tmp1)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 * tmp4
    tmp6 = x2
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1, 1], 64, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr0 + (64 + 128*x3 + 1024*y4 + (x2)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = -tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tmp6 >= tmp9
    tmp16 = tl.full([1, 1], 128, tl.int64)
    tmp17 = tmp6 < tmp16
    tmp18 = tl.load(in_ptr0 + (128*x3 + 1024*y4 + ((-64) + x2)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp10, tmp14, tmp18)
    tmp20 = tl_math.sin(tmp1)
    tmp21 = tmp20 * tmp3
    tmp22 = tmp19 * tmp21
    tmp23 = tmp5 + tmp22
    tl.store(out_ptr0 + (x5 + 1024*y4), tmp23, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (3, 32, 337), (10784, 337, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (32, 337, 32, 128), (1380352, 4096, 128, 1))
    assert_size_stride(primals_4, (32, 337, 8, 128), (345088, 1024, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 32, 1, 337), (10784, 337, 337, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_1, buf0, 32352, stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((96, 64, 337), (21568, 337, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_2, (96, 64, 1), (0, 1, 0), 0), reinterpret_tensor(buf0, (96, 1, 337), (337, 0, 1), 0), out=buf1)
        del buf0
        del primals_2
        buf2 = empty_strided_cuda((32, 337, 64), (21568, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf1, buf2, 2048, 337, stream=stream0)
        buf3 = empty_strided_cuda((3, 32, 337, 64), (690176, 21568, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_generalized_scatter_3], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(buf2, buf1, buf3, 6144, 337, stream=stream0)
        del buf1
        buf4 = reinterpret_tensor(buf2, (32, 337, 1, 64), (21568, 1, 21568, 337), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [emb], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(buf3, buf4, 10784, 64, stream=stream0)
        del buf3
        buf5 = empty_strided_cuda((32, 337, 32, 128), (1380352, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_2, cat_1, mul_3, q_embed], Original ATen: [aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_mul_4.run(primals_3, buf4, buf5, 10784, 4096, stream=stream0)
        del primals_3
        buf6 = empty_strided_cuda((32, 337, 8, 128), (345088, 1024, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_4, cat_2, mul_5, k_embed], Original ATen: [aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_mul_5.run(primals_4, buf4, buf6, 10784, 1024, stream=stream0)
        del primals_4
    return (buf5, buf6, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3, 32, 337), (10784, 337, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 337, 32, 128), (1380352, 4096, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 337, 8, 128), (345088, 1024, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
