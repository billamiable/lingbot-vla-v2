# AOT ID: ['19_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/tx/ctxxmqmwxyrti6ud7wtatistldnv6hvusnbvhipxgxzyqcdkshs7.py
# Topologically Sorted Source Nodes: [repeat], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   repeat => repeat
# Graph fragment:
#   %repeat : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_1, [32, 1, 1]), kwargs = {})
triton_poi_fused_repeat_0 = async_compile.triton('triton_poi_fused_repeat_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2662400}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 20480)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/4b/c4bcpcgeznqnuc3fj6qvvj6rkmgbg23l3235yzsqme5nrdnklpxj.py
# Topologically Sorted Source Nodes: [token_visual_masks], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   token_visual_masks => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 72], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_1 = async_compile.triton('triton_poi_fused_zeros_like_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 4608}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_like_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], False, tl.int1)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/wo/cworr6ogt4invfy7pj4h7aaibz6xgifz6t5euqv2hzkfwfo36jfz.py
# Topologically Sorted Source Nodes: [token_visual_masks_1], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   token_visual_masks_1 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 8], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_2 = async_compile.triton('triton_poi_fused_zeros_like_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_like_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], False, tl.int1)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ib/cib2oajf5vj4l3c6zubfvf2lscfowmed5rnhlez6luap3tregudu.py
# Topologically Sorted Source Nodes: [embs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   embs => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_2, %primals_4, %primals_6, %repeat], 1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_poi_fused_cat_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 281149440}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23429120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 2560) % 286)
    x0 = (xindex % 2560)
    x2 = xindex // 732160
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 198, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 2560*(x1) + 506880*x2), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 270, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 2560*((-198) + x1) + 184320*x2), tmp9, other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 278, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 2560*((-270) + x1) + 20480*x2), tmp14, other=0.0).to(tl.float32)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 286, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 2560*((-278) + x1) + 20480*x2), tmp16, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/b7/cb7ojbjyuasm4z5bfwdzw47a3sfyob4wr4hvfybk76743xdgnwj7.py
# Topologically Sorted Source Nodes: [pad_masks], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   pad_masks => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_8, %primals_5, %primals_7, %primals_7], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 54912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 286)
    x1 = xindex // 286
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 198, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (198*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 270, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (72*x1 + ((-198) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 278, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (8*x1 + ((-270) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 286, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr2 + (8*x1 + ((-278) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/o5/co56phtxritz3hnp27rwcshtrycyibd6ixrti4vhxqwf3yvyolea.py
# Topologically Sorted Source Nodes: [prefix_input_ids], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   prefix_input_ids => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_10, %primals_3, %primals_9, %primals_9], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 439296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 286)
    x1 = xindex // 286
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 198, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (198*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 270, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (72*x1 + ((-198) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 278, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (8*x1 + ((-270) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 286, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr2 + (8*x1 + ((-278) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/if/ciffblqirsmjdziixzwyxzs2ox2jx5jzaoxlnoj6jr2vmqlo6n3e.py
# Topologically Sorted Source Nodes: [full_visual_pos_masks], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   full_visual_pos_masks => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_11, %full_default, %full_default_1, %full_default_2], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 27456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 286)
    x1 = xindex // 286
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 198, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (198*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 270, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.full([1], False, tl.int1)
    tmp11 = tl.full(tmp10.shape, False, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp0 >= tmp7
    tmp14 = tl.full([1], 278, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = tl.full([1], False, tl.int1)
    tmp18 = tl.full(tmp17.shape, False, tmp17.dtype)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tmp0 >= tmp14
    tmp21 = tl.full([1], 286, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tl.full([1], False, tl.int1)
    tmp24 = tl.full(tmp23.shape, False, tmp23.dtype)
    tmp25 = tl.where(tmp20, tmp23, tmp24)
    tmp26 = tl.where(tmp16, tmp19, tmp25)
    tmp27 = tl.where(tmp9, tmp12, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/4j/c4jzcuk6yeknodneimuq6zk6gub77j5giiblku3z6jeqbbrkvk5a.py
# Topologically Sorted Source Nodes: [att_masks], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   att_masks => full_default_3
# Graph fragment:
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 286], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_7 = async_compile.triton('triton_poi_fused_ones_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18304}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], True, tl.int1)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (8, 2560), (2560, 1))
    assert_size_stride(primals_2, (32, 198, 2560), (506880, 2560, 1))
    assert_size_stride(primals_3, (32, 72), (72, 1))
    assert_size_stride(primals_4, (32, 72, 2560), (184320, 2560, 1))
    assert_size_stride(primals_5, (32, 72), (72, 1))
    assert_size_stride(primals_6, (32, 8, 2560), (20480, 2560, 1))
    assert_size_stride(primals_7, (32, 8), (8, 1))
    assert_size_stride(primals_8, (32, 198), (198, 1))
    assert_size_stride(primals_9, (32, 8), (8, 1))
    assert_size_stride(primals_10, (32, 198), (198, 1))
    assert_size_stride(primals_11, (32, 198), (198, 1))
    assert_size_stride(primals_12, (32, 3), (3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 8, 2560), (20480, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [repeat], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_0.run(primals_1, buf0, 655360, stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((32, 72), (72, 1), torch.bool)
        # Topologically Sorted Source Nodes: [token_visual_masks], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_1.run(buf1, 2304, stream=stream0)
        buf2 = empty_strided_cuda((32, 8), (8, 1), torch.bool)
        # Topologically Sorted Source Nodes: [token_visual_masks_1], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_2.run(buf2, 256, stream=stream0)
        buf3 = empty_strided_cuda((32, 8), (8, 1), torch.bool)
        # Topologically Sorted Source Nodes: [token_visual_masks_2], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_2.run(buf3, 256, stream=stream0)
        buf4 = empty_strided_cuda((32, 286, 2560), (732160, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [embs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_3.run(primals_2, primals_4, primals_6, buf0, buf4, 23429120, stream=stream0)
        del primals_2
        del primals_4
        del primals_6
        buf5 = empty_strided_cuda((32, 286), (286, 1), torch.bool)
        # Topologically Sorted Source Nodes: [pad_masks], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(primals_8, primals_5, primals_7, buf5, 9152, stream=stream0)
        del primals_5
        del primals_7
        del primals_8
        buf6 = empty_strided_cuda((32, 286), (286, 1), torch.int64)
        # Topologically Sorted Source Nodes: [prefix_input_ids], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(primals_10, primals_3, primals_9, buf6, 9152, stream=stream0)
        del primals_10
        del primals_9
        buf7 = empty_strided_cuda((32, 286), (286, 1), torch.bool)
        # Topologically Sorted Source Nodes: [full_visual_pos_masks], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(primals_11, buf7, 9152, stream=stream0)
        del primals_11
        buf8 = empty_strided_cuda((32, 286), (286, 1), torch.bool)
        # Topologically Sorted Source Nodes: [att_masks], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_7.run(buf8, 9152, stream=stream0)
    return (reinterpret_tensor(primals_12, (96, ), (1, ), 0), buf4, buf5, buf6, buf7, buf8, buf0, primals_3, buf1, buf2, buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((32, 198, 2560), (506880, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((32, 72), (72, 1), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((32, 72, 2560), (184320, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((32, 72), (72, 1), device='cuda:0', dtype=torch.bool)
    primals_6 = rand_strided((32, 8, 2560), (20480, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.bool)
    primals_8 = rand_strided((32, 198), (198, 1), device='cuda:0', dtype=torch.bool)
    primals_9 = rand_strided((32, 8), (8, 1), device='cuda:0', dtype=torch.int64)
    primals_10 = rand_strided((32, 198), (198, 1), device='cuda:0', dtype=torch.int64)
    primals_11 = rand_strided((32, 198), (198, 1), device='cuda:0', dtype=torch.bool)
    primals_12 = rand_strided((32, 3), (3, 1), device='cuda:0', dtype=torch.bool)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
