# AOT ID: ['23_inference']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/55/c55vudyl5twa5q3iyqwvprdhijxakcwrmuao27sb7fvegdc4tyun.py
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 517632}},
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/z3/cz3idqnrwjac6jlzcqiqgvf7ua5i3w25zit7padllgpw2hfyi475.py
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 13803520}},
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/4w/c4wq6fikriotcdnes2y5scm7jl7s7e6q7u6xwijzy3vqod7t7ww3.py
# Topologically Sorted Source Nodes: [cos, cos_1, sin, sin_1], Original ATen: [aten.cos, aten.mul, aten.sin]
# Source node to ATen node mapping:
#   cos => cos
#   cos_1 => mul
#   sin => sin
#   sin_1 => mul_1
# Graph fragment:
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_12,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_12,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
triton_poi_fused_cos_mul_sin_2 = async_compile.triton('triton_poi_fused_cos_mul_sin_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cos_mul_sin_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 27607040, 'x': 11042816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cos_mul_sin_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 337
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp2 = tl.load(in_ptr0 + (64*x2 + 21568*y1 + ((y0 % 64))), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (x2 + 337*((y0 % 64)) + 21568*y1), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1, 1], 0, tl.int32)
    tmp1 = tmp0 == tmp0
    tmp3 = (y3 % 64)
    tmp4 = tl.full([1, 1], 1, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1, 1], 60, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = (((-1) + ((y0 % 64))) % 3)
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tmp8 == tmp9
    tmp11 = tmp5 & tmp7
    tmp12 = tmp11 & tmp10
    tmp13 = tl.load(in_ptr1 + (690513 + x2 + 1011*(triton_helpers.div_floor_integer((-1) + ((y0 % 64)),  3)) + 21568*y1), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.where(tmp1, tmp15, tmp14)
    tmp17 = tl.where(tmp1, tmp2, tmp16)
    tmp18 = tl_math.cos(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tl_math.sin(tmp17)
    tmp22 = tmp21 * tmp19
    tl.store(out_ptr0 + (y0 + 128*x2 + 43136*y1), tmp20, xmask)
    tl.store(out_ptr1 + (y0 + 128*x2 + 43136*y1), tmp22, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (3, 32, 337), (10784, 337, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3, 32, 1, 337), (10784, 337, 337, 1), torch.float32)
        # Topologically Sorted Source Nodes: [position_ids_expanded], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 32352, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((96, 64, 337), (21568, 337, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg1_1, (96, 64, 1), (0, 1, 0), 0), reinterpret_tensor(buf0, (96, 1, 337), (337, 0, 1), 0), out=buf1)
        del arg1_1
        del buf0
        buf2 = empty_strided_cuda((32, 337, 64), (21568, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(buf1, buf2, 2048, 337, stream=stream0)
        buf3 = empty_strided_cuda((32, 337, 128), (43136, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 337, 128), (43136, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cos, cos_1, sin, sin_1], Original ATen: [aten.cos, aten.mul, aten.sin]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cos_mul_sin_2.run(buf2, buf1, buf3, buf4, 4096, 337, stream=stream0)
        del buf1
        del buf2
    return (buf3, buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 32, 337), (10784, 337, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
