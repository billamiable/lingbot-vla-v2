# AOT ID: ['25_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/ga/cgaso6373dk5km5uajbbe2qxmxjrr7ibo6cmir24qgngchbaseir.py
# Topologically Sorted Source Nodes: [full, _generalized_scatter, _generalized_scatter_1], Original ATen: [aten.neg, aten.slice_backward, aten.add, aten.mul]
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
#   _generalized_scatter_1 => slice_scatter_default_1
#   full => full_default
# Graph fragment:
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_17,), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 337, 8, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %neg_2, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default, %slice_18, 3, 0, 64), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default, %slice_scatter_default_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_2, %unsqueeze_5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_7), kwargs = {})
triton_poi_fused_add_mul_neg_slice_backward_0 = async_compile.triton('triton_poi_fused_add_mul_neg_slice_backward_0', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_neg_slice_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 220856320}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_neg_slice_backward_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10784
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 128)
    x5 = xindex
    y4 = yindex
    y0 = (yindex % 337)
    y1 = yindex // 337
    tmp25 = tl.load(in_ptr0 + (x5 + 1024*y4), xmask & ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), xmask & ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x5 + 1024*y4), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 * tmp7
    tmp9 = -tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (64 + x5 + 1024*y4), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl_math.sin(tmp16)
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 * tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp14, tmp20, tmp21)
    tmp23 = tl.where(tmp14, tmp22, tmp12)
    tmp24 = tmp13 + tmp23
    tmp27 = tl_math.cos(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tmp31 = tmp24 + tmp30
    tl.store(out_ptr0 + (x5 + 1024*y4), tmp31, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/xi/cxiqtt2nsjnrrqalgpndqrjujrcj2isphrkfzo7aaqe7rcqfwxoq.py
# Topologically Sorted Source Nodes: [full_2, _generalized_scatter_2, _generalized_scatter_3], Original ATen: [aten.neg, aten.slice_backward, aten.add, aten.mul]
# Source node to ATen node mapping:
#   _generalized_scatter_2 => slice_scatter_default_2
#   _generalized_scatter_3 => slice_scatter_default_3
#   full_2 => full_default_2
# Graph fragment:
#   %neg_3 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_19,), kwargs = {})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 337, 32, 128], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_2, %neg_3, 3, 64, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_2, %slice_20, 3, 0, 64), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_2, %slice_scatter_default_3), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %unsqueeze_5), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_9), kwargs = {})
triton_poi_fused_add_mul_neg_slice_backward_1 = async_compile.triton('triton_poi_fused_add_mul_neg_slice_backward_1', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_neg_slice_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 5521408, 'x': 883425280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_neg_slice_backward_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10784
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = tl.full([YBLOCK, XBLOCK], True, tl.int1)
    x2 = (xindex % 128)
    x5 = xindex
    y4 = yindex
    y0 = (yindex % 337)
    y1 = yindex // 337
    tmp25 = tl.load(in_ptr0 + (x5 + 4096*y4), ymask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 64, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-64) + x5 + 4096*y4), tmp2 & ymask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), tmp2 & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp3 * tmp7
    tmp9 = -tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = 0.0
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp14 = tmp0 < tmp1
    tmp15 = tl.load(in_ptr0 + (64 + x5 + 4096*y4), tmp14 & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (y0 + 337*((x2 % 64)) + 21568*y1), tmp14 & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl_math.sin(tmp16)
    tmp18 = 1.0
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 * tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp14, tmp20, tmp21)
    tmp23 = tl.where(tmp14, tmp22, tmp12)
    tmp24 = tmp13 + tmp23
    tmp27 = tl_math.cos(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp25 * tmp29
    tmp31 = tmp24 + tmp30
    tl.store(out_ptr0 + (x5 + 4096*y4), tmp31, ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    unsqueeze_4, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(unsqueeze_4, (32, 337, 1, 64), (21568, 1, 21568, 337))
    assert_size_stride(tangents_1, (32, 337, 32, 128), (1380352, 4096, 128, 1))
    assert_size_stride(tangents_2, (32, 337, 8, 128), (345088, 1024, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 337, 8, 128), (345088, 1024, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full, _generalized_scatter, _generalized_scatter_1], Original ATen: [aten.neg, aten.slice_backward, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_neg_slice_backward_0.run(tangents_2, unsqueeze_4, buf0, 10784, 1024, stream=stream0)
        del tangents_2
        buf1 = empty_strided_cuda((32, 337, 32, 128), (1380352, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_2, _generalized_scatter_2, _generalized_scatter_3], Original ATen: [aten.neg, aten.slice_backward, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_neg_slice_backward_1.run(tangents_1, unsqueeze_4, buf1, 10784, 4096, stream=stream0)
        del tangents_1
        del unsqueeze_4
    return (None, None, buf1, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    unsqueeze_4 = rand_strided((32, 337, 1, 64), (21568, 1, 21568, 337), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, 337, 32, 128), (1380352, 4096, 128, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((32, 337, 8, 128), (345088, 1024, 128, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([unsqueeze_4, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
