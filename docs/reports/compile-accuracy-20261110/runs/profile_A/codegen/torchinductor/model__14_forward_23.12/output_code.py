# AOT ID: ['14_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/jo/cjo75dijm3k4lzkasg36mfg7qqxd3ir2gfmm4sfklt6sss73yvlv.py
# Topologically Sorted Source Nodes: [img_chunks], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   img_chunks => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_3, %primals_2, %primals_1], 2), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 97331200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16220160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 2560) % 66)
    x0 = (xindex % 2560)
    x2 = xindex // 168960
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 65, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 2560*((-1) + x1) + 163840*x2), tmp9, other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 66, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0), tmp11, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/d2/cd27pw7ibrpujamv77seoyduxrkhoqh3tz6hkovx7adx2lgk3fbt.py
# Topologically Sorted Source Nodes: [image_visual_masks, setitem, _generalized_scatter, image_pad_masks_1], Original ATen: [aten.zeros_like, aten.copy, aten.clone]
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
#   image_pad_masks_1 => clone
#   image_visual_masks => full
#   setitem => copy
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 3, 66], False), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%slice_3, %expand_1), kwargs = {})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full, %copy, 2, 1, 65), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_copy_zeros_like_1 = async_compile.triton('triton_poi_fused_clone_copy_zeros_like_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_copy_zeros_like_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_copy_zeros_like_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 66
    x2 = xindex
    x0 = (xindex % 66)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = x0
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1], 65, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = tl.load(in_ptr0 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = tl.full([1], False, tl.int1)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
    tl.store(out_ptr1 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/r6/cr6u27n42qhhbqfvm6my7hglyunscc3i47myfcovtunal76k76x6.py
# Topologically Sorted Source Nodes: [fake_image_ids, setitem_1, _generalized_scatter_1, setitem_2, _generalized_scatter_2], Original ATen: [aten.full, aten.lift_fresh, aten.fill]
# Source node to ATen node mapping:
#   _generalized_scatter_1 => select_scatter_default
#   _generalized_scatter_2 => select_scatter_default_1
#   fake_image_ids => full_default
#   setitem_1 => copy_1, full_default_1
#   setitem_2 => copy_2, full_default_2
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([32, 3, 66], 151655), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 151652), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %full_default_1), kwargs = {})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default, %copy_1, 2, 0), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 151653), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_3, %full_default_2), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_2, 2, -1), kwargs = {})
triton_poi_fused_fill_full_lift_fresh_2 = async_compile.triton('triton_poi_fused_fill_full_lift_fresh_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_fill_full_lift_fresh_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 101376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_fill_full_lift_fresh_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 65, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp0 == tmp3
    tmp5 = tl.full([1], 151652, tl.int64)
    tmp6 = tl.full([1], 151655, tl.int64)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1], 151653, tl.int64)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 1, 2560), (0, 0, 2560, 1))
    assert_size_stride(primals_2, (32, 3, 64, 2560), (491520, 163840, 2560, 1))
    assert_size_stride(primals_3, (32, 3, 1, 2560), (0, 0, 2560, 1))
    assert_size_stride(primals_4, (32, 3), (3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 66, 2560), (506880, 168960, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [img_chunks], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_3, primals_2, primals_1, buf0, 16220160, stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((32, 3, 66), (198, 66, 1), torch.bool)
        buf3 = empty_strided_cuda((32, 3, 66), (198, 66, 1), torch.bool)
        # Topologically Sorted Source Nodes: [image_visual_masks, setitem, _generalized_scatter, image_pad_masks_1], Original ATen: [aten.zeros_like, aten.copy, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_copy_zeros_like_1.run(primals_4, buf1, buf3, 6336, stream=stream0)
        del primals_4
        buf2 = empty_strided_cuda((32, 3, 66), (198, 66, 1), torch.int64)
        # Topologically Sorted Source Nodes: [fake_image_ids, setitem_1, _generalized_scatter_1, setitem_2, _generalized_scatter_2], Original ATen: [aten.full, aten.lift_fresh, aten.fill]
        stream0 = get_raw_stream(0)
        triton_poi_fused_fill_full_lift_fresh_2.run(buf2, 6336, stream=stream0)
    return (reinterpret_tensor(buf0, (32, 198, 2560), (506880, 2560, 1), 0), reinterpret_tensor(buf1, (32, 198), (198, 1), 0), reinterpret_tensor(buf2, (32, 198), (198, 1), 0), reinterpret_tensor(buf3, (32, 198), (198, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 1, 2560), (0, 0, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((32, 3, 64, 2560), (491520, 163840, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((32, 3, 1, 2560), (0, 0, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((32, 3), (3, 1), device='cuda:0', dtype=torch.bool)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
