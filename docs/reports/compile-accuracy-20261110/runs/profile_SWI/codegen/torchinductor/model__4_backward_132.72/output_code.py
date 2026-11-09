# AOT ID: ['4_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/ew/cewesgr5chtbfmcxthwq3qplivvkn3rdcukjc3h5d6btwcn4pknt.py
# Topologically Sorted Source Nodes: [full], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   full => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([24576, 1024], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_zeros_0 = async_compile.triton('triton_poi_fused_new_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/wo/cwo3eq3bityboaejxhd4qooc5fttlr3bdxvo6sbvln2v3yqrrqcy.py
# Topologically Sorted Source Nodes: [full_5], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   full_5 => full_default_6
# Graph fragment:
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2304, 1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_embedding_dense_backward_1 = async_compile.triton('triton_poi_fused_embedding_dense_backward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18874368}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/nx/cnx3sujqfaqbsdc3k6i37ekahmws5i2hzlypxozoeqvbidsqgnix.py
# Topologically Sorted Source Nodes: [full_1, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2, _generalized_scatter_3, scalar_tensor, full_5], Original ATen: [aten.select_backward, aten.add, aten.mul, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   _generalized_scatter => select_scatter_default
#   _generalized_scatter_1 => select_scatter_default_1
#   _generalized_scatter_2 => select_scatter_default_2
#   _generalized_scatter_3 => select_scatter_default_3
#   full_1 => full_default_1
#   full_5 => full_default_6
#   scalar_tensor => full_default_5
# Graph fragment:
#   %full_default_1 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([4, 24576, 1024], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %index_add, 0, 3), kwargs = {})
#   %select_scatter_default_1 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %index_add, 0, 2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default, %select_scatter_default_1), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %index_add, 0, 1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %select_scatter_default_2), kwargs = {})
#   %select_scatter_default_3 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %index_add, 0, 0), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %select_scatter_default_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, %unsqueeze), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.float32), kwargs = {})
#   %full_default_5 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_1, %full_default_5, %convert_element_type), kwargs = {})
#   %full_default_6 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2304, 1024], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_6, [%primals_2], %where, True), kwargs = {})
triton_poi_fused_add_embedding_dense_backward_mul_select_backward_2 = async_compile.triton('triton_poi_fused_add_embedding_dense_backward_mul_select_backward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_mul_select_backward_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_embedding_dense_backward_mul_select_backward_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex // 1024
    x2 = xindex // 25165824
    x4 = (xindex % 25165824)
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([XBLOCK], 2304, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 2304), "index out of bounds: 0 <= tmp4 < 2304")
    tmp6 = tl.full([1], -1, tl.int64)
    tmp7 = tmp0 == tmp6
    tmp8 = x2
    tmp9 = tl.full([1], 3, tl.int32)
    tmp10 = tmp8 == tmp9
    tmp12 = 0.0
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = tl.full([1], 2, tl.int32)
    tmp15 = tmp8 == tmp14
    tmp16 = tl.where(tmp15, tmp11, tmp12)
    tmp17 = tmp13 + tmp16
    tmp18 = tl.full([1], 1, tl.int32)
    tmp19 = tmp8 == tmp18
    tmp20 = tl.where(tmp19, tmp11, tmp12)
    tmp21 = tmp17 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = tmp8 == tmp22
    tmp24 = tl.where(tmp23, tmp11, tmp12)
    tmp25 = tmp21 + tmp24
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.where(tmp7, tmp12, tmp28)
    tl.atomic_add(out_ptr0 + (x0 + 1024*tmp4), tmp29, None, sem='relaxed')
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/xu/cxu2w7miidlwwls327dtkzjieu6veabsg4yfe4ankpoddhuxtibw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%index_put, torch.bfloat16), kwargs = {})
triton_poi_fused_embedding_dense_backward_3 = async_compile.triton('triton_poi_fused_embedding_dense_backward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9437184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_3, primals_4, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (4, 24576), (24576, 1))
    assert_size_stride(primals_3, (4, 24576), (24576, 1))
    assert_size_stride(primals_4, (24576, ), (1, ))
    assert_size_stride(tangents_1, (24576, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [full], Original ATen: [aten.new_zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_0.run(buf0, 25165824, stream=stream0)
        # Topologically Sorted Source Nodes: [full], Original ATen: [aten.new_zeros, aten.index_add]
        buf1 = torch.ops.aten.index_add.default(buf0, 0, primals_4, tangents_1)
        del buf0
        del primals_4
        del tangents_1
        buf2 = buf1
        assert_size_stride(buf2, (24576, 1024), (1024, 1), 'torch.ops.aten.index_add.default')
        assert_alignment(buf2, 16, 'torch.ops.aten.index_add.default')
        del buf1
        buf3 = empty_strided_cuda((2304, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_5], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_1.run(buf3, 2359296, stream=stream0)
        # Topologically Sorted Source Nodes: [full_1, _generalized_scatter, _generalized_scatter_1, _generalized_scatter_2, _generalized_scatter_3, scalar_tensor, full_5], Original ATen: [aten.select_backward, aten.add, aten.mul, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_embedding_dense_backward_mul_select_backward_2.run(primals_2, buf2, primals_3, buf3, 100663296, stream=stream0)
        del buf2
        del primals_2
        del primals_3
        buf5 = empty_strided_cuda((2304, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_3.run(buf3, buf5, 2359296, stream=stream0)
        del buf3
    return (buf5, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((4, 24576), (24576, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((4, 24576), (24576, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((24576, ), (1, ), device='cuda:0', dtype=torch.int64)
    tangents_1 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_2, primals_3, primals_4, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
