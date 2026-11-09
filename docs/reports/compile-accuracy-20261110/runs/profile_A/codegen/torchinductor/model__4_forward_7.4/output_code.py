# AOT ID: ['4_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/cs/ccs4nri3fjxasjjc6a5rfndjfscsoqbaldlvzm2nkg6kuv7plkdl.py
# Topologically Sorted Source Nodes: [add, add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select, %select_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %select_2), kwargs = {})
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 1024
    x0 = (xindex % 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (24576 + x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (24576 + x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (49152 + x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (49152 + x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([XBLOCK], 2304, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 2304), "index out of bounds: 0 <= tmp4 < 2304")
    tmp6 = tl.load(in_ptr1 + (x0 + 1024*tmp4), None).to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tl.device_assert((0 <= tmp12) & (tmp12 < 2304), "index out of bounds: 0 <= tmp12 < 2304")
    tmp14 = tl.load(in_ptr1 + (x0 + 1024*tmp12), None).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 + tmp16
    tmp19 = tmp18 + tmp1
    tmp20 = tmp18 < 0
    tmp21 = tl.where(tmp20, tmp19, tmp18)
    tl.device_assert((0 <= tmp21) & (tmp21 < 2304), "index out of bounds: 0 <= tmp21 < 2304")
    tmp23 = tl.load(in_ptr1 + (x0 + 1024*tmp21), None).to(tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp17 + tmp25
    tl.store(out_ptr0 + (x2), tmp26, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/kx/ckxv76r77dwk6ngn2eqkx4qzysk5fgxr2i26uqxvrzoqyanlbrso.py
# Topologically Sorted Source Nodes: [positions, index_select], Original ATen: [aten.add, aten.index_select]
# Source node to ATen node mapping:
#   index_select => index
#   positions => add_2
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %select_3), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%add_2, [%primals_4]), kwargs = {})
triton_poi_fused_add_index_select_1 = async_compile.triton('triton_poi_fused_add_index_select_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_select_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_index_select_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 1024
    x0 = (xindex % 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 24576, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 24576), "index out of bounds: 0 <= tmp4 < 24576")
    tmp6 = tl.load(in_ptr1 + (x0 + 1024*tmp4), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (73728 + tmp4), None, eviction_policy='evict_last')
    tmp8 = tl.full([XBLOCK], 2304, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 2304), "index out of bounds: 0 <= tmp11 < 2304")
    tmp13 = tl.load(in_ptr3 + (x0 + 1024*tmp11), None).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (73728 + tmp4), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp6 + tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (2304, 1024), (1024, 1))
    assert_size_stride(primals_2, (4, 24576), (24576, 1))
    assert_size_stride(primals_3, (4, 24576), (24576, 1))
    assert_size_stride(primals_4, (24576, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [add, add_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(primals_2, primals_1, primals_3, buf0, 25165824, stream=stream0)
        buf1 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [positions, index_select], Original ATen: [aten.add, aten.index_select]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_index_select_1.run(primals_4, buf0, primals_2, primals_1, primals_3, buf1, 25165824, stream=stream0)
        del buf0
        del primals_1
    return (buf1, primals_2, primals_3, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2304, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((4, 24576), (24576, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((4, 24576), (24576, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((24576, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
