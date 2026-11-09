# AOT ID: ['180_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/tr/ctrnte7yyzb53uzo4hr2j4fpssie7zijmu6otnb3xddrqmlrzo5n.py
# Topologically Sorted Source Nodes: [float_3, l1_loss], Original ATen: [aten.div, aten.smooth_l1_loss_backward, aten._to_copy, aten.smooth_l1_loss, aten.add]
# Source node to ATen node mapping:
#   float_3 => convert_element_type_1
#   l1_loss => abs_1, lt, sub
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 8388608), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, 1.0), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sub), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, 1.0), kwargs = {})
#   %sign : [num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sign), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%abs_1, 1.0), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %div_2, %mul_3), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %where_1), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_div_smooth_l1_loss_smooth_l1_loss_backward_0 = async_compile.triton('triton_poi_fused__to_copy_add_div_smooth_l1_loss_smooth_l1_loss_backward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_smooth_l1_loss_smooth_l1_loss_backward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 117440512}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_smooth_l1_loss_smooth_l1_loss_backward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp5 = tl_math.abs(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 < tmp6
    tmp10 = 1.1920928955078125e-07
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp6
    tmp13 = tmp12 * tmp4
    tmp14 = tmp13 * tmp6
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = tmp15 < tmp4
    tmp17 = tmp16.to(tl.int8)
    tmp18 = tmp4 < tmp15
    tmp19 = tmp18.to(tl.int8)
    tmp20 = tmp17 - tmp19
    tmp21 = tmp20.to(tmp4.dtype)
    tmp22 = tmp12 * tmp21
    tmp23 = tl.where(tmp7, tmp14, tmp22)
    tmp24 = tmp0 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp25, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, convert_element_type, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_2, (32, 256, 1024), (262144, 1024, 1))
    assert_size_stride(convert_element_type, (32, 256, 1024), (262144, 1024, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (32, 256, 1024), (262144, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 256, 1024), (262144, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_3, l1_loss], Original ATen: [aten.div, aten.smooth_l1_loss_backward, aten._to_copy, aten.smooth_l1_loss, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_smooth_l1_loss_smooth_l1_loss_backward_0.run(tangents_2, convert_element_type, primals_2, tangents_1, buf0, 8388608, stream=stream0)
        del convert_element_type
        del primals_2
        del tangents_1
        del tangents_2
    return (buf0, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, convert_element_type, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
