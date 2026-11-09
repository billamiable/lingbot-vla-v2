# AOT ID: ['46_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/6d/c6dy4gdxpprcmor4mxdzahmq7g45elfpvbj63gqydqq3ovizxrm7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_37, %unsqueeze_37), kwargs = {})
#   %exp_36 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_36,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_36, %exp_36), kwargs = {})
triton_poi_fused_exp_mul_sub_0 = async_compile.triton('triton_poi_fused_exp_mul_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 6528, 'x': 626688}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_mul_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1632
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [YBLOCK, XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [YBLOCK, XBLOCK])
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x1 + 32*y0), xmask & ymask, eviction_policy='evict_last')
    tmp4 = 0.0001
    tmp5 = tmp3 * tmp4
    tmp6 = 0.027777777777777776
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 0.0006127450980392157
    tmp10 = tmp8 * tmp9
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 * tmp13
    tmp16 = tmp15 - tmp11
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp14 * tmp17
    tl.store(out_ptr0 + (x1 + 32*y0), tmp18, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_31, add_32, add_33, add_34, add_35, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37 = args
    args.clear()
    assert_size_stride(primals_2, (1632, 32), (32, 1))
    assert_size_stride(primals_3, (1632, 32), (32, 1))
    assert_size_stride(primals_4, (1632, 32), (32, 1))
    assert_size_stride(primals_5, (1632, 32), (32, 1))
    assert_size_stride(primals_6, (1632, 32), (32, 1))
    assert_size_stride(primals_7, (1632, 32), (32, 1))
    assert_size_stride(primals_8, (1632, 32), (32, 1))
    assert_size_stride(primals_9, (1632, 32), (32, 1))
    assert_size_stride(primals_10, (1632, 32), (32, 1))
    assert_size_stride(primals_11, (1632, 32), (32, 1))
    assert_size_stride(primals_12, (1632, 32), (32, 1))
    assert_size_stride(primals_13, (1632, 32), (32, 1))
    assert_size_stride(primals_14, (1632, 32), (32, 1))
    assert_size_stride(primals_15, (1632, 32), (32, 1))
    assert_size_stride(primals_16, (1632, 32), (32, 1))
    assert_size_stride(primals_17, (1632, 32), (32, 1))
    assert_size_stride(primals_18, (1632, 32), (32, 1))
    assert_size_stride(primals_19, (1632, 32), (32, 1))
    assert_size_stride(primals_20, (1632, 32), (32, 1))
    assert_size_stride(primals_21, (1632, 32), (32, 1))
    assert_size_stride(primals_22, (1632, 32), (32, 1))
    assert_size_stride(primals_23, (1632, 32), (32, 1))
    assert_size_stride(primals_24, (1632, 32), (32, 1))
    assert_size_stride(primals_25, (1632, 32), (32, 1))
    assert_size_stride(primals_26, (1632, 32), (32, 1))
    assert_size_stride(primals_27, (1632, 32), (32, 1))
    assert_size_stride(primals_28, (1632, 32), (32, 1))
    assert_size_stride(primals_29, (1632, 32), (32, 1))
    assert_size_stride(primals_30, (1632, 32), (32, 1))
    assert_size_stride(primals_31, (1632, 32), (32, 1))
    assert_size_stride(primals_32, (1632, 32), (32, 1))
    assert_size_stride(primals_33, (1632, 32), (32, 1))
    assert_size_stride(primals_34, (1632, 32), (32, 1))
    assert_size_stride(primals_35, (1632, 32), (32, 1))
    assert_size_stride(primals_36, (1632, 32), (32, 1))
    assert_size_stride(primals_37, (1632, 32), (32, 1))
    assert_size_stride(add, (1632, ), (1, ))
    assert_size_stride(add_1, (1632, ), (1, ))
    assert_size_stride(add_2, (1632, ), (1, ))
    assert_size_stride(add_3, (1632, ), (1, ))
    assert_size_stride(add_4, (1632, ), (1, ))
    assert_size_stride(add_5, (1632, ), (1, ))
    assert_size_stride(add_6, (1632, ), (1, ))
    assert_size_stride(add_7, (1632, ), (1, ))
    assert_size_stride(add_8, (1632, ), (1, ))
    assert_size_stride(add_9, (1632, ), (1, ))
    assert_size_stride(add_10, (1632, ), (1, ))
    assert_size_stride(add_11, (1632, ), (1, ))
    assert_size_stride(add_12, (1632, ), (1, ))
    assert_size_stride(add_13, (1632, ), (1, ))
    assert_size_stride(add_14, (1632, ), (1, ))
    assert_size_stride(add_15, (1632, ), (1, ))
    assert_size_stride(add_16, (1632, ), (1, ))
    assert_size_stride(add_17, (1632, ), (1, ))
    assert_size_stride(add_18, (1632, ), (1, ))
    assert_size_stride(add_19, (1632, ), (1, ))
    assert_size_stride(add_20, (1632, ), (1, ))
    assert_size_stride(add_21, (1632, ), (1, ))
    assert_size_stride(add_22, (1632, ), (1, ))
    assert_size_stride(add_23, (1632, ), (1, ))
    assert_size_stride(add_24, (1632, ), (1, ))
    assert_size_stride(add_25, (1632, ), (1, ))
    assert_size_stride(add_26, (1632, ), (1, ))
    assert_size_stride(add_27, (1632, ), (1, ))
    assert_size_stride(add_28, (1632, ), (1, ))
    assert_size_stride(add_29, (1632, ), (1, ))
    assert_size_stride(add_30, (1632, ), (1, ))
    assert_size_stride(add_31, (1632, ), (1, ))
    assert_size_stride(add_32, (1632, ), (1, ))
    assert_size_stride(add_33, (1632, ), (1, ))
    assert_size_stride(add_34, (1632, ), (1, ))
    assert_size_stride(add_35, (1632, ), (1, ))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (), ())
    assert_size_stride(tangents_3, (), ())
    assert_size_stride(tangents_4, (), ())
    assert_size_stride(tangents_5, (), ())
    assert_size_stride(tangents_6, (), ())
    assert_size_stride(tangents_7, (), ())
    assert_size_stride(tangents_8, (), ())
    assert_size_stride(tangents_9, (), ())
    assert_size_stride(tangents_10, (), ())
    assert_size_stride(tangents_11, (), ())
    assert_size_stride(tangents_12, (), ())
    assert_size_stride(tangents_13, (), ())
    assert_size_stride(tangents_14, (), ())
    assert_size_stride(tangents_15, (), ())
    assert_size_stride(tangents_16, (), ())
    assert_size_stride(tangents_17, (), ())
    assert_size_stride(tangents_18, (), ())
    assert_size_stride(tangents_19, (), ())
    assert_size_stride(tangents_20, (), ())
    assert_size_stride(tangents_21, (), ())
    assert_size_stride(tangents_22, (), ())
    assert_size_stride(tangents_23, (), ())
    assert_size_stride(tangents_24, (), ())
    assert_size_stride(tangents_25, (), ())
    assert_size_stride(tangents_26, (), ())
    assert_size_stride(tangents_27, (), ())
    assert_size_stride(tangents_28, (), ())
    assert_size_stride(tangents_29, (), ())
    assert_size_stride(tangents_30, (), ())
    assert_size_stride(tangents_31, (), ())
    assert_size_stride(tangents_32, (), ())
    assert_size_stride(tangents_33, (), ())
    assert_size_stride(tangents_34, (), ())
    assert_size_stride(tangents_35, (), ())
    assert_size_stride(tangents_36, (), ())
    assert_size_stride(tangents_37, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_37, tangents_1, add_35, primals_37, buf0, 1632, 32, stream=stream0)
        del add_35
        del primals_37
        del tangents_37
        buf1 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_36, tangents_1, add_34, primals_36, buf1, 1632, 32, stream=stream0)
        del add_34
        del primals_36
        del tangents_36
        buf2 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_35, tangents_1, add_33, primals_35, buf2, 1632, 32, stream=stream0)
        del add_33
        del primals_35
        del tangents_35
        buf3 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_34, tangents_1, add_32, primals_34, buf3, 1632, 32, stream=stream0)
        del add_32
        del primals_34
        del tangents_34
        buf4 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_33, tangents_1, add_31, primals_33, buf4, 1632, 32, stream=stream0)
        del add_31
        del primals_33
        del tangents_33
        buf5 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_32, tangents_1, add_30, primals_32, buf5, 1632, 32, stream=stream0)
        del add_30
        del primals_32
        del tangents_32
        buf6 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_31, tangents_1, add_29, primals_31, buf6, 1632, 32, stream=stream0)
        del add_29
        del primals_31
        del tangents_31
        buf7 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_30, tangents_1, add_28, primals_30, buf7, 1632, 32, stream=stream0)
        del add_28
        del primals_30
        del tangents_30
        buf8 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_29, tangents_1, add_27, primals_29, buf8, 1632, 32, stream=stream0)
        del add_27
        del primals_29
        del tangents_29
        buf9 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_28, tangents_1, add_26, primals_28, buf9, 1632, 32, stream=stream0)
        del add_26
        del primals_28
        del tangents_28
        buf10 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_27, tangents_1, add_25, primals_27, buf10, 1632, 32, stream=stream0)
        del add_25
        del primals_27
        del tangents_27
        buf11 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_26, tangents_1, add_24, primals_26, buf11, 1632, 32, stream=stream0)
        del add_24
        del primals_26
        del tangents_26
        buf12 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_25, tangents_1, add_23, primals_25, buf12, 1632, 32, stream=stream0)
        del add_23
        del primals_25
        del tangents_25
        buf13 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_24, tangents_1, add_22, primals_24, buf13, 1632, 32, stream=stream0)
        del add_22
        del primals_24
        del tangents_24
        buf14 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_23, tangents_1, add_21, primals_23, buf14, 1632, 32, stream=stream0)
        del add_21
        del primals_23
        del tangents_23
        buf15 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_22, tangents_1, add_20, primals_22, buf15, 1632, 32, stream=stream0)
        del add_20
        del primals_22
        del tangents_22
        buf16 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_21, tangents_1, add_19, primals_21, buf16, 1632, 32, stream=stream0)
        del add_19
        del primals_21
        del tangents_21
        buf17 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_20, tangents_1, add_18, primals_20, buf17, 1632, 32, stream=stream0)
        del add_18
        del primals_20
        del tangents_20
        buf18 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_19, tangents_1, add_17, primals_19, buf18, 1632, 32, stream=stream0)
        del add_17
        del primals_19
        del tangents_19
        buf19 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_18, tangents_1, add_16, primals_18, buf19, 1632, 32, stream=stream0)
        del add_16
        del primals_18
        del tangents_18
        buf20 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_17, tangents_1, add_15, primals_17, buf20, 1632, 32, stream=stream0)
        del add_15
        del primals_17
        del tangents_17
        buf21 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_16, tangents_1, add_14, primals_16, buf21, 1632, 32, stream=stream0)
        del add_14
        del primals_16
        del tangents_16
        buf22 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_15, tangents_1, add_13, primals_15, buf22, 1632, 32, stream=stream0)
        del add_13
        del primals_15
        del tangents_15
        buf23 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_14, tangents_1, add_12, primals_14, buf23, 1632, 32, stream=stream0)
        del add_12
        del primals_14
        del tangents_14
        buf24 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_13, tangents_1, add_11, primals_13, buf24, 1632, 32, stream=stream0)
        del add_11
        del primals_13
        del tangents_13
        buf25 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_12, tangents_1, add_10, primals_12, buf25, 1632, 32, stream=stream0)
        del add_10
        del primals_12
        del tangents_12
        buf26 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_11, tangents_1, add_9, primals_11, buf26, 1632, 32, stream=stream0)
        del add_9
        del primals_11
        del tangents_11
        buf27 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_10, tangents_1, add_8, primals_10, buf27, 1632, 32, stream=stream0)
        del add_8
        del primals_10
        del tangents_10
        buf28 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_9, tangents_1, add_7, primals_9, buf28, 1632, 32, stream=stream0)
        del add_7
        del primals_9
        del tangents_9
        buf29 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_8, tangents_1, add_6, primals_8, buf29, 1632, 32, stream=stream0)
        del add_6
        del primals_8
        del tangents_8
        buf30 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_7, tangents_1, add_5, primals_7, buf30, 1632, 32, stream=stream0)
        del add_5
        del primals_7
        del tangents_7
        buf31 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_6, tangents_1, add_4, primals_6, buf31, 1632, 32, stream=stream0)
        del add_4
        del primals_6
        del tangents_6
        buf32 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_5, tangents_1, add_3, primals_5, buf32, 1632, 32, stream=stream0)
        del add_3
        del primals_5
        del tangents_5
        buf33 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_4, tangents_1, add_2, primals_4, buf33, 1632, 32, stream=stream0)
        del add_2
        del primals_4
        del tangents_4
        buf34 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_3, tangents_1, add_1, primals_3, buf34, 1632, 32, stream=stream0)
        del add_1
        del primals_3
        del tangents_3
        buf35 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_sub_0.run(tangents_2, tangents_1, add, primals_2, buf35, 1632, 32, stream=stream0)
        del add
        del primals_2
        del tangents_1
        del tangents_2
    return (None, buf35, buf34, buf33, buf32, buf31, buf30, buf29, buf28, buf27, buf26, buf25, buf24, buf23, buf22, buf21, buf20, buf19, buf18, buf17, buf16, buf15, buf14, buf13, buf12, buf11, buf10, buf9, buf8, buf7, buf6, buf5, buf4, buf3, buf2, buf1, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    add = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_1 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_2 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_3 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_4 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_5 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_6 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_7 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_8 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_9 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_10 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_11 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_12 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_13 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_14 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_15 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_16 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_17 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_18 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_19 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_20 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_21 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_22 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_23 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_24 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_25 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_26 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_27 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_28 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_29 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_30 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_31 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_32 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_33 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_34 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    add_35 = rand_strided((1632, ), (1, ), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_4 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_5 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_6 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_7 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_8 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_9 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_10 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_11 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_12 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_13 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_14 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_15 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_16 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_17 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_18 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_19 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_20 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_21 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_22 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_23 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_24 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_25 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_26 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_27 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_28 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_29 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_30 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_31 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_32 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_33 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_34 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_35 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_36 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_37 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_31, add_32, add_33, add_34, add_35, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, tangents_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
