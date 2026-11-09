# AOT ID: ['187_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/lx/clx2bfqbzw5ne54bf6xfssjecrhrm3aeyedbhwwi3zspi4putp6z.py
# Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
# Source node to ATen node mapping:
#   logsumexp => abs_1, add, amax, eq, exp, full_default, log, sub, sum_1, where
# Graph fragment:
#   %amax : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%primals_2, [-1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax,), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default : [num_users=36] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %amax), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %where), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, %squeeze), kwargs = {})
triton_per_fused_logsumexp_0 = async_compile.triton('triton_per_fused_logsumexp_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_logsumexp_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13056, 'r0_': 208896}}
)
@triton.jit
def triton_per_fused_logsumexp_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1632
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp5 = tl_math.abs(tmp4)
    tmp6 = float("inf")
    tmp7 = tmp5 == tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp7, tmp8, tmp4)
    tmp10 = tmp0 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl_math.log(tmp15)
    tmp17 = tmp16 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/6t/c6tehpviincl35zwxhs22bo2istvvttoowffy5lqvxsbihhks7ww.py
# Topologically Sorted Source Nodes: [pow_1, mean, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
# Source node to ATen node mapping:
#   mean => mean
#   pow_1 => pow_1
#   stack => cat
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3, %unsqueeze_4, %unsqueeze_5, %unsqueeze_6, %unsqueeze_7, %unsqueeze_8, %unsqueeze_9, %unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14, %unsqueeze_15, %unsqueeze_16, %unsqueeze_17, %unsqueeze_18, %unsqueeze_19, %unsqueeze_20, %unsqueeze_21, %unsqueeze_22, %unsqueeze_23, %unsqueeze_24, %unsqueeze_25, %unsqueeze_26, %unsqueeze_27, %unsqueeze_28, %unsqueeze_29, %unsqueeze_30, %unsqueeze_31, %unsqueeze_32, %unsqueeze_33, %unsqueeze_34, %unsqueeze_35],), kwargs = {})
triton_red_fused_mean_pow_stack_1 = async_compile.triton('triton_red_fused_mean_pow_stack_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_stack_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 6528}}
)
@triton.jit
def triton_red_fused_mean_pow_stack_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 1632
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = 1632.0
    tmp6 = (tmp3 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/q4/cq4y74dhzzlx2sivqafctwgeg2hup6af6i67ioc3szkevbftxvar.py
# Topologically Sorted Source Nodes: [pow_2, mean_1, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
# Source node to ATen node mapping:
#   mean_1 => mean_1
#   pow_2 => pow_2
#   stack => cat
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1, 2), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_2,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze, %unsqueeze_1, %unsqueeze_2, %unsqueeze_3, %unsqueeze_4, %unsqueeze_5, %unsqueeze_6, %unsqueeze_7, %unsqueeze_8, %unsqueeze_9, %unsqueeze_10, %unsqueeze_11, %unsqueeze_12, %unsqueeze_13, %unsqueeze_14, %unsqueeze_15, %unsqueeze_16, %unsqueeze_17, %unsqueeze_18, %unsqueeze_19, %unsqueeze_20, %unsqueeze_21, %unsqueeze_22, %unsqueeze_23, %unsqueeze_24, %unsqueeze_25, %unsqueeze_26, %unsqueeze_27, %unsqueeze_28, %unsqueeze_29, %unsqueeze_30, %unsqueeze_31, %unsqueeze_32, %unsqueeze_33, %unsqueeze_34, %unsqueeze_35],), kwargs = {})
triton_red_fused_mean_pow_stack_2 = async_compile.triton('triton_red_fused_mean_pow_stack_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_stack_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 6528}}
)
@triton.jit
def triton_red_fused_mean_pow_stack_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 1632
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = 1632.0
    tmp6 = (tmp3 / tmp5)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp6, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/mj/cmjbupsskcmaa3zgleqgr5cdhstbuslokujcfd4jeqekujtq6sos.py
# Topologically Sorted Source Nodes: [mean_36, router_z_loss_1], Original ATen: [aten.mean, aten.mul]
# Source node to ATen node mapping:
#   mean_36 => mean_36
#   router_z_loss_1 => mul
# Graph fragment:
#   %mean_36 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%cat,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_36, 0.0001), kwargs = {})
triton_per_fused_mean_mul_3 = async_compile.triton('triton_per_fused_mean_mul_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 144}}
)
@triton.jit
def triton_per_fused_mean_mul_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 36
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 36.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 0.0001
    tmp8 = tmp6 * tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37 = args
    args.clear()
    assert_size_stride(primals_1, (32, 50, 55), (2750, 55, 1))
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [logsumexp], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf2, primals_2, 1632, 32, stream=stream0)
        buf6 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_1], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf7, primals_3, 1632, 32, stream=stream0)
        buf11 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_2], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf12, primals_4, 1632, 32, stream=stream0)
        buf16 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_3], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf17, primals_5, 1632, 32, stream=stream0)
        buf21 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_4], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf22, primals_6, 1632, 32, stream=stream0)
        buf26 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_5], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf27, primals_7, 1632, 32, stream=stream0)
        buf31 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_6], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf32, primals_8, 1632, 32, stream=stream0)
        buf36 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_7], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf37, primals_9, 1632, 32, stream=stream0)
        buf41 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_8], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf42, primals_10, 1632, 32, stream=stream0)
        buf46 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_9], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf47, primals_11, 1632, 32, stream=stream0)
        buf51 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_10], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf52, primals_12, 1632, 32, stream=stream0)
        buf56 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_11], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf57, primals_13, 1632, 32, stream=stream0)
        buf61 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_12], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf62, primals_14, 1632, 32, stream=stream0)
        buf66 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_13], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf67, primals_15, 1632, 32, stream=stream0)
        buf71 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_14], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf72, primals_16, 1632, 32, stream=stream0)
        buf76 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_15], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf77, primals_17, 1632, 32, stream=stream0)
        buf81 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_16], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf82, primals_18, 1632, 32, stream=stream0)
        buf86 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_17], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf87, primals_19, 1632, 32, stream=stream0)
        buf91 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_18], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf92, primals_20, 1632, 32, stream=stream0)
        buf96 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_19], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf97, primals_21, 1632, 32, stream=stream0)
        buf101 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_20], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf102, primals_22, 1632, 32, stream=stream0)
        buf106 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_21], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf107, primals_23, 1632, 32, stream=stream0)
        buf111 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_22], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf112, primals_24, 1632, 32, stream=stream0)
        buf116 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_23], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf117, primals_25, 1632, 32, stream=stream0)
        buf121 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_24], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf122, primals_26, 1632, 32, stream=stream0)
        buf126 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_25], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf127, primals_27, 1632, 32, stream=stream0)
        buf131 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_26], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf132, primals_28, 1632, 32, stream=stream0)
        buf136 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_27], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf137, primals_29, 1632, 32, stream=stream0)
        buf141 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf142 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_28], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf142, primals_30, 1632, 32, stream=stream0)
        buf146 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf147 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_29], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf147, primals_31, 1632, 32, stream=stream0)
        buf151 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf152 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_30], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf152, primals_32, 1632, 32, stream=stream0)
        buf156 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_31], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf157, primals_33, 1632, 32, stream=stream0)
        buf161 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_32], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf162, primals_34, 1632, 32, stream=stream0)
        buf166 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_33], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf167, primals_35, 1632, 32, stream=stream0)
        buf171 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf172 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_34], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf172, primals_36, 1632, 32, stream=stream0)
        buf176 = empty_strided_cuda((1632, ), (1, ), torch.float32)
        buf177 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [logsumexp, logsumexp_35], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_per_fused_logsumexp_0.run(buf177, primals_37, 1632, 32, stream=stream0)
        buf3 = empty_strided_cuda((), (), torch.float32)
        buf4 = buf3; del buf3  # reuse
        buf216 = empty_strided_cuda((36, ), (1, ), torch.float32)
        buf180 = reinterpret_tensor(buf216, (1, ), (1, ), 0)  # alias
        # Topologically Sorted Source Nodes: [pow_1, mean, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_1.run(buf4, buf2, buf180, 1, 1632, stream=stream0)
        buf8 = empty_strided_cuda((), (), torch.float32)
        buf9 = buf8; del buf8  # reuse
        buf181 = reinterpret_tensor(buf216, (1, ), (1, ), 1)  # alias
        # Topologically Sorted Source Nodes: [pow_2, mean_1, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf9, buf7, buf181, 1, 1632, stream=stream0)
        buf13 = empty_strided_cuda((), (), torch.float32)
        buf14 = buf13; del buf13  # reuse
        buf182 = reinterpret_tensor(buf216, (1, ), (1, ), 2)  # alias
        # Topologically Sorted Source Nodes: [pow_3, mean_2, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf14, buf12, buf182, 1, 1632, stream=stream0)
        buf18 = empty_strided_cuda((), (), torch.float32)
        buf19 = buf18; del buf18  # reuse
        buf183 = reinterpret_tensor(buf216, (1, ), (1, ), 3)  # alias
        # Topologically Sorted Source Nodes: [pow_4, mean_3, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf19, buf17, buf183, 1, 1632, stream=stream0)
        buf23 = empty_strided_cuda((), (), torch.float32)
        buf24 = buf23; del buf23  # reuse
        buf184 = reinterpret_tensor(buf216, (1, ), (1, ), 4)  # alias
        # Topologically Sorted Source Nodes: [pow_5, mean_4, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf24, buf22, buf184, 1, 1632, stream=stream0)
        buf28 = empty_strided_cuda((), (), torch.float32)
        buf29 = buf28; del buf28  # reuse
        buf185 = reinterpret_tensor(buf216, (1, ), (1, ), 5)  # alias
        # Topologically Sorted Source Nodes: [pow_6, mean_5, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf29, buf27, buf185, 1, 1632, stream=stream0)
        buf33 = empty_strided_cuda((), (), torch.float32)
        buf34 = buf33; del buf33  # reuse
        buf186 = reinterpret_tensor(buf216, (1, ), (1, ), 6)  # alias
        # Topologically Sorted Source Nodes: [pow_7, mean_6, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf34, buf32, buf186, 1, 1632, stream=stream0)
        buf38 = empty_strided_cuda((), (), torch.float32)
        buf39 = buf38; del buf38  # reuse
        buf187 = reinterpret_tensor(buf216, (1, ), (1, ), 7)  # alias
        # Topologically Sorted Source Nodes: [pow_8, mean_7, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf39, buf37, buf187, 1, 1632, stream=stream0)
        buf43 = empty_strided_cuda((), (), torch.float32)
        buf44 = buf43; del buf43  # reuse
        buf188 = reinterpret_tensor(buf216, (1, ), (1, ), 8)  # alias
        # Topologically Sorted Source Nodes: [pow_9, mean_8, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf44, buf42, buf188, 1, 1632, stream=stream0)
        buf48 = empty_strided_cuda((), (), torch.float32)
        buf49 = buf48; del buf48  # reuse
        buf189 = reinterpret_tensor(buf216, (1, ), (1, ), 9)  # alias
        # Topologically Sorted Source Nodes: [pow_10, mean_9, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf49, buf47, buf189, 1, 1632, stream=stream0)
        buf53 = empty_strided_cuda((), (), torch.float32)
        buf54 = buf53; del buf53  # reuse
        buf190 = reinterpret_tensor(buf216, (1, ), (1, ), 10)  # alias
        # Topologically Sorted Source Nodes: [pow_11, mean_10, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf54, buf52, buf190, 1, 1632, stream=stream0)
        buf58 = empty_strided_cuda((), (), torch.float32)
        buf59 = buf58; del buf58  # reuse
        buf191 = reinterpret_tensor(buf216, (1, ), (1, ), 11)  # alias
        # Topologically Sorted Source Nodes: [pow_12, mean_11, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf59, buf57, buf191, 1, 1632, stream=stream0)
        buf63 = empty_strided_cuda((), (), torch.float32)
        buf64 = buf63; del buf63  # reuse
        buf192 = reinterpret_tensor(buf216, (1, ), (1, ), 12)  # alias
        # Topologically Sorted Source Nodes: [pow_13, mean_12, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf64, buf62, buf192, 1, 1632, stream=stream0)
        buf68 = empty_strided_cuda((), (), torch.float32)
        buf69 = buf68; del buf68  # reuse
        buf193 = reinterpret_tensor(buf216, (1, ), (1, ), 13)  # alias
        # Topologically Sorted Source Nodes: [pow_14, mean_13, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf69, buf67, buf193, 1, 1632, stream=stream0)
        buf73 = empty_strided_cuda((), (), torch.float32)
        buf74 = buf73; del buf73  # reuse
        buf194 = reinterpret_tensor(buf216, (1, ), (1, ), 14)  # alias
        # Topologically Sorted Source Nodes: [pow_15, mean_14, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf74, buf72, buf194, 1, 1632, stream=stream0)
        buf78 = empty_strided_cuda((), (), torch.float32)
        buf79 = buf78; del buf78  # reuse
        buf195 = reinterpret_tensor(buf216, (1, ), (1, ), 15)  # alias
        # Topologically Sorted Source Nodes: [pow_16, mean_15, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf79, buf77, buf195, 1, 1632, stream=stream0)
        buf83 = empty_strided_cuda((), (), torch.float32)
        buf84 = buf83; del buf83  # reuse
        buf196 = reinterpret_tensor(buf216, (1, ), (1, ), 16)  # alias
        # Topologically Sorted Source Nodes: [pow_17, mean_16, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_1.run(buf84, buf82, buf196, 1, 1632, stream=stream0)
        buf88 = empty_strided_cuda((), (), torch.float32)
        buf89 = buf88; del buf88  # reuse
        buf197 = reinterpret_tensor(buf216, (1, ), (1, ), 17)  # alias
        # Topologically Sorted Source Nodes: [pow_18, mean_17, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf89, buf87, buf197, 1, 1632, stream=stream0)
        buf93 = empty_strided_cuda((), (), torch.float32)
        buf94 = buf93; del buf93  # reuse
        buf198 = reinterpret_tensor(buf216, (1, ), (1, ), 18)  # alias
        # Topologically Sorted Source Nodes: [pow_19, mean_18, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf94, buf92, buf198, 1, 1632, stream=stream0)
        buf98 = empty_strided_cuda((), (), torch.float32)
        buf99 = buf98; del buf98  # reuse
        buf199 = reinterpret_tensor(buf216, (1, ), (1, ), 19)  # alias
        # Topologically Sorted Source Nodes: [pow_20, mean_19, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf99, buf97, buf199, 1, 1632, stream=stream0)
        buf103 = empty_strided_cuda((), (), torch.float32)
        buf104 = buf103; del buf103  # reuse
        buf200 = reinterpret_tensor(buf216, (1, ), (1, ), 20)  # alias
        # Topologically Sorted Source Nodes: [pow_21, mean_20, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf104, buf102, buf200, 1, 1632, stream=stream0)
        buf108 = empty_strided_cuda((), (), torch.float32)
        buf109 = buf108; del buf108  # reuse
        buf201 = reinterpret_tensor(buf216, (1, ), (1, ), 21)  # alias
        # Topologically Sorted Source Nodes: [pow_22, mean_21, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf109, buf107, buf201, 1, 1632, stream=stream0)
        buf113 = empty_strided_cuda((), (), torch.float32)
        buf114 = buf113; del buf113  # reuse
        buf202 = reinterpret_tensor(buf216, (1, ), (1, ), 22)  # alias
        # Topologically Sorted Source Nodes: [pow_23, mean_22, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf114, buf112, buf202, 1, 1632, stream=stream0)
        buf118 = empty_strided_cuda((), (), torch.float32)
        buf119 = buf118; del buf118  # reuse
        buf203 = reinterpret_tensor(buf216, (1, ), (1, ), 23)  # alias
        # Topologically Sorted Source Nodes: [pow_24, mean_23, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf119, buf117, buf203, 1, 1632, stream=stream0)
        buf123 = empty_strided_cuda((), (), torch.float32)
        buf124 = buf123; del buf123  # reuse
        buf204 = reinterpret_tensor(buf216, (1, ), (1, ), 24)  # alias
        # Topologically Sorted Source Nodes: [pow_25, mean_24, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf124, buf122, buf204, 1, 1632, stream=stream0)
        buf128 = empty_strided_cuda((), (), torch.float32)
        buf129 = buf128; del buf128  # reuse
        buf205 = reinterpret_tensor(buf216, (1, ), (1, ), 25)  # alias
        # Topologically Sorted Source Nodes: [pow_26, mean_25, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf129, buf127, buf205, 1, 1632, stream=stream0)
        buf133 = empty_strided_cuda((), (), torch.float32)
        buf134 = buf133; del buf133  # reuse
        buf206 = reinterpret_tensor(buf216, (1, ), (1, ), 26)  # alias
        # Topologically Sorted Source Nodes: [pow_27, mean_26, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf134, buf132, buf206, 1, 1632, stream=stream0)
        buf138 = empty_strided_cuda((), (), torch.float32)
        buf139 = buf138; del buf138  # reuse
        buf207 = reinterpret_tensor(buf216, (1, ), (1, ), 27)  # alias
        # Topologically Sorted Source Nodes: [pow_28, mean_27, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf139, buf137, buf207, 1, 1632, stream=stream0)
        buf143 = empty_strided_cuda((), (), torch.float32)
        buf144 = buf143; del buf143  # reuse
        buf208 = reinterpret_tensor(buf216, (1, ), (1, ), 28)  # alias
        # Topologically Sorted Source Nodes: [pow_29, mean_28, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf144, buf142, buf208, 1, 1632, stream=stream0)
        buf148 = empty_strided_cuda((), (), torch.float32)
        buf149 = buf148; del buf148  # reuse
        buf209 = reinterpret_tensor(buf216, (1, ), (1, ), 29)  # alias
        # Topologically Sorted Source Nodes: [pow_30, mean_29, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf149, buf147, buf209, 1, 1632, stream=stream0)
        buf153 = empty_strided_cuda((), (), torch.float32)
        buf154 = buf153; del buf153  # reuse
        buf210 = reinterpret_tensor(buf216, (1, ), (1, ), 30)  # alias
        # Topologically Sorted Source Nodes: [pow_31, mean_30, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf154, buf152, buf210, 1, 1632, stream=stream0)
        buf158 = empty_strided_cuda((), (), torch.float32)
        buf159 = buf158; del buf158  # reuse
        buf211 = reinterpret_tensor(buf216, (1, ), (1, ), 31)  # alias
        # Topologically Sorted Source Nodes: [pow_32, mean_31, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf159, buf157, buf211, 1, 1632, stream=stream0)
        buf163 = empty_strided_cuda((), (), torch.float32)
        buf164 = buf163; del buf163  # reuse
        buf212 = reinterpret_tensor(buf216, (1, ), (1, ), 32)  # alias
        # Topologically Sorted Source Nodes: [pow_33, mean_32, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_1.run(buf164, buf162, buf212, 1, 1632, stream=stream0)
        buf168 = empty_strided_cuda((), (), torch.float32)
        buf169 = buf168; del buf168  # reuse
        buf213 = reinterpret_tensor(buf216, (1, ), (1, ), 33)  # alias
        # Topologically Sorted Source Nodes: [pow_34, mean_33, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf169, buf167, buf213, 1, 1632, stream=stream0)
        buf173 = empty_strided_cuda((), (), torch.float32)
        buf174 = buf173; del buf173  # reuse
        buf214 = reinterpret_tensor(buf216, (1, ), (1, ), 34)  # alias
        # Topologically Sorted Source Nodes: [pow_35, mean_34, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf174, buf172, buf214, 1, 1632, stream=stream0)
        buf178 = empty_strided_cuda((), (), torch.float32)
        buf179 = buf178; del buf178  # reuse
        buf215 = reinterpret_tensor(buf216, (1, ), (1, ), 35)  # alias
        # Topologically Sorted Source Nodes: [pow_36, mean_35, stack], Original ATen: [aten.pow, aten.mean, aten.stack]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_stack_2.run(buf179, buf177, buf215, 1, 1632, stream=stream0)
        buf217 = empty_strided_cuda((), (), torch.float32)
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [mean_36, router_z_loss_1], Original ATen: [aten.mean, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_3.run(buf218, buf216, 1, 36, stream=stream0)
        del buf180
        del buf181
        del buf182
        del buf183
        del buf184
        del buf185
        del buf186
        del buf187
        del buf188
        del buf189
        del buf190
        del buf191
        del buf192
        del buf193
        del buf194
        del buf195
        del buf196
        del buf197
        del buf198
        del buf199
        del buf200
        del buf201
        del buf202
        del buf203
        del buf204
        del buf205
        del buf206
        del buf207
        del buf208
        del buf209
        del buf210
        del buf211
        del buf212
        del buf213
        del buf214
        del buf215
        del buf216
    return (buf218, buf4, buf9, buf14, buf19, buf24, buf29, buf34, buf39, buf44, buf49, buf54, buf59, buf64, buf69, buf74, buf79, buf84, buf89, buf94, buf99, buf104, buf109, buf114, buf119, buf124, buf129, buf134, buf139, buf144, buf149, buf154, buf159, buf164, buf169, buf174, buf179, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, buf2, buf7, buf12, buf17, buf22, buf27, buf32, buf37, buf42, buf47, buf52, buf57, buf62, buf67, buf72, buf77, buf82, buf87, buf92, buf97, buf102, buf107, buf112, buf117, buf122, buf127, buf132, buf137, buf142, buf147, buf152, buf157, buf162, buf167, buf172, buf177, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 50, 55), (2750, 55, 1), device='cuda:0', dtype=torch.bfloat16)
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
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
