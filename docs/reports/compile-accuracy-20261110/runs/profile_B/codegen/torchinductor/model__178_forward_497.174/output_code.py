# AOT ID: ['178_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/om/com3ug57ts4dqdgs2bqhwbq5nt4lrlg5xg5sp2cv2naggqh53h5h.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_1 => add, add_1, convert_element_type_6, convert_element_type_7, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_6, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_8), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
triton_red_fused_native_layer_norm_0 = async_compile.triton('triton_red_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36864, 'r0_': 35399680}}
)
@triton.jit
def triton_red_fused_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2304
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tmp9 = 2560.0
    tmp10 = (tmp4 / tmp9)
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp13, xmask)
    x2 = (xindex % 72)
    x3 = xindex // 72
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp3
        tmp17 = tmp16 * tmp13
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2560*x2 + 839680*x3), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/tc/ctcfoix4gcxl7rjuybyvclbllqkjwcfhrxn5v5rw4xt3hucyy2po.py
# Topologically Sorted Source Nodes: [latents, kv_input], Original ATen: [aten.native_layer_norm, aten.cat]
# Source node to ATen node mapping:
#   kv_input => cat
#   latents => add_2, add_3, convert_element_type_8, convert_element_type_9, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_8, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_8, %getitem_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_9), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_10), kwargs = {})
#   %convert_element_type_9 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convert_element_type_7, %convert_element_type_9], -2), kwargs = {})
triton_red_fused_cat_native_layer_norm_1 = async_compile.triton('triton_red_fused_cat_native_layer_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_native_layer_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 209725440}}
)
@triton.jit
def triton_red_fused_cat_native_layer_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    tl.store(out_ptr0 + (x0), tmp3, None)
    tmp9 = 2560.0
    tmp10 = (tmp4 / tmp9)
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp13, None)
    x2 = (xindex % 256)
    x3 = xindex // 256
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp3
        tmp17 = tmp16 * tmp13
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp24, r0_mask)
        tl.store(out_ptr2 + (r0_1 + 2560*x2 + 839680*x3), tmp24, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/zz/czzenzl6scvrmndgkn3nch3bafoahtcn77wny2gz25ad22jx5fzr.py
# Topologically Sorted Source Nodes: [mul, weight], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   mul => mul_4
#   weight => clone
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, 0.4204482076268573), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_poi_fused_clone_mul_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 256)
    x2 = ((xindex // 8192) % 4)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 128*x1 + 32768*x3), None).to(tl.float32)
    tmp1 = 0.4204482076268573
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ob/cobhbpxiye7zgvnc4dpbsjlu3ngnbzjxd3aovk6xctsbodby6ay6.py
# Topologically Sorted Source Nodes: [weight], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   weight => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'y': 2686976, 'x': 5373952}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 328
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
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x2 + 83968*y1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = 0.4204482076268573
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 328*y3), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/b7/cb7xiovhml22oronmowwkpckmrlircoey62urwl32uqqu5ioo7sj.py
# Topologically Sorted Source Nodes: [float_1, , sub, exp, softmax, weight_1], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
# Source node to ATen node mapping:
#    => prepare_softmax_online_default
#   exp => exp_default
#   float_1 => convert_element_type_16
#   softmax => div
#   sub => sub_tensor
#   weight_1 => convert_element_type_17
# Graph fragment:
#   %convert_element_type_16 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_16, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_16, %getitem_10), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_11), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
triton_per_fused__softmax__to_copy_exp_prepare_softmax_online_sub_4 = async_compile.triton('triton_per_fused__softmax__to_copy_exp_prepare_softmax_online_sub_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax__to_copy_exp_prepare_softmax_online_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 524288, 'r0_': 64487424}}
)
@triton.jit
def triton_per_fused__softmax__to_copy_exp_prepare_softmax_online_sub_4(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 32768
    XBLOCK: tl.constexpr = 1
    r0_numel = 328
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 328*x0), r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, float("-inf"))
    tmp7 = triton_helpers.promote_to_tensor(triton_helpers.max2(tmp6, 0))
    tmp8 = tmp2 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK])
    tmp12 = tl.where(r0_mask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tmp1 - tmp7
    tmp15 = tl_math.exp(tmp14)
    tmp16 = (tmp15 / tmp13)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 328*x0), tmp17, r0_mask)
    tl.store(out_ptr0 + (x0), tmp7, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/nz/cnzesaghnuljh6ckp2ls7j67qkatxllvzmxiooeuwjkapthmkqtl.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   out => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 8060928}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1343488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 328)
    x2 = ((xindex // 10496) % 4)
    x3 = xindex // 41984
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (128 + x0 + 32*x2 + 256*x1 + 83968*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/nx/cnx5bgwvaxrgtb3uchqd4dl6kztoxs44p3gfvpu67jhov6rdjwyd.py
# Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   out_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_8,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 6291456}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 4)
    x2 = ((xindex // 128) % 256)
    x3 = xindex // 32768
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 8192*x1 + 32768*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/bz/cbzgmjq6oen6ruvndgx26ljafwq2yopsahsywbhgge7gqfzjsczm.py
# Topologically Sorted Source Nodes: [queries_1, input_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_1 => add_5, add_6, convert_element_type_22, convert_element_type_23, mul_6, mul_7, rsqrt_2, sub_3, var_mean_2
#   queries_1 => add_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_22, %view_1), kwargs = {})
#   %convert_element_type_22 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %getitem_7), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_14), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_15), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
triton_red_fused_add_native_layer_norm_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 251668480}}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask, tmp5_weight_next, tmp5_weight)
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp2, r0_mask)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tl.store(out_ptr0 + (x0), tmp5, None)
    tmp11 = 2560.0
    tmp12 = (tmp6 / tmp11)
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp15, None)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp5
        tmp19 = tmp18 * tmp15
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 * tmp21
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp26, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/2r/c2r2mafcwiictqgkuajn3vhfr6mfdsizkzhhmcrht4wdzrjaojbn.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   input_3 => add_7, convert_element_type_26, convert_element_type_27, erf, mul_10, mul_8, mul_9
# Graph fragment:
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_24, torch.float32), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, 0.5), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_9,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %add_7), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_8 = async_compile.triton('triton_poi_fused_gelu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 125829120}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20971520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = libdevice.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/yh/cyhutruocbpjvmhovkrrp4co5symmvccbsado7h3uscnge2n3egd.py
# Topologically Sorted Source Nodes: [queries_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   queries_2 => add_8
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_26, %add_4), kwargs = {})
triton_poi_fused_add_9 = async_compile.triton('triton_poi_fused_add_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 167772160}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20971520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/tb/ctblv5knmpjbixvzgqyijbleazysn6fpxw4jyihmdhmcorbvttcz.py
# Topologically Sorted Source Nodes: [queries_4], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   queries_4 => add_10, add_9, convert_element_type_33, convert_element_type_34, mul_11, mul_12, rsqrt_3, sub_4, var_mean_3
# Graph fragment:
#   %convert_element_type_33 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_28, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_33, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_33, %getitem_9), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %primals_20), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %primals_21), kwargs = {})
#   %convert_element_type_34 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
triton_per_fused_native_layer_norm_10 = async_compile.triton('triton_per_fused_native_layer_norm_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 50335744}}
)
@triton.jit
def triton_per_fused_native_layer_norm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 8192
    XBLOCK: tl.constexpr = 1
    r0_numel = 1024
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[:]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp22 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [R0_BLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = tl.full([1], 1024, tl.int32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = (tmp6 / tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [R0_BLOCK])
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 1024.0
    tmp16 = (tmp14 / tmp15)
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp1 - tmp9
    tmp21 = tmp20 * tmp19
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp21 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp19, None)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp28, None)
    tl.store(out_ptr0 + (x0), tmp9, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21 = args
    args.clear()
    assert_size_stride(primals_1, (2560, 2560), (2560, 1))
    assert_size_stride(primals_2, (2560, ), (1, ))
    assert_size_stride(primals_3, (32, 256, 2560), (655360, 2560, 1))
    assert_size_stride(primals_4, (2560, 2560), (2560, 1))
    assert_size_stride(primals_5, (2560, ), (1, ))
    assert_size_stride(primals_6, (32, 72, 2560), (184320, 2560, 1))
    assert_size_stride(primals_7, (2560, ), (1, ))
    assert_size_stride(primals_8, (2560, ), (1, ))
    assert_size_stride(primals_9, (2560, ), (1, ))
    assert_size_stride(primals_10, (2560, ), (1, ))
    assert_size_stride(primals_11, (128, 2560), (2560, 1))
    assert_size_stride(primals_12, (256, 2560), (2560, 1))
    assert_size_stride(primals_13, (2560, 128), (128, 1))
    assert_size_stride(primals_14, (2560, ), (1, ))
    assert_size_stride(primals_15, (2560, ), (1, ))
    assert_size_stride(primals_16, (2560, 2560), (2560, 1))
    assert_size_stride(primals_17, (2560, 2560), (2560, 1))
    assert_size_stride(primals_18, (1024, 2560), (2560, 1))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_21, (1024, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [queries], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_2, reinterpret_tensor(primals_3, (8192, 2560), (2560, 1), 0), reinterpret_tensor(primals_1, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((2304, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, reinterpret_tensor(primals_6, (2304, 2560), (2560, 1), 0), reinterpret_tensor(primals_4, (2560, 2560), (1, 2560), 0), alpha=1, beta=1, out=buf1)
        del primals_5
        buf2 = empty_strided_cuda((32, 72, 1), (72, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((32, 72, 1), (72, 1, 2304), torch.float32)
        buf5 = reinterpret_tensor(buf3, (32, 72, 1), (72, 1, 1), 0); del buf3  # reuse
        buf14 = empty_strided_cuda((32, 328, 2560), (839680, 2560, 1), torch.bfloat16)
        buf12 = reinterpret_tensor(buf14, (32, 72, 2560), (839680, 2560, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_0.run(buf5, buf1, primals_7, primals_8, buf2, buf12, 2304, 2560, stream=stream0)
        del primals_8
        buf6 = empty_strided_cuda((32, 256, 1), (256, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf9 = reinterpret_tensor(buf7, (32, 256, 1), (256, 1, 1), 0); del buf7  # reuse
        buf10 = empty_strided_cuda((32, 256, 2560), (655360, 2560, 1), torch.bfloat16)
        buf13 = reinterpret_tensor(buf14, (32, 256, 2560), (839680, 2560, 1), 184320)  # alias
        # Topologically Sorted Source Nodes: [latents, kv_input], Original ATen: [aten.native_layer_norm, aten.cat]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_native_layer_norm_1.run(buf9, buf0, primals_9, primals_10, buf6, buf10, buf13, 8192, 2560, stream=stream0)
        del primals_10
        buf11 = empty_strided_cuda((8192, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf10, (8192, 2560), (2560, 1), 0), reinterpret_tensor(primals_11, (2560, 128), (1, 2560), 0), out=buf11)
        buf15 = empty_strided_cuda((10496, 256), (256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (10496, 2560), (2560, 1), 0), reinterpret_tensor(primals_12, (2560, 256), (1, 2560), 0), out=buf15)
        buf16 = empty_strided_cuda((32, 4, 256, 32), (32768, 8192, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [mul, weight], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_2.run(buf11, buf16, 1048576, stream=stream0)
        buf17 = empty_strided_cuda((32, 4, 32, 328), (41984, 10496, 328, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [weight], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf15, buf17, 4096, 328, stream=stream0)
        buf18 = empty_strided_cuda((128, 256, 328), (83968, 328, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [weight], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (128, 256, 32), (8192, 32, 1), 0), reinterpret_tensor(buf17, (128, 32, 328), (10496, 328, 1), 0), out=buf18)
        buf19 = empty_strided_cuda((32, 4, 256, 1), (1024, 256, 1, 1), torch.float32)
        buf20 = empty_strided_cuda((32, 4, 256, 1), (1024, 256, 1, 1), torch.float32)
        buf21 = empty_strided_cuda((32, 4, 256, 328), (335872, 83968, 328, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [float_1, , sub, exp, softmax, weight_1], Original ATen: [aten._to_copy, prims.prepare_softmax_online, aten.sub, aten.exp, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax__to_copy_exp_prepare_softmax_online_sub_4.run(buf18, buf19, buf20, buf21, 32768, 328, stream=stream0)
        buf22 = empty_strided_cuda((32, 4, 328, 32), (41984, 10496, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf15, buf22, 1343488, stream=stream0)
        del buf15
        buf23 = reinterpret_tensor(buf11, (128, 256, 32), (8192, 32, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (128, 256, 328), (83968, 328, 1), 0), reinterpret_tensor(buf22, (128, 328, 32), (10496, 32, 1), 0), out=buf23)
        buf24 = empty_strided_cuda((32, 256, 4, 32), (32768, 128, 32, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf23, buf24, 1048576, stream=stream0)
        del buf23
        buf25 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf24, (8192, 128), (128, 1), 0), reinterpret_tensor(primals_13, (128, 2560), (1, 128), 0), out=buf25)
        buf26 = reinterpret_tensor(buf25, (32, 256, 2560), (655360, 2560, 1), 0); del buf25  # reuse
        buf27 = empty_strided_cuda((32, 256, 1), (256, 1, 1), torch.float32)
        buf28 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf30 = reinterpret_tensor(buf28, (32, 256, 1), (256, 1, 1), 0); del buf28  # reuse
        buf31 = empty_strided_cuda((32, 256, 2560), (655360, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [queries_1, input_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_7.run(buf26, buf30, buf0, primals_14, primals_15, buf27, buf31, 8192, 2560, stream=stream0)
        del primals_15
        buf32 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (8192, 2560), (2560, 1), 0), reinterpret_tensor(primals_16, (2560, 2560), (1, 2560), 0), out=buf32)
        buf33 = empty_strided_cuda((32, 256, 2560), (655360, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf32, buf33, 20971520, stream=stream0)
        buf34 = empty_strided_cuda((8192, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (8192, 2560), (2560, 1), 0), reinterpret_tensor(primals_17, (2560, 2560), (1, 2560), 0), out=buf34)
        buf35 = reinterpret_tensor(buf34, (32, 256, 2560), (655360, 2560, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [queries_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_9.run(buf35, buf26, 20971520, stream=stream0)
        buf36 = empty_strided_cuda((8192, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [queries_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf35, (8192, 2560), (2560, 1), 0), reinterpret_tensor(primals_18, (2560, 1024), (1, 2560), 0), alpha=1, beta=1, out=buf36)
        del primals_19
        buf37 = empty_strided_cuda((32, 256, 1), (256, 1, 1), torch.float32)
        buf38 = empty_strided_cuda((32, 256, 1), (256, 1, 8192), torch.float32)
        buf40 = reinterpret_tensor(buf38, (32, 256, 1), (256, 1, 1), 0); del buf38  # reuse
        buf41 = empty_strided_cuda((32, 256, 1024), (262144, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [queries_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_10.run(buf40, buf36, primals_20, primals_21, buf37, buf41, 8192, 1024, stream=stream0)
        del primals_21
    return (buf41, primals_1, primals_4, primals_7, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, reinterpret_tensor(primals_3, (8192, 2560), (2560, 1), 0), buf0, reinterpret_tensor(primals_6, (2304, 2560), (2560, 1), 0), buf1, buf2, buf5, buf6, buf9, reinterpret_tensor(buf10, (8192, 2560), (2560, 1), 0), reinterpret_tensor(buf14, (10496, 2560), (2560, 1), 0), buf18, buf19, buf20, reinterpret_tensor(buf24, (8192, 128), (128, 1), 0), buf26, buf27, buf30, reinterpret_tensor(buf31, (8192, 2560), (2560, 1), 0), buf32, reinterpret_tensor(buf33, (8192, 2560), (2560, 1), 0), reinterpret_tensor(buf35, (8192, 2560), (2560, 1), 0), buf36, buf37, buf40, reinterpret_tensor(buf21, (128, 328, 256), (83968, 1, 328), 0), reinterpret_tensor(buf22, (128, 32, 328), (10496, 1, 32), 0), reinterpret_tensor(buf16, (128, 32, 256), (8192, 1, 32), 0), reinterpret_tensor(buf17, (128, 328, 32), (10496, 1, 328), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((32, 256, 2560), (655360, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((32, 72, 2560), (184320, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((128, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((256, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((2560, 128), (128, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_15 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_17 = rand_strided((2560, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_18 = rand_strided((1024, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_20 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_21 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
