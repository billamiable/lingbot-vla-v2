# AOT ID: ['57_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/nl/cnl5efyxzua4npburqracgromqriphdujc7acfarf4zc7cms743l.py
# Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   out_emb => clone, view
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [9152, 4096]), kwargs = {})
triton_poi_fused__unsafe_view_clone_0 = async_compile.triton('triton_poi_fused__unsafe_view_clone_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 224919552}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37486592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4096*((x1 % 286)) + 1380352*(x1 // 286)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/mk/cmkt5ehpvicpaqux2rvh7zsmfwd54yipuxxolipkfnrr447ksade.py
# Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt, hidden_states_2, to_3, out_emb_2], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_1
#   hidden_states_1 => convert_element_type_2
#   hidden_states_2 => mul
#   out_emb_1 => add
#   out_emb_2 => mul_1
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_3 => convert_element_type_3
#   variance => mean
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %convert_element_type_3), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 73216, 'r0_': 187438080}}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 9152
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = 2560.0
        tmp14 = (tmp6 / tmp13)
        tmp15 = 1e-06
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp8 * tmp19
        tl.store(out_ptr1 + (r0_1 + 2560*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/kr/ckrwgzk7yqyudogkmsbdlbstlsw7kk4ol27r5gkgjmulqog7ghsh.py
# Topologically Sorted Source Nodes: [silu, mul_2, full], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   full => full_default
#   mul_2 => mul_3
#   silu => convert_element_type_6, convert_element_type_7, mul_2, sigmoid
# Graph fragment:
#   %convert_element_type_6 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_6,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %sigmoid), kwargs = {})
#   %convert_element_type_7 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %view_7), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %convert_element_type_7), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %view_7), kwargs = {})
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_5,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 286, 9728], 1), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default, %sigmoid_1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %sub), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_6, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %add_3), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %mul_7), kwargs = {})
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_2 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 1602551808}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 89030656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp4
    tmp9 = tmp7 * tmp5
    tmp10 = tl.sigmoid(tmp0)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp0 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp8, None)
    tl.store(out_ptr2 + (x0), tmp16, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/sm/csmi6g3nvjjwx7fnsfgxymbnjxqmjue2cj4mcy2nadnlfrz7mwlw.py
# Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt, hidden_states_2, to_3], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add_1
#   hidden_states_1 => convert_element_type_2
#   hidden_states_2 => mul
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_3 => convert_element_type_3
#   variance => mean
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %view_17), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %convert_element_type_3), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_10, [0, 1], True), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_3 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 188334080, 'r0_': 36608}}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 112640
    r0_numel = 208
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 2560)
    x1 = xindex // 2560
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_2 + 532480*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 2560*r0_2 + 532480*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 2560*r0_2 + 532480*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + 2560*r0_2 + 532480*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r0_2 + 208*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = 2560.0
        tmp9 = (tmp7 / tmp8)
        tmp10 = 1e-06
        tmp11 = tmp9 + tmp10
        tmp12 = libdevice.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp2 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/yh/cyhyi76vqyishedmdjjhus7kohjiyt47xoahvprkaxhwyuacsy32.py
# Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt, hidden_states_2, to_3], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add_1
#   hidden_states_1 => convert_element_type_2
#   hidden_states_2 => mul
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_3 => convert_element_type_3
#   variance => mean
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, %rsqrt), kwargs = {})
#   %convert_element_type_3 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %view_17), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %convert_element_type_3), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_10, [0, 1], True), kwargs = {})
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_sum_4 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_sum_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 460800, 'r0_': 0}}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2560
    r0_numel = 44
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2560*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/px/cpxnna6sqlvec7wgiocz5pgov2fe5atsl4ljebb2kdaf7cgig2zm.py
# Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add => add_1
#   hidden_states_1 => convert_element_type_2
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %view_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_4, %primals_4), kwargs = {})
#   %convert_element_type_24 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.float32), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %convert_element_type_2), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, %rsqrt), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_11, [2], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand, 2560), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_2, 1.0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_3, 2.0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_15), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_16), kwargs = {})
#   %convert_element_type_25 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_5, torch.bfloat16), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_25, %tangents_1), kwargs = {})
triton_red_fused__to_copy_add_div_mean_mul_pow_rsqrt_sum_5 = async_compile.triton('triton_red_fused__to_copy_add_div_mean_mul_pow_rsqrt_sum_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mean_mul_pow_rsqrt_sum_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 36608, 'r0_': 328012800}}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mean_mul_pow_rsqrt_sum_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 9152
    r0_numel = 2560
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp5 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr0 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr2 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr3 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr5 + (r0_1 + 2560*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = 2560.0
        tmp22 = (tmp20 / tmp21)
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp19 * tmp25
        tmp27 = -0.5
        tmp28 = tmp12 * tmp27
        tmp29 = tmp25 * tmp25
        tmp30 = tmp29 * tmp25
        tmp31 = tmp28 * tmp30
        tmp32 = 0.000390625
        tmp33 = tmp31 * tmp32
        tmp36 = tmp34 + tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = 2.0
        tmp39 = tmp37 * tmp38
        tmp40 = tmp33 * tmp39
        tmp41 = tmp26 + tmp40
        tmp42 = tmp41.to(tl.float32)
        tmp44 = tmp42 + tmp43
        tl.store(in_out_ptr0 + (r0_1 + 2560*x0), tmp44, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/vp/cvpmfnyq3gxlecs4kndibklz6jmhrcjgcpdozjxf3ycpfengonvf.py
# Topologically Sorted Source Nodes: [full_1, _generalized_scatter], Original ATen: [aten.slice_backward]
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
#   full_1 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 337, 4096], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_1, %view_20, 1, 0, 286), kwargs = {})
triton_poi_fused_slice_backward_6 = async_compile.triton('triton_poi_fused_slice_backward_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_backward_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 265027584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_slice_backward_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44171264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 337)
    x2 = xindex // 1380352
    x3 = (xindex % 1380352)
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 286, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x3 + 1171456*x2), tmp2, other=0.0).to(tl.float32)
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 286, 2560), (732160, 2560, 1))
    assert_size_stride(primals_2, (32, 337, 4096), (1380352, 4096, 1))
    assert_size_stride(primals_3, (2560, 4096), (4096, 1))
    assert_size_stride(primals_4, (2560, ), (1, ))
    assert_size_stride(primals_5, (9728, 2560), (2560, 1))
    assert_size_stride(primals_6, (9728, 2560), (2560, 1))
    assert_size_stride(primals_7, (2560, 9728), (9728, 1))
    assert_size_stride(tangents_1, (32, 286, 2560), (732160, 2560, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((9152, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_0.run(primals_2, buf0, 37486592, stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((9152, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4096, 2560), (1, 4096), 0), out=buf1)
        buf2 = empty_strided_cuda((32, 286, 1), (286, 1, 9152), torch.float32)
        buf3 = empty_strided_cuda((32, 286, 2560), (732160, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt, hidden_states_2, to_3, out_emb_2], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1.run(buf1, primals_1, primals_4, buf2, buf3, 9152, 2560, stream=stream0)
        buf4 = empty_strided_cuda((9152, 9728), (9728, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (9152, 2560), (2560, 1), 0), reinterpret_tensor(primals_5, (2560, 9728), (1, 2560), 0), out=buf4)
        buf5 = empty_strided_cuda((9152, 9728), (9728, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (9152, 2560), (2560, 1), 0), reinterpret_tensor(primals_6, (2560, 9728), (1, 2560), 0), out=buf5)
        buf8 = empty_strided_cuda((9152, 9728), (9728, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (9152, 2560), (2560, 1), 0), primals_7, out=buf8)
        del primals_7
        buf6 = empty_strided_cuda((32, 286, 9728), (2782208, 9728, 1), torch.bfloat16)
        buf9 = empty_strided_cuda((32, 286, 9728), (2782208, 9728, 1), torch.bfloat16)
        buf12 = empty_strided_cuda((32, 286, 9728), (2782208, 9728, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu, mul_2, full], Original ATen: [aten.silu, aten.mul, aten.sigmoid, aten.fill, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_2.run(buf4, buf5, buf8, buf6, buf9, buf12, 89030656, stream=stream0)
        del buf4
        del buf5
        del buf8
        buf7 = empty_strided_cuda((2560, 9728), (9728, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (2560, 9152), (1, 2560), 0), reinterpret_tensor(buf6, (9152, 9728), (9728, 1), 0), out=buf7)
        del buf6
        buf10 = empty_strided_cuda((9728, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (9728, 9152), (1, 9728), 0), reinterpret_tensor(buf3, (9152, 2560), (2560, 1), 0), out=buf10)
        buf11 = empty_strided_cuda((9152, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf9, (9152, 9728), (9728, 1), 0), primals_6, out=buf11)
        del buf9
        del primals_6
        buf13 = empty_strided_cuda((9728, 2560), (2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (9728, 9152), (1, 9728), 0), reinterpret_tensor(buf3, (9152, 2560), (2560, 1), 0), out=buf13)
        buf14 = reinterpret_tensor(buf3, (9152, 2560), (2560, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf12, (9152, 9728), (9728, 1), 0), primals_5, out=buf14)
        del buf12
        del primals_5
        buf15 = empty_strided_cuda((1, 1, 2560, 44), (112640, 112640, 1, 2560), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt, hidden_states_2, to_3], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_sum_3.run(buf11, buf14, buf1, primals_1, buf2, buf15, 112640, 208, stream=stream0)
        buf16 = empty_strided_cuda((1, 1, 2560), (2560, 2560, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt, hidden_states_2, to_3], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_sum_4.run(buf15, buf16, 2560, 44, stream=stream0)
        del buf15
        buf18 = reinterpret_tensor(buf11, (32, 286, 2560), (732160, 2560, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [out_emb_1, hidden_states_1, pow_1, variance, add, rsqrt], Original ATen: [aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_div_mean_mul_pow_rsqrt_sum_5.run(buf18, buf14, primals_4, buf1, primals_1, buf2, tangents_1, 9152, 2560, stream=stream0)
        del buf1
        del buf14
        del buf2
        del primals_1
        del primals_4
        del tangents_1
        buf19 = empty_strided_cuda((2560, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [permute_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (2560, 9152), (1, 2560), 0), buf0, out=buf19)
        buf20 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf18, (9152, 2560), (2560, 1), 0), primals_3, out=buf20)
        del primals_3
        buf21 = empty_strided_cuda((32, 337, 4096), (1380352, 4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [full_1, _generalized_scatter], Original ATen: [aten.slice_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_slice_backward_6.run(buf20, buf21, 44171264, stream=stream0)
        del buf20
    return (buf18, buf21, buf19, reinterpret_tensor(buf16, (2560, ), (1, ), 0), buf13, buf10, buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 286, 2560), (732160, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((32, 337, 4096), (1380352, 4096, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((2560, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((9728, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((9728, 2560), (2560, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_7 = rand_strided((2560, 9728), (9728, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32, 286, 2560), (732160, 2560, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
