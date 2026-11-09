# AOT ID: ['122_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/of/cofdby3xslsjq72phkmcnnhvecndqgm7tjuek4itxnpyoghcxgmg.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_10
# Graph fragment:
#   %triton_kernel_wrapper_mutation_10 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 308, grid: [(7, 1, 1)], tma_descriptor_metadata: {}, kwargs: {out_ptr: %full_default, x_ptr: %view_7}})
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/cx/ccxdirurfdyw5hi4ivwn2imis4jfuuiblr4ojkklnzmtd6gvms5z.py
# Topologically Sorted Source Nodes: [full_3], Original ATen: [aten._to_copy, aten.mul, aten.div, aten.add, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
#   full_3 => full_default_3
# Graph fragment:
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_25, torch.float32), kwargs = {})
#   %mul_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, 4.0), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_22, %add_6), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %expand), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1632, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_3, 1, %getitem_1, %add_14), kwargs = {})
triton_poi_fused__to_copy_add_div_mul_new_zeros_scatter_add_1 = async_compile.triton('triton_poi_fused__to_copy_add_div_mul_new_zeros_scatter_add_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_mul_new_zeros_scatter_add_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 417792}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_mul_new_zeros_scatter_add_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/hs/chsmaywxppppxzov74aumhuvkh6mjh5or55ycngw2sfxhqhtvhjm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_11
# Graph fragment:
#   %triton_kernel_wrapper_mutation_11 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 314, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %convert_element_type_11, O: %empty_7, index: %view_9}})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 10027008}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1253376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/mz/cmzs2eqiqj235yzxwwdy5m6r6dqe5h7kp6lgz4sao3zgovx6x3pm.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_9
# Graph fragment:
#   %triton_kernel_wrapper_mutation_9 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 315, grid: [(52, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %empty_7, b_ptr: %convert_element_type_6, c_ptr: %empty_8, cumsum_M: %cumsum}})
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/w2/cw2cafwpntntfcxaxsumwhjsnju6wayjrhiyh7wvrgizcb3h3iae.py
# Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   out_emb => clone, view
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1632, 4096]), kwargs = {})
triton_poi_fused__unsafe_view_clone_4 = async_compile.triton('triton_poi_fused__unsafe_view_clone_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 80216064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6684672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 4096)
    x1 = xindex // 4096
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1171456 + x0 + 4096*((x1 % 51)) + 1380352*(x1 // 51)), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ib/cibuph6uc6zbuxghapcj6cd7olgadlvy7tb2zwuoxhvhtbocaffn.py
# Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, hidden_states_3, add_1, mul_2, hidden_states_4, ], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_8
#   add => add_1
#   add_1 => add_2
#   hidden_states_2 => mul
#   hidden_states_3 => mul_1
#   hidden_states_4 => add_3
#   mul_2 => mul_2
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %mul), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %mul_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_1), kwargs = {})
#   %triton_kernel_wrapper_mutation_8 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 309, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %convert_element_type_2, O: %empty, index: %view_9}})
triton_per_fused_add_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13056, 'r0_': 45130752}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel):
    xnumel = 1632
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    x3 = xindex // 51
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 768*x3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp22 = tl.load(in_ptr5 + (r0_1 + 768*x3), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp10 = tmp8 + tmp9
    tmp11 = 1.0
    tmp12 = tmp10 + tmp11
    tmp14 = 768.0
    tmp15 = (tmp7 / tmp14)
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp2 * tmp18
    tmp20 = tmp13 * tmp19
    tmp21 = tmp12 * tmp20
    tmp24 = tmp22 + tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 768*x0), tmp25, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp26, r0_mask)
    tl.store(out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/7t/c7tsbnq3p6icfkioihk2aivjkoyyg35tlkgq3zmcehfnydjyorv2.py
# Topologically Sorted Source Nodes: [routing_scores, scores_for_choice], Original ATen: [aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   routing_scores => sigmoid
#   scores_for_choice => add_4
# Graph fragment:
#   %sigmoid : [num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, %unsqueeze_2), kwargs = {})
triton_poi_fused_add_sigmoid_6 = async_compile.triton('triton_poi_fused_add_sigmoid_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sigmoid_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 626816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sigmoid_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x2), tmp3, xmask)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/moe.py:26
_expert_histogram_kernel_0 = async_compile.triton('_expert_histogram_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_expert_histogram_kernel_0', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'out_ptr': '*i32', 'x_ptr': '*i64', 'num_elts': 'i32', 'num_bins': 'i32', 'NUM_BINS_LAST_UNUSED': 'constexpr', 'BLOCK_SIZE': 'constexpr', 'BLOCK_ALIGNED': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'NUM_BINS_LAST_UNUSED': 64, 'BLOCK_SIZE': 1024, 'BLOCK_ALIGNED': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _expert_histogram_kernel(
    out_ptr,
    x_ptr,
    num_elts,
    num_bins,
    NUM_BINS_LAST_UNUSED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ALIGNED: tl.constexpr,
):
    pid = tl.program_id(0)

    in_off = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = load_with_pred_1d(x_ptr + in_off, BLOCK_ALIGNED, in_off < num_elts, NUM_BINS_LAST_UNUSED - 1).to(tl.int32)

    tl.device_assert(
        data < num_bins or data == NUM_BINS_LAST_UNUSED - 1,
        "Out-of-bound element found.",
    )
    count = tl.histogram(data, NUM_BINS_LAST_UNUSED)

    out_off = tl.arange(0, NUM_BINS_LAST_UNUSED)
    tl.atomic_add(out_ptr + out_off, count, mask=out_off < num_bins, sem="relaxed")

@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/j6/cj6eu44hslr7z7xr3lqavuwq2ladzg36ao2zfzokjyi7e2yaeyyn.py
# Topologically Sorted Source Nodes: [one_hot, counts, to_8, add_], Original ATen: [aten.arange, aten.eq, aten._to_copy, aten.sum, aten.add, aten.copy_]
# Source node to ATen node mapping:
#   add_ => add_5
#   counts => sum_1
#   one_hot => convert_element_type, eq, iota
#   to_8 => convert_element_type_1
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_3, %iota), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq, torch.int64), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type, [0]), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_1, torch.float32), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_13, %convert_element_type_1), kwargs = {})
#   %copy__1 : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_13, %add_5), kwargs = {})
triton_red_fused__to_copy_add_arange_copy__eq_sum_7 = async_compile.triton('triton_red_fused__to_copy_add_arange_copy__eq_sum_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_arange_copy__eq_sum_7', 'mutated_arg_names': ['in_ptr1', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 384, 'r0_': 52224}}
)
@triton.jit
def triton_red_fused__to_copy_add_arange_copy__eq_sum_7(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 6528
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = x0
        tmp2 = tmp0 == tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tmp5.to(tl.float32)
    tmp9 = tmp7 + tmp8
    tl.store(out_ptr2 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ot/cotlfi7lhmdm75xwehgx7b2hujjwdoaclr6erfvfruh4rmed5lsz.py
# Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.cumsum]
# Source node to ATen node mapping:
#   final_hidden_states => cumsum
# Graph fragment:
#   %cumsum : [num_users=8] = call_function[target=torch.ops.aten.cumsum.default](args = (%full_default, 0), kwargs = {})
triton_per_fused_cumsum_8 = async_compile.triton('triton_per_fused_cumsum_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 512}}
)
@triton.jit
def triton_per_fused_cumsum_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp4, = tl.associative_scan((tmp3,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp4, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/kq/ckqtj7njanaurmrxfstz2ogt7563tgsimatcsiwqau2422374glt.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_11, triton_kernel_wrapper_mutation_8
# Graph fragment:
#   %triton_kernel_wrapper_mutation_11 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 314, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %convert_element_type_11, O: %empty_7, index: %view_9}})
#   %triton_kernel_wrapper_mutation_8 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 309, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %convert_element_type_2, O: %empty, index: %view_9}})
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 156672}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tl.store(out_ptr1 + (x0), tmp1, xmask)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/moe.py:250
_moe_scatter_kernel_1 = async_compile.triton('_moe_scatter_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_moe_scatter_kernel_1', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'O': '*bf16', 'index': '*i32', 'num_elts_in': 'i32', 'num_elts_out': 'i32', 'N': 'constexpr', 'TOPK': 'constexpr', 'STRIDE_XM': 'constexpr', 'STRIDE_XN': 'constexpr', 'STRIDE_OM': 'constexpr', 'STRIDE_ON': 'constexpr', 'STRIDE_IM': 'constexpr', 'STRIDE_IN': 'constexpr', 'BLOCK_N': 'constexpr', 'N_ALIGNED': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'N': 768, 'TOPK': 4, 'STRIDE_XM': 768, 'STRIDE_XN': 1, 'STRIDE_OM': 768, 'STRIDE_ON': 1, 'STRIDE_IM': 4, 'STRIDE_IN': 1, 'BLOCK_N': 1024, 'N_ALIGNED': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _moe_scatter_kernel(
    X,
    O,  # noqa
    index,
    num_elts_in,
    num_elts_out,
    N: tl.constexpr,  # hidden size
    TOPK: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XN: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    r"""
    X: m x n
    O: m * topk x n
    index: m x topk

    code:
        X: m x n -> repeat -> m x topk x n -> m * topk x n
            X[arange(m) * topk] = X[arange(m)]

        O[index] = X
            O[index[arange(m) * topk]] = X[arange(m) * topk]
    """

    pid_m = tl.program_id(axis=0)  # m
    block_idx = tl.program_id(axis=1)
    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    tl.device_assert(pid_m < num_elts_in, "Input OOB.")
    X = X + pid_m * STRIDE_XM + n * STRIDE_XN
    x = load_with_pred_1d(X, N_ALIGNED, mask=n < N, other=0)

    for i in tl.static_range(TOPK):
        o_index = tl.load(index + pid_m * STRIDE_IM + i * STRIDE_IN)
        tl.device_assert(o_index < num_elts_out, "Output OOB.")
        tmp_index = o_index.to(tl.int64) * STRIDE_OM
        # tl.device_print("tmp_index", tmp_index)
        out = O + tmp_index + n * STRIDE_ON

        # save one line
        store_with_pred_1d(out, x, N_ALIGNED, mask=n < N)

@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask: tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/group_gemm.py:53
group_gemm_same_nk_kernel_2 = async_compile.triton('group_gemm_same_nk_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'group_gemm_same_nk_kernel_2', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'a_ptr': '*bf16', 'b_ptr': '*bf16', 'c_ptr': '*bf16', 'act_ptr': 'constexpr', 'cumsum_M': '*i64', 'max_M': 'i32', 'total_M': 'i32', 'G': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'TRANSPOSE_A': 'constexpr', 'TRANSPOSE_B': 'constexpr', 'ACCUMULATE_TO_C': 'constexpr', 'GROUP': 'constexpr', 'N_ALIGNED': 'constexpr', 'K_ALIGNED': 'constexpr', 'ACTIVATION': 'constexpr', 'HAS_ACTIVATION': 'constexpr', 'SAVE_ACTIVATION': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'act_ptr': None, 'G': 32, 'N': 512, 'K': 768, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'TRANSPOSE_A': False, 'TRANSPOSE_B': False, 'ACCUMULATE_TO_C': False, 'GROUP': 8, 'N_ALIGNED': True, 'K_ALIGNED': True, 'ACTIVATION': None, 'HAS_ACTIVATION': False, 'SAVE_ACTIVATION': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def group_gemm_same_nk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    act_ptr,
    cumsum_M,
    max_M,
    total_M,  # Used for generating algo. key only.
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    # No need to support TRANPOSE_C, just ask user to calculate `c.t()` as `b.t() @ a.t()`.
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    K_ALIGNED: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
    SAVE_ACTIVATION: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_M + gid)
    m_size = (gtid_end - gtid_start).to(tl.uint64)

    if m * BLOCK_M >= m_size:
        return

    a_ptr += gtid_start * K
    b_ptr += gid * K * N
    c_ptr += gtid_start * N

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_am = offs_m % m_size.to(tl.int64)
    offs_bn = offs_n % N

    blk_k = tl.arange(0, BLOCK_K)

    stride_am, stride_ak = (K, 1) if not TRANSPOSE_A else (1, m_size)
    stride_bk, stride_bn = (N, 1) if not TRANSPOSE_B else (1, K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + blk_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (blk_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + N * offs_m[:, None] + 1 * offs_n[None, :]

    if ACCUMULATE_TO_C:
        c = load_with_pred_2d(
            c_ptrs,
            False,
            N_ALIGNED,
            offs_m[:, None] < m_size,
            offs_n[None, :] < N,
            other=0,
        )
    else:
        c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Really loading a 2D block. Using `load_with_pred_1d` as we only have one predicate.
        a = load_with_pred_1d(a_ptrs, K_ALIGNED, blk_k[None, :] < K - k * BLOCK_K, other=0)
        b = load_with_pred_1d(b_ptrs, K_ALIGNED, blk_k[:, None] < K - k * BLOCK_K, other=0)

        c = tl.dot(a, b, c)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_ACTIVATION:
        # Makes GELU_APPROX faster, not sure why..
        c = make_blocked(c, c_ptr.dtype.element_ty)
        if SAVE_ACTIVATION:
            store_with_pred_2d(
                act_ptr + gtid_start * N + N * offs_m[:, None] + offs_n[None, :],
                c,
                False,
                N_ALIGNED,
                offs_m[:, None] < m_size,
                offs_n[None, :] < N,
            )
        c = activation_fwd(c, ACTIVATION)

    store_with_pred_2d(c_ptrs, c, False, N_ALIGNED, offs_m[:, None] < m_size, offs_n[None, :] < N)

@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def load_with_pred_2d(
    ptr,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
    other=0,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_0 and mask_1, other=other)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, mask_0, other=other)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_1, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def make_blocked(t: tl.tensor, intermediate_type: tl.dtype) -> tl.tensor:
    """Forcibly convert tensor (from "mma" layout) into "blocked" layout.

    `intermediate_type` affects performance. Usually `tl.bfloat16` or `tl.float16` should be used.
    INTERNALLY `t` IS CONVERTED TO `intermediate_type` AND BACK  SO THE PRECISION CAN DROP.

    ATM Triton does such conversion prior to storing tensor into global memory. This usually doesn't
    matter as we usually only store the accumulator once. However, if we'd like to perform some
    element-wise operation on the accumulator and save both pre-op and post-op results, Triton will
    do the conversion twice, and hence hurt performance.

    In such cases, forcibly convert tensor eagerly can help performance. This is not guaranteed, so
    be sure to benchmark before applying this "optimization".

    NOTE: Once Triton can optimize away multiple layout conversions, this hack should be removed.
    """
    # This really relies on Triton's internal implementation.. See implementation of `expand_dims`
    # op, it triggers emission of `triton_gpu.convert_layout`.
    return t.to(intermediate_type).expand_dims(0).reshape(t.shape)

@triton.jit
def store_with_pred_2d(
    ptr,
    value,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_0 and mask_1)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, mask_0)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_1)
    else:
        tl.store(ptr, value)

@triton.jit
def activation_fwd(x: tl.tensor, ACTIVATION: tl.constexpr):
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    if ACTIVATION == "gelu":
        y = gelu(x)
    elif ACTIVATION == "gelu_new":
        y = gelu_new(x)
    elif ACTIVATION == "silu":
        y = silu(x)
    else:
        tl.static_assert(False, f"Unsupported activation of {ACTIVATION}")
    return y.to(orig_dtype)

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    x = x.to(tl.float32)
    return x * 0.5 * (1.0 + tl.erf(x * _sqrt1_2))

@triton.jit
def gelu_new(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def silu(x):
    """https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html"""
    x = x.to(tl.float32)
    return x * tl.sigmoid(x)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/dz/cdzjwoprwfnxp4ahndm5mkr5bonz24h4sduaitewskvpwtqvfq6n.py
# Topologically Sorted Source Nodes: [silu, mul_4, full_1], Original ATen: [aten.silu, aten.mul, aten.fill, aten.sub, aten.add]
# Source node to ATen node mapping:
#   full_1 => full_default_1
#   mul_4 => mul_8
#   silu => mul_7, sigmoid_2
# Graph fragment:
#   %sigmoid_2 : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_2,), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, %sigmoid_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %mm_3), kwargs = {})
#   %mul_9 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_6, %mul_7), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_6, %mm_3), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1632, 704], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%full_default_1, %sigmoid_2), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, %sub), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mul_11, 1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_2, %add_9), kwargs = {})
#   %mul_13 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %mul_12), kwargs = {})
triton_poi_fused_add_fill_mul_silu_sub_10 = async_compile.triton('triton_poi_fused_add_fill_mul_silu_sub_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_silu_sub_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 41361408}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_fill_mul_silu_sub_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1148928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp5 * tmp2
    tmp7 = tmp5 * tmp3
    tmp8 = 1.0
    tmp9 = tmp8 - tmp1
    tmp10 = tmp0 * tmp9
    tmp11 = tmp10 + tmp8
    tmp12 = tmp1 * tmp11
    tmp13 = tmp7 * tmp12
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr2 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/y6/cy6vddjoejtr4xx3dr2s76vpoujnfuweggngl2tb3td27f3upzrp.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_2, triton_kernel_wrapper_mutation_7
# Graph fragment:
#   %triton_kernel_wrapper_mutation_7 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 310, grid: [(204, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %empty, b_ptr: %convert_element_type_4, c_ptr: %empty_1, cumsum_M: %cumsum}})
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 319, grid: [(78, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %mul_21, b_ptr: %convert_element_type_4, c_ptr: %empty_13, cumsum_M: %cumsum}})
triton_poi_fused_11 = async_compile.triton('triton_poi_fused_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 150994944}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_11(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
    tl.store(out_ptr1 + (x0), tmp1, None)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/group_gemm.py:53
group_gemm_same_nk_kernel_3 = async_compile.triton('group_gemm_same_nk_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'group_gemm_same_nk_kernel_3', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'a_ptr': '*bf16', 'b_ptr': '*bf16', 'c_ptr': '*bf16', 'act_ptr': 'constexpr', 'cumsum_M': '*i64', 'max_M': 'i32', 'total_M': 'i32', 'G': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'TRANSPOSE_A': 'constexpr', 'TRANSPOSE_B': 'constexpr', 'ACCUMULATE_TO_C': 'constexpr', 'GROUP': 'constexpr', 'N_ALIGNED': 'constexpr', 'K_ALIGNED': 'constexpr', 'ACTIVATION': 'constexpr', 'HAS_ACTIVATION': 'constexpr', 'SAVE_ACTIVATION': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'act_ptr': None, 'G': 32, 'N': 512, 'K': 768, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'TRANSPOSE_A': False, 'TRANSPOSE_B': True, 'ACCUMULATE_TO_C': False, 'GROUP': 8, 'N_ALIGNED': True, 'K_ALIGNED': True, 'ACTIVATION': None, 'HAS_ACTIVATION': False, 'SAVE_ACTIVATION': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def group_gemm_same_nk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    act_ptr,
    cumsum_M,
    max_M,
    total_M,  # Used for generating algo. key only.
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    # No need to support TRANPOSE_C, just ask user to calculate `c.t()` as `b.t() @ a.t()`.
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    K_ALIGNED: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
    SAVE_ACTIVATION: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_M + gid)
    m_size = (gtid_end - gtid_start).to(tl.uint64)

    if m * BLOCK_M >= m_size:
        return

    a_ptr += gtid_start * K
    b_ptr += gid * K * N
    c_ptr += gtid_start * N

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_am = offs_m % m_size.to(tl.int64)
    offs_bn = offs_n % N

    blk_k = tl.arange(0, BLOCK_K)

    stride_am, stride_ak = (K, 1) if not TRANSPOSE_A else (1, m_size)
    stride_bk, stride_bn = (N, 1) if not TRANSPOSE_B else (1, K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + blk_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (blk_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + N * offs_m[:, None] + 1 * offs_n[None, :]

    if ACCUMULATE_TO_C:
        c = load_with_pred_2d(
            c_ptrs,
            False,
            N_ALIGNED,
            offs_m[:, None] < m_size,
            offs_n[None, :] < N,
            other=0,
        )
    else:
        c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Really loading a 2D block. Using `load_with_pred_1d` as we only have one predicate.
        a = load_with_pred_1d(a_ptrs, K_ALIGNED, blk_k[None, :] < K - k * BLOCK_K, other=0)
        b = load_with_pred_1d(b_ptrs, K_ALIGNED, blk_k[:, None] < K - k * BLOCK_K, other=0)

        c = tl.dot(a, b, c)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_ACTIVATION:
        # Makes GELU_APPROX faster, not sure why..
        c = make_blocked(c, c_ptr.dtype.element_ty)
        if SAVE_ACTIVATION:
            store_with_pred_2d(
                act_ptr + gtid_start * N + N * offs_m[:, None] + offs_n[None, :],
                c,
                False,
                N_ALIGNED,
                offs_m[:, None] < m_size,
                offs_n[None, :] < N,
            )
        c = activation_fwd(c, ACTIVATION)

    store_with_pred_2d(c_ptrs, c, False, N_ALIGNED, offs_m[:, None] < m_size, offs_n[None, :] < N)

@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def load_with_pred_2d(
    ptr,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
    other=0,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_0 and mask_1, other=other)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, mask_0, other=other)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_1, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def make_blocked(t: tl.tensor, intermediate_type: tl.dtype) -> tl.tensor:
    """Forcibly convert tensor (from "mma" layout) into "blocked" layout.

    `intermediate_type` affects performance. Usually `tl.bfloat16` or `tl.float16` should be used.
    INTERNALLY `t` IS CONVERTED TO `intermediate_type` AND BACK  SO THE PRECISION CAN DROP.

    ATM Triton does such conversion prior to storing tensor into global memory. This usually doesn't
    matter as we usually only store the accumulator once. However, if we'd like to perform some
    element-wise operation on the accumulator and save both pre-op and post-op results, Triton will
    do the conversion twice, and hence hurt performance.

    In such cases, forcibly convert tensor eagerly can help performance. This is not guaranteed, so
    be sure to benchmark before applying this "optimization".

    NOTE: Once Triton can optimize away multiple layout conversions, this hack should be removed.
    """
    # This really relies on Triton's internal implementation.. See implementation of `expand_dims`
    # op, it triggers emission of `triton_gpu.convert_layout`.
    return t.to(intermediate_type).expand_dims(0).reshape(t.shape)

@triton.jit
def store_with_pred_2d(
    ptr,
    value,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_0 and mask_1)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, mask_0)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_1)
    else:
        tl.store(ptr, value)

@triton.jit
def activation_fwd(x: tl.tensor, ACTIVATION: tl.constexpr):
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    if ACTIVATION == "gelu":
        y = gelu(x)
    elif ACTIVATION == "gelu_new":
        y = gelu_new(x)
    elif ACTIVATION == "silu":
        y = silu(x)
    else:
        tl.static_assert(False, f"Unsupported activation of {ACTIVATION}")
    return y.to(orig_dtype)

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    x = x.to(tl.float32)
    return x * 0.5 * (1.0 + tl.erf(x * _sqrt1_2))

@triton.jit
def gelu_new(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def silu(x):
    """https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html"""
    x = x.to(tl.float32)
    return x * tl.sigmoid(x)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/rd/crdpxqya5k5kkkpjj3rmfktblwjecucvup5syrmgpdvsyzolywvx.py
# Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.silu, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   final_hidden_states => convert_element_type_8, convert_element_type_9, mul_4, mul_5, sigmoid_1
# Graph fragment:
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%empty_1, torch.float32), kwargs = {})
#   %sigmoid_1 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_8,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %sigmoid_1), kwargs = {})
#   %convert_element_type_9 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_9, %empty_2), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %empty_8), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [-1]), kwargs = {})
triton_per_fused_mul_silu_sum_12 = async_compile.triton('triton_per_fused_mul_silu_sum_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_silu_sum_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26112, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_mul_silu_sum_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel):
    xnumel = 6528
    XBLOCK: tl.constexpr = 1
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r0_1 + 512*x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tl.store(out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/nf/cnfhj5uaho46eeneffqoaphpov243zj3xrmx3nloywyqicjytxat.py
# Topologically Sorted Source Nodes: [routing_scores, routing_weights, sum_2, add_4, routing_weights_1], Original ATen: [aten.sigmoid, aten.gather, aten.sum, aten.add, aten.div, aten._to_copy, aten.mul, aten.neg]
# Source node to ATen node mapping:
#   add_4 => add_6
#   routing_scores => sigmoid
#   routing_weights => gather
#   routing_weights_1 => div
#   sum_2 => sum_2
# Graph fragment:
#   %sigmoid : [num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %gather : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gather, [-1], True), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-20), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather, %add_6), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_25, torch.float32), kwargs = {})
#   %mul_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, 4.0), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %add_6), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_22,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %div_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [1], True), kwargs = {})
triton_poi_fused__to_copy_add_div_gather_mul_neg_sigmoid_sum_13 = async_compile.triton('triton_poi_fused__to_copy_add_div_gather_mul_neg_sigmoid_sum_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_gather_mul_neg_sigmoid_sum_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_gather_mul_neg_sigmoid_sum_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 32")
    tmp6 = tl.load(in_ptr1 + (tmp4 + 32*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.sigmoid(tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp8 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp8)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 32")
    tmp13 = tl.load(in_ptr1 + (tmp11 + 32*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp7 + tmp14
    tmp17 = tmp16 + tmp1
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tl.device_assert(((0 <= tmp19) & (tmp19 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp19 < 32")
    tmp21 = tl.load(in_ptr1 + (tmp19 + 32*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp15 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 32")
    tmp29 = tl.load(in_ptr1 + (tmp27 + 32*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp23 + tmp30
    tmp32 = 1e-20
    tmp33 = tmp31 + tmp32
    tmp35 = tmp34.to(tl.int32)
    tmp36 = tl.full([XBLOCK], 6528, tl.int32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tl.device_assert(((0 <= tmp39) & (tmp39 < 6528)) | ~(xmask), "index out of bounds: 0 <= tmp39 < 6528")
    tmp41 = tl.load(in_ptr3 + (tmp39), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = 4.0
    tmp44 = tmp42 * tmp43
    tmp45 = -tmp44
    tmp46 = (tmp7 / tmp33)
    tmp47 = (tmp46 / tmp33)
    tmp48 = tmp45 * tmp47
    tmp50 = tmp49.to(tl.int32)
    tmp51 = tmp50 + tmp36
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tl.device_assert(((0 <= tmp53) & (tmp53 < 6528)) | ~(xmask), "index out of bounds: 0 <= tmp53 < 6528")
    tmp55 = tl.load(in_ptr3 + (tmp53), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp43
    tmp58 = -tmp57
    tmp59 = (tmp14 / tmp33)
    tmp60 = (tmp59 / tmp33)
    tmp61 = tmp58 * tmp60
    tmp62 = tmp48 + tmp61
    tmp64 = tmp63.to(tl.int32)
    tmp65 = tmp64 + tmp36
    tmp66 = tmp64 < 0
    tmp67 = tl.where(tmp66, tmp65, tmp64)
    tl.device_assert(((0 <= tmp67) & (tmp67 < 6528)) | ~(xmask), "index out of bounds: 0 <= tmp67 < 6528")
    tmp69 = tl.load(in_ptr3 + (tmp67), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp70 * tmp43
    tmp72 = -tmp71
    tmp73 = (tmp22 / tmp33)
    tmp74 = (tmp73 / tmp33)
    tmp75 = tmp72 * tmp74
    tmp76 = tmp62 + tmp75
    tmp78 = tmp77.to(tl.int32)
    tmp79 = tmp78 + tmp36
    tmp80 = tmp78 < 0
    tmp81 = tl.where(tmp80, tmp79, tmp78)
    tl.device_assert(((0 <= tmp81) & (tmp81 < 6528)) | ~(xmask), "index out of bounds: 0 <= tmp81 < 6528")
    tmp83 = tl.load(in_ptr3 + (tmp81), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp84 = tmp83.to(tl.float32)
    tmp85 = tmp84 * tmp43
    tmp86 = -tmp85
    tmp87 = (tmp30 / tmp33)
    tmp88 = (tmp87 / tmp33)
    tmp89 = tmp86 * tmp88
    tmp90 = tmp76 + tmp89
    tl.store(out_ptr0 + (x0), tmp33, xmask)
    tl.store(out_ptr1 + (x0), tmp90, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/lf/clfzszqynwzxgaafirq6apl6xeyb6na4nkqnpxvpnm2vbs6ha4r2.py
# Topologically Sorted Source Nodes: [final_hidden_states, , full_3], Original ATen: [aten.index_put, aten._to_copy, aten.mul, aten.div, aten.add, aten.new_zeros, aten.scatter_add]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation
#   final_hidden_states => index_put
#   full_3 => full_default_3
# Graph fragment:
#   %index_put : [num_users=2] = call_function[target=torch.ops.aten.index_put_.default](args = (%permute_4, [%view_11], %view_10), kwargs = {})
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 3, constant_args_idx: 321, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %add_12, Y: %empty_15, index: %view_9}})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_25, torch.float32), kwargs = {})
#   %mul_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, 4.0), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_22, %add_6), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %expand), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1632, 32], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter_add : [num_users=1] = call_function[target=torch.ops.aten.scatter_add.default](args = (%full_default_3, 1, %getitem_1, %add_14), kwargs = {})
triton_poi_fused__to_copy_add_div_index_put_mul_new_zeros_scatter_add_14 = async_compile.triton('triton_poi_fused__to_copy_add_div_index_put_mul_new_zeros_scatter_add_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'out_ptr0': '*i32', 'out_ptr1': '*bf16', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_div_index_put_mul_new_zeros_scatter_add_14', 'mutated_arg_names': ['out_ptr1', 'out_ptr2'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_div_index_put_mul_new_zeros_scatter_add_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask)
    tmp15 = tl.load(in_ptr3 + (x0 // 4), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int32)
    tmp2 = tl.full([XBLOCK], 6528, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 6528)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 6528")
    tmp8 = tl.full([XBLOCK], 32, tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert(((0 <= tmp11) & (tmp11 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp11 < 32")
    tmp13 = tl.load(in_ptr2 + (tmp11 + 32*(x0 // 4)), xmask, eviction_policy='evict_last')
    tmp14 = tl.sigmoid(tmp13)
    tmp16 = (tmp14 / tmp15)
    tmp17 = 4.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tl.device_assert(((0 <= tmp7) & (tmp7 < 32)) | ~(xmask), "index out of bounds: 0 <= tmp7 < 32")
    tmp21 = tl.load(in_ptr4 + (tmp5), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 * tmp17
    tmp25 = (tmp23 / tmp24)
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr0 + (x0), tmp1, xmask)
    tl.store(out_ptr1 + (tl.broadcast_to(tmp5, [XBLOCK])), tmp19, xmask)
    tl.atomic_add(out_ptr2 + (tmp7 + 32*x2), tmp27, xmask, sem='relaxed')
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ks/cksctvgjth7mh62pa7zo4ltod7syhxv73hbdboqptffcocvanhqd.py
# Topologically Sorted Source Nodes: [routing_scores], Original ATen: [aten.sigmoid, aten.sigmoid_backward, aten.add]
# Source node to ATen node mapping:
#   routing_scores => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=4] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub_2), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_add, %mul_24), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %mul_25), kwargs = {})
triton_poi_fused_add_sigmoid_sigmoid_backward_15 = async_compile.triton('triton_poi_fused_add_sigmoid_sigmoid_backward_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sigmoid_sigmoid_backward_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 835584}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sigmoid_sigmoid_backward_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp6 = tmp3 * tmp5
    tmp7 = tmp1 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/yk/cykbh535joumtsjxn3n6f5ddbmimwo3lknfpk7566rd7yezdjiqb.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_1, triton_kernel_wrapper_mutation_2, triton_kernel_wrapper_mutation_3, triton_kernel_wrapper_mutation_4, triton_kernel_wrapper_mutation_5
# Graph fragment:
#   %triton_kernel_wrapper_mutation_5 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 4, constant_args_idx: 316, grid: [(24, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %empty_7, b_ptr: %mul_6, c_ptr: %permute_22, cumsum_K: %cumsum}})
#   %triton_kernel_wrapper_mutation_4 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 317, grid: [(78, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %mul_18, b_ptr: %convert_element_type_5, c_ptr: %empty_10, cumsum_M: %cumsum}})
#   %triton_kernel_wrapper_mutation_3 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 4, constant_args_idx: 318, grid: [(24, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %mul_18, b_ptr: %empty, c_ptr: %permute_23, cumsum_K: %cumsum}})
#   %triton_kernel_wrapper_mutation_2 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 319, grid: [(78, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %mul_21, b_ptr: %convert_element_type_4, c_ptr: %empty_13, cumsum_M: %cumsum}})
#   %triton_kernel_wrapper_mutation_1 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 4, constant_args_idx: 320, grid: [(24, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %mul_21, b_ptr: %empty, c_ptr: %permute_25, cumsum_K: %cumsum}})
triton_poi_fused_16 = async_compile.triton('triton_poi_fused_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'out_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 66846720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3342336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp7
    tmp11 = tmp4 * tmp10
    tmp12 = tmp10 * tmp5
    tmp13 = tl.sigmoid(tmp0)
    tmp14 = 1.0
    tmp15 = tmp14 - tmp13
    tmp16 = tmp0 * tmp15
    tmp17 = tmp16 + tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp12 * tmp18
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp11, None)
    tl.store(out_ptr2 + (x2), tmp11, None)
    tl.store(out_ptr3 + (x2), tmp19, None)
    tl.store(out_ptr4 + (x2), tmp19, None)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/group_gemm.py:53
group_gemm_same_nk_kernel_4 = async_compile.triton('group_gemm_same_nk_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'group_gemm_same_nk_kernel_4', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'a_ptr': '*bf16', 'b_ptr': '*bf16', 'c_ptr': '*bf16', 'act_ptr': 'constexpr', 'cumsum_M': '*i64', 'max_M': 'i32', 'total_M': 'i32', 'G': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'TRANSPOSE_A': 'constexpr', 'TRANSPOSE_B': 'constexpr', 'ACCUMULATE_TO_C': 'constexpr', 'GROUP': 'constexpr', 'N_ALIGNED': 'constexpr', 'K_ALIGNED': 'constexpr', 'ACTIVATION': 'constexpr', 'HAS_ACTIVATION': 'constexpr', 'SAVE_ACTIVATION': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'act_ptr': None, 'G': 32, 'N': 768, 'K': 512, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'TRANSPOSE_A': False, 'TRANSPOSE_B': False, 'ACCUMULATE_TO_C': False, 'GROUP': 8, 'N_ALIGNED': True, 'K_ALIGNED': True, 'ACTIVATION': None, 'HAS_ACTIVATION': False, 'SAVE_ACTIVATION': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def group_gemm_same_nk_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    act_ptr,
    cumsum_M,
    max_M,
    total_M,  # Used for generating algo. key only.
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    # No need to support TRANPOSE_C, just ask user to calculate `c.t()` as `b.t() @ a.t()`.
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    N_ALIGNED: tl.constexpr,
    K_ALIGNED: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_ACTIVATION: tl.constexpr,
    SAVE_ACTIVATION: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), max_M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_M + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_M + gid)
    m_size = (gtid_end - gtid_start).to(tl.uint64)

    if m * BLOCK_M >= m_size:
        return

    a_ptr += gtid_start * K
    b_ptr += gid * K * N
    c_ptr += gtid_start * N

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_am = offs_m % m_size.to(tl.int64)
    offs_bn = offs_n % N

    blk_k = tl.arange(0, BLOCK_K)

    stride_am, stride_ak = (K, 1) if not TRANSPOSE_A else (1, m_size)
    stride_bk, stride_bn = (N, 1) if not TRANSPOSE_B else (1, K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + blk_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (blk_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    c_ptrs = c_ptr + N * offs_m[:, None] + 1 * offs_n[None, :]

    if ACCUMULATE_TO_C:
        c = load_with_pred_2d(
            c_ptrs,
            False,
            N_ALIGNED,
            offs_m[:, None] < m_size,
            offs_n[None, :] < N,
            other=0,
        )
    else:
        c = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Really loading a 2D block. Using `load_with_pred_1d` as we only have one predicate.
        a = load_with_pred_1d(a_ptrs, K_ALIGNED, blk_k[None, :] < K - k * BLOCK_K, other=0)
        b = load_with_pred_1d(b_ptrs, K_ALIGNED, blk_k[:, None] < K - k * BLOCK_K, other=0)

        c = tl.dot(a, b, c)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_ACTIVATION:
        # Makes GELU_APPROX faster, not sure why..
        c = make_blocked(c, c_ptr.dtype.element_ty)
        if SAVE_ACTIVATION:
            store_with_pred_2d(
                act_ptr + gtid_start * N + N * offs_m[:, None] + offs_n[None, :],
                c,
                False,
                N_ALIGNED,
                offs_m[:, None] < m_size,
                offs_n[None, :] < N,
            )
        c = activation_fwd(c, ACTIVATION)

    store_with_pred_2d(c_ptrs, c, False, N_ALIGNED, offs_m[:, None] < m_size, offs_n[None, :] < N)

@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def load_with_pred_2d(
    ptr,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
    other=0,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_0 and mask_1, other=other)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, mask_0, other=other)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, mask_1, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def make_blocked(t: tl.tensor, intermediate_type: tl.dtype) -> tl.tensor:
    """Forcibly convert tensor (from "mma" layout) into "blocked" layout.

    `intermediate_type` affects performance. Usually `tl.bfloat16` or `tl.float16` should be used.
    INTERNALLY `t` IS CONVERTED TO `intermediate_type` AND BACK  SO THE PRECISION CAN DROP.

    ATM Triton does such conversion prior to storing tensor into global memory. This usually doesn't
    matter as we usually only store the accumulator once. However, if we'd like to perform some
    element-wise operation on the accumulator and save both pre-op and post-op results, Triton will
    do the conversion twice, and hence hurt performance.

    In such cases, forcibly convert tensor eagerly can help performance. This is not guaranteed, so
    be sure to benchmark before applying this "optimization".

    NOTE: Once Triton can optimize away multiple layout conversions, this hack should be removed.
    """
    # This really relies on Triton's internal implementation.. See implementation of `expand_dims`
    # op, it triggers emission of `triton_gpu.convert_layout`.
    return t.to(intermediate_type).expand_dims(0).reshape(t.shape)

@triton.jit
def store_with_pred_2d(
    ptr,
    value,
    skip_boundary_check_0: tl.constexpr,
    skip_boundary_check_1: tl.constexpr,
    mask_0: tl.tensor,
    mask_1: tl.tensor,
):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_0 and mask_1)
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, mask_0)
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, mask_1)
    else:
        tl.store(ptr, value)

@triton.jit
def activation_fwd(x: tl.tensor, ACTIVATION: tl.constexpr):
    orig_dtype = x.dtype
    x = x.to(tl.float32)
    if ACTIVATION == "gelu":
        y = gelu(x)
    elif ACTIVATION == "gelu_new":
        y = gelu_new(x)
    elif ACTIVATION == "silu":
        y = silu(x)
    else:
        tl.static_assert(False, f"Unsupported activation of {ACTIVATION}")
    return y.to(orig_dtype)

@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    x = x.to(tl.float32)
    return x * 0.5 * (1.0 + tl.erf(x * _sqrt1_2))

@triton.jit
def gelu_new(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(_sqrt2pi * x * (1.0 + 0.044715 * x * x)))

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def silu(x):
    """https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html"""
    x = x.to(tl.float32)
    return x * tl.sigmoid(x)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/group_gemm.py:240
group_gemm_same_mn_kernel_5 = async_compile.triton('group_gemm_same_mn_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'group_gemm_same_mn_kernel_5', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'a_ptr': '*bf16', 'b_ptr': '*bf16', 'c_ptr': '*bf16', 'cumsum_K': '*i64', 'total_K': 'i32', 'G': 'constexpr', 'M': 'constexpr', 'N': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'TRANSPOSE_A': 'constexpr', 'TRANSPOSE_B': 'constexpr', 'ACCUMULATE_TO_C': 'constexpr', 'GROUP': 'constexpr', 'M_ALIGNED': 'constexpr', 'N_ALIGNED': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'G': 32, 'M': 768, 'N': 512, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'TRANSPOSE_A': True, 'TRANSPOSE_B': False, 'ACCUMULATE_TO_C': False, 'GROUP': 8, 'M_ALIGNED': True, 'N_ALIGNED': True}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def group_gemm_same_mn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    cumsum_K,
    total_K,  # Used for generating algo. key only.
    G: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    M_ALIGNED: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_K + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_K + gid)
    k = (gtid_end - gtid_start).to(tl.uint64)

    if TRANSPOSE_A:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + gtid_start * M,
            shape=(M, k),
            strides=(1, M),
            offsets=(m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(0, 1),
        )
    else:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + gtid_start * M,
            shape=(M, k),
            strides=(k, 1),
            offsets=(m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
    if TRANSPOSE_B:
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + gtid_start * N,
            shape=(k, N),
            strides=(1, k),
            offsets=(0, n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
    else:
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + gtid_start * N,
            shape=(k, N),
            strides=(N, 1),
            offsets=(0, n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + gid * M * N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(m * BLOCK_M, n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Special case: no GEMM needed.
    if k == 0:
        if not ACCUMULATE_TO_C:
            # Zero out the corresponding output region.
            store_block_with_pred_2d(
                c_block_ptr,
                # tl.zeros(..., dtype=c_block_ptr.dtype.element_ty) raises "not implemented".
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(c_block_ptr.dtype.element_ty),
                M_ALIGNED,
                N_ALIGNED,
            )
        else:
            # Nothing to do then, just leave the kernel.
            pass

        return

    if ACCUMULATE_TO_C:
        out = tl.load(c_block_ptr).to(tl.float32)
    else:
        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # FIXME: Weird type conversion.
    for _ in range(tl.cdiv(k.to(tl.int64), BLOCK_K)):
        a = load_block_with_pred_2d(a_block_ptr, M_ALIGNED, False)
        b = load_block_with_pred_2d(b_block_ptr, False, N_ALIGNED)

        out += tl.dot(a, b)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    store_block_with_pred_2d(c_block_ptr, out.to(c_block_ptr.dtype.element_ty), M_ALIGNED, N_ALIGNED)

@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def store_block_with_pred_2d(ptr, value, skip_boundary_check_0: tl.constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(1,))
    else:
        tl.store(ptr, value)

@triton.jit
def load_block_with_pred_2d(ptr, skip_boundary_check_0: tl.constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(1,))
    else:
        return tl.load(ptr)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/group_gemm.py:240
group_gemm_same_mn_kernel_6 = async_compile.triton('group_gemm_same_mn_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': 'group_gemm_same_mn_kernel_6', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'a_ptr': '*bf16', 'b_ptr': '*bf16', 'c_ptr': '*bf16', 'cumsum_K': '*i64', 'total_K': 'i32', 'G': 'constexpr', 'M': 'constexpr', 'N': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'TRANSPOSE_A': 'constexpr', 'TRANSPOSE_B': 'constexpr', 'ACCUMULATE_TO_C': 'constexpr', 'GROUP': 'constexpr', 'M_ALIGNED': 'constexpr', 'N_ALIGNED': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'G': 32, 'M': 512, 'N': 768, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'TRANSPOSE_A': True, 'TRANSPOSE_B': False, 'ACCUMULATE_TO_C': False, 'GROUP': 8, 'M_ALIGNED': True, 'N_ALIGNED': True}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def group_gemm_same_mn_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    cumsum_K,
    total_K,  # Used for generating algo. key only.
    G: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TRANSPOSE_A: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    ACCUMULATE_TO_C: tl.constexpr,
    GROUP: tl.constexpr,
    M_ALIGNED: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    m, n = get_pid_mn(tl.program_id(axis=0), M, N, BLOCK_M, BLOCK_N, GROUP)
    gid = tl.program_id(1).to(tl.uint64)
    gtid_start = tl.load(cumsum_K + gid - 1, mask=gid > 0, other=0)
    gtid_end = tl.load(cumsum_K + gid)
    k = (gtid_end - gtid_start).to(tl.uint64)

    if TRANSPOSE_A:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + gtid_start * M,
            shape=(M, k),
            strides=(1, M),
            offsets=(m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(0, 1),
        )
    else:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr + gtid_start * M,
            shape=(M, k),
            strides=(k, 1),
            offsets=(m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
    if TRANSPOSE_B:
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + gtid_start * N,
            shape=(k, N),
            strides=(1, k),
            offsets=(0, n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
    else:
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + gtid_start * N,
            shape=(k, N),
            strides=(N, 1),
            offsets=(0, n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + gid * M * N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(m * BLOCK_M, n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Special case: no GEMM needed.
    if k == 0:
        if not ACCUMULATE_TO_C:
            # Zero out the corresponding output region.
            store_block_with_pred_2d(
                c_block_ptr,
                # tl.zeros(..., dtype=c_block_ptr.dtype.element_ty) raises "not implemented".
                tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32).to(c_block_ptr.dtype.element_ty),
                M_ALIGNED,
                N_ALIGNED,
            )
        else:
            # Nothing to do then, just leave the kernel.
            pass

        return

    if ACCUMULATE_TO_C:
        out = tl.load(c_block_ptr).to(tl.float32)
    else:
        out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # FIXME: Weird type conversion.
    for _ in range(tl.cdiv(k.to(tl.int64), BLOCK_K)):
        a = load_block_with_pred_2d(a_block_ptr, M_ALIGNED, False)
        b = load_block_with_pred_2d(b_block_ptr, False, N_ALIGNED)

        out += tl.dot(a, b)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    store_block_with_pred_2d(c_block_ptr, out.to(c_block_ptr.dtype.element_ty), M_ALIGNED, N_ALIGNED)

@triton.jit
def get_pid_mn(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.jit
def store_block_with_pred_2d(ptr, value, skip_boundary_check_0: tl.constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        tl.store(ptr, value, boundary_check=(1,))
    else:
        tl.store(ptr, value)

@triton.jit
def load_block_with_pred_2d(ptr, skip_boundary_check_0: tl.constexpr, skip_boundary_check_1: tl.constexpr):
    if not skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0, 1))
    elif not skip_boundary_check_0 and skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(0,))
    elif skip_boundary_check_0 and not skip_boundary_check_1:
        return tl.load(ptr, boundary_check=(1,))
    else:
        return tl.load(ptr)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ai/caimk33ciceslhgqxpyynelurgmsf7l4qst4h7cv6rqseege27ph.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation
# Graph fragment:
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 3, constant_args_idx: 321, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %add_12, Y: %empty_15, index: %view_9}})
triton_poi_fused_17 = async_compile.triton('triton_poi_fused_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20054016}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5013504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/moe.py:84
_moe_gather_kernel_7 = async_compile.triton('_moe_gather_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_moe_gather_kernel_7', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {'X': '*bf16', 'Y': '*bf16', 'index': '*i32', 'num_elts_in': 'i32', 'num_elts_out': 'i32', 'N': 'constexpr', 'TOPK': 'constexpr', 'STRIDE_XM': 'constexpr', 'STRIDE_XN': 'constexpr', 'STRIDE_OM': 'constexpr', 'STRIDE_ON': 'constexpr', 'STRIDE_IM': 'constexpr', 'STRIDE_IN': 'constexpr', 'BLOCK_N': 'constexpr', 'N_ALIGNED': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'N': 768, 'TOPK': 4, 'STRIDE_XM': 768, 'STRIDE_XN': 1, 'STRIDE_OM': 768, 'STRIDE_ON': 1, 'STRIDE_IM': 4, 'STRIDE_IN': 1, 'BLOCK_N': 1024, 'N_ALIGNED': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def _moe_gather_kernel(
    X,
    Y,
    index,
    num_elts_in,
    num_elts_out,
    N: tl.constexpr,  # hidden size
    TOPK: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XN: tl.constexpr,
    STRIDE_OM: tl.constexpr,
    STRIDE_ON: tl.constexpr,
    STRIDE_IM: tl.constexpr,
    STRIDE_IN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_ALIGNED: tl.constexpr,
):
    r"""
    X: m * topk x n
    Y: m x n
    index: m x topk
    code:
        repeated-X: m * topk x n -> reduce(sum_over_topk) -> m x n
        Y: Y[arange(m)] = sum_over_topk(repeated-X[arange(m) * topk])
    """
    pid_m = tl.program_id(axis=0).to(tl.int64)  # m
    block_idx = tl.program_id(axis=1).to(tl.int64)
    n = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    y = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        x_index = tl.load(index + pid_m.to(tl.int64) * STRIDE_IM + i * STRIDE_IN)
        tl.device_assert(x_index < num_elts_in, "Input OOB")
        x = load_with_pred_1d(
            X + x_index.to(tl.int64) * STRIDE_XM + n.to(tl.int64) * STRIDE_XN, N_ALIGNED, mask=n < N, other=0
        )
        y += x
    # save one line
    tl.device_assert(pid_m < num_elts_out, "Output OOB")
    Y = Y + pid_m.to(tl.int64) * STRIDE_OM + n.to(tl.int64) * STRIDE_ON  # noqa
    store_with_pred_1d(Y, y, N_ALIGNED, mask=n < N)

@triton.jit
def load_with_pred_1d(ptr, skip_boundary_check: tl.constexpr, mask: tl.tensor, other=0):
    if not skip_boundary_check:
        return tl.load(ptr, mask, other=other)
    else:
        return tl.load(ptr)

@triton.jit
def store_with_pred_1d(ptr, value, skip_boundary_check: tl.constexpr, mask: tl.tensor):
    if not skip_boundary_check:
        tl.store(ptr, value, mask)
    else:
        tl.store(ptr, value)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/yb/cyba7kytzoe6eymv755ss5fcifiv5sd27tofpmqbo6yuv2hsfwqv.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_8, %mm_10), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%empty_15, torch.float32), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %convert_element_type_18), kwargs = {})
triton_poi_fused__to_copy_add_18 = async_compile.triton('triton_poi_fused__to_copy_add_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20054016}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1253376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 + tmp4
    tl.store(in_out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/v6/cv6s7xnet62rcupfwlwmfq7rqejdkw2y2l6qfbqow6lwfnldaxkp.py
# Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, add_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_2
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %mul_26 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %add_2), kwargs = {})
#   %mul_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %primals_5), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %add), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %rsqrt), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_30, [2], True), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_1, 768), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 1.0), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_3, 2.0), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %mul_34), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %mul_35), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_20), kwargs = {})
triton_per_fused_add_div_mean_mul_pow_rsqrt_sum_19 = async_compile.triton('triton_per_fused_add_div_mean_mul_pow_rsqrt_sum_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mean_mul_pow_rsqrt_sum_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 35100672}}
)
@triton.jit
def triton_per_fused_add_div_mean_mul_pow_rsqrt_sum_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, r0_numel):
    xnumel = 1632
    XBLOCK: tl.constexpr = 1
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 51
    tmp0 = tl.load(in_ptr0 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 768*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp9 = tl.load(in_ptr4 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp10 = tl.load(in_ptr5 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp17 = tl.load(in_ptr6 + (x3), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr7 + (r0_2 + 768*x3), r0_mask, other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 768.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = 1e-06
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp8 * tmp22
    tmp24 = -0.5
    tmp25 = tmp16 * tmp24
    tmp26 = tmp22 * tmp22
    tmp27 = tmp26 * tmp22
    tmp28 = tmp25 * tmp27
    tmp29 = 0.0013020833333333333
    tmp30 = tmp28 * tmp29
    tmp31 = 2.0
    tmp32 = tmp11 * tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = tmp23 + tmp33
    tmp36 = tmp34 + tmp35
    tl.store(in_out_ptr0 + (r0_2 + 768*x3), tmp36, r0_mask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/6c/c6ccgvpkw3fkrunyhzyjwjmpgukgit2asobm7m6hhnqnqn4d52lr.py
# Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, add_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_2
#   hidden_states_2 => mul
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %mul_26 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %add_2), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %mul), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_29, [0, 1], True), kwargs = {})
triton_red_fused_add_mean_mul_pow_rsqrt_sum_20 = async_compile.triton('triton_red_fused_add_mean_mul_pow_rsqrt_sum_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mean_mul_pow_rsqrt_sum_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 20210688, 'r0_': 6552}}
)
@triton.jit
def triton_red_fused_add_mean_mul_pow_rsqrt_sum_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 9984
    r0_numel = 126
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 768
    x0 = (xindex % 768)
    _tmp24 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 126*x1
        tmp1 = tl.full([1, 1], 1632, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 768*(((r0_2 + 126*x1) % 1632))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (x0 + 768*((((r0_2 + 126*x1) // 51) % 32))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr2 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp4 + tmp5
        tmp7 = 1.0
        tmp8 = tmp6 + tmp7
        tmp9 = tmp3 * tmp8
        tmp10 = tl.load(in_ptr3 + (x0 + 768*(((r0_2 + 126*x1) % 1632))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr4 + (x0 + 768*(((r0_2 + 126*x1) % 1632))), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.load(in_ptr5 + (((r0_2 + 126*x1) % 1632)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = 768.0
        tmp15 = (tmp13 / tmp14)
        tmp16 = 1e-06
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp12 * tmp18
        tmp20 = tmp9 * tmp19
        tmp21 = tl.full(tmp20.shape, 0, tmp20.dtype)
        tmp22 = tl.where(tmp2, tmp20, tmp21)
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK, R0_BLOCK])
        tmp25 = _tmp24 + tmp23
        _tmp24 = tl.where(r0_mask & xmask, tmp25, _tmp24)
    tmp24 = tl.sum(_tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/u6/cu62t2su5ros6vmj5bl64imuqcnwcm7hcg6eetrl5or75setx3f3.py
# Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, hidden_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add_1
#   hidden_states_2 => mul
#   hidden_states_3 => mul_1
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %mul), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_28, [1], True), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %mul_1), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_27, [1], True), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_sum_21 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_sum_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_sum_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 15436800, 'r0_': 6528}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_sum_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 24576
    r0_numel = 51
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
    r0_2 = r0_index
    x0 = (xindex % 768)
    x1 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_2 + 39168*x1), r0_mask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0 + 768*r0_2 + 39168*x1), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr3 + (x0 + 768*r0_2 + 39168*x1), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr4 + (r0_2 + 51*x1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp8 = tmp6 + tmp7
    tmp10 = 768.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tmp16 = tmp5 * tmp15
    tmp17 = tmp0 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/74/c7456axxk7obukmhqsjb74glhjxrw4nldp4rx4q2j2vx2rrodhi3.py
# Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, add_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   add => add_1
#   add_1 => add_2
#   hidden_states_2 => mul
#   out_emb_1 => add
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   variance => mean
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=3] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %mul_26 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, %add_2), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %mul), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_29, [0, 1], True), kwargs = {})
triton_per_fused_add_mean_mul_pow_rsqrt_sum_22 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_sum_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 16},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_sum_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 46080, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_sum_22(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
    r0_numel = 13
    R0_BLOCK: tl.constexpr = 16
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
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ok/coklkk7vik6ljw4nzyijqcq6ngna26dybl53fbkzrueomtj6laez.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%squeeze, [0], True), kwargs = {})
triton_per_fused_sum_23 = async_compile.triton('triton_per_fused_sum_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 104448, 'r0_': 0}}
)
@triton.jit
def triton_per_fused_sum_23(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 768
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
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ht/chtysqv3nvl76yhqydfp3644ni5kwmzlylwip3mleg5oykbibpdz.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_22, torch.float32), kwargs = {})
triton_poi_fused__to_copy_24 = async_compile.triton('triton_poi_fused__to_copy_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/cg/ccgtojretrlr2tn5yyjd4aenoyvwxecpebdqk73x37ztnpufuf5x.py
# Topologically Sorted Source Nodes: [full_4, _generalized_scatter], Original ATen: [aten.slice_backward]
# Source node to ATen node mapping:
#   _generalized_scatter => slice_scatter_default
#   full_4 => full_default_4
# Graph fragment:
#   %full_default_4 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 337, 4096], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_4, %view_33, 1, 286, 337), kwargs = {})
triton_poi_fused_slice_backward_25 = async_compile.triton('triton_poi_fused_slice_backward_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_backward_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 530055168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_slice_backward_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-1171456) + x3 + 208896*x2), tmp2, other=0.0)
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4), tmp5, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_1, (32, 51, 768), (39168, 768, 1))
    assert_size_stride(primals_2, (32, 337, 4096), (1380352, 4096, 1))
    assert_size_stride(primals_3, (32, 768), (768, 1))
    assert_size_stride(primals_4, (768, 4096), (4096, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, 768), (768, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, 768), (768, 1))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (32, 768), (768, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_15, (32, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_16, (32, 768, 512), (393216, 512, 1))
    assert_size_stride(primals_17, (704, 768), (768, 1))
    assert_size_stride(primals_18, (704, 768), (768, 1))
    assert_size_stride(primals_19, (768, 704), (704, 1))
    assert_size_stride(tangents_1, (32, 51, 768), (39168, 768, 1))
    assert_size_stride(tangents_2, (1632, 32), (32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf33 = empty_strided_cuda((32, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(buf33, 32, stream=stream0)
        buf50 = empty_strided_cuda((6528, 1), (1, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf3)
        del primals_6
        buf4 = empty_strided_cuda((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf4)
        del primals_8
        buf77 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_3], Original ATen: [aten._to_copy, aten.mul, aten.div, aten.add, aten.new_zeros, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_mul_new_zeros_scatter_add_1.run(buf77, 52224, stream=stream0)
        buf29 = empty_strided_cuda((1632, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(tangents_1, buf29, 1253376, stream=stream0)
        buf69 = empty_strided_cuda((1632, 768), (768, 1), torch.bfloat16)
        buf10 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (1632, 768), (768, 1), 0), primals_19, out=buf10)
        del primals_19
        buf32 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf44 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf47 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf17 = empty_strided_cuda((6528, 768), (768, 1), torch.bfloat16)
        buf40 = empty_strided_cuda((6528, 768), (768, 1), torch.bfloat16)
        buf55 = empty_strided_cuda((6528, 768), (768, 1), torch.bfloat16)
        buf62 = empty_strided_cuda((6528, 768), (768, 1), torch.bfloat16)
        buf36 = empty_strided_cuda((32, 768, 512), (393216, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_16, buf36, 12582912, stream=stream0)
        del primals_16
        buf38 = empty_strided_cuda((32, 768, 512), (393216, 512, 1), torch.bfloat16)
        buf59 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.bfloat16)
        buf66 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.bfloat16)
        buf0 = empty_strided_cuda((1632, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_4.run(primals_2, buf0, 6684672, stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((1632, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_4, (4096, 768), (1, 4096), 0), out=buf1)
        buf2 = empty_strided_cuda((32, 51, 1), (51, 1, 1632), torch.float32)
        buf5 = empty_strided_cuda((32, 51, 768), (39168, 768, 1), torch.float32)
        buf41 = empty_strided_cuda((1632, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, hidden_states_3, add_1, mul_2, hidden_states_4, ], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_5.run(buf1, primals_1, buf3, primals_7, primals_5, buf4, primals_9, buf2, buf5, buf41, 1632, 768, stream=stream0)
        del primals_9
        buf18 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [router_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 32), (1, 768), 0), out=buf18)
        buf19 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [routing_scores, scores_for_choice], Original ATen: [aten.sigmoid, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sigmoid_6.run(buf18, primals_11, buf19, 52224, stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [routing_scores, scores_for_choice, topk], Original ATen: [aten.sigmoid, aten.add, aten.topk]
        buf20 = torch.ops.aten.topk.default(buf19, 4)
        del buf19
        buf22 = buf20[1]
        assert_size_stride(buf22, (1632, 4), (4, 1), 'torch.ops.aten.topk.default')
        assert_alignment(buf22, 16, 'torch.ops.aten.topk.default')
        del buf20
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _expert_histogram_kernel_0.run(buf33, reinterpret_tensor(buf22, (6528, ), (1, ), 0), 6528, 32, 64, 1024, False, 7, 1, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [one_hot, counts, to_8, add_], Original ATen: [aten.arange, aten.eq, aten._to_copy, aten.sum, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_arange_copy__eq_sum_7.run(buf22, primals_13, primals_13, 32, 6528, stream=stream0)
        del primals_13
        buf35 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_8.run(buf33, buf35, 1, 32, stream=stream0)
        del buf33
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.sort]
        buf23 = torch.ops.aten.sort.stable(reinterpret_tensor(buf22, (6528, ), (1, ), 0), stable=True, dim=0, descending=False)
        buf25 = buf23[1]
        assert_size_stride(buf25, (6528, ), (1, ), 'torch.ops.aten.sort.stable')
        assert_alignment(buf25, 16, 'torch.ops.aten.sort.stable')
        del buf23
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.sort]
        buf26 = torch.ops.aten.sort.stable(buf25, stable=False, dim=0, descending=False)
        del buf25
        buf28 = buf26[1]
        assert_size_stride(buf28, (6528, ), (1, ), 'torch.ops.aten.sort.default')
        assert_alignment(buf28, 16, 'torch.ops.aten.sort.default')
        del buf26
        buf30 = empty_strided_cuda((1632, 4), (4, 1), torch.int32)
        buf42 = empty_strided_cuda((1632, 4), (4, 1), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf28, buf30, buf42, 6528, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _moe_scatter_kernel_1.run(buf29, buf17, buf30, 1632, 6528, 768, 4, 768, 1, 768, 1, 4, 1, 1024, False, 1632, 1, 1, stream=stream0)
        del buf29
        del buf30
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_2.run(buf17, buf36, buf32, -1, buf35, 1632, 6528, 32, 512, 768, 128, 128, 32, False, False, False, 8, True, True, -1, False, False, 52, 32, 1, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _moe_scatter_kernel_1.run(buf41, buf40, buf42, 1632, 6528, 768, 4, 768, 1, 768, 1, 4, 1, 1024, False, 1632, 1, 1, stream=stream0)
        del buf41
        buf6 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 704), (1, 768), 0), out=buf6)
        buf7 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), reinterpret_tensor(primals_18, (768, 704), (1, 768), 0), out=buf7)
        buf8 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        buf11 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        buf14 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu, mul_4, full_1], Original ATen: [aten.silu, aten.mul, aten.fill, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_fill_mul_silu_sub_10.run(buf6, buf7, buf10, buf8, buf11, buf14, 1148928, stream=stream0)
        del buf10
        del buf6
        del buf7
        buf9 = empty_strided_cuda((768, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [silu, mul_4, permute_12], Original ATen: [aten.silu, aten.mul, aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (768, 1632), (1, 768), 0), buf8, out=buf9)
        del buf8
        buf12 = empty_strided_cuda((704, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (704, 1632), (1, 704), 0), reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), out=buf12)
        buf13 = empty_strided_cuda((1632, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf11, primals_18, out=buf13)
        del buf11
        del primals_18
        buf15 = empty_strided_cuda((704, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf14, (704, 1632), (1, 704), 0), reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), out=buf15)
        buf16 = empty_strided_cuda((1632, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, primals_17, out=buf16)
        del buf14
        del primals_17
        buf45 = reinterpret_tensor(buf36, (32, 512, 768), (393216, 768, 1), 0); del buf36  # reuse
        buf64 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_14, buf45, buf64, 12582912, stream=stream0)
        del primals_14
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_3.run(buf40, buf45, buf44, -1, buf35, 6528, 6528, 32, 512, 768, 128, 128, 32, False, True, False, 8, True, True, -1, False, False, 204, 32, 1, stream=stream0)
        buf48 = buf45; del buf45  # reuse
        buf57 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_11.run(primals_15, buf48, buf57, 12582912, stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_3.run(buf40, buf48, buf47, -1, buf35, 6528, 6528, 32, 512, 768, 128, 128, 32, False, True, False, 8, True, True, -1, False, False, 204, 32, 1, stream=stream0)
        del buf48
        buf54 = empty_strided_cuda((6528, ), (1, ), torch.bfloat16)
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.silu, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_mul_silu_sum_12.run(buf44, buf47, buf32, buf54, 6528, 512, stream=stream0)
        buf39 = empty_strided_cuda((1632, 1), (1, 1632), torch.float32)
        buf76 = empty_strided_cuda((1632, 1), (1, 1632), torch.float32)
        # Topologically Sorted Source Nodes: [routing_scores, routing_weights, sum_2, add_4, routing_weights_1], Original ATen: [aten.sigmoid, aten.gather, aten.sum, aten.add, aten.div, aten._to_copy, aten.mul, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_gather_mul_neg_sigmoid_sum_13.run(buf22, buf18, buf28, buf54, buf39, buf76, 1632, stream=stream0)
        buf71 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [final_hidden_states, , full_3], Original ATen: [aten.index_put, aten._to_copy, aten.mul, aten.div, aten.add, aten.new_zeros, aten.scatter_add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_div_index_put_mul_new_zeros_scatter_add_14.run(buf28, buf22, buf18, buf39, buf54, buf76, buf71, buf50, buf77, 6528, stream=stream0)
        del buf22
        del buf28
        del buf39
        del buf54
        del buf76
        buf79 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [routing_scores], Original ATen: [aten.sigmoid, aten.sigmoid_backward, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sigmoid_sigmoid_backward_15.run(buf79, tangents_2, buf77, 52224, stream=stream0)
        del buf77
        del tangents_2
        buf80 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [permute_29], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (32, 1632), (1, 32), 0), reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), out=buf80)
        del buf5
        buf52 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf56 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf60 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf63 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf67 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_16.run(buf44, buf47, buf50, buf32, buf52, buf56, buf60, buf63, buf67, 3342336, stream=stream0)
        del buf32
        del buf44
        del buf47
        del buf50
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_4.run(buf56, buf57, buf55, -1, buf35, 1632, 6528, 32, 768, 512, 128, 128, 32, False, False, False, 8, True, True, -1, False, False, 78, 32, 1, stream=stream0)
        del buf56
        del buf57
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_4.run(buf63, buf64, buf62, -1, buf35, 1632, 6528, 32, 768, 512, 128, 128, 32, False, False, False, 8, True, True, -1, False, False, 78, 32, 1, stream=stream0)
        del buf63
        del buf64
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_mn_kernel_5.run(buf17, buf52, buf38, buf35, 6528, 32, 768, 512, 128, 128, 32, True, False, False, 8, True, True, 24, 32, 1, stream=stream0)
        del buf17
        del buf52
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_mn_kernel_6.run(buf60, buf40, buf59, buf35, 6528, 32, 512, 768, 128, 128, 32, True, False, False, 8, True, True, 24, 32, 1, stream=stream0)
        del buf60
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_mn_kernel_6.run(buf67, buf40, buf66, buf35, 6528, 32, 512, 768, 128, 128, 32, True, False, False, 8, True, True, 24, 32, 1, stream=stream0)
        del buf35
        del buf67
        buf70 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_17.run(buf62, buf55, buf70, 5013504, stream=stream0)
        del buf55
        del buf62
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _moe_gather_kernel_7.run(buf70, buf69, buf71, 6528, 1632, 768, 4, 768, 1, 768, 1, 4, 1, 1024, False, 1632, 1, 1, stream=stream0)
        del buf70
        del buf71
        buf81 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_18.run(buf81, buf16, buf69, 1253376, stream=stream0)
        del buf69
        buf82 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [addmm], Original ATen: [aten.add, aten._to_copy]
        extern_kernels.addmm(buf81, buf79, primals_10, alpha=1, beta=1, out=buf82)
        del buf79
        del primals_10
        buf92 = reinterpret_tensor(buf81, (32, 51, 768), (39168, 768, 1), 0); del buf81  # reuse
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, add_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mean_mul_pow_rsqrt_sum_19.run(buf93, buf82, buf3, primals_7, primals_5, buf1, primals_1, buf2, tangents_1, 1632, 768, stream=stream0)
        del tangents_1
        buf94 = empty_strided_cuda((768, 4096), (4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_39], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (768, 1632), (1, 768), 0), buf0, out=buf94)
        buf89 = empty_strided_cuda((1, 1, 768, 13), (9984, 9984, 1, 768), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, add_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mean_mul_pow_rsqrt_sum_20.run(buf82, buf3, primals_7, buf1, primals_1, buf2, buf89, 9984, 126, stream=stream0)
        del primals_7
        buf83 = reinterpret_tensor(buf3, (32, 1, 768), (768, 768, 1), 0); del buf3  # reuse
        buf84 = empty_strided_cuda((32, 1, 768), (768, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, hidden_states_3], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_sum_21.run(buf82, primals_5, buf1, primals_1, buf2, buf83, buf84, 24576, 51, stream=stream0)
        del buf1
        del buf2
        del buf82
        del primals_1
        del primals_5
        buf90 = empty_strided_cuda((1, 1, 768), (768, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, add_1], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_sum_22.run(buf89, buf90, 768, 13, stream=stream0)
        del buf89
        buf86 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_23.run(buf83, buf86, 768, 32, stream=stream0)
        buf88 = empty_strided_cuda((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_per_fused_sum_23.run(buf84, buf88, 768, 32, stream=stream0)
        buf85 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (768, 32), (1, 768), 0), primals_3, out=buf85)
        del buf83
        buf87 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_35], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf84, (768, 32), (1, 768), 0), primals_3, out=buf87)
        del buf84
        del primals_3
        buf95 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (1632, 768), (768, 1), 0), primals_4, out=buf95)
        del primals_4
        buf73 = empty_strided_cuda((32, 768, 512), (393216, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf38, buf73, 12582912, stream=stream0)
        del buf38
        buf74 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf59, buf74, 12582912, stream=stream0)
        del buf59
        buf75 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_24.run(buf66, buf75, 12582912, stream=stream0)
        del buf66
        buf96 = empty_strided_cuda((32, 337, 4096), (1380352, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_4, _generalized_scatter], Original ATen: [aten.slice_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_slice_backward_25.run(buf95, buf96, 44171264, stream=stream0)
        del buf95
    return (buf93, buf96, None, buf94, reinterpret_tensor(buf90, (768, ), (1, ), 0), buf87, reinterpret_tensor(buf88, (768, ), (1, ), 0), buf85, reinterpret_tensor(buf86, (768, ), (1, ), 0), buf80, None, None, None, buf75, buf74, buf73, buf15, buf12, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 51, 768), (39168, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 337, 4096), (1380352, 4096, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((768, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((704, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((704, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 704), (704, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, 51, 768), (39168, 768, 1), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
