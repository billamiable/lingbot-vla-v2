# AOT ID: ['126_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/5s/c5s5ag52njtwbw5ep5siyi7rckiovzooisjbtenrlnz7naczmgbf.py
# Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   out_emb => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 80216064}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6684672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 208896)
    x1 = xindex // 208896
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (1171456 + x0 + 1380352*x1), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ca/ccakzftjrrwb5udq7onbsmyop2infnwxknrps537xgg7e5x4y7ec.py
# Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, hidden_states_3, add_1, mul_2, hidden_states_4, ], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_4
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
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %mul), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze, 1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %mul_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_1), kwargs = {})
#   %triton_kernel_wrapper_mutation_4 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 323, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %convert_element_type_2, O: %empty, index: %view_9}})
triton_per_fused_add_mean_mul_pow_rsqrt_1 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 0, 'r0_': 45130752}}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, r0_numel):
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
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/j6/cj6ugwqhj6dxyjbkjcrn44yfq2m77rmk6wrxsyl45fl27iamat6n.py
# Topologically Sorted Source Nodes: [routing_scores, scores_for_choice], Original ATen: [aten.sigmoid, aten.add]
# Source node to ATen node mapping:
#   routing_scores => sigmoid
#   scores_for_choice => add_4
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, %unsqueeze_2), kwargs = {})
triton_poi_fused_add_sigmoid_2 = async_compile.triton('triton_poi_fused_add_sigmoid_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sigmoid_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 626816}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sigmoid_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/gw/cgwxpzeghuva63hrf2t7wcoawvit6qirlou4j2bhbsxqnztttcuv.py
# Topologically Sorted Source Nodes: [routing_scores, routing_weights, sum_2, add_4], Original ATen: [aten.sigmoid, aten.gather, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_6
#   routing_scores => sigmoid
#   routing_weights => gather
#   sum_2 => sum_2
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %gather : [num_users=3] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%gather, [-1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-20), kwargs = {})
triton_poi_fused_add_gather_sigmoid_sum_3 = async_compile.triton('triton_poi_fused_add_gather_sigmoid_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gather_sigmoid_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_gather_sigmoid_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x0), tmp33, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ks/cksblyl3shhwg6pluhuxzi3t7p6szp3zdfrcc7awafby3i5gl5fj.py
# Topologically Sorted Source Nodes: [routing_scores, routing_weights, avg_score, , final_hidden_states], Original ATen: [aten.sigmoid, aten.gather, aten.mean, aten.index_put, aten.copy_]
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation, triton_kernel_wrapper_mutation_4
#   avg_score => mean_1
#   final_hidden_states => index_put
#   routing_scores => sigmoid
#   routing_weights => gather
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_1,), kwargs = {})
#   %gather : [num_users=3] = call_function[target=torch.ops.aten.gather.default](args = (%sigmoid, 1, %getitem_1), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%gather,), kwargs = {})
#   %triton_kernel_wrapper_mutation_4 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 1, constant_args_idx: 323, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %convert_element_type_2, O: %empty, index: %view_9}})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%permute_4, [%view_11], %view_10), kwargs = {})
#   %triton_kernel_wrapper_mutation : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 3, constant_args_idx: 327, grid: [(1632, 1, 1)], tma_descriptor_metadata: {}, kwargs: {X: %empty_4, Y: %empty_5, index: %view_9}})
#   %copy_ : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%primals_12, %view_5), kwargs = {})
triton_red_fused_copy__gather_index_put_mean_sigmoid_4 = async_compile.triton('triton_red_fused_copy__gather_index_put_mean_sigmoid_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'out_ptr1': '*i32', 'out_ptr2': '*bf16', 'out_ptr3': '*i32', 'out_ptr4': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_copy__gather_index_put_mean_sigmoid_4', 'mutated_arg_names': ['out_ptr2', 'out_ptr4'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_copy__gather_index_put_mean_sigmoid_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 6528
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        r0_1 = r0_index // 4
        tmp0 = tl.load(in_ptr0 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp20 = tl.load(in_ptr3 + (r0_2 // 4), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, R0_BLOCK], 32, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 32)) | ~(r0_mask), "index out of bounds: 0 <= tmp4 < 32")
        tmp6 = tl.load(in_ptr1 + (tmp4 + 32*r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.sigmoid(tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
        tmp12 = tmp11.to(tl.int32)
        tmp13 = tl.full([XBLOCK, R0_BLOCK], 6528, tl.int32)
        tmp14 = tmp12 + tmp13
        tmp15 = tmp12 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp12)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 6528)) | ~(r0_mask), "index out of bounds: 0 <= tmp16 < 6528")
        tmp18 = tl.load(in_ptr1 + (tmp4 + 32*(r0_2 // 4)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.sigmoid(tmp18)
        tmp21 = (tmp19 / tmp20)
        tmp22 = 4.0
        tmp23 = tmp21 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp12, r0_mask)
        tl.store(out_ptr2 + (tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])), tmp24, r0_mask)
        tl.store(out_ptr3 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), tmp12, r0_mask)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp25 = 6528.0
    tmp26 = (tmp9 / tmp25)
    tl.store(out_ptr4 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp26, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/tp/ctps6iwaatpdqogionneopeeetxga65rinrsp4ahkqb35kvtwlj2.py
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
triton_red_fused__to_copy_add_arange_copy__eq_sum_5 = async_compile.triton('triton_red_fused__to_copy_add_arange_copy__eq_sum_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_arange_copy__eq_sum_5', 'mutated_arg_names': ['in_ptr1', 'out_ptr2'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 384, 'r0_': 52224}}
)
@triton.jit
def triton_red_fused__to_copy_add_arange_copy__eq_sum_5(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/e4/ce4reqxrvatbeuhctbf3uscwkmwt66mutywdo5iolnkdrxcxohyl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_5
# Graph fragment:
#   %triton_kernel_wrapper_mutation_5 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 322, grid: [(7, 1, 1)], tma_descriptor_metadata: {}, kwargs: {out_ptr: %full_default, x_ptr: %view_7}})
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 256}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/5t/c5ty2nrhkbfsctep37czhgm5ilxmzsg4teza3aa5bgiyg7modti6.py
# Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.cumsum]
# Source node to ATen node mapping:
#   final_hidden_states => cumsum
# Graph fragment:
#   %cumsum : [num_users=3] = call_function[target=torch.ops.aten.cumsum.default](args = (%full_default, 0), kwargs = {})
triton_per_fused_cumsum_7 = async_compile.triton('triton_per_fused_cumsum_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 512}}
)
@triton.jit
def triton_per_fused_cumsum_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/w4/cw4jny6kq4nefyeqnr66pnoeyberegoee2x4skbsrn4f3lge5uyl.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_3
# Graph fragment:
#   %triton_kernel_wrapper_mutation_3 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 324, grid: [(204, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %empty, b_ptr: %convert_element_type_4, c_ptr: %empty_1, cumsum_M: %cumsum}})
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 100663296}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/4g/c4gxb4hfkyufrpaobmrdv2laqyircdcjewv3t6magl6oukc6dz4x.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
#    => triton_kernel_wrapper_mutation_1
# Graph fragment:
#   %triton_kernel_wrapper_mutation_1 : [num_users=0] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_mutation](args = (), kwargs = {kernel_idx: 2, constant_args_idx: 326, grid: [(306, 32, 1)], tma_descriptor_metadata: {}, kwargs: {a_ptr: %mul_6, b_ptr: %convert_element_type_6, c_ptr: %empty_4, cumsum_M: %cumsum}})
triton_poi_fused_9 = async_compile.triton('triton_poi_fused_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13369344}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3342336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
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
    triton_meta={'signature': {'a_ptr': '*bf16', 'b_ptr': '*bf16', 'c_ptr': '*bf16', 'act_ptr': 'constexpr', 'cumsum_M': '*i64', 'max_M': 'i32', 'total_M': 'i32', 'G': 'constexpr', 'N': 'constexpr', 'K': 'constexpr', 'BLOCK_M': 'constexpr', 'BLOCK_N': 'constexpr', 'BLOCK_K': 'constexpr', 'TRANSPOSE_A': 'constexpr', 'TRANSPOSE_B': 'constexpr', 'ACCUMULATE_TO_C': 'constexpr', 'GROUP': 'constexpr', 'N_ALIGNED': 'constexpr', 'K_ALIGNED': 'constexpr', 'ACTIVATION': 'constexpr', 'HAS_ACTIVATION': 'constexpr', 'SAVE_ACTIVATION': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {'act_ptr': None, 'G': 32, 'N': 768, 'K': 512, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'TRANSPOSE_A': False, 'TRANSPOSE_B': True, 'ACCUMULATE_TO_C': False, 'GROUP': 8, 'N_ALIGNED': True, 'K_ALIGNED': True, 'ACTIVATION': None, 'HAS_ACTIVATION': False, 'SAVE_ACTIVATION': False}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
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


# Original path: /root/experiments/lingbot-compile-accuracy-20261110/src/lingbotvla/ops/group_gemm/kernel/moe.py:84
_moe_gather_kernel_4 = async_compile.triton('_moe_gather_kernel', '''

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'kernel_name': '_moe_gather_kernel_4', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/xy/cxygvcm5s2uvo7cx4y4io4itnfh5pqmqxdv7glmzg4j6fq6vh4g4.py
# Topologically Sorted Source Nodes: [silu, mul_4], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul_4 => mul_8
#   silu => mul_7, sigmoid_2
# Graph fragment:
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%mm_2,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_2, %sigmoid_2), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %mm_3), kwargs = {})
triton_poi_fused_mul_silu_10 = async_compile.triton('triton_poi_fused_mul_silu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 18382848}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1148928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/ti/ctitbx54xdwqitltz27acbukbsdxshvtfxxitfnu7ntw6r5ighjc.py
# Topologically Sorted Source Nodes: [out_emb_1, out_emb_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_emb_1 => add
#   out_emb_3 => add_8
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_1), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %add), kwargs = {})
triton_poi_fused_add_11 = async_compile.triton('triton_poi_fused_add_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 25067520}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1253376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_out_ptr0 + (x0), None)
    tmp4 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 + tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19 = args
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
    assert_size_stride(primals_12, (1, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_15, (32, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_16, (32, 768, 512), (393216, 512, 1))
    assert_size_stride(primals_17, (704, 768), (768, 1))
    assert_size_stride(primals_18, (704, 768), (768, 1))
    assert_size_stride(primals_19, (768, 704), (704, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 51, 4096), (208896, 4096, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(primals_2, buf0, 6684672, stream=stream0)
        buf1 = empty_strided_cuda((1632, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_emb], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1632, 4096), (4096, 1), 0), reinterpret_tensor(primals_4, (4096, 768), (1, 4096), 0), out=buf1)
        del buf0
        buf3 = empty_strided_cuda((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_6, (768, 768), (1, 768), 0), out=buf3)
        buf4 = empty_strided_cuda((32, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(primals_3, reinterpret_tensor(primals_8, (768, 768), (1, 768), 0), out=buf4)
        buf5 = empty_strided_cuda((32, 51, 768), (39168, 768, 1), torch.float32)
        buf23 = empty_strided_cuda((1632, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_emb_1, pow_1, variance, add, rsqrt, hidden_states_2, hidden_states_3, add_1, mul_2, hidden_states_4, ], Original ATen: [aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_mul_pow_rsqrt_1.run(buf1, primals_1, buf3, primals_7, primals_5, buf4, primals_9, buf5, buf23, 1632, 768, stream=stream0)
        del buf3
        del buf4
        buf6 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [router_logits], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 32), (1, 768), 0), out=buf6)
        buf7 = empty_strided_cuda((1632, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [routing_scores, scores_for_choice], Original ATen: [aten.sigmoid, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sigmoid_2.run(buf6, primals_11, buf7, 52224, stream=stream0)
        # Topologically Sorted Source Nodes: [routing_scores, scores_for_choice, topk], Original ATen: [aten.sigmoid, aten.add, aten.topk]
        buf8 = torch.ops.aten.topk.default(buf7, 4)
        del buf7
        buf10 = buf8[1]
        assert_size_stride(buf10, (1632, 4), (4, 1), 'torch.ops.aten.topk.default')
        assert_alignment(buf10, 16, 'torch.ops.aten.topk.default')
        del buf8
        buf13 = empty_strided_cuda((1632, 1), (1, 1632), torch.float32)
        # Topologically Sorted Source Nodes: [routing_scores, routing_weights, sum_2, add_4], Original ATen: [aten.sigmoid, aten.gather, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_gather_sigmoid_sum_3.run(buf10, buf6, buf13, 1632, stream=stream0)
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.sort]
        buf16 = torch.ops.aten.sort.stable(reinterpret_tensor(buf10, (6528, ), (1, ), 0), stable=True, dim=0, descending=False)
        buf18 = buf16[1]
        assert_size_stride(buf18, (6528, ), (1, ), 'torch.ops.aten.sort.stable')
        assert_alignment(buf18, 16, 'torch.ops.aten.sort.stable')
        del buf16
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.sort]
        buf19 = torch.ops.aten.sort.stable(buf18, stable=False, dim=0, descending=False)
        del buf18
        buf21 = buf19[1]
        assert_size_stride(buf21, (6528, ), (1, ), 'torch.ops.aten.sort.default')
        assert_alignment(buf21, 16, 'torch.ops.aten.sort.default')
        del buf19
        buf33 = empty_strided_cuda((6528, 1), (1, 1), torch.bfloat16)
        buf24 = empty_strided_cuda((1632, 4), (4, 1), torch.int32)
        buf40 = empty_strided_cuda((1632, 4), (4, 1), torch.int32)
        # Topologically Sorted Source Nodes: [routing_scores, routing_weights, avg_score, , final_hidden_states], Original ATen: [aten.sigmoid, aten.gather, aten.mean, aten.index_put, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused_copy__gather_index_put_mean_sigmoid_4.run(buf10, buf6, buf21, buf13, buf24, buf33, buf40, primals_12, 1, 6528, stream=stream0)
        del buf13
        del buf21
        del primals_12
        # Topologically Sorted Source Nodes: [one_hot, counts, to_8, add_], Original ATen: [aten.arange, aten.eq, aten._to_copy, aten.sum, aten.add, aten.copy_]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_arange_copy__eq_sum_5.run(buf10, primals_13, primals_13, 32, 6528, stream=stream0)
        buf14 = empty_strided_cuda((32, ), (1, ), torch.int32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(buf14, 32, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _expert_histogram_kernel_0.run(buf14, reinterpret_tensor(buf10, (6528, ), (1, ), 0), 6528, 32, 64, 1024, False, 7, 1, 1, stream=stream0)
        del buf10
        buf22 = empty_strided_cuda((6528, 768), (768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _moe_scatter_kernel_1.run(buf23, buf22, buf24, 1632, 6528, 768, 4, 768, 1, 768, 1, 4, 1, 1024, False, 1632, 1, 1, stream=stream0)
        del buf24
        buf26 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [final_hidden_states], Original ATen: [aten.cumsum]
        stream0 = get_raw_stream(0)
        triton_per_fused_cumsum_7.run(buf14, buf26, 1, 32, stream=stream0)
        del buf14
        buf27 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf28 = empty_strided_cuda((32, 512, 768), (393216, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_14, buf28, 12582912, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_2.run(buf22, buf28, buf27, -1, buf26, 6528, 6528, 32, 512, 768, 128, 128, 32, False, True, False, 8, True, True, -1, False, False, 204, 32, 1, stream=stream0)
        buf30 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        buf31 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_15, buf31, 12582912, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_2.run(buf22, buf31, buf30, -1, buf26, 6528, 6528, 32, 512, 768, 128, 128, 32, False, True, False, 8, True, True, -1, False, False, 204, 32, 1, stream=stream0)
        buf35 = buf22; del buf22  # reuse
        buf36 = empty_strided_cuda((6528, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_9.run(buf27, buf30, buf33, buf36, 3342336, stream=stream0)
        del buf27
        del buf30
        del buf33
        buf37 = reinterpret_tensor(buf31, (32, 768, 512), (393216, 512, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_16, buf37, 12582912, stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        group_gemm_same_nk_kernel_3.run(buf36, buf37, buf35, -1, buf26, 6528, 6528, 32, 768, 512, 128, 128, 32, False, True, False, 8, True, True, -1, False, False, 306, 32, 1, stream=stream0)
        del buf26
        del buf36
        del buf37
        buf39 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        _moe_gather_kernel_4.run(buf35, buf39, buf40, 6528, 1632, 768, 4, 768, 1, 768, 1, 4, 1, 1024, False, 1632, 1, 1, stream=stream0)
        del buf35
        del buf40
        buf42 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), reinterpret_tensor(primals_17, (768, 704), (1, 768), 0), out=buf42)
        buf43 = empty_strided_cuda((1632, 704), (704, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (1632, 768), (768, 1), 0), reinterpret_tensor(primals_18, (768, 704), (1, 768), 0), out=buf43)
        buf44 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [silu, mul_4], Original ATen: [aten.silu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_10.run(buf44, buf43, 1148928, stream=stream0)
        del buf43
        buf45 = reinterpret_tensor(buf5, (1632, 768), (768, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [silu, mul_4, shared_expert_output], Original ATen: [aten.silu, aten.mul, aten.mm]
        extern_kernels.mm(buf44, reinterpret_tensor(primals_19, (704, 768), (1, 704), 0), out=buf45)
        del buf44
        buf46 = reinterpret_tensor(buf45, (32, 51, 768), (39168, 768, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [out_emb_1, out_emb_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_11.run(buf46, buf39, buf1, primals_1, 1253376, stream=stream0)
        del buf1
        del buf39
    return (buf46, buf6, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, )


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
    primals_12 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 768, 512), (393216, 512, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((704, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((704, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((768, 704), (704, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
