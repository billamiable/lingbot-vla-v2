# AOT ID: ['5_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/b5/cb5b53sllp5hqel6zx5frnjjiglmcgdqouggyyrk752m3d7u5ovu.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [1]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_2), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.bfloat16), kwargs = {})
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 150999040}}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 24576
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
    tmp17 = 1e-06
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/mr/cmrve3lvxkir5atlwhk4rz5duban4nqrvf25jl642oymcml4zid4.py
# Topologically Sorted Source Nodes: [q, k, mul, cat, mul_1, q_embed, mul_2, cat_1, mul_3, k_embed, q_embed_1, k_embed_1], Original ATen: [aten._to_copy, aten.mul, aten.cat, aten.add]
# Source node to ATen node mapping:
#   cat => cat
#   cat_1 => cat_1
#   k => convert_element_type_6
#   k_embed => add_3
#   k_embed_1 => convert_element_type_8
#   mul => mul_2
#   mul_1 => mul_3
#   mul_2 => mul_4
#   mul_3 => mul_5
#   q => convert_element_type_5
#   q_embed => add_2
#   q_embed_1 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_2, torch.float32), kwargs = {})
#   %convert_element_type_6 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_3, torch.float32), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_5, %unsqueeze), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, %unsqueeze), kwargs = {})
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_3], -1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_1), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.bfloat16), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_add_cat_mul_1 = async_compile.triton('triton_poi_fused__to_copy_add_cat_mul_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_cat_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 515899392}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_cat_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex // 1024
    x3 = (xindex % 1024)
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 16)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + 3072*x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (1024 + x3 + 3072*x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tl.load(in_ptr0 + (32 + 64*x1 + 3072*x2 + (x0)), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = -tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tmp4 >= tmp7
    tmp15 = tl.full([1], 64, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tl.load(in_ptr0 + (64*x1 + 3072*x2 + ((-32) + x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tl.where(tmp8, tmp13, tmp20)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp3 + tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 * tmp2
    tmp29 = tl.load(in_ptr0 + (1056 + 64*x1 + 3072*x2 + (x0)), tmp8, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = -tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp8, tmp31, tmp32)
    tmp34 = tl.load(in_ptr0 + (1024 + 64*x1 + 3072*x2 + ((-32) + x0)), tmp14, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp14, tmp35, tmp36)
    tmp38 = tl.where(tmp8, tmp33, tmp37)
    tmp39 = tmp38 * tmp22
    tmp40 = tmp28 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp25, None)
    tl.store(out_ptr1 + (x4), tmp41, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/yb/cyb5yekyscftrka4osjbvkpr5i2usuug7ew2mxauzhln2bdrgvor.py
# Topologically Sorted Source Nodes: [hidden_states, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   hidden_states => add_4
#   layer_norm_1 => add_5, add_6, convert_element_type_12, convert_element_type_13, mul_6, mul_7, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %addmm_1), kwargs = {})
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_12, [1]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %getitem_10), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_11), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_12), kwargs = {})
#   %convert_element_type_13 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_6, torch.bfloat16), kwargs = {})
triton_per_fused_add_native_layer_norm_2 = async_compile.triton('triton_per_fused_add_native_layer_norm_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 393216, 'r0_': 201330688}}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel):
    xnumel = 24576
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [R0_BLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.full([1], 1024, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [R0_BLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 1024.0
    tmp18 = (tmp16 / tmp17)
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp3 - tmp11
    tmp23 = tmp22 * tmp21
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, None)
    tl.store(out_ptr1 + (r0_1 + 1024*x0), tmp30, None)
    tl.store(out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/og/cog4ng2obmmesbhesowb2h6pgct3vfzbuzwkq2qkwas66tore7dh.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_7, add_8, convert_element_type_17, convert_element_type_18, mul_10, mul_11, mul_12, mul_13, mul_8, mul_9, tanh
# Graph fragment:
#   %convert_element_type_17 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%addmm_2, torch.float32), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, %convert_element_type_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %convert_element_type_17), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, 0.044715), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_17, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, 0.7978845608028654), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_17, 0.5), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_11,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %add_8), kwargs = {})
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
triton_poi_fused_gelu_3 = async_compile.triton('triton_poi_fused_gelu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 603979776}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100663296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp1 * tmp1
    tmp5 = tmp4 * tmp1
    tmp6 = 0.044715
    tmp7 = tmp5 * tmp6
    tmp8 = tmp1 + tmp7
    tmp9 = 0.7978845608028654
    tmp10 = tmp8 * tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp15, None)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/w2/cw2fraiyhnisj3jqrlikq5sms7odvnwkkvdam6gqw7cymw7tz6wq.py
# Topologically Sorted Source Nodes: [hidden_states, add, hidden_states_1], Original ATen: [aten.add, aten.addmm]
# Source node to ATen node mapping:
#   add => add_tensor
#   hidden_states => add_4
#   hidden_states_1 => add_9
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %addmm_1), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_16), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %add_tensor), kwargs = {})
triton_poi_fused_add_addmm_4 = async_compile.triton('triton_poi_fused_add_addmm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 251660288}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25165824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (24576, 1024), (1024, 1))
    assert_size_stride(primals_4, (3072, 1024), (1024, 1))
    assert_size_stride(primals_5, (3072, ), (1, ))
    assert_size_stride(primals_6, (24576, 64), (64, 1))
    assert_size_stride(primals_7, (24576, 64), (64, 1))
    assert_size_stride(primals_8, (97, ), (1, ))
    assert_size_stride(primals_9, (1024, 1024), (1024, 1))
    assert_size_stride(primals_10, (1024, ), (1, ))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (4096, 1024), (1024, 1))
    assert_size_stride(primals_14, (4096, ), (1, ))
    assert_size_stride(primals_15, (1024, 4096), (4096, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24576, 1), (1, 1), torch.float32)
        buf1 = empty_strided_cuda((24576, 1), (1, 24576), torch.float32)
        buf3 = reinterpret_tensor(buf1, (24576, 1), (1, 1), 0); del buf1  # reuse
        buf4 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_0.run(buf3, primals_3, primals_1, primals_2, buf0, buf4, 24576, 1024, stream=stream0)
        del primals_2
        buf5 = empty_strided_cuda((24576, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf4, reinterpret_tensor(primals_4, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf5)
        del primals_5
        buf6 = empty_strided_cuda((24576, 16, 64), (1024, 64, 1), torch.bfloat16)
        buf7 = empty_strided_cuda((24576, 16, 64), (1024, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [q, k, mul, cat, mul_1, q_embed, mul_2, cat_1, mul_3, k_embed, q_embed_1, k_embed_1], Original ATen: [aten._to_copy, aten.mul, aten.cat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_cat_mul_1.run(buf5, primals_6, primals_7, buf6, buf7, 25165824, stream=stream0)
        # Topologically Sorted Source Nodes: [out], Original ATen: [flash_attn._flash_attn_varlen_forward]
        buf8 = torch.ops.flash_attn._flash_attn_varlen_forward.default(buf6, buf7, reinterpret_tensor(buf5, (24576, 16, 64), (3072, 64, 1), 2048), primals_8, primals_8, 256, 256, 0.0, 0.125, False, window_size_left=-1, window_size_right=-1, softcap=0.0, alibi_slopes=None, return_softmax=False, block_table=None, leftpad_k=None, seqused_k=None, zero_tensors=False)
        buf9 = buf8[0]
        assert_size_stride(buf9, (24576, 16, 64), (1024, 64, 1), 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
        assert_alignment(buf9, 16, 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
        buf10 = buf8[1]
        assert_size_stride(buf10, (16, 24576), (24576, 1), 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
        assert_alignment(buf10, 16, 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
        buf12 = buf8[3]
        assert_size_stride(buf12, (2, ), (1, ), 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
        assert_alignment(buf12, 16, 'torch.ops.flash_attn._flash_attn_varlen_forward.default')
        del buf8
        buf13 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [attn_output_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf9, (24576, 1024), (1024, 1), 0), reinterpret_tensor(primals_9, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf13)
        del primals_10
        buf14 = empty_strided_cuda((24576, 1), (1, 1), torch.float32)
        buf15 = empty_strided_cuda((24576, 1), (1, 24576), torch.float32)
        buf17 = reinterpret_tensor(buf15, (24576, 1), (1, 1), 0); del buf15  # reuse
        buf18 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_2.run(buf17, primals_3, buf13, primals_11, primals_12, buf14, buf18, 24576, 1024, stream=stream0)
        del primals_12
        buf19 = empty_strided_cuda((24576, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, buf18, reinterpret_tensor(primals_13, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf19)
        del primals_14
        buf20 = empty_strided_cuda((24576, 4096), (4096, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_3.run(buf19, buf20, 100663296, stream=stream0)
        buf21 = empty_strided_cuda((24576, 1024), (1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.addmm]
        extern_kernels.mm(buf20, reinterpret_tensor(primals_15, (4096, 1024), (1, 4096), 0), out=buf21)
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [hidden_states, add, hidden_states_1], Original ATen: [aten.add, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_4.run(buf22, primals_3, buf13, primals_16, 25165824, stream=stream0)
        del primals_16
    return (buf22, primals_1, primals_3, primals_4, primals_6, primals_7, primals_8, primals_9, primals_11, primals_13, primals_15, buf0, buf3, buf4, buf6, buf7, reinterpret_tensor(buf5, (24576, 16, 64), (3072, 64, 1), 2048), buf9, buf10, buf12, buf13, buf14, buf17, buf18, buf19, buf20, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_3 = rand_strided((24576, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((3072, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((24576, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24576, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((97, ), (1, ), device='cuda:0', dtype=torch.int32)
    primals_9 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((4096, 1024), (1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_15 = rand_strided((1024, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_16 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
