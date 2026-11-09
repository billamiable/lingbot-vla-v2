# AOT ID: ['185_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/je/cjecxawtvnald5wdiap7scb46qs33f4ln5q6my5fxvcf3kepmc73.py
# Topologically Sorted Source Nodes: [target, target_norm, cosine_similarity, pred_norm, scalar_tensor, mse_loss], Original ATen: [aten._to_copy, aten.div, aten.mul, aten.clamp_min, aten.neg, aten.sum, aten.eq, aten.masked_fill, aten.add, aten.ge, aten.where, aten.mse_loss, aten.mse_loss_backward]
# Source node to ATen node mapping:
#   cosine_similarity => clamp_min_2, div_2, div_3
#   mse_loss => sub
#   pred_norm => div
#   scalar_tensor => full_default
#   target => convert_element_type_1
#   target_norm => div_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_1, %expand_1), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_1, %clamp_min_3), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_3, %div_2), kwargs = {})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %expand), kwargs = {})
#   %clamp_min_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%pow_7, 1e-06), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %clamp_min_2), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_3, %clamp_min_2), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mul_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %div_6), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, %clamp_min_2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [2], True), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %pow_7), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_7, 0), kwargs = {})
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div_8), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_6, %where), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_7, %mul_8), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %expand), kwargs = {})
#   %neg_2 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%add_1,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_2, %div_10), kwargs = {})
#   %div_11 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1, %expand), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tangents_2, %div_11), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_9, [2], True), kwargs = {})
#   %ge : [num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%pow_3, 1e-06), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %sum_7, %full_default), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type, %pow_3), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%pow_3, 0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default, %div_12), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %where_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, 1.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %convert_element_type_1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, 2.384185791015625e-07), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %mul_11), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_13), kwargs = {})
#   %convert_element_type_3 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.bfloat16), kwargs = {})
triton_per_fused__to_copy_add_clamp_min_div_eq_ge_masked_fill_mse_loss_mse_loss_backward_mul_neg_sum_where_0 = async_compile.triton('triton_per_fused__to_copy_add_clamp_min_div_eq_ge_masked_fill_mse_loss_mse_loss_backward_mul_neg_sum_where_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clamp_min_div_eq_ge_masked_fill_mse_loss_mse_loss_backward_mul_neg_sum_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 131072, 'r0_': 117440512}}
)
@triton.jit
def triton_per_fused__to_copy_add_clamp_min_div_eq_ge_masked_fill_mse_loss_mse_loss_backward_mul_neg_sum_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, xnumel, r0_numel):
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
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK])
    tmp7 = tl.load(in_ptr1 + (r0_1 + 1024*x0), None).to(tl.float32)
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (r0_1 + 1024*x0), None)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr7 + (r0_1 + 1024*x0), None)
    tmp2 = 0.2
    tmp3 = tmp1 * tmp2
    tmp4 = -tmp3
    tmp5 = 0.0001220703125
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = (tmp8 / tmp9)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tmp6 * tmp12
    tmp14 = -tmp13
    tmp17 = 1e-06
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = (tmp15 / tmp18)
    tmp21 = triton_helpers.maximum(tmp20, tmp17)
    tmp22 = (tmp19 / tmp21)
    tmp23 = (tmp22 / tmp21)
    tmp24 = tmp14 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [R0_BLOCK])
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp28 = (tmp13 / tmp21)
    tmp29 = 0.0
    tmp30 = tmp20 == tmp29
    tmp31 = (tmp19 / tmp20)
    tmp32 = tl.where(tmp30, tmp29, tmp31)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp28 + tmp33
    tmp35 = -tmp34
    tmp36 = (tmp19 / tmp18)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.broadcast_to(tmp37, [R0_BLOCK])
    tmp40 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp42 = (tmp34 / tmp18)
    tmp43 = tmp41 + tmp42
    tmp44 = tmp16 >= tmp17
    tmp45 = tl.where(tmp44, tmp40, tmp29)
    tmp46 = tmp16 == tmp29
    tmp47 = (tmp15 / tmp16)
    tmp48 = tl.where(tmp46, tmp29, tmp47)
    tmp49 = tmp45 * tmp48
    tmp50 = tmp43 + tmp49
    tmp51 = tmp15 - tmp8
    tmp52 = 2.384185791015625e-07
    tmp53 = tmp51 * tmp52
    tmp54 = 1.0
    tmp55 = tmp1 * tmp54
    tmp56 = tmp53 * tmp55
    tmp57 = tmp50 + tmp56
    tmp58 = tmp57.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 1024*x0), tmp58, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, convert_element_type, pow_3, clamp_min_1, pow_7, clamp_min_3, tangents_1, tangents_2 = args
    args.clear()
    assert_size_stride(primals_2, (32, 256, 1024), (262144, 1024, 1))
    assert_size_stride(convert_element_type, (32, 256, 1024), (262144, 1024, 1))
    assert_size_stride(pow_3, (32, 256, 1), (256, 1, 1))
    assert_size_stride(clamp_min_1, (32, 256, 1), (256, 1, 1))
    assert_size_stride(pow_7, (32, 256, 1), (256, 1, 1))
    assert_size_stride(clamp_min_3, (32, 256, 1), (256, 1, 1))
    assert_size_stride(tangents_1, (), ())
    assert_size_stride(tangents_2, (32, 256, 1024), (262144, 1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((32, 256, 1024), (262144, 1024, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [target, target_norm, cosine_similarity, pred_norm, scalar_tensor, mse_loss], Original ATen: [aten._to_copy, aten.div, aten.mul, aten.clamp_min, aten.neg, aten.sum, aten.eq, aten.masked_fill, aten.add, aten.ge, aten.where, aten.mse_loss, aten.mse_loss_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_clamp_min_div_eq_ge_masked_fill_mse_loss_mse_loss_backward_mul_neg_sum_where_0.run(tangents_1, primals_2, clamp_min_1, clamp_min_3, convert_element_type, pow_3, pow_7, tangents_2, buf3, 8192, 1024, stream=stream0)
        del clamp_min_1
        del clamp_min_3
        del convert_element_type
        del pow_3
        del pow_7
        del primals_2
        del tangents_1
        del tangents_2
    return (buf3, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    convert_element_type = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.float32)
    pow_3 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    clamp_min_1 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    pow_7 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    clamp_min_3 = rand_strided((32, 256, 1), (256, 1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((32, 256, 1024), (262144, 1024, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, convert_element_type, pow_3, clamp_min_1, pow_7, clamp_min_3, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
