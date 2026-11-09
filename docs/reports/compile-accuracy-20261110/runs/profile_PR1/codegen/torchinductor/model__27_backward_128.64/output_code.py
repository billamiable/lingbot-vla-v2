# AOT ID: ['27_backward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/k7/ck7jqczfotdtrm6ldnt2n3zymqyxaie2vxk4r32p2bikridcywkp.py
# Topologically Sorted Source Nodes: [full_10], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_10 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 32, 337], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_1, %primals_2, %primals_3, %getitem, %getitem_1, %permute_1, %full_default, %fw_graph0, %joint_graph0, (337, 337, %primals_5, %primals_4, %primals_7, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, 128, 128, %mask_graph0), 0.08838834764831845, {BLOCK_M: 32, BLOCK_N: 64, num_warps: 4, num_stages: 2, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True}, (), (%primals_6,)), kwargs = {})
triton_red_fused_zeros_0 = async_compile.triton('triton_red_fused_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 524288, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_zeros_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 2760704, 'r0_': 353370112}}
)
@triton.jit
def triton_red_fused_zeros_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 345088
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 337)
    x1 = ((xindex // 337) % 32)
    x2 = xindex // 10784
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_3 + 128*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 128*x1 + 4096*x0 + 1380352*x2), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 0.0
    tmp7 = tmp4 - tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp7, xmask)
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/uy/cuy3lqshvgqrobov3vsojg2a2qsioxaz74su2s25due7epfonpej.py
# Topologically Sorted Source Nodes: [full_10], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   full_10 => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([32, 32, 337], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_1, %primals_2, %primals_3, %getitem, %getitem_1, %permute_1, %full_default, %fw_graph0, %joint_graph0, (337, 337, %primals_5, %primals_4, %primals_7, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, 128, 128, %mask_graph0), 0.08838834764831845, {BLOCK_M: 32, BLOCK_N: 64, num_warps: 4, num_stages: 2, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True}, (), (%primals_6,)), kwargs = {})
triton_tem_fused_zeros_1 = async_compile.triton('triton_tem_fused_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_Q': '*fp32', 'arg_K': '*fp32', 'arg_V': '*fp32', 'arg_LSE': '*fp32', 'arg_DELTA': '*fp32', 'arg_DO': '*fp32', 'arg_DQ': '*fp32', 'arg_DV': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_Q_NUM_BLKS': '*i32', 'arg_Q_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'arg_FULL_Q_NUM_BLKS': '*i32', 'arg_FULL_Q_IDX': '*i32', 'in_ptr16': '*i1', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_zeros_1', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2']},

)
@triton.jit
def triton_tem_fused_zeros_1(arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0):
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 16
    BLOCK_N1 : tl.constexpr = 16
    BLOCK_M2 : tl.constexpr = 16
    BLOCK_N2 : tl.constexpr = 16
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    DELTA = arg_DELTA
    DO = arg_DO
    DQ = arg_DQ
    DV = arg_DV
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    Q_NUM_BLKS = arg_Q_NUM_BLKS
    Q_IDX = arg_Q_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX
    FULL_Q_NUM_BLKS = arg_FULL_Q_NUM_BLKS
    FULL_Q_IDX = arg_FULL_Q_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = 1380352, 43136, 128, 1
    stride_kz, stride_kh, stride_kn, stride_kd = 345088, 43136, 128, 1
    stride_vz, stride_vh, stride_vn, stride_vd = 345088, 43136, 128, 1
    stride_doz, stride_doh, stride_dom, stride_dod = 1380352, 128, 4096, 1

    stride_dqz, stride_dqh, stride_dqm, stride_dqd = 1380352, 43136, 128, 1
    stride_dvz, stride_dvh, stride_dvm, stride_dvd = 345088, 43136, 128, 1

    ZQ = 32
    HQ = 32
    HKV = 8
    Q_LEN = 337
    ZKV = 32
    KV_LEN = 337

    MATMUL_PRECISION = Q.dtype.element_ty

    pid = tl.program_id(0)
    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_Q_BLOCKS = tl.cdiv(Q_LEN, BLOCK_M2)

    off_hz = tl.program_id(2)
    off_zq = off_hz // HKV # q batch idx
    off_hkv = off_hz % HKV # kv head idx
    off_zkv = off_zq % ZKV # kv batch idx

    SPARSE_Z = 32
    SPARSE_HQ = 32

    sparse_idx_z = off_zq % SPARSE_Z

    k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
    v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
    # first compute broadcasted dv of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
    # then reduce to dv of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
    dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)

    # offset K, V, DV pointers for batch/kv-head
    K += k_adj
    V += v_adj
    DV += dv_adj

    RCP_LN2 = 1.44269504
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    if pid >= NUM_KV_BLOCKS:
        off_pid = pid - NUM_KV_BLOCKS
        # THIS BLOCK DOES DQ
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M2)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
        off_hq2 = off_pid // NUM_Q_BLOCKS + off_hkv * GQA_SHARED_HEADS
        start_m2_block = off_pid % NUM_Q_BLOCKS
        off_pid_mask = start_m2_block // SPARSE_Q_MULTIPLE
        stride_kv_num_blks_h = 3
        stride_kv_idx_h = 9
        stride_kv_idx_m = 3

        sparse_idx_hq2 = off_hq2 % SPARSE_HQ
        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq2

        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + off_pid_mask
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + off_pid_mask * stride_kv_idx_m  # noqa: B950

        # Offset Q, DQ, DO, DELTA & LSE. These inputs are offsetted by query heads.
        q_adj2 = (stride_qh * off_hq2 + stride_qz * off_zq).to(tl.int64)
        do_adj2 = (stride_doh * off_hq2 + stride_doz * off_zq).to(tl.int64)
        dq_adj2 = (stride_dqh * off_hq2 + stride_dqz * off_zq).to(tl.int64)
        off_chz2 = ((off_zq * HQ + off_hq2) * Q_LEN).to(tl.int64)

        Q2 = Q + q_adj2
        DO2 = DO + do_adj2
        # TODO: This does not work if DQ is not the same layout as Q (for example,
        # if Q is broadcasted)
        DQ2 = DQ + dq_adj2
        LSE2 = LSE + off_chz2
        DELTA2 = DELTA + off_chz2

        # dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM], dtype=tl.float32)
        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_m2 = start_m2_block * BLOCK_M2
        offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

        # load Q and do: they stay in SRAM throughout the inner loop.
        q = load_checked_2d(Q2, offs_m2, offs_k, stride_qm, stride_qd, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM)
        do = load_checked_2d(DO2, offs_m2, offs_v, stride_dom, stride_dod, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)

        if PRESCALE_QK:
            q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        if IS_DIVISIBLE:
            Di = tl.load(DELTA2 + offs_m2)
            lse = tl.load(LSE2 + offs_m2)
        else:
            Di = tl.load(DELTA2 + offs_m2, mask=offs_m2 < Q_LEN)
            lse = tl.load(LSE2 + offs_m2, mask=offs_m2 < Q_LEN)
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        lse = lse[:, None]

        # ~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # KV_IDX and KV_NUM_BLKS are always contiguous.
        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        sparse_kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
        dq = bwd_dq_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
            K, V,
            dq, q, do, Di, lse,
            off_zq, off_hq2, offs_m2, offs_n2,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=False,
        )

        if HAS_FULL_BLOCKS:
            # ~~~~~~~~~~~ partial unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
            sparse_kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)

            offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
            dq = bwd_dq_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                K, V,
                dq, q, do, Di, lse,
                off_zq, off_hq2, offs_m2, offs_n2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=True,
            )

        # Write back dQ.
        dq_ptrs = DQ2 + offs_m2[:, None] * stride_dqm + offs_k[None, :] * stride_dqd
        dq *= SM_SCALE
        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=(offs_m2[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM))
    else:
        # THIS BLOCK DOES DK & DV
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

        pid_mask = pid // SPARSE_KV_MULTIPLE

        stride_q_num_blks_h = 3
        stride_q_idx_h = 9
        stride_q_idx_n = 3


        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM_ROUNDED], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_n1 = pid * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)

        # load K and V: they stay in SRAM throughout the inner loop.
        k = load_checked_2d(K, offs_n1, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)
        v = load_checked_2d(V, offs_n1, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)

        if PRESCALE_QK:
            k = (k * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            # Offset Q, DQ, DO, DELTA & LSE. These inputs are offsetted by query heads.
            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            dq_adj1 = (stride_dqh * off_hq1 + stride_dqz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            # TODO: This does not work if DQ is not the same layout as Q (for example,
            # if Q is broadcasted)
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1

            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n  # noqa: B950

            # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Q_IDX and Q_NUM_BLKS are always contiguous.
            q_indices = Q_IDX + sparse_q_idx_offset
            q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)

            offs_m1 = q_start + tl.arange(0, BLOCK_M1)
            dk, dv = bwd_dkdv_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                Q1, DO1, DELTA1, LSE1,
                dk, dv, k, v,
                off_zq, off_hq1, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=False,
            )


            if HAS_FULL_BLOCKS:
                # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FULL_Q_IDX and FULL_Q_NUM_BLKS are always contiguous.
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)

                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                dk, dv = bwd_dkdv_inner(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                    Q1, DO1, DELTA1, LSE1,
                    dk, dv, k, v,
                    off_zq, off_hq1, offs_n1, offs_m1,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION,
                    IS_FULL_BLOCKS=True,
                )

        # Write back dV and dK.
        dv_ptrs = DV + offs_n1[:, None] * stride_dvm + offs_v[None, :] * stride_dvd

        index_n = offs_n1[:, None]
        index_k = offs_k[None, :]
        index_v = offs_v[None, :]

        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM))

        dk *= SM_SCALE

        if SAFE_HEAD_DIM:
            mask = index_n < KV_LEN
        else:
            mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)

        # first compute broadcasted dk of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
        # then reduce to dk of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
        xindex = index_k + 128*index_n + 43136*off_hkv + 345088*off_zq
        tl.store(out_ptr0 + (tl.broadcast_to(xindex, dk.shape)), dk, mask)

@triton.jit
def bwd_dq_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    K, V,  # pointers
    dq, q, do, Di, lse,
    off_z, off_hq, offs_m2, offs_n2,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 16
    BLOCK_N1 : tl.constexpr = 16
    BLOCK_M2 : tl.constexpr = 16
    BLOCK_N2 : tl.constexpr = 16
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = 337
    KV_LEN = 337

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    kT_ptrs = K + offs_n2[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n2[None, :] * stride_vn + offs_v[:, None] * stride_vd
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    hi = tl.minimum(sparse_kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N2), 1))
    if not IS_DIVISIBLE:
        if hi >= 1:
            for start_n in range(0, hi - 1):
                dq = bwd_dq_block_mn(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                    off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
                    stride_kn, stride_kd, stride_vn, stride_vd,
                    kv_indices, sparse_kv_num_blocks,
                    MATMUL_PRECISION, RCP_LN2,
                    IS_FULL_BLOCKS,
                )

                # Increment pointers.
                offset = get_offset_for_next_block(
                    start_n, kv_indices, sparse_kv_num_blocks,
                    SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2, BLOCKS_ARE_CONTIGUOUS
                )

                kT_ptrs += offset * stride_kn
                vT_ptrs += offset * stride_vn

                offs_n2 += offset

            dq = bwd_dq_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
    else:
        for start_n in range(0, hi):
            dq = bwd_dq_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
                off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )

            # Increment pointers.
            offset = get_offset_for_next_block(
                start_n, kv_indices, sparse_kv_num_blocks,
                SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2, BLOCKS_ARE_CONTIGUOUS
            )

            kT_ptrs += offset * stride_kn
            vT_ptrs += offset * stride_vn

            offs_n2 += offset

    return dq


@triton.jit
def bwd_dq_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
    off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 16
    BLOCK_N1 : tl.constexpr = 16
    BLOCK_M2 : tl.constexpr = 16
    BLOCK_N2 : tl.constexpr = 16
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128


    # NB reversed order to since K is transposed
    kT = load_checked_2d(kT_ptrs, offs_k, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, KV_LEN)
    qk = tl.dot(q, kT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    pre_mod_scores = qk
    n = get_bounded_indices(offs_n2[None, :], KV_LEN if CHECK_BLOCK_BOUNDARY else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across N dim
    # that the M reads out of bounds prior to the last loop
    m = get_bounded_indices(offs_m2[:, None], Q_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

    tmp0 = (qk)
    post_mod_scores = tmp0


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n2[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = (n)
        tmp2 = (m)
        tmp3 = (off_hq)
        tmp4 = (off_z)
        tmp5 = tl.load(in_ptr16 + tmp1 + 337*tmp2 + 113569*tmp3 + 3634208*tmp4)
        mask_mod_output = tmp5


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, False)
        # apply mask for partial masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    p = tl.math.exp2(post_mod_scores - lse)
    # Compute dP and dS.
    # NB reversed order to since V is transposed
    vT = load_checked_2d(vT_ptrs, offs_v, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, V_HEAD_DIM, KV_LEN)

    dp = tl.dot(do, vT, input_precision=FLOAT32_PRECISION)
    ds = p * (dp - Di[:, None])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    tmp6 = (ds)
    grad_scores = tmp6

    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n2[None, :] < KV_LEN, grad_scores, 0.0)

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if WRITE_DQ:
        scatter_mask = offs_m2[:, None] < Q_LEN and offs_n2[None, :] < KV_LEN

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = grad_scores

    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n2[None, :] < KV_LEN, mask_mod_output, False)
        # (grads) apply mask for partially unmasked block
        ds = tl.where(mask_mod_output, ds, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = ds.to(MATMUL_PRECISION)
    # Compute dQ.
    dq += tl.dot(ds, tl.trans(kT), input_precision=FLOAT32_PRECISION)

    return dq


@triton.jit
def bwd_dkdv_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    Q, DO, DELTA, LSE, # pointers
    dk, dv, k, v,
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 16
    BLOCK_N1 : tl.constexpr = 16
    BLOCK_M2 : tl.constexpr = 16
    BLOCK_N2 : tl.constexpr = 16
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = 337
    KV_LEN = 337

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    qT_ptrs = Q + offs_m1[None, :] * stride_qm + offs_k[:, None] * stride_qd
    do_ptrs = DO + offs_m1[:, None] * stride_dom + offs_v[None, :] * stride_dod
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    hi = tl.minimum(sparse_q_num_blocks * SPARSE_Q_MULTIPLE, tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1))

    if not IS_DIVISIBLE:
        if hi >= 1:
            for start_m in range(0, hi - 1):
                dk, dv = bwd_dkdv_block_mn(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                    off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION, RCP_LN2,
                    IS_FULL_BLOCKS,
                )
                # Increment pointers.
                offset = get_offset_for_next_block(
                    start_m, q_indices, sparse_q_num_blocks,
                    SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1, BLOCKS_ARE_CONTIGUOUS
                )

                qT_ptrs += offset * stride_qm
                do_ptrs += offset * stride_dom

                offs_m1 += offset

            dk, dv = bwd_dkdv_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )
    else:
        for start_m in range(0, hi):
            dk, dv = bwd_dkdv_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
                dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
                off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
            # Increment pointers.
            offset = get_offset_for_next_block(
                start_m, q_indices, sparse_q_num_blocks,
                SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1, BLOCKS_ARE_CONTIGUOUS
            )

            qT_ptrs += offset * stride_qm
            do_ptrs += offset * stride_dom

            offs_m1 += offset

    return dk, dv


@triton.jit
def bwd_dkdv_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, out_ptr0,
    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
    off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 64
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    FLOAT32_PRECISION : tl.constexpr = 'tf32'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 16
    BLOCK_N1 : tl.constexpr = 16
    BLOCK_M2 : tl.constexpr = 16
    BLOCK_N2 : tl.constexpr = 16
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128


    # NB reversed order since Q is transposed
    qT = load_checked_2d(qT_ptrs, offs_k, offs_m1, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, Q_LEN)
    # Load LSE before computing qk to reduce pipeline stall.
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN)
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(k, qT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    m = get_bounded_indices(offs_m1[None, :], Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across M dim
    # that the n reads out of bounds prior to the last loop
    n = get_bounded_indices(offs_n1[:, None], KV_LEN if (not IS_DIVISIBLE or CHECK_BLOCK_BOUNDARY) else None)

    pre_mod_scores = qkT
    tmp7 = (qkT)
    post_mod_scores = tmp7


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n1[:, None] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp8 = (n)
        tmp9 = (m)
        tmp10 = (off_hq)
        tmp11 = (off_z)
        tmp12 = tl.load(in_ptr16 + tmp8 + 337*tmp9 + 113569*tmp10 + 3634208*tmp11)
        mask_mod_output = tmp12

        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, False)
        # (grads) apply mask for fully masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    pT = tl.math.exp2(post_mod_scores - lse[None, :])
    do = load_checked_2d(do_ptrs, offs_m1, offs_v, None, None, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)
    # Compute dV.
    ppT = pT
    dv += tl.dot(ppT.to(MATMUL_PRECISION), do, input_precision=FLOAT32_PRECISION)
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN)
    # Compute dP and dS.
    dpT = tl.dot(v, tl.trans(do), input_precision=FLOAT32_PRECISION)
    dsT = pT * (dpT - Di[None, :])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    tmp13 = (dsT)
    grad_scores = tmp13


    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if not WRITE_DQ:
        idx_b = off_z
        idx_h = off_hq
        idx_m = m
        idx_n = n
        scatter_mask = offs_m1[None, :] < Q_LEN and offs_n1[:, None] < KV_LEN

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if CHECK_BLOCK_BOUNDARY:
        grad_scores = tl.where(offs_n1[:, None] < KV_LEN, grad_scores, 0.0)

    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n1[:, None] < KV_LEN, mask_mod_output, False)
        # (grads) apply mask for partially unmasked block
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dk += tl.dot(dsT.to(MATMUL_PRECISION), tl.trans(qT), input_precision=FLOAT32_PRECISION)

    return dk, dv

@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_DIM: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_DIM), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_DIM), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, getitem, getitem_1, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, 32, 337, 128), (1380352, 43136, 128, 1))
    assert_size_stride(primals_2, (32, 8, 337, 128), (345088, 43136, 128, 1))
    assert_size_stride(primals_3, (32, 8, 337, 128), (345088, 43136, 128, 1))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_6, (32, 32, 337, 337), (3634208, 113569, 337, 1))
    assert_size_stride(primals_7, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_8, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_9, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_10, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_11, (32, 32, 3), (96, 3, 1))
    assert_size_stride(primals_12, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(getitem, (32, 32, 337, 128), (1380352, 43136, 128, 1))
    assert_size_stride(getitem_1, (32, 32, 337), (10784, 337, 1))
    assert_size_stride(tangents_1, (32, 337, 4096), (1380352, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 337), (10784, 337, 1), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [full_10], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_red_fused_zeros_0.run(buf1, getitem, tangents_1, 345088, 128, stream=stream0)
        del getitem
        buf3 = empty_strided_cuda((32, 32, 337, 128), (1380352, 43136, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 8, 337, 128), (345088, 43136, 128, 1), torch.float32)
        buf5 = empty_strided_cuda((32, 8, 337, 128), (345088, 43136, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [full_10], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_tem_fused_zeros_1.run(primals_1, primals_2, primals_3, getitem_1, buf1, tangents_1, buf3, buf4, primals_5, primals_4, primals_9, primals_10, primals_7, primals_8, primals_11, primals_12, primals_6, buf5, 110, 1, 256, stream=stream0)
        del buf1
        del getitem_1
        del primals_1
        del primals_10
        del primals_11
        del primals_12
        del primals_2
        del primals_3
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
        del tangents_1
    return (buf3, buf5, buf4, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 32, 337, 128), (1380352, 43136, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, 8, 337, 128), (345088, 43136, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, 8, 337, 128), (345088, 43136, 128, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_5 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_6 = rand_strided((32, 32, 337, 337), (3634208, 113569, 337, 1), device='cuda:0', dtype=torch.bool)
    primals_7 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_8 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_9 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_10 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_11 = rand_strided((32, 32, 3), (96, 3, 1), device='cuda:0', dtype=torch.int32)
    primals_12 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.int32)
    getitem = rand_strided((32, 32, 337, 128), (1380352, 43136, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((32, 32, 337), (10784, 337, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, 337, 4096), (1380352, 4096, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, getitem, getitem_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
