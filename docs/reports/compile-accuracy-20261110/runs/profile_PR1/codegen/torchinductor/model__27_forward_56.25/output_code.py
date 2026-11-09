# AOT ID: ['27_forward']
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/ey/ceygbvi76ganbvvvmllhulo6iy5gmasxhtjhp2wbqc2vwmaf6ytl.py
# Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
# Source node to ATen node mapping:
#   flex_attention => flex_attention
# Graph fragment:
#   %flex_attention : [num_users=2] = call_function[target=torch.ops.higher_order.flex_attention](args = (%primals_1, %primals_2, %primals_3, %sdpa_score0, (337, 337, %primals_5, %primals_4, %primals_7, %primals_8, %primals_9, %primals_10, %primals_11, %primals_12, 128, 128, %sdpa_mask0), 0.08838834764831845, {BLOCK_M: 32, BLOCK_N: 64, num_warps: 4, num_stages: 2, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True}, (), (%primals_6,)), kwargs = {})
triton_tem_fused_0 = async_compile.triton('triton_tem_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=2,
num_warps=4,
triton_meta={'signature': {'arg_Q': '*fp32', 'arg_K': '*fp32', 'arg_V': '*fp32', 'arg_LSE': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'in_ptr8': '*i1', 'out_ptr0': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_0', 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2']},

)
@triton.jit
def triton_tem_fused_0(arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0):
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
    USE_TMA : tl.constexpr = False
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # BLOCKS_ARE_CONTIGUOUS: Is it guaranteed that all blocks in the mask are
    # contiguous? If so, we don't need to do an indirect jump for every block

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = 1380352, 43136, 128, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 345088, 43136, 128, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 345088, 43136, 128, 1

    ZQ = 32
    HQ = 32
    Q_LEN = 337
    ZKV = 32
    KV_LEN = 337

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0)
    off_zq = tl.program_id(1) // HQ
    off_hq = tl.program_id(1) % HQ


    # Setting up the TMA descriptors for Q, K, V
    desc_q = None
    desc_k = None
    desc_v = None


    # We support two cases for batch dimension. a) (ZKV == ZQ) where off_zkv = off_zq.
    # b) (ZKV == 1 and ZQ > 1) where KV is broadcasted along the batch dimension and off_zkv=0.
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    SPARSE_Z = 32
    SPARSE_HQ = 32

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    stride_kv_num_blks_h = 3
    stride_kv_idx_h = 9
    stride_kv_idx_m = 3

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m  # noqa: B950
    K_block_ptr = None
    V_block_ptr = None
    Q_block_ptr = None

    if not USE_TMA:
        Q_block_ptr = tl.make_block_ptr(
            base=Q ,
            shape=(Q_LEN, QK_HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(q_start * BLOCK_M, 0),
            block_shape=(BLOCK_M, QK_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )
        q = load_checked_block(Q_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)

    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))


    if not USE_TMA:
        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
            order=(0, 1)
        )

        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
            order=(1, 0)
        )

    offs_n = kv_start + tl.arange(0, BLOCK_N)


    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0,
        q, K_block_ptr, V_block_ptr,
        desc_k, desc_v, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_start,
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        if not USE_TMA:
            K_block_ptr = tl.make_block_ptr(
                base=K,
                shape=(QK_HEAD_DIM, KV_LEN),
                strides=(stride_kk, stride_kn),
                offsets=(0, kv_start),
                block_shape=(QK_HEAD_DIM_ROUNDED, BLOCK_N),
                order=(0, 1)
            )
            V_block_ptr = tl.make_block_ptr(
                base=V,
                shape=(KV_LEN, V_HEAD_DIM),
                strides=(stride_vn, stride_vk),
                offsets=(kv_start, 0),
                block_shape=(BLOCK_N, V_HEAD_DIM_ROUNDED),
                order=(1, 0)
            )
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0,
            q, K_block_ptr, V_block_ptr,
            desc_k, desc_v, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_start,
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_zq = tl.program_id(1) // HQ
    idx_hq = tl.program_id(1) % HQ
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :]

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

    xindex = idx_d + 128*idx_m + 43136*idx_hq + 1380352*idx_zq
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)

    if OUTPUT_LOGSUMEXP:
        off_hz = tl.program_id(1)
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)

@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0,
    q, K_block_ptr, V_block_ptr,
    desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
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
    USE_TMA : tl.constexpr = False
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128


    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        # Here IS_DIVISIBLE acts are the start_n = tl.multiple_of(start_n, BLOCK_N) from triton_fused_attention.
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0,
                q, K_block_ptr, V_block_ptr, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                start_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0,
                q, K_block_ptr, V_block_ptr, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                start_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )



        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        offs_n = offs_n + offset
        if not USE_TMA:
            K_block_ptr = tl.advance(K_block_ptr, (0, offset))
            V_block_ptr = tl.advance(V_block_ptr, (offset, 0))


    return acc, l_i, m_i


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
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr8, out_ptr0,
    q, K_block_ptr, V_block_ptr, desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    start_n,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
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
    USE_TMA : tl.constexpr = False
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128


    # -- load k --
    # NB reversed order to since K is transposed
    k = load_checked_block(K_block_ptr, SAFE_HEAD_DIM, IS_DIVISIBLE)

    if USE_TMA:
        k = tl.trans(k)
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    tmp0 = (qk)
    post_mod_scores = tmp0


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = (n)
        tmp2 = (m)
        tmp3 = (off_h)
        tmp4 = (off_z)
        tmp5 = tl.load(in_ptr8 + tmp1 + 337*tmp2 + 113569*tmp3 + 3634208*tmp4)
        mask_mod_output = tmp5


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    v = load_checked_block(V_block_ptr, IS_DIVISIBLE, SAFE_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i


@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/hz/chzn73fwfsvwfpta56s634src5kbtgmhpsph2zkx63h3wlklr3xc.py
# Topologically Sorted Source Nodes: [reshape], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   reshape => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 530055168}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44171264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 32)
    x2 = ((xindex // 4096) % 337)
    x3 = xindex // 1380352
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 43136*x1 + 1380352*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 337), (10784, 337, 1), torch.float32)
        buf1 = empty_strided_cuda((32, 32, 337, 128), (1380352, 43136, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_tem_fused_0.run(primals_1, primals_2, primals_3, buf0, primals_5, primals_4, primals_7, primals_8, primals_6, buf1, 11, 1024, 1, stream=stream0)
        buf3 = empty_strided_cuda((32, 337, 32, 128), (1380352, 4096, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [reshape], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf1, buf3, 44171264, stream=stream0)
    return (reinterpret_tensor(buf3, (32, 337, 4096), (1380352, 4096, 1), 0), primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, buf1, buf0, )


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
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
