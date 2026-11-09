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


# kernel path: /root/qihank/.cache/sync_bench/inductor/lk/clkplhvp3xetkxa3tkhfmabn76xvrfpdglyd6zj3hjie5cwjm3ci.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_for_fused_0 = async_compile.triton('triton_for_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.foreach(
    num_warps=8,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (32,): [['tt.divisibility', 16]], (36,): [['tt.divisibility', 16]], (52,): [['tt.divisibility', 16]], (68,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 36, 'min_blocks': 0, 'default_config': {'XBLOCK': 1024}, 'no_x_dim_0': False, 'xnumel_0': 1, 'no_x_dim_1': False, 'xnumel_1': 1, 'no_x_dim_2': False, 'xnumel_2': 1, 'no_x_dim_3': False, 'xnumel_3': 1, 'no_x_dim_4': False, 'xnumel_4': 1, 'no_x_dim_5': False, 'xnumel_5': 1, 'no_x_dim_6': False, 'xnumel_6': 1, 'no_x_dim_7': False, 'xnumel_7': 1, 'no_x_dim_8': False, 'xnumel_8': 1, 'no_x_dim_9': False, 'xnumel_9': 1, 'no_x_dim_10': False, 'xnumel_10': 1, 'no_x_dim_11': False, 'xnumel_11': 1, 'no_x_dim_12': False, 'xnumel_12': 1, 'no_x_dim_13': False, 'xnumel_13': 1, 'no_x_dim_14': False, 'xnumel_14': 1, 'no_x_dim_15': False, 'xnumel_15': 1, 'no_x_dim_16': False, 'xnumel_16': 1, 'no_x_dim_17': False, 'xnumel_17': 1, 'no_x_dim_18': False, 'xnumel_18': 1, 'no_x_dim_19': False, 'xnumel_19': 1, 'no_x_dim_20': False, 'xnumel_20': 1, 'no_x_dim_21': False, 'xnumel_21': 1, 'no_x_dim_22': False, 'xnumel_22': 1, 'no_x_dim_23': False, 'xnumel_23': 1, 'no_x_dim_24': False, 'xnumel_24': 1, 'no_x_dim_25': False, 'xnumel_25': 1, 'no_x_dim_26': False, 'xnumel_26': 1, 'no_x_dim_27': False, 'xnumel_27': 1, 'no_x_dim_28': False, 'xnumel_28': 1, 'no_x_dim_29': False, 'xnumel_29': 1, 'no_x_dim_30': False, 'xnumel_30': 1, 'no_x_dim_31': False, 'xnumel_31': 1, 'no_x_dim_32': False, 'xnumel_32': 1, 'no_x_dim_33': False, 'xnumel_33': 1, 'no_x_dim_34': False, 'xnumel_34': 1, 'no_x_dim_35': False, 'xnumel_35': 1}, 'kernel_name': 'triton_for_fused_0', 'mutated_arg_names': [], 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_for_fused_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35):
    pid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    num_xblocks_0 = tl.cdiv(1, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(1, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(1, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(1, XBLOCK)
    num_xblocks_4 = num_xblocks_3 + tl.cdiv(1, XBLOCK)
    num_xblocks_5 = num_xblocks_4 + tl.cdiv(1, XBLOCK)
    num_xblocks_6 = num_xblocks_5 + tl.cdiv(1, XBLOCK)
    num_xblocks_7 = num_xblocks_6 + tl.cdiv(1, XBLOCK)
    num_xblocks_8 = num_xblocks_7 + tl.cdiv(1, XBLOCK)
    num_xblocks_9 = num_xblocks_8 + tl.cdiv(1, XBLOCK)
    num_xblocks_10 = num_xblocks_9 + tl.cdiv(1, XBLOCK)
    num_xblocks_11 = num_xblocks_10 + tl.cdiv(1, XBLOCK)
    num_xblocks_12 = num_xblocks_11 + tl.cdiv(1, XBLOCK)
    num_xblocks_13 = num_xblocks_12 + tl.cdiv(1, XBLOCK)
    num_xblocks_14 = num_xblocks_13 + tl.cdiv(1, XBLOCK)
    num_xblocks_15 = num_xblocks_14 + tl.cdiv(1, XBLOCK)
    num_xblocks_16 = num_xblocks_15 + tl.cdiv(1, XBLOCK)
    num_xblocks_17 = num_xblocks_16 + tl.cdiv(1, XBLOCK)
    num_xblocks_18 = num_xblocks_17 + tl.cdiv(1, XBLOCK)
    num_xblocks_19 = num_xblocks_18 + tl.cdiv(1, XBLOCK)
    num_xblocks_20 = num_xblocks_19 + tl.cdiv(1, XBLOCK)
    num_xblocks_21 = num_xblocks_20 + tl.cdiv(1, XBLOCK)
    num_xblocks_22 = num_xblocks_21 + tl.cdiv(1, XBLOCK)
    num_xblocks_23 = num_xblocks_22 + tl.cdiv(1, XBLOCK)
    num_xblocks_24 = num_xblocks_23 + tl.cdiv(1, XBLOCK)
    num_xblocks_25 = num_xblocks_24 + tl.cdiv(1, XBLOCK)
    num_xblocks_26 = num_xblocks_25 + tl.cdiv(1, XBLOCK)
    num_xblocks_27 = num_xblocks_26 + tl.cdiv(1, XBLOCK)
    num_xblocks_28 = num_xblocks_27 + tl.cdiv(1, XBLOCK)
    num_xblocks_29 = num_xblocks_28 + tl.cdiv(1, XBLOCK)
    num_xblocks_30 = num_xblocks_29 + tl.cdiv(1, XBLOCK)
    num_xblocks_31 = num_xblocks_30 + tl.cdiv(1, XBLOCK)
    num_xblocks_32 = num_xblocks_31 + tl.cdiv(1, XBLOCK)
    num_xblocks_33 = num_xblocks_32 + tl.cdiv(1, XBLOCK)
    num_xblocks_34 = num_xblocks_33 + tl.cdiv(1, XBLOCK)
    num_xblocks_35 = num_xblocks_34 + tl.cdiv(1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp0 = tl.load(in_ptr0 + (0))
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
        tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp1, None)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp2 = tl.load(in_ptr1 + (0))
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
        tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp4 = tl.load(in_ptr2 + (0))
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
        tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, None)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp6 = tl.load(in_ptr3 + (0))
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
        tl.store(out_ptr3 + (tl.full([XBLOCK], 0, tl.int32)), tmp7, None)
    elif pid < num_xblocks_4:
        pid_offset = pid - num_xblocks_3
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp8 = tl.load(in_ptr4 + (0))
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
        tl.store(out_ptr4 + (tl.full([XBLOCK], 0, tl.int32)), tmp9, None)
    elif pid < num_xblocks_5:
        pid_offset = pid - num_xblocks_4
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp10 = tl.load(in_ptr5 + (0))
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
        tl.store(out_ptr5 + (tl.full([XBLOCK], 0, tl.int32)), tmp11, None)
    elif pid < num_xblocks_6:
        pid_offset = pid - num_xblocks_5
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp12 = tl.load(in_ptr6 + (0))
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
        tl.store(out_ptr6 + (tl.full([XBLOCK], 0, tl.int32)), tmp13, None)
    elif pid < num_xblocks_7:
        pid_offset = pid - num_xblocks_6
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp14 = tl.load(in_ptr7 + (0))
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
        tl.store(out_ptr7 + (tl.full([XBLOCK], 0, tl.int32)), tmp15, None)
    elif pid < num_xblocks_8:
        pid_offset = pid - num_xblocks_7
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp16 = tl.load(in_ptr8 + (0))
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
        tl.store(out_ptr8 + (tl.full([XBLOCK], 0, tl.int32)), tmp17, None)
    elif pid < num_xblocks_9:
        pid_offset = pid - num_xblocks_8
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp18 = tl.load(in_ptr9 + (0))
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
        tl.store(out_ptr9 + (tl.full([XBLOCK], 0, tl.int32)), tmp19, None)
    elif pid < num_xblocks_10:
        pid_offset = pid - num_xblocks_9
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp20 = tl.load(in_ptr10 + (0))
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
        tl.store(out_ptr10 + (tl.full([XBLOCK], 0, tl.int32)), tmp21, None)
    elif pid < num_xblocks_11:
        pid_offset = pid - num_xblocks_10
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp22 = tl.load(in_ptr11 + (0))
        tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
        tl.store(out_ptr11 + (tl.full([XBLOCK], 0, tl.int32)), tmp23, None)
    elif pid < num_xblocks_12:
        pid_offset = pid - num_xblocks_11
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp24 = tl.load(in_ptr12 + (0))
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
        tl.store(out_ptr12 + (tl.full([XBLOCK], 0, tl.int32)), tmp25, None)
    elif pid < num_xblocks_13:
        pid_offset = pid - num_xblocks_12
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp26 = tl.load(in_ptr13 + (0))
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
        tl.store(out_ptr13 + (tl.full([XBLOCK], 0, tl.int32)), tmp27, None)
    elif pid < num_xblocks_14:
        pid_offset = pid - num_xblocks_13
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp28 = tl.load(in_ptr14 + (0))
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
        tl.store(out_ptr14 + (tl.full([XBLOCK], 0, tl.int32)), tmp29, None)
    elif pid < num_xblocks_15:
        pid_offset = pid - num_xblocks_14
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp30 = tl.load(in_ptr15 + (0))
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
        tl.store(out_ptr15 + (tl.full([XBLOCK], 0, tl.int32)), tmp31, None)
    elif pid < num_xblocks_16:
        pid_offset = pid - num_xblocks_15
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp32 = tl.load(in_ptr16 + (0))
        tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
        tl.store(out_ptr16 + (tl.full([XBLOCK], 0, tl.int32)), tmp33, None)
    elif pid < num_xblocks_17:
        pid_offset = pid - num_xblocks_16
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp34 = tl.load(in_ptr17 + (0))
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
        tl.store(out_ptr17 + (tl.full([XBLOCK], 0, tl.int32)), tmp35, None)
    elif pid < num_xblocks_18:
        pid_offset = pid - num_xblocks_17
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp36 = tl.load(in_ptr18 + (0))
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
        tl.store(out_ptr18 + (tl.full([XBLOCK], 0, tl.int32)), tmp37, None)
    elif pid < num_xblocks_19:
        pid_offset = pid - num_xblocks_18
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp38 = tl.load(in_ptr19 + (0))
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
        tl.store(out_ptr19 + (tl.full([XBLOCK], 0, tl.int32)), tmp39, None)
    elif pid < num_xblocks_20:
        pid_offset = pid - num_xblocks_19
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp40 = tl.load(in_ptr20 + (0))
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
        tl.store(out_ptr20 + (tl.full([XBLOCK], 0, tl.int32)), tmp41, None)
    elif pid < num_xblocks_21:
        pid_offset = pid - num_xblocks_20
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp42 = tl.load(in_ptr21 + (0))
        tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
        tl.store(out_ptr21 + (tl.full([XBLOCK], 0, tl.int32)), tmp43, None)
    elif pid < num_xblocks_22:
        pid_offset = pid - num_xblocks_21
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp44 = tl.load(in_ptr22 + (0))
        tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
        tl.store(out_ptr22 + (tl.full([XBLOCK], 0, tl.int32)), tmp45, None)
    elif pid < num_xblocks_23:
        pid_offset = pid - num_xblocks_22
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp46 = tl.load(in_ptr23 + (0))
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
        tl.store(out_ptr23 + (tl.full([XBLOCK], 0, tl.int32)), tmp47, None)
    elif pid < num_xblocks_24:
        pid_offset = pid - num_xblocks_23
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp48 = tl.load(in_ptr24 + (0))
        tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
        tl.store(out_ptr24 + (tl.full([XBLOCK], 0, tl.int32)), tmp49, None)
    elif pid < num_xblocks_25:
        pid_offset = pid - num_xblocks_24
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp50 = tl.load(in_ptr25 + (0))
        tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
        tl.store(out_ptr25 + (tl.full([XBLOCK], 0, tl.int32)), tmp51, None)
    elif pid < num_xblocks_26:
        pid_offset = pid - num_xblocks_25
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp52 = tl.load(in_ptr26 + (0))
        tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
        tl.store(out_ptr26 + (tl.full([XBLOCK], 0, tl.int32)), tmp53, None)
    elif pid < num_xblocks_27:
        pid_offset = pid - num_xblocks_26
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp54 = tl.load(in_ptr27 + (0))
        tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
        tl.store(out_ptr27 + (tl.full([XBLOCK], 0, tl.int32)), tmp55, None)
    elif pid < num_xblocks_28:
        pid_offset = pid - num_xblocks_27
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp56 = tl.load(in_ptr28 + (0))
        tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
        tl.store(out_ptr28 + (tl.full([XBLOCK], 0, tl.int32)), tmp57, None)
    elif pid < num_xblocks_29:
        pid_offset = pid - num_xblocks_28
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp58 = tl.load(in_ptr29 + (0))
        tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
        tl.store(out_ptr29 + (tl.full([XBLOCK], 0, tl.int32)), tmp59, None)
    elif pid < num_xblocks_30:
        pid_offset = pid - num_xblocks_29
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp60 = tl.load(in_ptr30 + (0))
        tmp61 = tl.broadcast_to(tmp60, [XBLOCK])
        tl.store(out_ptr30 + (tl.full([XBLOCK], 0, tl.int32)), tmp61, None)
    elif pid < num_xblocks_31:
        pid_offset = pid - num_xblocks_30
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp62 = tl.load(in_ptr31 + (0))
        tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
        tl.store(out_ptr31 + (tl.full([XBLOCK], 0, tl.int32)), tmp63, None)
    elif pid < num_xblocks_32:
        pid_offset = pid - num_xblocks_31
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp64 = tl.load(in_ptr32 + (0))
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
        tl.store(out_ptr32 + (tl.full([XBLOCK], 0, tl.int32)), tmp65, None)
    elif pid < num_xblocks_33:
        pid_offset = pid - num_xblocks_32
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp66 = tl.load(in_ptr33 + (0))
        tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
        tl.store(out_ptr33 + (tl.full([XBLOCK], 0, tl.int32)), tmp67, None)
    elif pid < num_xblocks_34:
        pid_offset = pid - num_xblocks_33
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp68 = tl.load(in_ptr34 + (0))
        tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
        tl.store(out_ptr34 + (tl.full([XBLOCK], 0, tl.int32)), tmp69, None)
    elif pid < num_xblocks_35:
        pid_offset = pid - num_xblocks_34
        xnumel = 1
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        tmp70 = tl.load(in_ptr35 + (0))
        tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
        tl.store(out_ptr35 + (tl.full([XBLOCK], 0, tl.int32)), tmp71, None)
    else:
        pass
''', device_str='cuda')


# kernel path: /root/qihank/.cache/sync_bench/inductor/oz/coz7sjh6lpq7377c7zd3di66qmtkayry6zjalz23aizpy44klolh.py
# Topologically Sorted Source Nodes: [inv_lens, f_i, all_scores, all_probs], Original ATen: [aten.reciprocal, aten.mul, aten.div, aten.gather, aten.sigmoid, aten.neg, aten.sum, aten.add, aten.sigmoid_backward]
# Source node to ATen node mapping:
#   all_probs => div
#   all_scores => sigmoid
#   f_i => mul_1
#   inv_lens => mul, reciprocal
# Graph fragment:
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%clamp_min,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 8.0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem, %mul), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %mul_1), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, %clamp_min), kwargs = {})
#   %gather : [num_users=2] = call_function[target=torch.ops.aten.gather.default](args = (%div_3, 0, %expand), kwargs = {})
#   %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%primals_4,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sigmoid, %sum_1), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %sum_1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%gather,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %div_5), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%gather, %sum_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_6, %expand_3), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %sub), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %mul_5), kwargs = {})
triton_per_fused_add_div_gather_mul_neg_reciprocal_sigmoid_sigmoid_backward_sum_1 = async_compile.triton('triton_per_fused_add_div_gather_mul_neg_reciprocal_sigmoid_sigmoid_backward_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_gather_mul_neg_reciprocal_sigmoid_sigmoid_backward_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_gather_mul_neg_reciprocal_sigmoid_sigmoid_backward_sum_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 58752
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, R0_BLOCK], 1152, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 1152)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 1152")
    tmp6 = tl.load(in_ptr1 + (((tmp4 // 32) % 36)), xmask, eviction_policy='evict_last')
    tmp7 = 0.03125
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr2 + (r0_1 + 32*tmp4), xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (tmp4), xmask, eviction_policy='evict_last')
    tmp11 = tl.full([1, 1], 1, tl.int32)
    tmp12 = (tmp11 / tmp10)
    tmp13 = 8.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp9 * tmp14
    tmp16 = tmp8 * tmp15
    tmp17 = (tmp16 / tmp10)
    tmp18 = -tmp17
    tmp20 = tl.sigmoid(tmp19)
    tmp22 = (tmp20 / tmp21)
    tmp23 = (tmp22 / tmp21)
    tmp24 = tmp18 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = (tmp17 / tmp21)
    tmp30 = tmp29 + tmp28
    tmp31 = 1.0
    tmp32 = tmp31 - tmp20
    tmp33 = tmp20 * tmp32
    tmp34 = tmp30 * tmp33
    tl.store(out_ptr1 + (r0_1 + 32*x0), tmp34, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_4, sum_1, clamp_min, getitem, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36 = args
    args.clear()
    assert_size_stride(primals_1, (58752, ), (1, ))
    assert_size_stride(primals_4, (58752, 32), (32, 1))
    assert_size_stride(sum_1, (58752, 1), (1, 1))
    assert_size_stride(clamp_min, (1152, 1), (1, 1))
    assert_size_stride(getitem, (1152, 32), (32, 1))
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf36 = empty_strided_cuda((36, ), (1, ), torch.float32)
        buf0 = reinterpret_tensor(buf36, (1, ), (1, ), 0)  # alias
        buf1 = reinterpret_tensor(buf36, (1, ), (1, ), 1)  # alias
        buf2 = reinterpret_tensor(buf36, (1, ), (1, ), 2)  # alias
        buf3 = reinterpret_tensor(buf36, (1, ), (1, ), 3)  # alias
        buf4 = reinterpret_tensor(buf36, (1, ), (1, ), 4)  # alias
        buf5 = reinterpret_tensor(buf36, (1, ), (1, ), 5)  # alias
        buf6 = reinterpret_tensor(buf36, (1, ), (1, ), 6)  # alias
        buf7 = reinterpret_tensor(buf36, (1, ), (1, ), 7)  # alias
        buf8 = reinterpret_tensor(buf36, (1, ), (1, ), 8)  # alias
        buf9 = reinterpret_tensor(buf36, (1, ), (1, ), 9)  # alias
        buf10 = reinterpret_tensor(buf36, (1, ), (1, ), 10)  # alias
        buf11 = reinterpret_tensor(buf36, (1, ), (1, ), 11)  # alias
        buf12 = reinterpret_tensor(buf36, (1, ), (1, ), 12)  # alias
        buf13 = reinterpret_tensor(buf36, (1, ), (1, ), 13)  # alias
        buf14 = reinterpret_tensor(buf36, (1, ), (1, ), 14)  # alias
        buf15 = reinterpret_tensor(buf36, (1, ), (1, ), 15)  # alias
        buf16 = reinterpret_tensor(buf36, (1, ), (1, ), 16)  # alias
        buf17 = reinterpret_tensor(buf36, (1, ), (1, ), 17)  # alias
        buf18 = reinterpret_tensor(buf36, (1, ), (1, ), 18)  # alias
        buf19 = reinterpret_tensor(buf36, (1, ), (1, ), 19)  # alias
        buf20 = reinterpret_tensor(buf36, (1, ), (1, ), 20)  # alias
        buf21 = reinterpret_tensor(buf36, (1, ), (1, ), 21)  # alias
        buf22 = reinterpret_tensor(buf36, (1, ), (1, ), 22)  # alias
        buf23 = reinterpret_tensor(buf36, (1, ), (1, ), 23)  # alias
        buf24 = reinterpret_tensor(buf36, (1, ), (1, ), 24)  # alias
        buf25 = reinterpret_tensor(buf36, (1, ), (1, ), 25)  # alias
        buf26 = reinterpret_tensor(buf36, (1, ), (1, ), 26)  # alias
        buf27 = reinterpret_tensor(buf36, (1, ), (1, ), 27)  # alias
        buf28 = reinterpret_tensor(buf36, (1, ), (1, ), 28)  # alias
        buf29 = reinterpret_tensor(buf36, (1, ), (1, ), 29)  # alias
        buf30 = reinterpret_tensor(buf36, (1, ), (1, ), 30)  # alias
        buf31 = reinterpret_tensor(buf36, (1, ), (1, ), 31)  # alias
        buf32 = reinterpret_tensor(buf36, (1, ), (1, ), 32)  # alias
        buf33 = reinterpret_tensor(buf36, (1, ), (1, ), 33)  # alias
        buf34 = reinterpret_tensor(buf36, (1, ), (1, ), 34)  # alias
        buf35 = reinterpret_tensor(buf36, (1, ), (1, ), 35)  # alias
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_for_fused_0.run(tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, stream=stream0)
        del tangents_1
        del tangents_10
        del tangents_11
        del tangents_12
        del tangents_13
        del tangents_14
        del tangents_15
        del tangents_16
        del tangents_17
        del tangents_18
        del tangents_19
        del tangents_2
        del tangents_20
        del tangents_21
        del tangents_22
        del tangents_23
        del tangents_24
        del tangents_25
        del tangents_26
        del tangents_27
        del tangents_28
        del tangents_29
        del tangents_3
        del tangents_30
        del tangents_31
        del tangents_32
        del tangents_33
        del tangents_34
        del tangents_35
        del tangents_36
        del tangents_4
        del tangents_5
        del tangents_6
        del tangents_7
        del tangents_8
        del tangents_9
        buf38 = empty_strided_cuda((58752, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [inv_lens, f_i, all_scores, all_probs], Original ATen: [aten.reciprocal, aten.mul, aten.div, aten.gather, aten.sigmoid, aten.neg, aten.sum, aten.add, aten.sigmoid_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_gather_mul_neg_reciprocal_sigmoid_sigmoid_backward_sum_1.run(primals_1, buf36, getitem, clamp_min, primals_4, sum_1, buf38, 58752, 32, stream=stream0)
        del buf0
        del buf1
        del buf10
        del buf11
        del buf12
        del buf13
        del buf14
        del buf15
        del buf16
        del buf17
        del buf18
        del buf19
        del buf2
        del buf20
        del buf21
        del buf22
        del buf23
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del buf29
        del buf3
        del buf30
        del buf31
        del buf32
        del buf33
        del buf34
        del buf35
        del buf36
        del buf4
        del buf5
        del buf6
        del buf7
        del buf8
        del buf9
        del clamp_min
        del getitem
        del primals_1
        del primals_4
        del sum_1
    return (None, None, None, buf38, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((58752, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((58752, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    sum_1 = rand_strided((58752, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    clamp_min = rand_strided((1152, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    getitem = rand_strided((1152, 32), (32, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([primals_1, primals_4, sum_1, clamp_min, getitem, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, tangents_22, tangents_23, tangents_24, tangents_25, tangents_26, tangents_27, tangents_28, tangents_29, tangents_30, tangents_31, tangents_32, tangents_33, tangents_34, tangents_35, tangents_36])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
