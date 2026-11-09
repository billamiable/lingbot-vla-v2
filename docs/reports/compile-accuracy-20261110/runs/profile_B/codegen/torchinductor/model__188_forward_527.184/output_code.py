# AOT ID: ['188_forward']
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
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
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


# kernel path: /root/qihank/.cache/sync_bench/inductor/zr/czrcyz7o3ghzjtbol22hxz7fy2lkrkdacnpbliu67eka3s2xlhtd.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=110, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]], (20,): [['tt.divisibility', 16]], (21,): [['tt.divisibility', 16]], (22,): [['tt.divisibility', 16]], (23,): [['tt.divisibility', 16]], (24,): [['tt.divisibility', 16]], (25,): [['tt.divisibility', 16]], (26,): [['tt.divisibility', 16]], (27,): [['tt.divisibility', 16]], (28,): [['tt.divisibility', 16]], (29,): [['tt.divisibility', 16]], (30,): [['tt.divisibility', 16]], (31,): [['tt.divisibility', 16]], (32,): [['tt.divisibility', 16]], (33,): [['tt.divisibility', 16]], (34,): [['tt.divisibility', 16]], (35,): [['tt.divisibility', 16]], (36,): [['tt.divisibility', 16]], (37,): [['tt.divisibility', 16]], (38,): [['tt.divisibility', 16]], (39,): [['tt.divisibility', 16]], (40,): [['tt.divisibility', 16]], (41,): [['tt.divisibility', 16]], (42,): [['tt.divisibility', 16]], (43,): [['tt.divisibility', 16]], (44,): [['tt.divisibility', 16]], (45,): [['tt.divisibility', 16]], (46,): [['tt.divisibility', 16]], (47,): [['tt.divisibility', 16]], (48,): [['tt.divisibility', 16]], (49,): [['tt.divisibility', 16]], (50,): [['tt.divisibility', 16]], (51,): [['tt.divisibility', 16]], (52,): [['tt.divisibility', 16]], (53,): [['tt.divisibility', 16]], (54,): [['tt.divisibility', 16]], (55,): [['tt.divisibility', 16]], (56,): [['tt.divisibility', 16]], (57,): [['tt.divisibility', 16]], (58,): [['tt.divisibility', 16]], (59,): [['tt.divisibility', 16]], (60,): [['tt.divisibility', 16]], (61,): [['tt.divisibility', 16]], (62,): [['tt.divisibility', 16]], (63,): [['tt.divisibility', 16]], (64,): [['tt.divisibility', 16]], (65,): [['tt.divisibility', 16]], (66,): [['tt.divisibility', 16]], (67,): [['tt.divisibility', 16]], (68,): [['tt.divisibility', 16]], (69,): [['tt.divisibility', 16]], (70,): [['tt.divisibility', 16]], (71,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 36, 'min_blocks': 0, 'default_config': {'XBLOCK': 1024}, 'no_x_dim_0': False, 'xnumel_0': 52224, 'no_x_dim_1': False, 'xnumel_1': 52224, 'no_x_dim_2': False, 'xnumel_2': 52224, 'no_x_dim_3': False, 'xnumel_3': 52224, 'no_x_dim_4': False, 'xnumel_4': 52224, 'no_x_dim_5': False, 'xnumel_5': 52224, 'no_x_dim_6': False, 'xnumel_6': 52224, 'no_x_dim_7': False, 'xnumel_7': 52224, 'no_x_dim_8': False, 'xnumel_8': 52224, 'no_x_dim_9': False, 'xnumel_9': 52224, 'no_x_dim_10': False, 'xnumel_10': 52224, 'no_x_dim_11': False, 'xnumel_11': 52224, 'no_x_dim_12': False, 'xnumel_12': 52224, 'no_x_dim_13': False, 'xnumel_13': 52224, 'no_x_dim_14': False, 'xnumel_14': 52224, 'no_x_dim_15': False, 'xnumel_15': 52224, 'no_x_dim_16': False, 'xnumel_16': 52224, 'no_x_dim_17': False, 'xnumel_17': 52224, 'no_x_dim_18': False, 'xnumel_18': 52224, 'no_x_dim_19': False, 'xnumel_19': 52224, 'no_x_dim_20': False, 'xnumel_20': 52224, 'no_x_dim_21': False, 'xnumel_21': 52224, 'no_x_dim_22': False, 'xnumel_22': 52224, 'no_x_dim_23': False, 'xnumel_23': 52224, 'no_x_dim_24': False, 'xnumel_24': 52224, 'no_x_dim_25': False, 'xnumel_25': 52224, 'no_x_dim_26': False, 'xnumel_26': 52224, 'no_x_dim_27': False, 'xnumel_27': 52224, 'no_x_dim_28': False, 'xnumel_28': 52224, 'no_x_dim_29': False, 'xnumel_29': 52224, 'no_x_dim_30': False, 'xnumel_30': 52224, 'no_x_dim_31': False, 'xnumel_31': 52224, 'no_x_dim_32': False, 'xnumel_32': 52224, 'no_x_dim_33': False, 'xnumel_33': 52224, 'no_x_dim_34': False, 'xnumel_34': 52224, 'no_x_dim_35': False, 'xnumel_35': 52224}, 'kernel_name': 'triton_for_fused_0', 'mutated_arg_names': [], 'backend_hash': 'ADD422BD172A0988DC6B2FB6CF49464040C5993A7FDC4BFD9B7F4CB74AEE5ACF', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_for_fused_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35):
    pid = tl.program_id(0)
    XBLOCK: tl.constexpr = 1024
    num_xblocks_0 = tl.cdiv(52224, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(52224, XBLOCK)
    num_xblocks_2 = num_xblocks_1 + tl.cdiv(52224, XBLOCK)
    num_xblocks_3 = num_xblocks_2 + tl.cdiv(52224, XBLOCK)
    num_xblocks_4 = num_xblocks_3 + tl.cdiv(52224, XBLOCK)
    num_xblocks_5 = num_xblocks_4 + tl.cdiv(52224, XBLOCK)
    num_xblocks_6 = num_xblocks_5 + tl.cdiv(52224, XBLOCK)
    num_xblocks_7 = num_xblocks_6 + tl.cdiv(52224, XBLOCK)
    num_xblocks_8 = num_xblocks_7 + tl.cdiv(52224, XBLOCK)
    num_xblocks_9 = num_xblocks_8 + tl.cdiv(52224, XBLOCK)
    num_xblocks_10 = num_xblocks_9 + tl.cdiv(52224, XBLOCK)
    num_xblocks_11 = num_xblocks_10 + tl.cdiv(52224, XBLOCK)
    num_xblocks_12 = num_xblocks_11 + tl.cdiv(52224, XBLOCK)
    num_xblocks_13 = num_xblocks_12 + tl.cdiv(52224, XBLOCK)
    num_xblocks_14 = num_xblocks_13 + tl.cdiv(52224, XBLOCK)
    num_xblocks_15 = num_xblocks_14 + tl.cdiv(52224, XBLOCK)
    num_xblocks_16 = num_xblocks_15 + tl.cdiv(52224, XBLOCK)
    num_xblocks_17 = num_xblocks_16 + tl.cdiv(52224, XBLOCK)
    num_xblocks_18 = num_xblocks_17 + tl.cdiv(52224, XBLOCK)
    num_xblocks_19 = num_xblocks_18 + tl.cdiv(52224, XBLOCK)
    num_xblocks_20 = num_xblocks_19 + tl.cdiv(52224, XBLOCK)
    num_xblocks_21 = num_xblocks_20 + tl.cdiv(52224, XBLOCK)
    num_xblocks_22 = num_xblocks_21 + tl.cdiv(52224, XBLOCK)
    num_xblocks_23 = num_xblocks_22 + tl.cdiv(52224, XBLOCK)
    num_xblocks_24 = num_xblocks_23 + tl.cdiv(52224, XBLOCK)
    num_xblocks_25 = num_xblocks_24 + tl.cdiv(52224, XBLOCK)
    num_xblocks_26 = num_xblocks_25 + tl.cdiv(52224, XBLOCK)
    num_xblocks_27 = num_xblocks_26 + tl.cdiv(52224, XBLOCK)
    num_xblocks_28 = num_xblocks_27 + tl.cdiv(52224, XBLOCK)
    num_xblocks_29 = num_xblocks_28 + tl.cdiv(52224, XBLOCK)
    num_xblocks_30 = num_xblocks_29 + tl.cdiv(52224, XBLOCK)
    num_xblocks_31 = num_xblocks_30 + tl.cdiv(52224, XBLOCK)
    num_xblocks_32 = num_xblocks_31 + tl.cdiv(52224, XBLOCK)
    num_xblocks_33 = num_xblocks_32 + tl.cdiv(52224, XBLOCK)
    num_xblocks_34 = num_xblocks_33 + tl.cdiv(52224, XBLOCK)
    num_xblocks_35 = num_xblocks_34 + tl.cdiv(52224, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0), tmp0, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x1 = xindex
        tmp1 = tl.load(in_ptr1 + (x1), xmask)
        tl.store(out_ptr1 + (x1), tmp1, xmask)
    elif pid < num_xblocks_2:
        pid_offset = pid - num_xblocks_1
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        tmp2 = tl.load(in_ptr2 + (x2), xmask)
        tl.store(out_ptr2 + (x2), tmp2, xmask)
    elif pid < num_xblocks_3:
        pid_offset = pid - num_xblocks_2
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x3 = xindex
        tmp3 = tl.load(in_ptr3 + (x3), xmask)
        tl.store(out_ptr3 + (x3), tmp3, xmask)
    elif pid < num_xblocks_4:
        pid_offset = pid - num_xblocks_3
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x4 = xindex
        tmp4 = tl.load(in_ptr4 + (x4), xmask)
        tl.store(out_ptr4 + (x4), tmp4, xmask)
    elif pid < num_xblocks_5:
        pid_offset = pid - num_xblocks_4
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x5 = xindex
        tmp5 = tl.load(in_ptr5 + (x5), xmask)
        tl.store(out_ptr5 + (x5), tmp5, xmask)
    elif pid < num_xblocks_6:
        pid_offset = pid - num_xblocks_5
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x6 = xindex
        tmp6 = tl.load(in_ptr6 + (x6), xmask)
        tl.store(out_ptr6 + (x6), tmp6, xmask)
    elif pid < num_xblocks_7:
        pid_offset = pid - num_xblocks_6
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x7 = xindex
        tmp7 = tl.load(in_ptr7 + (x7), xmask)
        tl.store(out_ptr7 + (x7), tmp7, xmask)
    elif pid < num_xblocks_8:
        pid_offset = pid - num_xblocks_7
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x8 = xindex
        tmp8 = tl.load(in_ptr8 + (x8), xmask)
        tl.store(out_ptr8 + (x8), tmp8, xmask)
    elif pid < num_xblocks_9:
        pid_offset = pid - num_xblocks_8
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x9 = xindex
        tmp9 = tl.load(in_ptr9 + (x9), xmask)
        tl.store(out_ptr9 + (x9), tmp9, xmask)
    elif pid < num_xblocks_10:
        pid_offset = pid - num_xblocks_9
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x10 = xindex
        tmp10 = tl.load(in_ptr10 + (x10), xmask)
        tl.store(out_ptr10 + (x10), tmp10, xmask)
    elif pid < num_xblocks_11:
        pid_offset = pid - num_xblocks_10
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x11 = xindex
        tmp11 = tl.load(in_ptr11 + (x11), xmask)
        tl.store(out_ptr11 + (x11), tmp11, xmask)
    elif pid < num_xblocks_12:
        pid_offset = pid - num_xblocks_11
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x12 = xindex
        tmp12 = tl.load(in_ptr12 + (x12), xmask)
        tl.store(out_ptr12 + (x12), tmp12, xmask)
    elif pid < num_xblocks_13:
        pid_offset = pid - num_xblocks_12
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x13 = xindex
        tmp13 = tl.load(in_ptr13 + (x13), xmask)
        tl.store(out_ptr13 + (x13), tmp13, xmask)
    elif pid < num_xblocks_14:
        pid_offset = pid - num_xblocks_13
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x14 = xindex
        tmp14 = tl.load(in_ptr14 + (x14), xmask)
        tl.store(out_ptr14 + (x14), tmp14, xmask)
    elif pid < num_xblocks_15:
        pid_offset = pid - num_xblocks_14
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x15 = xindex
        tmp15 = tl.load(in_ptr15 + (x15), xmask)
        tl.store(out_ptr15 + (x15), tmp15, xmask)
    elif pid < num_xblocks_16:
        pid_offset = pid - num_xblocks_15
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x16 = xindex
        tmp16 = tl.load(in_ptr16 + (x16), xmask)
        tl.store(out_ptr16 + (x16), tmp16, xmask)
    elif pid < num_xblocks_17:
        pid_offset = pid - num_xblocks_16
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x17 = xindex
        tmp17 = tl.load(in_ptr17 + (x17), xmask)
        tl.store(out_ptr17 + (x17), tmp17, xmask)
    elif pid < num_xblocks_18:
        pid_offset = pid - num_xblocks_17
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x18 = xindex
        tmp18 = tl.load(in_ptr18 + (x18), xmask)
        tl.store(out_ptr18 + (x18), tmp18, xmask)
    elif pid < num_xblocks_19:
        pid_offset = pid - num_xblocks_18
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x19 = xindex
        tmp19 = tl.load(in_ptr19 + (x19), xmask)
        tl.store(out_ptr19 + (x19), tmp19, xmask)
    elif pid < num_xblocks_20:
        pid_offset = pid - num_xblocks_19
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x20 = xindex
        tmp20 = tl.load(in_ptr20 + (x20), xmask)
        tl.store(out_ptr20 + (x20), tmp20, xmask)
    elif pid < num_xblocks_21:
        pid_offset = pid - num_xblocks_20
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x21 = xindex
        tmp21 = tl.load(in_ptr21 + (x21), xmask)
        tl.store(out_ptr21 + (x21), tmp21, xmask)
    elif pid < num_xblocks_22:
        pid_offset = pid - num_xblocks_21
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x22 = xindex
        tmp22 = tl.load(in_ptr22 + (x22), xmask)
        tl.store(out_ptr22 + (x22), tmp22, xmask)
    elif pid < num_xblocks_23:
        pid_offset = pid - num_xblocks_22
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x23 = xindex
        tmp23 = tl.load(in_ptr23 + (x23), xmask)
        tl.store(out_ptr23 + (x23), tmp23, xmask)
    elif pid < num_xblocks_24:
        pid_offset = pid - num_xblocks_23
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x24 = xindex
        tmp24 = tl.load(in_ptr24 + (x24), xmask)
        tl.store(out_ptr24 + (x24), tmp24, xmask)
    elif pid < num_xblocks_25:
        pid_offset = pid - num_xblocks_24
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x25 = xindex
        tmp25 = tl.load(in_ptr25 + (x25), xmask)
        tl.store(out_ptr25 + (x25), tmp25, xmask)
    elif pid < num_xblocks_26:
        pid_offset = pid - num_xblocks_25
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x26 = xindex
        tmp26 = tl.load(in_ptr26 + (x26), xmask)
        tl.store(out_ptr26 + (x26), tmp26, xmask)
    elif pid < num_xblocks_27:
        pid_offset = pid - num_xblocks_26
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x27 = xindex
        tmp27 = tl.load(in_ptr27 + (x27), xmask)
        tl.store(out_ptr27 + (x27), tmp27, xmask)
    elif pid < num_xblocks_28:
        pid_offset = pid - num_xblocks_27
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x28 = xindex
        tmp28 = tl.load(in_ptr28 + (x28), xmask)
        tl.store(out_ptr28 + (x28), tmp28, xmask)
    elif pid < num_xblocks_29:
        pid_offset = pid - num_xblocks_28
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x29 = xindex
        tmp29 = tl.load(in_ptr29 + (x29), xmask)
        tl.store(out_ptr29 + (x29), tmp29, xmask)
    elif pid < num_xblocks_30:
        pid_offset = pid - num_xblocks_29
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x30 = xindex
        tmp30 = tl.load(in_ptr30 + (x30), xmask)
        tl.store(out_ptr30 + (x30), tmp30, xmask)
    elif pid < num_xblocks_31:
        pid_offset = pid - num_xblocks_30
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x31 = xindex
        tmp31 = tl.load(in_ptr31 + (x31), xmask)
        tl.store(out_ptr31 + (x31), tmp31, xmask)
    elif pid < num_xblocks_32:
        pid_offset = pid - num_xblocks_31
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x32 = xindex
        tmp32 = tl.load(in_ptr32 + (x32), xmask)
        tl.store(out_ptr32 + (x32), tmp32, xmask)
    elif pid < num_xblocks_33:
        pid_offset = pid - num_xblocks_32
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x33 = xindex
        tmp33 = tl.load(in_ptr33 + (x33), xmask)
        tl.store(out_ptr33 + (x33), tmp33, xmask)
    elif pid < num_xblocks_34:
        pid_offset = pid - num_xblocks_33
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x34 = xindex
        tmp34 = tl.load(in_ptr34 + (x34), xmask)
        tl.store(out_ptr34 + (x34), tmp34, xmask)
    elif pid < num_xblocks_35:
        pid_offset = pid - num_xblocks_34
        xnumel = 52224
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x35 = xindex
        tmp35 = tl.load(in_ptr35 + (x35), xmask)
        tl.store(out_ptr35 + (x35), tmp35, xmask)
    else:
        pass
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36 = args
    args.clear()
    assert_size_stride(primals_1, (1632, 32), (32, 1))
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf36 = empty_strided_cuda((58752, 32), (32, 1), torch.float32)
        buf0 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 0)  # alias
        buf1 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 52224)  # alias
        buf2 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 104448)  # alias
        buf3 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 156672)  # alias
        buf4 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 208896)  # alias
        buf5 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 261120)  # alias
        buf6 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 313344)  # alias
        buf7 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 365568)  # alias
        buf8 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 417792)  # alias
        buf9 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 470016)  # alias
        buf10 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 522240)  # alias
        buf11 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 574464)  # alias
        buf12 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 626688)  # alias
        buf13 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 678912)  # alias
        buf14 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 731136)  # alias
        buf15 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 783360)  # alias
        buf16 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 835584)  # alias
        buf17 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 887808)  # alias
        buf18 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 940032)  # alias
        buf19 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 992256)  # alias
        buf20 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1044480)  # alias
        buf21 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1096704)  # alias
        buf22 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1148928)  # alias
        buf23 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1201152)  # alias
        buf24 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1253376)  # alias
        buf25 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1305600)  # alias
        buf26 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1357824)  # alias
        buf27 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1410048)  # alias
        buf28 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1462272)  # alias
        buf29 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1514496)  # alias
        buf30 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1566720)  # alias
        buf31 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1618944)  # alias
        buf32 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1671168)  # alias
        buf33 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1723392)  # alias
        buf34 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1775616)  # alias
        buf35 = reinterpret_tensor(buf36, (1632, 32), (32, 1), 1827840)  # alias
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_for_fused_0.run(primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, stream=stream0)
        del primals_1
        del primals_10
        del primals_11
        del primals_12
        del primals_13
        del primals_14
        del primals_15
        del primals_16
        del primals_17
        del primals_18
        del primals_19
        del primals_2
        del primals_20
        del primals_21
        del primals_22
        del primals_23
        del primals_24
        del primals_25
        del primals_26
        del primals_27
        del primals_28
        del primals_29
        del primals_3
        del primals_30
        del primals_31
        del primals_32
        del primals_33
        del primals_34
        del primals_35
        del primals_36
        del primals_4
        del primals_5
        del primals_6
        del primals_7
        del primals_8
        del primals_9
    return (buf36, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1632, 32), (32, 1), device='cuda:0', dtype=torch.float32)
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
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
