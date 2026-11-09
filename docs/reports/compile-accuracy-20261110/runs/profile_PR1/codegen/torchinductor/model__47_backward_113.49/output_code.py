# AOT ID: ['47_backward']
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


async_compile.wait(globals())
del async_compile

def call(args):
    tangents_1, = args
    args.clear()
    assert_size_stride(tangents_1, (58752, 32), (32, 1))
    return (reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 0), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 52224), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 104448), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 156672), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 208896), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 261120), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 313344), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 365568), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 417792), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 470016), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 522240), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 574464), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 626688), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 678912), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 731136), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 783360), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 835584), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 887808), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 940032), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 992256), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1044480), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1096704), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1148928), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1201152), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1253376), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1305600), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1357824), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1410048), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1462272), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1514496), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1566720), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1618944), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1671168), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1723392), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1775616), reinterpret_tensor(tangents_1, (1632, 32), (32, 1), 1827840), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    tangents_1 = rand_strided((58752, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
