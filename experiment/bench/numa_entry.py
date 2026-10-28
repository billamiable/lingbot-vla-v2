"""Bind a torchrun worker to the CPUs of its GPU's NUMA node, then exec the given script.

    torchrun ... numa_entry.py SCRIPT [ARGS ...]

bench.sh exports BENCH_CPUS_<local rank> (a cpulist such as 0-63,128-191) for every GPU; the
binding is inherited by the DataLoader workers. Without the variable the worker runs unbound.
"""

import os
import sys


def parse(cpulist):
    cpus = set()
    for part in cpulist.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


cpulist = os.environ.get(f"BENCH_CPUS_{os.environ['LOCAL_RANK']}")
if cpulist:
    os.sched_setaffinity(0, parse(cpulist))
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
