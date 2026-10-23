"""Measure what a disk delivers for random reads, to check the emulated cloud disk.

    python read_bench.py <file> [--seconds 15]

Uses O_DIRECT so the page cache does not answer. Three tests:
  lat   4 KiB reads, one at a time            -> latency
  iops  4 KiB reads, 64 threads               -> IOPS
  bw    2 MiB reads (one cache row), 16 threads -> MB/s
"""

import argparse
import mmap
import os
import random
import threading
import time


def run(path, size, threads, seconds):
    n_blocks = os.path.getsize(path) // size
    lat, stop = [], time.time() + seconds

    def worker(seed):
        rng = random.Random(seed)
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = mmap.mmap(-1, size)
        local = []
        while time.time() < stop:
            t = time.perf_counter()
            os.preadv(fd, [buf], rng.randrange(n_blocks) * size)
            local.append(time.perf_counter() - t)
        os.close(fd)
        lat.extend(local)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    lat.sort()
    return len(lat) / seconds, len(lat) * size / seconds / 1e6, lat[len(lat) // 2] * 1e3, lat[int(len(lat) * 0.99)] * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--seconds", type=float, default=15)
    a = p.parse_args()
    for name, size, threads in (("lat", 4096, 1), ("iops", 4096, 64), ("bw", 2 << 20, 16)):
        iops, mbs, p50, p99 = run(a.file, size, threads, a.seconds)
        print(f"{name:5s} {size >> 10:5d} KiB x{threads:3d}: {iops:9.0f} IOPS {mbs:8.1f} MB/s  p50 {p50:6.2f} ms  p99 {p99:6.2f} ms", flush=True)


if __name__ == "__main__":
    main()
