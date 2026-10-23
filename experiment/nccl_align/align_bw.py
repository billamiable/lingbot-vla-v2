"""Does a 16-byte misaligned per-rank chunk slow NCCL all-gather / reduce-scatter down?

    torchrun --nproc_per_node 8 experiment/nccl_align/align_bw.py [--gap 4]

NCCL copies with 16-byte vector loads/stores only where source and destination are both
16-byte aligned. In all_gather_into_tensor, rank r's chunk starts at r * chunk_bytes in the
output, so a chunk whose byte size is not a multiple of 16 puts most chunks at a misaligned
address. The per-rank element counts below differ by a few elements only (the data volume is
the same to 1e-7), so any time difference between them comes from the alignment:

    119,465,984  bf16 chunk % 16 = 0   (what nccl-tests ran: it rounds the count down)
    119,465,991  bf16 chunk % 16 = 14  (the root FSDP module of the trainer)
    119,465,992  bf16 chunk % 16 = 0
    119,465,985  bf16 chunk % 16 = 2

Collectives:
    ag_inplace   all-gather whose input is the rank's own slice of the output, as FSDP2 does
    ag           all-gather with a separate input buffer (nccl-tests "out-of-place")
    rs           reduce-scatter (AVG, as FSDP2 with fp32 reduce_dtype)
each in bf16 and fp32. ptr_off runs the aligned count with the input starting one element
past an aligned address, to separate a misaligned pointer from a misaligned chunk size.

Every call is timed alone (barrier, then CUDA events around the collective) and the slowest
rank is reported. --gap adds a pass that sleeps that many seconds before each call, to test
whether a collective after an idle GPU (one training step apart) runs slower.
"""

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

COUNTS = (119_465_984, 119_465_991, 119_465_992, 119_465_985)


def timed(fn, iters, warmup, gap=0.0):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        if gap:
            time.sleep(gap)
            dist.barrier()  # all ranks leave the idle period together
            torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        t = torch.tensor(start.elapsed_time(end), device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times.append(t.item())
    return times


def make(kind, n, dtype, world, rank, ptr_off=False):
    """Return (fn, full, inp): full is the gathered / pre-reduce buffer, inp the collective's input."""
    if kind == "ag_inplace":
        out = torch.empty(n * world, dtype=dtype, device="cuda")
        inp = out.narrow(0, n * rank, n)
        inp.normal_()
        return (lambda: dist.all_gather_into_tensor(out, inp)), out, inp
    if kind == "ag":
        out = torch.empty(n * world, dtype=dtype, device="cuda")
        base = torch.randn(n + 8, dtype=dtype, device="cuda")
        inp = base[1 : n + 1] if ptr_off else base[:n]
        return (lambda: dist.all_gather_into_tensor(out, inp)), out, inp
    if kind == "rs":
        inp = torch.randn(n * world, dtype=dtype, device="cuda")
        out = torch.empty(n, dtype=dtype, device="cuda")
        return (lambda: dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.AVG)), inp, out
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gap", type=float, default=0.0, help="seconds idle before each call in the gap pass")
    ap.add_argument("--gap-iters", type=int, default=5)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if rank == 0:
        print(f"torch {torch.__version__}  nccl {'.'.join(map(str, torch.cuda.nccl.version()))}  world {world}", flush=True)
        print(
            f"{'collective':11s} {'dtype':8s} {'count/rank':>12s} {'chunk%16':>8s} {'in%16':>5s} "
            f"{'median ms':>9s} {'min ms':>8s} {'busbw GB/s':>10s}",
            flush=True,
        )

    def run(kind, dtype, n, ptr_off=False, gap=0.0, iters=None):
        fn, full, inp = make(kind, n, dtype, world, rank, ptr_off)
        times = timed(fn, iters or args.iters, args.warmup, gap)
        full_bytes = full.numel() * full.element_size()
        # Pointer offsets differ by rank for ag_inplace; report the largest one.
        in_mod = torch.tensor(inp.data_ptr() % 16, device="cuda")
        dist.all_reduce(in_mod, op=dist.ReduceOp.MAX)
        med = statistics.median(times)
        busbw = full_bytes / (med / 1e3) * (world - 1) / world / 1e9
        if rank == 0:
            label = kind + ("+ptr_off" if ptr_off else "") + (f" gap{gap:g}s" if gap else "")
            print(
                f"{label:11s} {str(dtype).split('.')[1]:8s} {n:12,d} {n * dtype.itemsize % 16:8d} {in_mod.item():5d} "
                f"{med:9.1f} {min(times):8.1f} {busbw:10.2f}",
                flush=True,
            )
        del fn, full, inp
        torch.cuda.empty_cache()

    for dtype in (torch.bfloat16, torch.float32):
        for kind in ("ag_inplace", "ag", "rs"):
            for n in COUNTS:
                run(kind, dtype, n)
        run("ag", dtype, COUNTS[0], ptr_off=True)
    if args.gap:
        for kind, dtype in (("ag_inplace", torch.bfloat16), ("rs", torch.float32)):
            for n in COUNTS[:2]:
                run(kind, dtype, n, gap=args.gap, iters=args.gap_iters)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
