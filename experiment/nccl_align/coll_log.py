"""Run a training script while logging the size, alignment and time of every all-gather and
reduce-scatter it issues.

    torchrun ... experiment/nccl_align/coll_log.py tasks/vla/train_lingbotvla.py <args>

Wraps torch.distributed.all_gather_into_tensor and reduce_scatter_tensor (what FSDP2 calls)
and writes $COLL_LOG_DIR/coll_rank<R>.csv with one row per call:

    call,op,dtype,in_numel,out_numel,chunk_bytes,chunk_mod16,in_mod16,out_mod16,ms

chunk_bytes is the per-rank chunk (the all-gather input, the reduce-scatter output); in_mod16
and out_mod16 are the buffer addresses modulo 16. ms is measured with CUDA events on the
stream the call was issued on, so it also counts time spent waiting for slower ranks. Only the
first COLL_LOG_CALLS calls (default 5000) are logged. The trainer itself runs unchanged.
"""

import os
import runpy
import sys

import torch
import torch.distributed as dist


def log_misaligned_groups(path):
    """For every FSDP2 unit whose per-rank all-gather chunk is not a multiple of 16 bytes (bf16),
    list the parameters whose own per-rank shard has an element count that is not a multiple of 8."""
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    orig_init = FSDPParamGroup.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        sizes = [(p, p.padded_sharded_param_size.numel()) for p in self.fsdp_params]
        total = sum(n for _, n in sizes)
        if total % 8:
            with open(path, "a") as f:
                names = ", ".join(type(m).__name__ for m in self.modules)
                f.write(f"unit [{names}]: {len(sizes)} params, {total:,} elements per rank, % 8 = {total % 8}\n")
                for p, n in sizes:
                    if n % 8:
                        info = p._module_info
                        f.write(
                            f"  {type(info.module).__name__}.{info.param_name} {tuple(p._orig_size)} "
                            f"-> {n:,} per rank, % 8 = {n % 8}\n"
                        )

    FSDPParamGroup.__init__ = init


def main():
    script, sys.argv = sys.argv[1], sys.argv[1:]
    out_dir = os.environ["COLL_LOG_DIR"]
    limit = int(os.environ.get("COLL_LOG_CALLS", "5000"))
    path = os.path.join(out_dir, f"coll_rank{os.environ.get('RANK', '0')}.csv")
    f = open(path, "w", buffering=1)
    f.write("call,op,dtype,in_numel,out_numel,chunk_bytes,chunk_mod16,in_mod16,out_mod16,ms\n")
    pending = []
    state = {"calls": 0}

    def flush(block=False):
        while pending and (block or pending[0][2].query()):
            row, start, end = pending.pop(0)
            end.synchronize()
            f.write(f"{row},{start.elapsed_time(end):.3f}\n")

    def wrap(op, orig, chunk_of):
        def wrapper(*args, **kwargs):
            if state["calls"] >= limit:
                return orig(*args, **kwargs)
            out = kwargs.get("output_tensor", kwargs.get("output", args[0] if args else None))
            inp = kwargs.get("input_tensor", kwargs.get("input", args[1] if len(args) > 1 else None))
            chunk = chunk_of(inp, out) * inp.element_size()
            row = (
                f"{state['calls']},{op},{str(inp.dtype).split('.')[1]},{inp.numel()},{out.numel()},"
                f"{chunk},{chunk % 16},{inp.data_ptr() % 16},{out.data_ptr() % 16}"
            )
            state["calls"] += 1
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = orig(*args, **kwargs)
            end.record()
            pending.append((row, start, end))
            flush()
            if state["calls"] == limit:
                flush(block=True)
            return result

        return wrapper

    log_misaligned_groups(os.path.join(out_dir, f"coll_groups_rank{os.environ.get('RANK', '0')}.txt"))
    dist.all_gather_into_tensor = wrap("allgather", dist.all_gather_into_tensor, lambda i, o: i.numel())
    dist.reduce_scatter_tensor = wrap("reducescatter", dist.reduce_scatter_tensor, lambda i, o: o.numel())
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        flush(block=True)
        f.close()


if __name__ == "__main__":
    main()
