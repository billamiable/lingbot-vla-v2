"""Compare the runs of run_essd.sh: step time, data wait, and how much the cache disk was read.

    python experiment/teacher_cache/essd_sim/summarize_essd.py OUT_ROOT

Steps 31 to the last (the first 30 include compilation). Disk reads (GB, peak MB/s) come from
iostat on the device the run read the cache from (md2 for local NVMe, dm-0 for the emulated disk).
"""

import os
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from summarize_ab import FIRST, disk_read_mb_s, p95, parse  # noqa: E402

RUNS = ("nvme", "pl1", "pl0", "pl0_slow")


def main():
    root = sys.argv[1]
    print(f"{'run':9s} {'steps':>7s} {'step s':>7s} {'p95 s':>7s} {'wall s':>7s} {'wait ms':>8s} {'max wait':>9s} {'read GB':>8s} {'max MB/s':>9s}")
    for r in RUNS:
        d = os.path.join(root, f"essd_{r}")
        steps = parse(d) if os.path.exists(os.path.join(d, "train.log")) else {}
        last = max(steps, default=0)
        if last < FIRST:
            continue
        window = [steps[k] for k in range(FIRST, last + 1) if k in steps]
        times = [w["step_time"] for w in window]
        waits = [w["data_wait"] * 1e3 for w in window if w["data_wait"] is not None]

        def ts(k):
            return datetime.strptime(steps[k]["stamp"], "%m/%d/%Y %H:%M:%S").timestamp()

        wall = (ts(last) - ts(FIRST - 1)) / (last - FIRST + 1) if FIRST - 1 in steps else float("nan")
        io = disk_read_mb_s(d, "md2" if r == "nvme" else "dm-0") or {"total_gb": float("nan"), "max_mb_s": float("nan")}
        print(
            f"{r:9s} {FIRST}-{last:<4d} {statistics.mean(times):7.3f} {p95(times):7.3f} {wall:7.3f} "
            f"{statistics.mean(waits):8.1f} {max(waits):9.1f} {io['total_gb']:8.1f} {io['max_mb_s']:9.0f}"
        )


if __name__ == "__main__":
    main()
