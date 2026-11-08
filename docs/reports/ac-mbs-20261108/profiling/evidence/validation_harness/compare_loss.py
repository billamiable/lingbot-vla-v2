"""Compare the per-step losses and grad norm of two bench.sh runs (steps 1-30, rank 0).

    python compare_loss.py RUN_A RUN_B [RUN_C RUN_D ...]     (pairs: A vs B, C vs D, ...)

Prints, per quantity, the mean and max relative difference |b - a| / |a| and whether step 1
is bit-identical. Two runs of the same code already differ a little after step 1 (nondeterministic
kernels), so compare against such a same-code pair.
"""

import json
import statistics
import sys
from pathlib import Path

KEYS = ("loss", "vla_loss", "depth_loss", "future_depth_loss", "future_video_loss", "grad_norm")


def load(run):
    rows = [json.loads(line) for line in Path(run, "rank0.jsonl").read_text().splitlines()]
    return {r["step"]: r for r in rows if r["step"] <= 30}


def main():
    runs = sys.argv[1:]
    for a_dir, b_dir in zip(runs[::2], runs[1::2]):
        a, b = load(a_dir), load(b_dir)
        steps = sorted(set(a) & set(b))
        print(f"{Path(a_dir).name} vs {Path(b_dir).name}, steps {steps[0]}-{steps[-1]}")
        for k in KEYS:
            rel = [abs(b[s][k] - a[s][k]) / max(abs(a[s][k]), 1e-12) for s in steps]
            same1 = a[steps[0]][k] == b[steps[0]][k]
            print(f"  {k:18s} mean rel {100 * statistics.mean(rel):7.3f}%  max rel {100 * max(rel):7.3f}%  "
                  f"step 1 {'identical' if same1 else f'{a[steps[0]][k]:.6g} vs {b[steps[0]][k]:.6g}'}")


if __name__ == "__main__":
    main()
