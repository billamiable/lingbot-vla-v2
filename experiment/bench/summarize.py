"""Summarize bench.sh runs: per step the slowest rank, mean over the observed steps.

    python summarize.py RUN_DIR [RUN_DIR ...]

Steps 1-10 are warm-up (compilation), 11-30 are observed, the last step is excluded. Also
prints the mean teacher time (rank-max) and the peak GPU memory sampled by nvidia-smi.
"""

import json
import statistics
import sys
from pathlib import Path

FIRST, LAST = 11, 30


def summarize(run):
    run = Path(run)
    ranks = []
    for path in sorted(run.glob("rank*.jsonl")):
        ranks.append({r["step"]: r for r in map(json.loads, path.read_text().splitlines())})
    if len(ranks) != 8 or any(LAST not in r for r in ranks):
        return None
    steps = range(FIRST, LAST + 1)
    step_max = [max(r[s]["elapsed_s"] for r in ranks) for s in steps]
    teacher = [max(r[s]["teacher_s"] for r in ranks) for s in steps]
    mem = 0
    smi = run / "smi.csv"
    if smi.exists():
        mem = max((int(line.split(",")[2]) for line in smi.read_text().splitlines() if line.count(",") >= 4), default=0)
    mean = statistics.mean(step_max)
    return dict(
        run=run.name,
        mean_s=mean,
        median_s=statistics.median(step_max),
        cv_pct=100 * statistics.pstdev(step_max) / mean,
        samples_per_s=256 / mean,
        teacher_s=statistics.mean(teacher),
        peak_mib=mem,
        loss_step30=ranks[0][LAST]["loss"],
    )


def main():
    print(f"{'run':40s} {'mean s':>8s} {'median':>8s} {'CV %':>6s} {'samp/s':>7s} {'teacher':>8s} {'peak MiB':>9s} {'loss@30':>8s}")
    for run in sys.argv[1:]:
        s = summarize(run)
        if s is None:
            print(f"{Path(run).name:40s} incomplete")
            continue
        print(
            f"{s['run']:40s} {s['mean_s']:8.4f} {s['median_s']:8.4f} {s['cv_pct']:6.3f} {s['samples_per_s']:7.3f} "
            f"{s['teacher_s']:8.3f} {s['peak_mib']:9d} {s['loss_step30']:8.4f}"
        )
        Path(run, "summary.json").write_text(json.dumps(s, indent=2) + "\n")


if __name__ == "__main__":
    main()
