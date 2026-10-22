"""Compare the runs of run_ab.sh: step time, teacher and data-wait time, disk reads, loss.

    python experiment/teacher_cache/summarize_ab.py OUT_ROOT [--ref-300w RUN_DIR]

Step statistics cover steps 31 to the last step (the first 30 include compilation). Wall clock
per step is the spacing of the log timestamps, so it includes waiting for data. --ref-300w is
an online-teacher 500-step run at 300 W to compare the loss of tc_pl300_500 with.
"""

import argparse
import os
import re
import statistics

STEP = re.compile(
    r"(\d\d/\d\d/\d{4} \d\d:\d\d:\d\d) - INFO - __main__ - Step (\d+)/\d+, .*?Loss ([\d.]+), VLA_Loss ([\d.]+), "
    r"Depth_Loss ([\d.]+), Future_Depth_Loss ([\d.]+), FutureVideo_Loss ([\d.]+),.*?StepTime ([\d.]+)s, "
    r"Depth_Forward_Time\s+([\d.]+)s(?:.*?Data_Wait ([\d.]+)s)?"
)
VERIFY = re.compile(r"TeacherCacheVerify step (\d+): (.*)")
FIRST = 31


def parse(run_dir):
    steps = {}
    with open(os.path.join(run_dir, "train.log"), errors="ignore") as f:
        for line in f:
            m = STEP.search(line)
            if m:
                s = int(m.group(2))
                steps[s] = {
                    "loss": float(m.group(3)),
                    "vla": float(m.group(4)),
                    "depth": float(m.group(5)),
                    "fdepth": float(m.group(6)),
                    "fvideo": float(m.group(7)),
                    "step_time": float(m.group(8)),
                    "teacher": float(m.group(9)),
                    "data_wait": float(m.group(10)) if m.group(10) else None,
                    "stamp": m.group(1),
                }
    return steps


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))]


def disk_read_mb_s(run_dir, device):
    """Mean and max read MB/s of the device over the samples with any activity."""
    path = os.path.join(run_dir, "iostat.txt")
    if not os.path.exists(path):
        return None
    header, reads = None, []
    first_report = True
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Device":
                header = parts
                continue
            if header and parts[0] == device:
                if first_report:  # the first report is the average since boot
                    first_report = False
                    continue
                reads.append(float(parts[header.index("rMB/s")]))
    total = sum(reads)
    return {"total_gb": total / 1024, "max_mb_s": max(reads, default=0.0)}


def summarize(name, run_dir):
    steps = parse(run_dir)
    if not steps:
        print(f"{name}: no steps")
        return steps
    last = max(steps)
    if last < FIRST:
        print(f"{name:16s} only {last} steps, no timing window")
        return steps
    window = [steps[s] for s in range(FIRST, last + 1) if s in steps]
    times = [w["step_time"] for w in window]
    from datetime import datetime

    def ts(s):
        return datetime.strptime(steps[s]["stamp"], "%m/%d/%Y %H:%M:%S").timestamp()

    wall = (ts(last) - ts(FIRST - 1)) / (last - FIRST + 1) if FIRST - 1 in steps else float("nan")
    io_str = ""
    for dev, label in (("md2", "cache disk"), ("md1", "data disk")):
        io = disk_read_mb_s(run_dir, dev)
        if io:
            io_str += f"  {label} read {io['total_gb']:.1f} GB (max {io['max_mb_s']:.0f} MB/s)"
    print(
        f"{name:16s} steps {FIRST}-{last}: StepTime mean {statistics.mean(times):.3f}s p95 {p95(times):.3f}s  "
        f"wall/step {wall:.3f}s  teacher {mean([w['teacher'] for w in window]) * 1e3:.1f} ms  "
        f"data_wait {mean([w['data_wait'] for w in window]) * 1e3:.1f} ms{io_str}"
    )
    return steps


def compare_loss(a_name, a, b_name, b):
    common = sorted(set(a) & set(b))
    if not common:
        return
    print(f"loss {a_name} vs {b_name} over {len(common)} common steps:")
    for key in ("loss", "vla", "depth", "fdepth", "fvideo"):
        diffs = [abs(a[s][key] - b[s][key]) for s in common]
        rel = [abs(a[s][key] - b[s][key]) / max(abs(b[s][key]), 1e-12) for s in common]
        print(f"  {key:7s} max |diff| {max(diffs):.4f}  mean |diff| {statistics.mean(diffs):.4f}  mean rel {statistics.mean(rel) * 100:.2f}%")
    tail = [s for s in common if s > max(common) - 50]
    print(
        f"  last 50 steps mean loss: {a_name} {mean([a[s]['loss'] for s in tail]):.4f}  "
        f"{b_name} {mean([b[s]['loss'] for s in tail]):.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--ref-300w", default=None)
    ap.add_argument("--ref-350w", default=None, help="an earlier online 130-step run, for run-to-run loss noise")
    args = ap.parse_args()
    runs = {}
    for name in ("online_130", "verify_30", "cold_130", "warm_130", "pl300_500", "verify300_500"):
        d = os.path.join(args.out_root, f"tc_{name}")
        if os.path.exists(os.path.join(d, "train.log")):
            runs[name] = summarize(name, d)
    if args.ref_300w:
        runs["ref_300w"] = summarize("online_300w_ref", args.ref_300w)
    if args.ref_350w:
        runs["ref_350w"] = summarize("online_350w_ref", args.ref_350w)

    verify = os.path.join(args.out_root, "tc_verify_30", "train.log")
    if os.path.exists(verify):
        lines = [VERIFY.search(l) for l in open(verify, errors="ignore")]
        lines = [m for m in lines if m]
        print(f"verify: {len(lines)} steps compared; last: step {lines[-1].group(1)}: {lines[-1].group(2)}" if lines else "verify: none")
        worst = {}
        for m in lines:
            for part in m.group(2).split("; "):
                key = part.split()[0]
                d = float(re.search(r"max\|diff\| (\S+)", part).group(1))
                e = float(re.search(r"exact (\S+)", part).group(1))
                w = worst.setdefault(key, [0.0, 1.0])
                w[0], w[1] = max(w[0], d), min(w[1], e)
        for key, (d, e) in worst.items():
            print(f"  {key:24s} worst max|diff| {d:.3e}  worst exact fraction {e:.6f}")

    if "online_130" in runs:
        for name in ("cold_130", "warm_130", "verify_30", "ref_350w"):
            if name in runs:
                compare_loss(name, runs[name], "online_130", runs["online_130"])
    for name in ("verify300_500", "pl300_500"):
        if name in runs and "ref_300w" in runs:
            compare_loss(name, runs[name], "ref_300w", runs["ref_300w"])


if __name__ == "__main__":
    main()
