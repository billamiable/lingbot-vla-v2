"""Group the collectives logged by coll_log.py by size and show which ones are misaligned.

    python experiment/nccl_align/summarize_coll_log.py RUN_DIR [--steps N]

A row is one (op, dtype, per-rank chunk size). misaligned means the chunk size or any rank's
input/output address is not a multiple of 16 bytes. ms is the slowest rank per call; "per
step" divides the sum by --steps (the number of training steps the log covers), so it includes
the first, compiling steps.
"""

import argparse
import collections
import csv
import glob
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--min-ms", type=float, default=20.0, help="hide aligned groups below this total")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.run_dir, "coll_rank*.csv")))
    ranks = [list(csv.DictReader(open(p))) for p in paths]
    n = min(len(r) for r in ranks)
    groups = collections.OrderedDict()
    for i in range(n):
        calls = [r[i] for r in ranks]
        c = calls[0]
        key = (c["op"], c["dtype"], int(c["chunk_bytes"]))
        g = groups.setdefault(key, {"calls": 0, "ms": [], "in": set(), "out": set(), "first": i})
        g["calls"] += 1
        g["ms"].append(max(float(x["ms"]) for x in calls))
        g["in"].update(int(x["in_mod16"]) for x in calls)
        g["out"].update(int(x["out_mod16"]) for x in calls)

    print(f"{len(paths)} ranks, {n} calls each")
    print(
        f"{'op':13s} {'dtype':9s} {'chunk bytes':>13s} {'%16':>3s} {'calls':>5s} {'in%16 (ranks)':>22s} "
        f"{'out%16':>6s} {'median ms':>9s} {'ms/step':>8s} {'first call':>10s}"
    )
    totals = collections.defaultdict(lambda: [0, 0.0])
    for (op, dtype, chunk), g in sorted(groups.items(), key=lambda kv: -sum(kv[1]["ms"])):
        bad = chunk % 16 or max(g["in"]) or max(g["out"])
        t = totals[(op, "misaligned" if bad else "aligned")]
        t[0] += g["calls"]
        t[1] += sum(g["ms"])
        if bad or sum(g["ms"]) >= args.min_ms:
            print(
                f"{op:13s} {dtype:9s} {chunk:13,d} {chunk % 16:3d} {g['calls']:5d} {str(sorted(g['in'])):>22s} "
                f"{str(sorted(g['out'])):>6s} {statistics.median(g['ms']):9.1f} {sum(g['ms']) / args.steps:8.1f} "
                f"{g['first']:10d}"
            )
    print("totals:")
    for (op, kind), (calls, ms) in sorted(totals.items()):
        print(f"  {op:13s} {kind:10s} {calls:6d} calls  {ms / args.steps:8.1f} ms/step")


if __name__ == "__main__":
    main()
