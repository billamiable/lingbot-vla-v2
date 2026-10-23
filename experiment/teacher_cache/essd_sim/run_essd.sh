#!/usr/bin/env bash
# Teacher cache read from local NVMe vs from an emulated cloud disk (essd_sim.sh), 8x RTX PRO 5000.
# Set the disk up first (essd_sim.sh setup), then:
#
#   PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... CACHE=/raid/teacher_cache/robotwin_clean \
#   SIM_CACHE=/mnt/essd_sim/robotwin_clean bash experiment/teacher_cache/essd_sim/run_essd.sh [run ...]
#
# Runs (default: all, in this order), 130 steps at 300 W, each after dropping the page cache:
#   nvme       cache on local NVMe, no limits                       (baseline)
#   pl1        emulated disk, 350 MB/s, 50k IOPS                    (ESSD PL1 per-disk maximum)
#   pl0        180 MB/s, 10k IOPS                                   (ESSD PL0 per-disk maximum)
#   pl0_slow   PL0 + dm-delay 1 ms, which this kernel rounds up to ~8 ms per read (timer ticks),
#              far slower than a cloud disk (sub-ms): a worst case, not an emulation
# Each run first measures the disk it reads (read_bench.py) under the same limits. Limits apply to
# the trainer's cgroup and only to the emulated disk; the videos stay on the system disk.
set -euo pipefail
: "${CACHE:?set CACHE}" "${SIM_CACHE:?set SIM_CACHE}" "${OUT_ROOT:?set OUT_ROOT}" "${PY:?set PY}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
DEV=/dev/mapper/${DM:-essd_sim}
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(nvme pl1 pl0 pl0_slow)
trap 'nvidia-smi -pl 350 >/dev/null || true' EXIT

run() {  # run <name> <cache> <MB/s or 0> <IOPS> <delay ms>
  local name=essd_$1 cache=$2 bw=$3 iops=$4 delay=$5 out=$OUT_ROOT/essd_$1 limits=()
  mkdir -p "$out"
  bash "$HERE/essd_sim.sh" delay "$delay"
  [ "$bw" != 0 ] && limits=(-p "IOReadBandwidthMax=$DEV ${bw}M" -p "IOReadIOPSMax=$DEV $iops")
  sync; echo 3 > /proc/sys/vm/drop_caches
  systemd-run --scope --quiet "${limits[@]}" "$PY" "$HERE/read_bench.py" "$cache/depth.npy" --seconds 10 \
    | tee "$out/read_bench.txt"
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)" -lt 40 ]; do sleep 20; done
  nvidia-smi -pl 300 >/dev/null
  sync; echo 3 > /proc/sys/vm/drop_caches
  iostat -dxm 1 > "$out/iostat.txt" &
  local io=$!
  echo "$(date +%T) start $name (bw ${bw} MB/s, ${iops} IOPS, +${delay} ms)"
  MAX_STEPS=130 TIMING_ONLY=1 OPTIMIZER=dist_muon MEM_PROBE=0 \
    systemd-run --scope --quiet "${limits[@]}" \
    bash "$REPO/experiment/single_node_repro/train_pro5000.sh" "$name" --train.teacher_cache_dir "$cache" || true
  kill $io 2>/dev/null || true
  echo "$(date +%T) done $name"
}

for r in "${RUNS[@]}"; do
  case $r in
    nvme)     run nvme "$CACHE" 0 0 0 ;;
    pl1)      run pl1 "$SIM_CACHE" 350 50000 0 ;;
    pl0)      run pl0 "$SIM_CACHE" 180 10000 0 ;;
    pl0_slow) run pl0_slow "$SIM_CACHE" 180 10000 1 ;;
    *) echo "unknown run $r" >&2; exit 1 ;;
  esac
done
echo ALL_DONE
