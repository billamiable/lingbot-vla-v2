#!/usr/bin/env bash
# Online teachers vs the offline teacher cache on one 8-GPU RTX PRO 5000 node, DistMuon.
# Build the cache first (see README.md), then:
#
#   PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... CACHE=/raid/teacher_cache/robotwin_clean \
#     bash experiment/teacher_cache/run_ab.sh [run ...]
#
# Runs (default: all, in this order):
#   online_130    teachers every step, 130 steps                     (speed baseline)
#   verify_30     teachers every step + compare with the cache, 30 steps (target equality)
#   cold_130      read the cache right after dropping the page cache  (cache served from disk)
#   warm_130      read the cache again, now in the page cache         (cache served from memory)
#   pl300_500     read the cache at a 300 W power limit, 500 steps    (speed and loss over a longer run)
#   verify300_500 verify mode at 300 W, 500 steps: teachers loaded and run, training on the cached
#                 targets, so the flow-matching noise matches an online run (loss vs the 300 W online run)
# Each run waits for the GPUs to cool below 40 C first; iostat and free are sampled every second.
set -euo pipefail
: "${CACHE:?set CACHE to the cache directory}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OUT_ROOT:?set OUT_ROOT}"
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(online_130 verify_30 cold_130 warm_130 pl300_500 verify300_500)
trap 'nvidia-smi -pl 350 >/dev/null || true' EXIT

run() {  # run <name> <steps> <power limit> [trainer args...]
  local name=tc_$1 steps=$2 pl=$3; shift 3
  local out=$OUT_ROOT/$name
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)" -lt 40 ]; do sleep 20; done
  nvidia-smi -pl "$pl" >/dev/null
  mkdir -p "$out"
  iostat -dxm 1 > "$out/iostat.txt" &
  local io=$!
  (while true; do echo "$(date +%T) $(free -g | awk '/Mem:/ {print "used", $3, "buff/cache", $6}')"; sleep 1; done) > "$out/free.txt" &
  local fr=$!
  echo "$(date +%T) start $name (${steps} steps, ${pl} W)"
  MAX_STEPS=$steps TIMING_ONLY=1 OPTIMIZER=dist_muon \
    bash "$REPO/experiment/single_node_repro/train_pro5000.sh" "$name" "$@" || true
  kill $io $fr 2>/dev/null || true
  echo "$(date +%T) done $name"
}

for r in "${RUNS[@]}"; do
  case $r in
    online_130) run online_130 130 350 ;;
    verify_30)  run verify_30 30 350 --train.teacher_cache_dir "$CACHE" --train.teacher_cache_mode verify ;;
    cold_130)   sync; echo 3 > /proc/sys/vm/drop_caches
                run cold_130 130 350 --train.teacher_cache_dir "$CACHE" ;;
    warm_130)   run warm_130 130 350 --train.teacher_cache_dir "$CACHE" ;;
    pl300_500)  run pl300_500 500 300 --train.teacher_cache_dir "$CACHE" ;;
    verify300_500) run verify300_500 500 300 --train.teacher_cache_dir "$CACHE" --train.teacher_cache_mode verify ;;
    *) echo "unknown run $r" >&2; exit 1 ;;
  esac
done
echo ALL_DONE
