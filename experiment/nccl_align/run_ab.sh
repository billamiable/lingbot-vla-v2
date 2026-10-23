#!/usr/bin/env bash
# --train.fsdp_align_root on vs off on one 8-GPU RTX PRO 5000 node, DistMuon, online teachers.
#
#   PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... bash experiment/nccl_align/run_ab.sh [run ...]
#
# Runs (default: all, in this order):
#   align_log_5   align on, 5 steps, every collective logged (coll_log.py): the root
#                 all-gather chunk should now be a multiple of 16 bytes
#   base_130      align off, 130 steps (speed baseline)
#   align_130     align on, 130 steps
#   base300_130, align300_130  the same pair at a 300 W power limit, where the GPUs do not
#                 thermally throttle (at 350 W one GPU can, which spoils the step times)
# Each run waits for the GPUs to cool below 40 C first. PYTHONPATH points at this checkout so
# an editable install of another checkout does not shadow lingbotvla.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${OUT_ROOT:?set OUT_ROOT}"
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(align_log_5 base_130 align_130)
trap 'nvidia-smi -pl 350 >/dev/null || true' EXIT

run() {  # run <name> <steps> <power limit W> [trainer args...]
  local name=$1 steps=$2 pl=$3; shift 3
  until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)" -lt 40 ]; do sleep 20; done
  nvidia-smi -pl "$pl" >/dev/null
  echo "$(date +%T) start $name ($steps steps, $pl W)"
  MAX_STEPS=$steps TIMING_ONLY=1 OPTIMIZER=dist_muon MEM_PROBE=0 \
    bash "$REPO/experiment/single_node_repro/train_pro5000.sh" "$name" "$@" || true
  echo "$(date +%T) done $name"
}

for r in "${RUNS[@]}"; do
  case $r in
    align_log_5)  COLL_LOG=1 run align_log_5 5 350 --train.fsdp_align_root true ;;
    base_130)     run base_130 130 350 ;;
    align_130)    run align_130 130 350 --train.fsdp_align_root true ;;
    base300_130)  run base300_130 130 300 ;;
    align300_130) run align300_130 130 300 --train.fsdp_align_root true ;;
    *) echo "unknown run $r" >&2; exit 1 ;;
  esac
done
echo ALL_DONE
