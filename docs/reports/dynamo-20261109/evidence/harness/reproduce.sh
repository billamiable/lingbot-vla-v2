#!/usr/bin/env bash
# Reproduce the reported step times on one 8-GPU node, in alternating order so that drift
# between runs does not favour one version:
#
#   vit:   baseline, improv1, improv2, improv2, improv1, baseline, improv2, align, align, improv2
#   build: one-off offline teacher cache build (~1 h on 8 GPUs, ~1.5 MB per frame on disk)
#   cache: baseline, cache, cache, baseline, verify
#
#   SRC=<clone of this repository with all branches fetched> RUNS_ROOT=<empty dir> PY=<python> \
#   MODEL_ROOT=<dir with lingbot-vla-v2-6b, Qwen3-VL-4B-Instruct, moge-2-vitb-normal> \
#   DATA_LIST=<train_path list> CACHE=<dir for the teacher cache> \
#     bash experiment/bench/reproduce.sh [vit] [build] [cache]
#
# SRC is switched between commits, so use a separate clone from the one this script runs from.
# summarize.py prints the step time of every run; compare_loss.py the loss / grad-norm drift.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SRC:?set SRC}" "${RUNS_ROOT:?set RUNS_ROOT}" "${PY:?set PY}" "${MODEL_ROOT:?set MODEL_ROOT}" "${DATA_LIST:?set DATA_LIST}"
export SRC RUNS_ROOT PY MODEL_ROOT DATA_LIST
STAGES=" ${*:-vit build cache} "

BASELINE=be969b8   # official main
IMPROV1=412b978    # vision / embedding / alignment modules as separate FSDP units, prefetch
IMPROV2=42e963e    # grid metadata cache and vectorized position interpolation
ALIGN=6d18cfa      # improv2 + --train.fsdp_align_root (branch nccl-align-on-keddy)
TCACHE=d66fb35     # official main + offline teacher cache (branch teacher-cache-on-main)

if [[ $STAGES == *" vit "* ]]; then
  bash "$HERE/sequence.sh" base-1=$BASELINE imp1-1=$IMPROV1 imp2-1=$IMPROV2 imp2-2=$IMPROV2 \
    imp1-2=$IMPROV1 base-2=$BASELINE imp2-3=$IMPROV2 \
    "align-1=$ALIGN --train.fsdp_align_root true" "align-2=$ALIGN --train.fsdp_align_root true" imp2-4=$IMPROV2
  "$PY" "$HERE/summarize.py" "$RUNS_ROOT"/{base,imp1,imp2,align}-*
  "$PY" "$HERE/compare_loss.py" "$RUNS_ROOT"/base-1 "$RUNS_ROOT"/base-2 "$RUNS_ROOT"/imp2-3 "$RUNS_ROOT"/align-1 \
    "$RUNS_ROOT"/imp2-4 "$RUNS_ROOT"/align-2
fi

if [[ $STAGES == *" build "* ]]; then
  : "${CACHE:?set CACHE}"
  # The build reuses the benchmark config; it resumes where it stopped and is CPU-bound on
  # image decoding, hence 24 DataLoader workers per rank.
  config=$(ls "$RUNS_ROOT"/*/config.yaml 2>/dev/null | head -1)
  [ -n "$config" ] || { RUN=$RUNS_ROOT/config-only BENCH_STEPS=1 bash "$HERE/bench.sh"; config=$RUNS_ROOT/config-only/config.yaml; }
  git -C "$SRC" checkout -q --detach $TCACHE
  mkdir -p "$RUNS_ROOT/cache-build"
  echo "$(date +%T) start cache build"
  ( cd "$SRC"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_MIN_CTAS=8 NCCL_NTHREADS=128 NCCL_P2P_READ_ENABLE=1 \
      TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
      PYTHONPATH=$SRC:$SRC/lingbotvla/models/vla/vision_models/lingbot-depth:$SRC/lingbotvla/models/vla/vision_models/MoGe
    "$PY" -m torch.distributed.run --standalone --nproc-per-node 8 tasks/vla/train_lingbotvla.py "$config" \
      --train.output_dir "$RUNS_ROOT/cache-build/output" --train.teacher_cache_dir "$CACHE" \
      --train.teacher_cache_mode build --data.num_workers 24 >> "$RUNS_ROOT/cache-build/train.log" 2>&1 ) \
    || { echo "cache build failed, see $RUNS_ROOT/cache-build/train.log" >&2; exit 1; }
  echo "$(date +%T) cache build done"
fi

if [[ $STAGES == *" cache "* ]]; then
  : "${CACHE:?set CACHE}"
  bash "$HERE/sequence.sh" cbase-1=$BASELINE "cache-1=$TCACHE --train.teacher_cache_dir $CACHE" \
    "cache-2=$TCACHE --train.teacher_cache_dir $CACHE" cbase-2=$BASELINE \
    "verify-1=$TCACHE --train.teacher_cache_dir $CACHE --train.teacher_cache_mode verify"
  "$PY" "$HERE/summarize.py" "$RUNS_ROOT"/{cbase,cache,verify}-*
  "$PY" "$HERE/compare_loss.py" "$RUNS_ROOT"/cbase-1 "$RUNS_ROOT"/cbase-2 "$RUNS_ROOT"/cbase-1 "$RUNS_ROOT"/verify-1
fi
