#!/usr/bin/env bash
# Reproduce against locally provisioned checkpoints/data/cache. Reserves an idle 8-GPU node.
# Numeric reference tensors require ~48 GB scratch space; never commit them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BASELINE_SRC:?official be969b8 checkout}" "${CANDIDATE_SRC:?perf/teacher-cache checkout}"
: "${PY:?training environment python}" "${MODEL_ROOT:?checkpoint directory}" "${DATA_LIST:?clean50 PNG data list}"
: "${TEACHER_CACHE:?completed cache}" "${OUT:?new output directory}"
[ ! -e "$OUT" ] || { echo 'OUT already exists' >&2; exit 1; }
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
unset LINGBOT_COMPILE_RMSNORM LINGBOT_REGIONAL_COMPILE TORCH_LOGS TORCHDYNAMO_VERBOSE
export NUMERIC_REFERENCE="$OUT/reference"
common=(--train.micro_batch_size 32 --train.global_batch_size 256 --train.gradient_accumulation_steps 1 --train.enable_gradient_checkpointing true)
run() {
  local label=$1 kind=$2 numeric=$3 reference=$4 steps=$5
  local src=$CANDIDATE_SRC
  local flags=()
  if [ "$kind" = base ]; then src=$BASELINE_SRC; fi
  if [ "$kind" = read ] || [ "$kind" = verify ]; then
    flags=(--train.teacher_cache_dir "$TEACHER_CACHE" --train.teacher_cache_mode "$kind")
  fi
  SRC="$src" RUN="$OUT/$label" BENCH_STEPS="$steps" \
    STUDY_ALIGNED_FLOW="$numeric" STUDY_NUMERIC="$numeric" STUDY_REFERENCE="$reference" \
    CACHE_BUILD_SUBSET=0 CACHE_BUILD_ONLY=0 \
    bash "$HERE/harness/bench.sh" "${common[@]}" "${flags[@]}"
}
# Explicit common-distribution flow inputs, gradients before optimizer, weights after optimizer.
run num_base_1 base 1 1 3
run num_off_fixed off 1 0 3
run num_verify_fixed verify 1 0 3
run num_read_fixed read 1 0 3
run num_base_2 base 1 0 3
# Natural training RNG, identical flags and timing window; ABBA order.
run final_base_1 base 0 0 31
run final_cache_1 read 0 0 31
run final_cache_2 read 0 0 31
run final_base_2 base 0 0 31
run verify_full_fixed verify 0 0 31
"$PY" "$HERE/harness/summarize.py" "$OUT/final_base_1" "$OUT/final_cache_1" "$OUT/final_cache_2" "$OUT/final_base_2"
echo "Results: $OUT (numeric references are private local tensors, not publishable artifacts)"
