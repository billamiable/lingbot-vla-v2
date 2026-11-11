#!/usr/bin/env bash
# Run bench.sh for several checkouts in a row, e.g. in the alternating order A B B A:
#
#   SRC=<git checkout used for every run> RUNS_ROOT=... PY=... MODEL_ROOT=... DATA_LIST=... \
#     bash experiment/bench/sequence.sh baseline-r1=be969b8 opt-r1=42e963e \
#       'cache-r1=teacher-cache-on-main --train.teacher_cache_dir /path/to/cache'
#
# Each entry is NAME=REF [trainer args]; SRC is switched to REF (detached) before the run.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SRC:?set SRC}" "${RUNS_ROOT:?set RUNS_ROOT}"
for entry in "$@"; do
  read -r -a words <<< "$entry"
  name=${words[0]%%=*} ref=${words[0]#*=}
  git -C "$SRC" checkout -q --detach "$ref" || { echo "cannot check out $ref" >&2; exit 1; }
  echo "$(date +%T) start $name ($ref $(git -C "$SRC" rev-parse --short HEAD)) ${words[*]:1}"
  RUN=$RUNS_ROOT/$name bash "$HERE/bench.sh" "${words[@]:1}"
done
echo "$(date +%T) ALL_DONE"
