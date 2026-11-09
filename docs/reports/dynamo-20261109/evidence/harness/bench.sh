#!/usr/bin/env bash
# Short, comparable speed test of one checkout on one 8-GPU node: the official
# robotwin_dist_muon.yaml with paths filled in, micro 32 x 8 GPUs = global 256, activation
# recompute, NCCL_MIN_CTAS=8 NCCL_NTHREADS=128 NCCL_P2P_READ_ENABLE=1, every rank bound to its
# GPU's NUMA node, 31 steps; summarize.py averages the slowest rank over steps 11-30.
#
#   SRC=<checkout to test> RUN=<new run dir> PY=... MODEL_ROOT=... DATA_LIST=... \
#     bash experiment/bench/bench.sh [--key value ...]
#
# Extra arguments go to the trainer (e.g. --train.teacher_cache_dir DIR). The test refuses to
# start while another process uses a GPU, and waits for every GPU to cool below 40 C.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SRC:?set SRC}" "${RUN:?set RUN}" "${PY:?set PY}" "${MODEL_ROOT:?set MODEL_ROOT}" "${DATA_LIST:?set DATA_LIST}"
export BENCH_STEPS=${BENCH_STEPS:-31} BENCH_OUT=$RUN
[ -e "$RUN" ] && { echo "$RUN exists" >&2; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "a GPU is in use" >&2; exit 1; }
mkdir -p "$RUN"
git -C "$SRC" log -1 --format='%H %s' > "$RUN/source.txt"
git -C "$SRC" status --porcelain --untracked-files=no >> "$RUN/source.txt"

"$PY" - "$SRC/configs/vla/robotwin/robotwin_dist_muon.yaml" "$RUN/config.yaml" <<EOF
import sys, yaml
src, dst = sys.argv[1:]
c = yaml.safe_load(open(src))
m, d, t = c["model"], c["data"], c["train"]
m.update(model_path="$MODEL_ROOT/lingbot-vla-v2-6b", tokenizer_path="$MODEL_ROOT/Qwen3-VL-4B-Instruct")
d.update(train_path="$DATA_LIST", num_workers=8, prefetch_factor=2)
t.update(micro_batch_size=32, global_batch_size=256, gradient_accumulation_steps=1,
         enable_gradient_checkpointing=True, enable_full_shard=False, enable_resume=False,
         save_steps=0, save_epochs=0, save_hf_weights=False, async_save_hf_weights=False,
         output_dir="$RUN/output")
t.pop("load_checkpoint_path", None)
t["align_params"]["depth"].update(moge_path="$MODEL_ROOT/moge-2-vitb-normal/model.pt",
                                  morgbd_path="$MODEL_ROOT/lingbot-vla-v2-6b/depth/model.pt")
t["align_params"]["video"].update(ckpt_path="$MODEL_ROOT/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth",
                                  config_path="$MODEL_ROOT/lingbot-vla-v2-6b/dino_video/config.yaml")
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
EOF

# NUMA node CPUs of every GPU, in PCI order (CUDA_DEVICE_ORDER=PCI_BUS_ID matches nvidia-smi).
while IFS=', ' read -r idx bus; do
  bdf=$(echo "${bus#0000}" | tr 'A-F' 'a-f')
  node=$(cat "/sys/bus/pci/devices/$bdf/numa_node")
  export "BENCH_CPUS_$idx=$(cat /sys/devices/system/node/node$node/cpulist)"
done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)
env | grep '^BENCH_CPUS_' | sort > "$RUN/numa.txt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_MIN_CTAS=8 NCCL_NTHREADS=128 NCCL_P2P_READ_ENABLE=1 \
  TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
  PYTHONPATH=$SRC:$SRC/lingbotvla/models/vla/vision_models/lingbot-depth:$SRC/lingbotvla/models/vla/vision_models/MoGe
until [ "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)" -lt 40 ]; do sleep 20; done
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.limit,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown \
  --format=csv,noheader,nounits -l 1 > "$RUN/smi.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
cd "$SRC"
"$PY" -m torch.distributed.run --standalone --nproc-per-node 8 "$HERE/numa_entry.py" "$HERE/step_timer.py" \
  "$RUN/config.yaml" "$@" > "$RUN/train.log" 2>&1 || echo "trainer exited with $?" >&2
"$PY" "$HERE/summarize.py" "$RUN"
