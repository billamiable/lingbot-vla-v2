# Step-time benchmark

Short, comparable speed test of any checkout on one 8-GPU node. Each run trains 31 steps of
the official `configs/vla/robotwin/robotwin_dist_muon.yaml` (micro batch 32 x 8 GPUs = global
256, activation checkpointing, Distributed Muon) and reports the slowest rank's full step time,
averaged over steps 11-30. Nothing is saved.

| File | Purpose |
|---|---|
| `reproduce.sh` | Runs every version in alternating order and prints the summaries |
| `sequence.sh` | Runs `bench.sh` for a list of `NAME=REF [trainer args]` entries |
| `bench.sh` | One run: config, NUMA binding, NCCL env, waits for GPUs < 40 C, logs `nvidia-smi` |
| `step_timer.py` | Executes the trainer with per-step timing and loss/grad-norm logging (in memory, no source edits) |
| `numa_entry.py` | Binds each rank (and its DataLoader workers) to its GPU's NUMA node |
| `summarize.py` | Mean / median / CV of the step time, teacher time, peak memory |
| `compare_loss.py` | Per-step relative difference of the five losses and grad norm between two runs |

## Setup

- `SRC`: a second clone of this repository with every branch fetched (`git fetch --all`); the
  scripts switch it between commits, so do not run them from it.
- `PY`: Python of the training environment (tested with PyTorch 2.8.0+cu128, NCCL 2.27.3).
- `MODEL_ROOT`: contains `lingbot-vla-v2-6b`, `Qwen3-VL-4B-Instruct` and `moge-2-vitb-normal`.
- `DATA_LIST`: the `data.train_path` list file, one `robotwin <lerobot dataset dir>` line per task.
  The reported numbers use the 50 RoboTwin 2.0 `aloha-agilex_clean_50` tasks converted with the
  official RoboTwin pi0 pipeline (2,500 episodes, 549,787 frames).
- `CACHE`: an empty directory with ~1.5 MB per frame free (806 GiB for the data above).
- No other process may use the GPUs; `bench.sh` refuses to start otherwise.

```bash
SRC=... RUNS_ROOT=... PY=... MODEL_ROOT=... DATA_LIST=... CACHE=... \
  bash experiment/bench/reproduce.sh            # or a subset: vit | build | cache
```

`vit` takes ~2 h (10 runs), `build` ~1 h, `cache` ~1 h.

## Reference results (8 x RTX PRO 5000, 350 W, max 64 C, no throttling)

| Version | Commit | Step time | vs. baseline | Peak MiB |
|---|---|---:|---:|---:|
| baseline | `be969b8` | 4.5080 s | — | 71,799 |
| improv1 | `412b978` | 4.2840 s | +5.23% | 63,451 |
| improv2 | `42e963e` | 4.2288 s | +6.60% | 63,507 |
| improv2 + `--train.fsdp_align_root true` | `6d18cfa` | 4.2268 s | +6.65% | — |
| baseline + teacher cache | `d66fb35` | 3.9745 s | +13.44% (vs. 4.5087 s) | 65,867 |

Throughput gain = baseline step time / step time - 1. Runs of the same commit agree within
0.05%. Step-1 losses are bit-identical across all versions; over steps 1-30 the loss drift
stays within that of two runs of the same code (~0.1-0.2%). In teacher cache `read` mode the
VLA loss differs from step 1 because the flow-matching noise sequence shifts (the teachers no
longer consume random numbers); `verify` mode keeps the online teachers and checks every cached
target bit for bit.
