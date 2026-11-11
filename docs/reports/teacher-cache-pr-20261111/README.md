# Teacher-cache PR validation — 2026-11-11

## Summary

**Category:** frozen-teacher computation / training IO. **Base:** `perf/training-efficiency` (`be969b8`).

The MoGe2 → MoRGBD depth teachers and the DINO video teacher are frozen. With deterministic images and fixed future-frame selection, their targets can be computed once rather than recomputed every SFT update. This PR adds an optional offline target cache with `build`, `read`, and `verify` modes. It does not depend on the subsequent ViT, kernel-fusion, FSDP, or attention PRs.

Fresh validation on 8 RTX PRO 5000 GPUs: **4.5210 → 3.9827 s/update; throughput +13.52%**, measured end to end including data loading. The cache also avoids loading the frozen teachers in read mode. All cache options are **off by default**.

## Files changed

- `lingbotvla/data/vla_data/teacher_cache.py`: BF16-preserving memory-mapped storage, episode-aware future-frame indices, configuration/completion validation, resumable distributed construction, and dataset wrapper.
- `multi_vla_dataset.py`: optionally return the actual sample index, including fallback reads.
- `tasks/vla/train_lingbotvla.py`: default-off configuration, build/read/verify routing, target layout restoration, and a `Data_Wait` log field.
- `tests/test_teacher_cache.py`: 12 CPU tests covering storage, indexing, interruptions, multiple writers/workers, guards, and target layout.
- `docs/teacher_cache.md`: configuration, commands, format, resumption, and limitations.

## How to enable

Use the usual model/data/teacher paths. With the tested 8-GPU configuration:

```bash
# Precompute targets and exit. Repeating this command resumes an incomplete cache.
torchrun --standalone --nproc-per-node 8 tasks/vla/train_lingbotvla.py \
  configs/vla/robotwin/robotwin_dist_muon.yaml \
  --train.enable_gradient_checkpointing true \
  --train.micro_batch_size 32 --train.global_batch_size 256 \
  --train.gradient_accumulation_steps 1 \
  --train.teacher_cache_dir /path/to/teacher-cache \
  --train.teacher_cache_mode build
```

Then use the same command with `--train.teacher_cache_mode read` to train without loading/running the teachers, or `verify` to run both paths and log target comparisons across all ranks. Omit `teacher_cache_dir` to retain the original online behavior. Build and read with the same micro batch size. [Full documentation](https://github.com/billamiable/lingbot-vla-v2/blob/perf/teacher-cache/docs/teacher_cache.md).

## Implementation and numerical behavior

Target arrays store the original BF16 **bit patterns as int16**, without additional quantization. Future depth reuses the depth row at the episode-clamped future index. Reads follow the actual returned sample index; construction rejects substituted samples rather than silently writing them to the wrong row. Each rank writes a disjoint range and flushes targets before publishing completion every 50 batches. Partial batches are padded to the configured teacher micro batch size.

Depth target **strides also matter**. The online `get_depth_target` returns BTD features with channels-first storage, whereas ordinary cache collation produces contiguous BTD. An initial prototype preserved values but changed strides and showed about 0.296% first-step gradient relative L2 difference. This PR restores the online depth layout before computing losses; `verify` checks both strides and values. This avoids changing compiled loss/reduction kernels simply because the targets came from disk.

## Testing (numerics)

Fresh tests against the official baseline, using the submitted code:

- All 12 CPU tests pass, both normally and with `python -O`; no GPU/checkpoints required for these tests.
- Real 8-GPU construction of a **1,150-frame complete-episode subset** succeeds. All three target arrays and future indices are bit-identical to the matching prefix of the existing full cache; the new subset also passes online verification.
- Full-dataset online verification: **31 updates × 4 target types**, with differences reduced over all 8 ranks. Every check reports `max|diff| 0`, `exact 1`, and matching target strides.
- Controlled first-update comparison uses identical student inputs, weights, explicit flow-matching noise/time, and all **1,672 trainable logical parameter tensors** across 8 rank shards. Gradients are compared after clipping and immediately before the optimizer; weights are compared immediately after the optimizer.

| Compared with original baseline | Gradient relative L2 | Updated-weight relative L2 | Bit-exact gradient tensors |
|---|---:|---:|---:|
| Cache verify | 7.37e-8 | 3.18e-8 | 1,664 / 1,672 |
| Cache read | 4.87e-8 | 3.39e-8 | 1,664 / 1,672 |
| Unchanged baseline, repeated | 8.88e-4 | 4.89e-6 | 908 / 1,672 |
| Candidate with cache disabled | 1.67e-3 | 6.20e-6 | 908 / 1,672 |

The first-step total/VLA/depth/future-depth/future-video loss scalars match exactly across these controlled runs: `0.405029296875 / 0.3740234375 / 3.414154052734375 / 3.326416015625 / 0.7472038269042969`.

The uncached controls themselves show run-to-run gradient variation; its source was not isolated in this study. The cache comparisons above are smaller, but this is **not** a claim that training is fully bitwise deterministic. Read mode skips teacher initialization and changes RNG consumption, so a common seed alone does not match flow noise/time. Performance runs use the natural training RNG; their per-step VLA losses are not used as a numerical equivalence test. These are target and first-update checks, not full-convergence or downstream-task validation.

## Performance (fresh November 11 measurements)

| Run | Mode | Full update (s) | Samples/s | Data wait (s) | Peak GPU memory (MiB) |
|---|---|---:|---:|---:|---:|
| final_base_1 | online | 4.5223 | 56.61 | 0.0005 | 61,481 |
| final_cache_1 | cache read | 3.9840 | 64.26 | 0.0005 | 58,867 |
| final_cache_2 | cache read | 3.9814 | 64.30 | 0.0005 | 58,867 |
| final_base_2 | online | 4.5197 | 56.64 | 0.0005 | 61,481 |
| **base mean** | | **4.5210** | **56.62** | | **61,481** |
| **cache mean** | | **3.9827** | **64.28** | | **58,867** |

Configuration: 8 RTX PRO 5000 engineering boards (NVML reports `NVIDIA Graphics Device`, PCI device `2BB3`, 73,415 MiB each), RoboTwin clean50 PNG / 549,787 frames, official DistMuon recipe, activation checkpointing on, micro batch 32, accumulation 1, global batch 256. PyTorch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3, Transformers 4.57.3, Triton 3.4.0, driver 580.95.05. Both paths use `expandable_segments:True`, identical NUMA binding and NCCL settings (`MIN_CTAS=8`, `NTHREADS=128`, `P2P_READ_ENABLE=1`), and the same local RAID storage.

ABBA order: online → cache → cache → online. Each run completes 31 updates on all 8 ranks; updates 1–10 warm up compilation and 11–30 are measured. Each update uses the **slowest rank's synchronized full-update wall time**, including data wait and logging, then the two runs are averaged. This differs from the trainer's `StepTime`, which excludes `next(data_iterator)` waiting. Peak memory is the maximum across all GPUs sampled by `nvidia-smi` at one-second intervals. Verification and numerical-audit runs are excluded from performance numbers.

Update time decreases by 11.91%. Peak sampled GPU memory falls by 2.55 GiB. Steady-state data waits are shown above; no thermal slowdown flags were observed in the sampled performance runs.

## Limitations and review notes

- Supported scope: map-style multi datasets; depth + future-depth + future-video alignment; current/future DINO patch targets; random image augmentation disabled. CLS targets are not cached.
- The full cache occupies **805.4 GiB**. This exchanges repeated teacher computation for storage and IO; speedup depends on the storage system. The small build validates construction, not full-dataset preprocessing cost or amortization.
- Metadata checks configured dataset identities/order/frame counts, camera order, future offset, checkpoint paths, and recorded video settings. It is not a data/weight content hash. Rebuild after changing data, preprocessing, or teacher weights/settings, even at the same paths. Legacy prototype caches without micro batch metadata are accepted and should be verified before reuse.
- All builder ranks share a writable directory; readers share the completed cache. Do not run independent builders concurrently. A restart can change world size only after all old writers stop. Distributed construction was tested on one 8-GPU node.
- Teacher-dependent image visualizations are skipped in read mode; verify retains online images/visualizations.

[Raw logs, numerical comparisons, environment, and reproduction scripts](https://github.com/billamiable/lingbot-vla-v2/tree/teacher-cache-bench/docs/reports/teacher-cache-pr-20261111). The evidence README identifies final runs separately from the retained layout diagnostic/prototype runs.

## Evidence navigation and reproduction

- `results.json`: primary summary. Only `final_base_1`, `final_cache_1`, `final_cache_2`, and `final_base_2` contribute to performance.
- `runs/<run>/rank0.jsonl` through `rank7.jsonl`: raw full-update timing, data wait, and loss records. `train.log.gz` and `*.launcher.log.gz` retain original output. `smi.csv` records all GPU memory, temperature, and slowdown samples.
- `num_base_1` is the first-update reference. `num_verify_fixed`, `num_read_fixed`, `num_off_fixed`, and `num_base_2` supply the comparisons in the table. `grad_rank*.json` / `parameter_rank*.json` contain all per-tensor comparisons, not the model tensors themselves. `*.inputs.json` records matched model inputs across all ranks for three updates.
- `verify_small_fixed.receipt.json`, `verify_full_fixed.receipt.json`, and corresponding compressed train logs record target equality and stride validation.
- `build-prefix-compare.json` records comparison of the new 1,150-frame cache against the existing full cache. The existing full cache was verified rather than rebuilt in its entirety.
- `unit-tests.log.gz` / `unit-tests-optimized.log.gz`: 12 CPU tests, normal / `-O`.
- `candidate.patch`, `tested_candidate_commit.json`, and `source_freeze.json`: source identity. Early run logs may show the draft commit before the final source was committed; the freeze hashes and final five-file check record the tested implementation.
- `results.before_layout.json`, `diag_layout`, `num_verify`, `num_read`, and earlier `base_1` are retained diagnostics of the initial contiguous-layout prototype. They are **not** the primary performance or final numerical results. `fixed-queue.log.gz` also retains the diagnostic gate stopping on disabled-cache control variability; final runs completed via `finish-bench.py`. No production numerical changes were made to suppress that control variability.
- `manifest.json`: original exported-file SHA256 hashes. `published-manifest.json` additionally covers the readable report and portable reproduction entrypoint. Frozen numeric tensor references (~48 GB), model checkpoints, cache arrays, and output images are excluded.

To reproduce, provision the same Python/model/data environment and two checkouts (official `be969b8` and submitted `a51af8759b80e6a196e82f802dc0c4bf2538ca99`), then build/verify the full teacher cache as documented in `docs/teacher_cache.md`. Reserve an idle 8-GPU node and run:

```bash
BASELINE_SRC=/path/to/baseline CANDIDATE_SRC=/path/to/candidate \
PY=/path/to/env/bin/python MODEL_ROOT=/path/to/models \
DATA_LIST=/path/to/robotwin-clean50-image-training.txt \
TEACHER_CACHE=/path/to/teacher-cache OUT=/path/to/new-results \
  bash reproduce.sh
```

`reproduce.sh` runs the fixed-flow first-update controls, natural-RNG ABBA performance runs, and 31-step online target verification using the exported harness. It stops if another process owns a GPU, waits for GPUs to cool below 40 C, and disables checkpoint saving. The 3-step numerical runs do not produce a 30-step timing summary. The original `driver.py` / setup scripts retain machine-specific paths and diagnostic chronology; use `reproduce.sh` as the portable entrypoint. Relative-L2 aggregation is `sqrt(sum(diff²) / sum(reference²))` across rank shards; do not average per-tensor relative errors.
