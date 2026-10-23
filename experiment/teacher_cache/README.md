# Teacher 离线缓存

每个训练步开头，都要先运行三个冻结的 teacher（深度用 MoGe2 和 MoRGBD，未来帧用 DINO video 模型），生成深度和视频的对齐 target。这些 teacher 的输入只取决于样本本身：图像增强关闭，未来帧固定在 49 帧之后。所以 target 可以预先算一遍，训练时由 DataLoader 直接读取。

缓存是可选功能。不加 `--train.teacher_cache_dir` 时，训练行为完全不变。

| 参数 | 作用 |
|---|---|
| `--train.teacher_cache_dir DIR --train.teacher_cache_mode build` | 对每个样本运行一次 teacher，写入 `DIR` 后退出 |
| `--train.teacher_cache_dir DIR`（`read` 模式，默认） | 从 `DIR` 读取 target 训练，不加载 teacher |
| `--train.teacher_cache_dir DIR --train.teacher_cache_mode verify` | 照常运行 teacher，记录缓存和在线结果的差异，并用缓存的 target 训练 |

缓存内容，每个训练样本一行，bf16：

- `depth.npy`：当前帧深度 target，256 × 1024
- `video_current.npy`、`video_future.npy`：当前帧和未来帧的 DINO patch target，各 256 × 1024
- `future_index.npy`：当前帧正好是本样本未来帧的那个样本编号。未来帧深度 target 直接读 `depth.npy` 的这一行，不单独存。未来帧在 episode 末尾的截断规则和 LeRobot 处理 delta timestamps 的方式一致。

每个样本存 1.5 MB。`meta.json` 记录了数据集、帧数、相机顺序和 teacher 权重，和训练配置不一致时会拒绝读取。图像增强必须保持关闭。

`read` 模式不构建 teacher，而构建 teacher 会消耗随机数，所以同一个随机种子下，`read` 运行的 flow matching 噪声和在线运行不同：数据和 target 相同，训练在统计上等价，但不能逐位一致。`verify` 模式保留了 teacher，噪声也保持一致，所以要和在线运行对比 loss 时用这个模式。

## 构建缓存

```bash
PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... MEM_PROBE=0 \
  bash experiment/single_node_repro/train_pro5000.sh cache_build \
  --train.teacher_cache_dir /raid/teacher_cache/robotwin_clean --train.teacher_cache_mode build \
  --data.num_workers 12
```

每张 GPU 处理一段连续的样本区间。中断后再次运行，会跳过已经标记完成的行，从断点继续。每个 batch 都会补满到 micro batch 大小：深度 teacher 经过 `torch.compile`，不同 batch 大小会选用不同的 kernel。如果希望缓存的 target 和在线运行逐位一致，构建时请使用和训练相同的 micro batch 大小。

## A/B 实验

```bash
PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... CACHE=/raid/teacher_cache/robotwin_clean \
  bash experiment/teacher_cache/run_ab.sh
python experiment/teacher_cache/summarize_ab.py $OUT_ROOT --ref-300w <300 W 下在线 teacher 的 500 步运行目录>
```

`run_ab.sh` 依次运行：在线 teacher 基线；一次 verify；清空 page cache 后的 read（缓存从磁盘读）；缓存已在内存中的 read；以及 300 W 下各 500 步的 read 和 verify。结果见 [REPORT.md](REPORT.md)。

## 云盘模拟

```bash
bash experiment/teacher_cache/essd_sim/essd_sim.sh setup /raid/teacher_cache/robotwin_clean   # 建模拟盘并拷入缓存
PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... CACHE=/raid/teacher_cache/robotwin_clean \
  SIM_CACHE=/mnt/essd_sim/robotwin_clean bash experiment/teacher_cache/essd_sim/run_essd.sh
python experiment/teacher_cache/essd_sim/summarize_essd.py $OUT_ROOT
bash experiment/teacher_cache/essd_sim/essd_sim.sh teardown                                     # 删除模拟盘
```

`run_essd.sh` 依次运行：缓存在本地 NVMe；模拟云盘限速到 ESSD PL1、PL0 的每盘上限；PL0 再加读取延迟。需要 root 权限（loop 设备、device-mapper、cgroup 限速）。结果见 [REPORT.md](REPORT.md) 第 6 节。

检查脚本：

- `check_future_frames.py`：比较每个样本的未来帧，和它的未来帧深度所在那一行的当前帧，确认两者像素一致。
- `check_teacher_determinism.py`：对同一批输入运行两次深度 teacher，并把两次结果都和缓存比较。

在容器里使用缓存时，需要给 DataLoader 的 worker 足够的共享内存（`--shm-size` 或 `--ipc=host`）：每张卡的每个 batch 会多带 32 × 2 MB 的 target。
