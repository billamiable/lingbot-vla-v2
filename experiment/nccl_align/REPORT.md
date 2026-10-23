# NCCL 16 字节对齐实验报告（RTX PRO 5000 单机 8 卡，DistMuon）

## 结论

- **提速 6.6%（当前代码下）**：每步从 4.958 秒降到 4.629 秒（300 W，第 31–130 步，无降频）。350 W 下 p50 从 4.802 秒降到 4.485 秒，方向和幅度一致。省下的时间等于 root AllGather 省下的时间：403 → 74 ms/步。
  - 这个数字以当前主线的 FSDP 切分为基线：root 未拆分，AllGather 整段暴露在每步开头。
  - 如果同时采用 `fsdp_split_root`（拆分 root 并预取）等其他 AllGather 优化，root 剩下的参数会变少，这一项的额外收益会变小，需要在那个基线上重新测。
- **根因已确认是 16 字节不对齐，不是冷启动**：每个 rank 的元素数只差几个，不对齐时 AllGather 380 ms（4.4 GB/s），对齐时 65 ms（25.5 GB/s）；空闲 4 秒后耗时不变。
- **只有一处不对齐**：训练中每个 rank 的 1459 次通信里，只有 root 单元的 AllGather 和 ReduceScatter 不对齐。原因是 `action_out_proj.bias`（形状 55）切到每个 rank 只有 7 个元素，见"根本原因"和第 2 节。
- **修复不改模型**：把 `action_out_proj` 单独切成一个 FSDP 单元，开关是 `--train.fsdp_align_root`，默认关闭。参数、state_dict、计算都不变，checkpoint 兼容；只多一次约 86 KB 的小通信。
- **loss 和显存不变**：同一种子下 loss 平均相对差异 0.37%，和两次相同配置运行之间的差异相同。每卡显存峰值 69.98 → 69.81 GiB。
- **backward 顺带对齐，但本来就不慢**：fp32 的 ReduceScatter 对不对齐几乎不敏感，root 的 ReduceScatter 只从 144 ms 降到 131 ms。这也解释了原 profile 里"backward 比 forward 更快"的现象。
- **净收益**：按 5k 步算，300 W 下每次训练省约 27 分钟（6.89 → 6.43 小时），不需要任何预处理。和 teacher 离线缓存作用在不同环节，预计可以叠加，但没有一起实测。

forward 开头的 root AllGather（ViT + embed_tokens + 对齐头，收齐后 1.91 GB）每步约 400 ms，只有约 5 GB/s。同一台机器上 nccl-tests 跑同样大小能到 27 GB/s。本文用对照实验确认原因是 16 字节不对齐，并给出单独修复它的收益。

## 根本原因

**NCCL 需要 16 字节对齐才能全速搬运。** NCCL 的 kernel 用 128 bit 向量指令读写，一次搬 16 字节，前提是源地址和目标地址都是 16 的倍数。不满足时会退回更窄的拷贝，指令数成倍增加。

**FSDP2 的 AllGather 布局让每块的大小决定了对齐。** 一个 FSDP 单元里，每个参数沿第 0 维切成 world_size 份，每个 rank 拿 `ceil(dim0 / world_size)` 行。每个 rank 把自己的所有分片拼成一块，AllGather 再把 8 块依次排进输出：

```
输出： [ rank0 的块 ][ rank1 的块 ][ rank2 的块 ] ... [ rank7 的块 ]
地址：  0             C             2C                    7C       （C = 每块字节数）
```

FSDP2 是 in-place AllGather，每个 rank 的输入就是输出里自己那一段。所以只要 C 不是 16 的倍数，除 rank 0 外每一块的起点都会错位。

**元凶是 `action_out_proj.bias`。** root 单元有 411 个参数，其他参数每个 rank 的元素数加起来是 8 的倍数。`action_out_proj`（动作专家的输出层，768 → 55 维）的情况如下：

| 参数 | 形状 | 每个 rank 的元素数 | 8 的倍数 |
|---|---|---|---|
| `action_out_proj.weight` | [55, 768] | 7 × 768 = 5,376 | 是 |
| `action_out_proj.bias` | (55,) | 7 | **否** |

root 每个 rank 的那块因此是 119,465,991 个元素，bf16 下 C = 238,931,982 字节，% 16 = 14。就因为这 7 个元素，1.9 GB 的 AllGather 全部走窄拷贝，从 65 ms 变成约 400 ms。

## 解决方法

在给 root 切分之前，先对 `action_out_proj` 单独调用一次 `fully_shard`，让它自成一个 FSDP 单元。`fully_shard` 只收还没有归属的参数，所以之后的 root 单元里不再包含它：

```python
for layer in llm_layers:
    fully_shard(layer)
fully_shard(model.action_out_proj)   # 新增：4.2 万参数的小单元
fully_shard(model)                   # root：剩下的参数
```

| 单元 | 每个 rank 元素数 | 每块字节数 | % 16 | AllGather |
|---|---|---|---|---|
| root（修复前） | 119,465,991 | 238,931,982 | 14 | ~400 ms |
| root（修复后） | 119,460,608 | 238,921,216 | **0** | ~74 ms |
| action_out_proj（新单元） | 5,383 | 10,766 | 14 | 约 86 KB，可忽略 |

- **不对齐被挪到了小单元。** 新单元自身仍不对齐，但窄拷贝的代价和数据量成正比，从 1.9 GB 那次挪到约 86 KB 这次，代价就可以忽略。
- **不改模型。** 参数形状、state_dict 和计算都不变，checkpoint 兼容。如果改成把 bias 补到 56 维，或者给 root 塞一个占位参数，就会改变模型或 checkpoint。
- **backward 自动一起对齐。** forward 和 backward 共用同一套单元划分，所以 root 的 ReduceScatter 也对齐了。

**收益**：root AllGather 403 → 74 ms；端到端每步 4.958 → 4.629 s（**−0.329 s，−6.6%**），loss 和显存都不变。这一项不需要拆 root，可以单独开启。

## 和 `fsdp_split_root` 的关系

另有一个 `fsdp_split_root` 方案：把 ViT 的 24 个 block、merger、embed_tokens、对齐头拆成独立的 FSDP 单元，配合预取，让 root 通信和计算重叠。其中的 `fsdp_split_action_out_proj` 选项和上面的修复做的是同一件事。下文侧重该方案之外、本实验独有的结论：

1. 用对照组确认慢在对齐，排除冷启动
2. 训练中所有通信的对齐情况和唯一的元凶参数
3. 只修对齐、不拆 root 时的端到端收益
4. 为什么 backward 的 ReduceScatter 不慢

## 1. 对照实验：慢在对齐，不在冷启动（`align_bw.py`）

之前的冷启动测试一直用训练里的真实大小 119,465,991 个元素，它本身就不对齐，所以不管空闲多久都是约 380 ms，分辨不出原因。这里让每个 rank 的元素数只差几个（数据量相差不到 1e-7），因此耗时差异只可能来自对齐。8 卡，每组 20 次取中位数，取最慢的 rank。

| 通信 | 对齐（984 / 992） | 不对齐（991 / 985） | 倍数 |
|---|---|---|---|
| AllGather bf16，in-place（FSDP2 的用法） | 65 ms，25.5 GB/s | **380 ms，4.4 GB/s** | 5.8× |
| AllGather bf16，out-of-place | 65 ms | 393 ms | 6.0× |
| AllGather fp32，in-place | 128 ms | 694 ms | 5.4× |
| ReduceScatter bf16 | 66 ms | 120 ms | 1.8× |
| ReduceScatter fp32（训练用的 reduce_dtype） | 128–135 ms | 136 ms | ~1.0× |
| 每块对齐，只让输入指针偏 2/4 字节（bf16/fp32） | 65 / 129 ms | 114 / 233 ms | 1.8× |
| 每次调用前空闲 4 秒 | 66 ms | 383 ms | 和连续调用相同 |

- 不对齐时的 380 ms / 4.4 GB/s 和训练里的 371–398 ms / 5 GB/s 吻合；对齐时的 65 ms 和 nccl-tests 吻合。nccl-tests 会把元素数向下取整到 119,465,984，正好对齐，所以测不出这个问题。
- 空闲 4 秒后耗时不变，冷启动可以排除。
- AllGather 不管 bf16 还是 fp32 都慢 5–6 倍；ReduceScatter 只慢 1.0–1.8 倍。主要差别在通信类型，而不在 dtype。（推测：AllGather 是纯搬运，最受拷贝宽度限制；ReduceScatter 每一步要读两份再相加，本来就更接近计算瓶颈。未验证。）

## 2. 训练里实际的通信（`coll_log.py`）

包装 `all_gather_into_tensor` / `reduce_scatter_tensor`，跑 5 步，记录每次调用的每块大小、地址 % 16 和耗时；同时 hook FSDP2 的参数分组，打印导致单元不对齐的参数。不改训练代码。

- 每个 rank 共 1459 次调用，只有 root 的两次不对齐，各 decoder layer 和 DistMuon 的通信全部对齐：

  | root 单元 | 修复前 | 修复后 |
  |---|---|---|
  | AllGather bf16，每块 | 238,931,982 B，% 16 = 14，403 ms | 238,921,216 B，% 16 = 0，**74 ms** |
  | ReduceScatter fp32，每块 | 477,863,964 B，% 16 = 12，144 ms | 477,842,432 B，% 16 = 0，131 ms |

- 唯一的不对齐参数：`Linear.bias (55,) -> 7 per rank`，即 `action_out_proj.bias`。weight `[55, 768]` 每个 rank 7 × 768 = 5,376 个元素，是 8 的倍数，不影响对齐。
- forward 和 backward 共用同一个单元划分，所以修复同时对齐了两边；backward 本来就不敏感，只省了约 13 ms。
- 拆出来的 `action_out_proj` 单元自身仍不对齐（每块 10,766 B），但只有约 86 KB，代价可以忽略。端到端省下的 0.329 s 正好等于 root AllGather 省下的 403 − 74 = 329 ms。

## 3. 端到端 A/B（`run_ab.sh`）

DistMuon，在线 teacher，micro 32 × 8 卡，130 步，统计第 31–130 步。

| 运行 | 功率 | StepTime 均值 | p50 | 降频 | 显存峰值 |
|---|---|---|---|---|---|
| base300_130 | 300 W | 4.958 s | 4.962 s | 无 | 69.98 GiB |
| **align300_130** | 300 W | **4.629 s** | 4.635 s | 无 | 69.81 GiB |
| base_130 | 350 W | 4.801 s | 4.802 s | 无 | 69.98 GiB |
| align_130 | 350 W | 6.311 s | 4.485 s | GPU 6 降频 355 s，均值不可用 | 69.81 GiB |

- 300 W 那组每步省 0.329 s（−6.6%）。350 W 的 align 运行中一张卡过热降频，只看 p50：少 0.317 s，方向和幅度一致。
- loss：align 和 base 各项 loss 的平均相对差异为 0.37%（总 loss）、0.40%（VLA）、0.03–0.17%（对齐头），和两次相同配置运行之间的差异（0.37%）相同。
- 按 5000 步估算：300 W 下 6.89 h → 6.43 h，每次训练省约 27 分钟。可以和 teacher 离线缓存（每步约 0.55 s）叠加。

## 4. 本分支的实现：`--train.fsdp_align_root`

`lingbotvla/distributed/torch_parallelize.py` 里，在 `fully_shard(model)` 之前，遍历仍归 root 管的参数（已切分的子单元参数此时已是 DTensor），按 `ceil(dim0 / world_size) × 其余维度` 算出每个 rank 的元素数；不是 8 的倍数的，把它所属的模块单独 `fully_shard`。默认关闭；state_dict 的 key 不变，checkpoint 兼容。

和 `fsdp_split_action_out_proj` 的对比：

| | 本分支 `fsdp_align_root` | `fsdp_split_action_out_proj` |
|---|---|---|
| 依赖 | 独立开启 | 必须同时开 `fsdp_split_root` |
| 如何找到问题参数 | 按 world size 自动计算 | 写死 `action_out_proj` |
| `action_fp32=True` | **未检查**（这种模式下 `_fp32_linear` 绕过了 FSDP hook，单独成单元会出错；当前配置未开启） | 报错拒绝 |

两者是互补关系：
- 对齐修复让 root AllGather 搬得快（403 → 74 ms）。
- `fsdp_split_root` 让剩下的通信可以和 ViT 计算重叠。

拆 root 之后，root 里剩下的参数很少，单独做对齐的额外收益会比本文测的 0.33 s 小（推算，未实测）。

统一写法时的建议：
- 以 `fsdp_split_root` 为主体。
- 对齐修复改为可以单独开启。
- 不写死模块，改用按 world size 自动检测。
- 保留 `action_fp32` 检查。

## 5. 复现

```bash
# 1. 对照实验
torchrun --nproc_per_node 8 experiment/nccl_align/align_bw.py --gap 4
# 2. 训练中每次通信的大小、对齐、耗时，以及导致单元不对齐的参数（coll_groups_rank*.txt）
COLL_LOG=1 MAX_STEPS=5 TIMING_ONLY=1 PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... \
  bash experiment/single_node_repro/train_pro5000.sh coll_log_5
python experiment/nccl_align/summarize_coll_log.py $OUT_ROOT/coll_log_5 --steps 5
# 3. 端到端 A/B
PY=... MODEL_ROOT=... DATA_LIST=... OUT_ROOT=... \
  bash experiment/nccl_align/run_ab.sh align_log_5 base300_130 align300_130
```

环境：torch 2.8.0+cu128，NCCL 2.27.3，8× RTX PRO 5000 72GB。
