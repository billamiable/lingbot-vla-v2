# 激活重算与 batch 配置对照实验

[最终报告](report.md)比较单机8卡、global batch 256下的两种配置：重算开启、mbs32、GA1；重算关闭、mbs16、GA2。官方代码和Qihan优化版分别测量两轮，每轮200次完整参数更新。

本分支以官方 baseline `be969b8` 为基点，新增内容仅在本目录。它是独立实验归档，引用了 `bench-repro@9dc0677` 的核心计时实现，但运行长度、主统计窗口、重复顺序和分配器设置不同。

## 阅读与离线复核

- `report.md`：完整报告，含配置、性能、显存、三套统计窗口和数值检查。
- `evidence/runs/`：逐rank原始JSONL、训练日志、配置、输入指纹、显存采样与实际执行trainer。
- `evidence/reference/`：用于核对的Qihan历史记录。
- `evidence/*.receipt.json`：各次运行是否完成及8个rank记录数。
- `evidence/{summary_all,aggregate,consistency_audit,verification}.json`：当时生成的统计与审计。
- `evidence/driver.py`、`evidence/harness/`、`evidence/study_harness/`：实际使用的原始运行脚本；保留当时绝对路径。
- `SHA256SUMS`：全部436个原始证据文件的校验和。原始日志未改写。

在本目录执行，无需GPU，仅使用Python标准库：

```bash
sha256sum -c SHA256SUMS
python3 scripts/reanalyze.py --output /tmp/lingbot-ac-reanalysis
```

输出目录必须尚不存在。脚本在临时目录运行原始统计逻辑，读取原始日志，输出三份JSON，并要求最终aggregate与归档结果完全一致。它不调用原始训练脚本，也不重新运行依赖原机器源码、模型与数据的`verify_final.py`。

## 在GPU机器复现训练

原始机器：8卡RTX PRO 5000，驱动580.95.05；Python3.12.15、PyTorch2.8.0+cu128、Transformers4.57.3、Triton3.4.0、FlashAttention2.8.3。完整版本见`evidence/environment.json`。

代码、模型和数据不包含在本目录。需要自行准备：

1. 两份独立源码：main `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`；优化版 `71a41bca6be05b15a170b5358e3f6b43c4523898`。若优化版commit不可获取，可在第二份baseline副本应用`evidence/optimized-vs-main.patch`，恢复本次优化版源码内容。
2. LingBot-VLA-v2-6b、Qwen3-VL-4B-Instruct和MoGe权重；RoboTwin clean50 PNG数据；对应teacher cache。原始路径见`evidence/manifest.json`和每次运行的`effective_args.json`。
3. 对应代码依赖与可用的8卡机器。运行脚本会拒绝在已有GPU进程时启动。

原始控制器的实验根目录是`/root/experiments/lingbot-ac-mbs-20261108`。在新的实验根目录放置`driver.py`、`harness/`、`study_harness/`和两份源码目录`main/`、`opt/`，并修改控制器中的`ROOT`、`PY`、模型/数据/teacher-cache路径和编译缓存路径。使用新的输出目录，不覆盖归档中的`runs/`。

准备后，在新根目录依次执行：

```bash
python3 driver.py reproduce    # 原样harness，main/opt各80次更新，默认分配器
python3 driver.py study_alloc  # 四配置各两次、200次更新，统一expandable_segments
```

第一阶段需核对31–80中位数及receipt，确认复现后再启动正式对照。第二阶段的精确开关、反序运行顺序及分配器设置均保存在`driver.py`；统计入口见上面的离线复核。`bench.sh`可能吞掉trainer退出码，因此应以receipt中的passed和8个rank完整记录数判断成功。

`prepare_harness.py`是早期构建脚本；最终实际使用的观测补丁以本目录的`study_harness/`为准，不要用早期构建器覆盖它。

## 结果范围

主比较只合并`alloc_`前缀的8次完整运行，默认分配器的OOM、smoke和复现记录独立保留。模型文件验证的是size/mtime，输入指纹仅覆盖前3次更新。200步训练健康检查不代表完整收敛或下游成功率验证。
