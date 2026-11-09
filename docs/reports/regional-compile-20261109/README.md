# 恢复 Decoder 区域编译：可行性与性能验证

目的：验证将FSDP hooks保留在图外、单独编译decoder纯计算的checkpoint区域，能否恢复捕获并改善训练性能。性能结论仅采用A→B；C、D有明显数据等待和长尾，保留原始数据但不作性能归因。

| 组 | Decoder 区域编译 | 手写 RMSNorm / SwiGLU / QKV 准备融合 |
| --- | --- | --- |
| A | 关 | 关 |
| B | 开 | 关 |
| C | 关 | 开 |
| D | 开 | 开 |

最终结论见 `report.md`；`results.json` 提供机器可读数据。`followup_status.json` 和各次 `*.receipt.json` 确认是否真正完成；进行中的报告不能当最终结果。

## 执行与复现

源代码基于 `71a41bc`，在独立 checkout 应用 `study.patch`（包含 `regional_runtime.py`）。`source_freeze.json` 记录正式运行前源代码和 harness 的 SHA256；`manifest.json` 记录分发证据的 SHA256。运行环境、路径与命令见 `harness/bench.sh`、`runs/*/effective_args.json`、`runs/*/source.txt` 和 `driver.py`。

原实验目录 `/root/experiments/lingbot-regional-20261109`。脚本中使用该绝对路径；异机复现需同时修改 `driver.py`、`followup.py` 的 ROOT，并提供模型、训练数据及 teacher cache。使用 `/root/qihank/envs/lingbotvla/bin/python`，torch 2.8.0+cu128、transformers 4.57.3、triton 3.4。

1. `python3 driver.py smoke`：每组3步数值检查，A先生成参考，其余组对照第一步全部梯度。
2. `python3 followup.py`：等待 smoke 通过，重复A，校验所有组输入哈希，运行8轮200步测量并生成报告。
3. `python3 summarize.py`：随时重新汇总已完成的数据。

`resume_smoke.py` 是本次失败恢复用脚本，不是从零复现的起点。已存在的运行目录会阻止覆盖；失败后应保留并移动到独立 attempts 目录。

## 对照范围

四组统一将 FSDP hooks 放在纯 forward checkpoint 区域之外；A/C执行 eager checkpoint，B/D编译 checkpoint 整个区域。区域编译开关全程 `fullgraph=True`；编译异常或8卡任一缺少图记录会阻止后续测量。共享入口的重编译上限为256，累计上限4096。

四组采用无 sequence padding 的 FP32 joint attention，避免将取消padding的收益误计为QKV融合收益。训练均开启激活重算、mbs32、GA1、global batch256、teacher cache和相同FSDP参数。不能直接与旧HF checkpoint或带padding的历史报告混比。

A→B将平均更新时间由3.3358秒降到2.8599秒，步时降低14.27%，吞吐提升16.64%。每组两轮、每轮200次更新，主窗口31–200；A、B的data wait均值约0.0005秒。C、D相应均值约0.86–0.88秒，不能用于判断手写融合的增益、损失或与区域编译的性能交互。

A、B均保留teacher cache、ViT/FSDP拆分与视觉位置缓存、PR2通信调优，关闭PR1三项主要融合，但保留无padding路径并共享新checkpoint边界。A不是官方原始baseline。区域编译可能同时影响融合与checkpoint保存/重算策略，尚未分解收益；B相对A单步梯度相对L2差10.98%，来源未定位，不能宣称数值等价。

短运行固定 noise/time 只用于数值对照，正式性能运行采用原训练分布。数值比较在梯度裁剪后、optimizer前进行；A_repeat用于观察重复基线自身波动。短程loss/梯度对照不是模型效果或比赛可用性的保证。

## 证据

`runs/` 包含各rank逐步JSONL、训练日志、显存采样、输入哈希、第一步梯度差异统计及区域编译图目标；大梯度参考 `.pt` 和训练输出目录只保留在原机器，不分发。`attempts/` 保留未进入正式测量的失败记录：v1 checkpoint metadata不一致，以及v2默认重编译次数上限触发。
