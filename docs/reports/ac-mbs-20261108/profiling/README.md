# 激活重算性能归因

[完整分析报告](report.md)解释此前四组性能结果：重算32 GA1虽然增加计算，却将FSDP通信轮次减半，减少的暴露通信/同步时间超过额外计算成本。新增重算16 GA2控制组和独立无Profiler复核用于区分这两部分。

## 文件

- `report.md`、`attribution.png`：结论、控制实验、全rank时间分解与图表。
- `evidence/analysis.json`：6组×8rank×3更新的逐步分析，以及2组无Profiler复核计时。
- `evidence/attribution.json`：表格与配置间差值。
- `evidence/runs/`：逐rank计时、训练日志、有效参数、输入指纹、显存采样、执行trainer和每rank的trace分析结果。
- `evidence/profile_verification.json`：完整性、有限值、配置、输入一致性与热降频审计。
- `evidence/TRACE_SHA256SUMS`：48份完整Kineto trace的SHA256。
- `scripts/`：实际采集入口、分析器、复核与报告生成脚本。

完整Kineto trace为大型文件，保存在以下位置；本分支保存原始逐步训练日志和每份trace的分析输出：

- 原机器：`/root/experiments/lingbot-profile-20261108/runs/<配置>/trace_rank<R>.json.gz`
- 本地：`/home/yujie/workspace/yujie/iCode/VLA/robbyant/reports/ac-mbs-20261108/profiling/evidence/runs/<配置>/trace_rank<R>.json.gz`

## 离线复核

只使用Python标准库，重新聚合已保存的每rank事件统计，并要求与归档analysis完全一致：

```bash
python3 scripts/reaggregate.py --output /tmp/profile-reaggregate.json
```

如果已经将48份完整trace补回`evidence/runs/`，可重新解析原始事件：

```bash
cd evidence
sha256sum -c TRACE_SHA256SUMS
cd ..
python3 scripts/reaggregate.py --from-raw --output /tmp/profile-from-raw.json
```

默认复核读取缓存的逐rank分析结果；只有`--from-raw`才重新解析Kineto原始事件。原始事件以CPU `user_annotation`更新区间为边界，排除重复的`gpu_user_annotation`。计算、NCCL和内存操作先各自取区间并集，再算交集，避免将重叠kernel耗时简单相加。

## 在GPU机器重新采集

环境与模型、数据和源码前提沿用[上一轮复现说明](../README.md)。源码为main `be969b8`、优化版`71a41bc`。实际采集根目录：`/root/experiments/lingbot-profile-20261108`。

`evidence/driver.py`顺序运行6组，每组60次完整更新；`evidence/harness/step_timer.py`在第30次Profiler预热、第31–33次采集全部8rank。记录CPU与CUDA，关闭shape、stack、memory tracing，不新增每阶段同步。AST只包装阶段标记，不修改训练计算。

`evidence/validate_controls.py`等待六组完成，再用`validation_harness`独立运行main/opt重算16 GA2各80次更新。它调用的原始控制器同时保存在`evidence/validation_driver_reference.py`；原始导入位置是上一轮实验的`driver.py`。

迁移到新机器时，先改脚本中的ROOT、SOURCE、PY、模型/数据/teacher-cache和编译缓存路径，准备干净源码副本与全新输出目录。检查8卡空闲后再运行采集控制器。原始脚本包含当时绝对路径，不能直接在归档目录启动覆盖已有结果。

## 范围

这些是同机同配置的执行路径证据。“未被计算覆盖的NCCL活动”包含传输、同步与rank等待，不等于纯网络传输；GPU kernel活动区间也不等于硬件算术单元利用率。固定microbatch的控制组证明重算本身有净成本，本报告不把重算概括为普遍提速开关。
