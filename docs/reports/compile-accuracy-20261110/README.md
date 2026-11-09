# Decoder区域编译：数值影响与收益来源

在相同的checkpoint/FSDP边界与既有优化基础上，比较A（decoder区域eager）和B（decoder区域Inductor编译）。PR1三个手写融合均关闭；teacher cache、ViT/FSDP拆分、PR2通信参数、无padding FP32 joint attention保持一致。8卡PRO5000、激活重算开启、mbs32、GA1、global batch256。

## 精度判断

参考Qihan 2026-11-05性能报告第4.1节的方法，分两层：

1. 固定同一权重和相同输入，核对第一步全rank的全部可训练参数梯度、optimizer更新后权重的逐位一致性及相对L2/最大绝对差；重复A量化本底波动。
2. A/B各6次重新加载初始权重、每次300步，交替次序ABBA/BAAB/ABBA。对每次运行分别计算1–300、151–300、251–300的平均loss，然后进行双侧精确置换检验（6对6共924种分法）。以运行作为样本，不把同一轨迹的300步当成300个独立实验。Qihan原报告部分比较为8对6；本轮采用均衡6对6，统计方法相同、样本数不同。

训练flow输入显式对齐：独立CPU RNG抽取标准正态noise，以及Beta(1.5,1)*0.999+0.001的time；分布保持原定义。CPU抽样实现不同于原CUDA实现，也不同于前轮固定linspace time的数值fixture，不能将绝对首步loss/梯度差直接与前轮混比。逐步记录8rank的全部CPU张量哈希，下一组与基线对齐。未覆盖完整收敛和下游成功率。

短检查额外包含B_cast（仅区域编译启用emulate_precision_casts）以及PR1全开。B_cast用于检验保留中间低精度舍入的影响，不代表严格数值等价；也不改变外层编译配置。短检查rank0记录两条流第0/1/17/35层输入、输出、输出反向梯度，帮助定位最早的分歧边界。

## 编译收益来源

独立profile A、B、PR1仅RMSNorm、仅SwiGLU、仅QKV准备和PR1全开：各18步，仅profile的rank0关闭FX graph cache以确保生成IR/代码，保留Triton二进制缓存；rank0抓取第11–13步CPU/CUDA活动，同时保存FX及Inductor生成代码/IR。只比较算子、kernel、融合边界与重算行为，不根据这些profile运行的端到端均值判断速度，尤其不重复前轮C/D受data wait影响的性能推论。

RMSNorm/SwiGLU位于decoder内部；跨VLM/action的cast、cat、RoPE在decoder层调用外部。必须分别辨认新增区域编译、原有外层compile和PR1手写kernel；时间接近不能证明三项优化全部被替代。生成代码及timeline需要人工结合阅读，不能仅凭FX节点数解释全部加速。

## 运行和证据

实验机器目录：`/root/experiments/lingbot-compile-accuracy-20261110`。

- `study_runner.py` 顺序执行数值短检查、profile及12次训练；每组启动前要求全部GPU空闲，任何失败即停止后续组。
- `status.json`、`*.receipt.json`、`*.inputs.json` 记录状态、完成检查和输入校验。
- `summarize.py` 汇总loss、参数/梯度误差和算子profile；报告自动更新为 `report.md`，未完成时不作精度结论。
- `src`基于71a41bc应用`study.patch`；`source_freeze.json`记录实际源码与harness的SHA256。
- `runs/`保存逐rank训练日志/逐步数据、参数差异统计、层边界统计、FX、生成代码、IR及压缩profiler trace。
- 大参考张量保留在 `/raid/experiments/lingbot-compile-accuracy-20261110/references`，不随报告分发。

需要相同模型、数据、teacher cache及Python环境；具体路径在study_runner.py和harness/bench.sh。首次准备使用setup.py/launch.py；已存在运行目录禁止覆盖。失败恢复需保留对应目录和日志，再显式选择续跑阶段。
