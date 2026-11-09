# Decoder Dynamo捕获根因实验

本分支基于官方baseline `be969b8`，只归档decoder编译捕获问题；激活重算与batch性能实验保留在独立的 `ac-batch-bench` 分支。

[完整报告](report.md)给出真实训练回退栈、单层捕获矩阵、双卡FSDP因果对照，以及joint/fused MoE的独立障碍。

核心结论：HF checkpoint包住了带FSDP hooks的`Module.__call__`；这些hooks默认被Dynamo禁用，使decoder半层回退eager。单独的joint计算在隔离mask与MoE包装障碍后可以捕获并由Inductor执行。

## 文件

- `evidence/runs/real_main_ac32/`：真实8卡、3步训练日志、配置、逐rank计时和实际trainer。
- `evidence/capture/`：真实decoder半层及joint的严格捕获，FX图与错误栈。
- `evidence/fsdp_probe/`：两卡FSDP、记录FX图的eager后端对照。
- `evidence/fsdp_inductor/`：相同两卡对照，实际Inductor代码生成与前后向。
- `evidence/capture_moe_initialized/`：正确初始化并行状态后的原始MoE失败。
- `evidence/capture_moe_noop_removed/`：移除无效lambda断言后，下一个日志/内省失败。
- `evidence/capture_moe_noop_removed_no_warn/`：再关闭预调优日志后的捕获成功。
- `evidence/inductor_joint_dense/`、`evidence/inductor_joint_moe_diagnostic/`：joint的Inductor前后向成功对照。
- `evidence/environment.json`、`source_identity.json`、`dependency_source.json`：实际版本、代码一致性和HF/FSDP源码。
- `evidence/verification.json`、`SHA256SUMS`：审计结果与原始证据校验和。

## 复现

原机器根目录：`/root/experiments/lingbot-dynamo-20261109`。源码环境与模型目录沿用[前一轮实验](https://github.com/billamiable/lingbot-vla-v2/blob/ac-batch-bench/docs/reports/ac-mbs-20261108/README.md)。真实训练用main be969b8；最小复现用opt 71a41bc，三个相关decoder/MoE类已经逐AST确认与baseline一致。

在源码根目录、原Python环境下运行以下入口。迁移时先修改脚本内的源码/模型/输出路径，使用全新输出目录，并确认GPU空闲。`probe.py`和`fsdp_probe.py`不做吞吐测试，输出中的seconds包含编译耗时。

```bash
export PYTHONPATH="$PWD:$PWD/lingbotvla/models/vla/vision_models/lingbot-depth:$PWD/lingbotvla/models/vla/vision_models/MoGe"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export TORCH_LOGS=graph_breaks,recompiles
PY=/root/qihank/envs/lingbotvla/bin/python
EXP=/root/experiments/lingbot-dynamo-20261109

# 真实模型3步，启动前检查real_path.py中的ROOT与引用路径
python3 "$EXP/real_path.py"

# 单层无FSDP捕获；capture后端只记录FX并执行，不做Inductor融合
CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP/probe.py" \
  --backend capture --out "$EXP/capture-new"

# 预构mask的joint，实际Inductor前后向
CUDA_VISIBLE_DEVICES=0 "$PY" "$EXP/probe.py" \
  --backend inductor --case joint_precomputed --out "$EXP/joint-inductor-new"

# 双卡FSDP对照，按顺序执行；脚本输出根目录需先改成全新目录
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$EXP/fsdp_probe.py" --backend capture
CUDA_VISIBLE_DEVICES=0,1 "$PY" -m torch.distributed.run --standalone --nproc-per-node=2 \
  "$EXP/fsdp_probe.py" --backend inductor

# MoE最小复现必须初始化parallel_state，避免测试环境产生logger伪断图
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.run --standalone --nproc-per-node=1 \
  "$EXP/probe.py" --backend capture --case moe --init-parallel --out "$EXP/moe-original-new"

# 进程内移除无效lambda断言，仍保留原调试日志
CUDA_VISIBLE_DEVICES=0 "$PY" -m torch.distributed.run --standalone --nproc-per-node=1 \
  "$EXP/probe.py" --backend capture --case moe --init-parallel --strip-noop-assert \
  --out "$EXP/moe-noop-removed-new"

# 再关闭调试日志，验证joint+fused MoE可由Inductor执行
CUDA_VISIBLE_DEVICES=0 BPEX_NO_WARN_ON_UNTUNED_CASE=1 "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=1 "$EXP/probe.py" --backend inductor --case joint_moe \
  --init-parallel --strip-noop-assert --out "$EXP/joint-moe-inductor-new"
```

上述命令只描述依赖与开关，模型/数据/源码不包含在归档中。`real_path.py`引用的前一轮控制器也保存在`reference_driver.py`；迁移时可改为导入该副本。完整日志保存了每次试探，包括未初始化并行状态的初始MoE试探；报告不把那一项当作真实模型根因。

## 数值与范围

捕获后端和Inductor分别有自己的无checkpoint编译控制组。对照中输出与输入梯度最大差0，指同一后端内部的比较，不代表Inductor相对原始eager逐位一致。

区域编译是已验证可行的诊断成功对照，尚未作为生产修复应用到36层完整训练。真实batch性能、显存、重编译、全参数梯度和最终模型效果需要下一阶段独立验证。
