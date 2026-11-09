# LingBot decoder编译捕获失败：根因实验

2026-11-09 · PyTorch 2.8.0+cu128 · Transformers 4.57.3 · RTX PRO 5000

## 结论

**真实训练中decoder半层回退eager的直接原因，是HF checkpoint把Module.__call__连同FSDP hook包进了捕获区域；FSDP hook默认被Dynamo禁用，导致该checkpoint区域无法捕获。joint self-attention的数学计算不是这个回退的必要条件。**

另外还有两个独立障碍：joint路径中的build_block_mask被显式compiler.disable；Action fused MoE的Python包装包含无效lambda断言和预调优日志内省。它们需要分别处理，不能把所有断图统称为joint attention不支持。

## 真实8卡训练证据

官方baseline be969b8，沿用此前数据/模型/8卡配置，开启激活重算、mbs32、global batch256、expandable_segments，额外开启TORCH_LOGS=graph_breaks,recompiles及TORCHDYNAMO_VERBOSE=1。3次完整更新，8个rank各有3条记录，正常结束，不保存checkpoint。

真实日志中的调用链：

```text
QwenvlWithExpertV2Model.forward
  → VLM layer(..., compute_kqv=True)
    → HF GradientCheckpointingLayer.__call__
      → checkpoint(partial(super().__call__, **kwargs), *args)
        → Module.__call__ / forward pre-hook
          → FSDP fsdp_hook_wrapper
            → torch._dynamo.disable(..., recursive=True)
```

该断图发生在生成QKV的层调用，早于本轮joint attention执行。原始日志见[evidence/runs/real_main_ac32/train.log](evidence/runs/real_main_ac32/train.log)。HF与FSDP的实际源码快照见[dependency_source.json](evidence/dependency_source.json)。

## 单层严格捕获：joint与checkpoint是否天然不兼容

使用仓库真实decoder类、真实hidden/head维度，合成输入B=1、VLM长度8、Action长度4。VLM hidden2560、Action hidden768、32个query heads、8个KV heads、head_dim128。VLM参数bf16，Action参数fp32；这是捕获根因测试，不是完整训练规模性能测试。

先使用fullgraph=True与记录FX图的eager后端，只验证Dynamo捕获；再使用Inductor检查代码生成及反向执行。joint测试直接复用模型forward方法，KV cache关闭、DeepStack关闭；dense Action作为MoE前的控制组。

| 测试 | 不重算 | 开启checkpoint | 结论 |
| --- | --- | --- | --- |
| VLM QKV半层 | 捕获+反向通过 | 捕获+反向通过 | checkpoint本身可编译 |
| VLM O投影+MLP半层 | 捕获+反向通过 | 捕获+反向通过 | 同一模块两阶段分支本身可编译 |
| Action QKV与dense后半层 | 均通过 | 均通过 | 两流分别计算本身可编译 |
| joint，原mask构建 | fullgraph失败 | 未单独重复 | 显式disable函数是边界 |
| joint，mask在图外预构建 | 捕获和Inductor前后向均通过 | 捕获和Inductor前后向均通过 | joint计算不是必然失败的原因 |

joint+checkpoint的FX图实际包含4个tag_activation_checkpoint节点，对应两条流、每条流的两次层调用；不是测试时误关了重算。参见[capture/joint_precomputed_ac.fx.py](evidence/capture/joint_precomputed_ac.fx.py)。

## FSDP的单因素对照

使用同一真实VLM层、相同随机初始化、两卡FSDP2及bf16 mixed precision，参数在forward后保留到backward，符合原训练的reshard_after_forward=False。QKV半层与O+MLP半层分别测试；捕获后端和Inductor均运行，两个rank结果一致。

| 配置 | forward/backward | QKV捕获图数 | O+MLP捕获图数 |
| --- | --- | --- | --- |
| FSDP，关闭checkpoint | 正常 | 自动1 | 自动1 |
| FSDP，开启HF checkpoint | 正常，但decoder回退eager | 自动0 | 自动0 |
| FSDP+HF checkpoint，显式compile(layer.forward) | 正常，恢复区域编译 | 显式1 | 显式2 |

O+MLP显式编译产生2张图，日志显示att_output在首次forward和重算时出现bf16/fp32输入dtype差异，触发了特化重编译。它不是捕获失败。这个现象提示未来实现还需要控制图特化数量。

这里的“正常”只表示执行正常。第二行虽然不报训练错误，后端却收不到decoder图，正是需要诊断的静默回退。两个后端、两个rank、两种半层的完整结果见[fsdp_probe](evidence/fsdp_probe/)及[fsdp_inductor](evidence/fsdp_inductor/)。

各后端内部，开启checkpoint及显式forward编译的测试输出与输入梯度，相对该后端的无checkpoint编译控制组最大绝对差均为0。**这不表示Inductor相对原始eager逐位一致**：后端之间没有做完整数值等价验收，也没有做全模型参数梯度/收敛/性能验证。

## 为什么FSDP hooks落进了checkpoint

实际HF基类在__call__中执行checkpoint(partial(super().__call__, ...))。super().__call__还会调用模块注册的forward hooks。FSDP2在这些hooks中做参数收集与状态管理；PyTorch 2.8默认skip_fsdp_hooks=True，其包装器明确使用torch._dynamo.disable(recursive=True)。

不加checkpoint时，Dynamo可以跨过被跳过的hook后继续捕获forward计算；被HF checkpoint包起来后，该hook变成checkpoint子图的一部分，阻断了这部分的整体捕获。显式编译纯forward建立了独立入口，因此在外层回退后仍能编译张量计算。

这条因果链不需要joint attention：只调用一次QKV半层也能复现。因此joint的两阶段结构是复杂性背景，也会影响FSDP通信，但不是该捕获回退的必要条件。

## Action fused MoE的独立障碍与成功对照

| 阶段 | 结果 | 具体位置 |
| --- | --- | --- |
| 原始fused MoE，正确初始化并行状态 | 严格捕获失败 | moe_scatter中的assert lambda，NestedUserFunctionVariable无法转proxy |
| 仅在进程内移除无效lambda断言 | 暴露下一个失败 | Pretuned.run调试日志调用qualified_name，hasattr(TritonKernelVariable, fn)不支持 |
| 再设置BPEX_NO_WARN_ON_UNTUNED_CASE=1 | MoE与joint+MoE、checkpoint开/关均捕获并反向通过 | 无数学计算修改 |
| 上述诊断条件下，joint+fused MoE使用Inductor | checkpoint开/关前后向均通过 | 证明存在可编译的真实MoE联合计算路径 |

`assert lambda: condition`只断言lambda对象为真，不执行condition。诊断脚本只在自己的进程中删除这个无效语句；没有编辑仓库源码。关闭预调优日志也不改变kernel选择和数学计算。真实生产修复仍需独立实现、审查和验证。

最初未初始化parallel_state的MoE试探出现logger错误，这是测试环境缺失，不能算模型根因；完整保留了初始记录，实际MoE结论使用capture_moe_initialized及之后的对照。

## 复现与证据

原机器目录：`/root/experiments/lingbot-dynamo-20261109`。环境、原始脚本、FX图、完整回退栈和逐rank结果均保存在[evidence](evidence/)。完整性审计见[verification.json](evidence/verification.json)。

真实训练使用main be969b8；最小复现导入opt 71a41bc中的类，已逐AST确认Qwen3VLTextDecoderLayer、Qwen2DecoderLayer与Qwen2TokenMoeBlock和baseline相同。RMSNorm独立编译、SwiGLU/QKV融合均未在这些最小复现中启用。

单卡脚本probe.py支持--backend capture/inductor、--case过滤、--init-parallel及--strip-noop-assert。MoE需正确初始化并行状态；以torchrun --standalone --nproc-per-node=1启动并传--init-parallel。FSDP脚本fsdp_probe.py以两进程torchrun启动，--backend capture/inductor，输出分别在fsdp_probe/fsdp_inductor。

脚本包含当时的源码/模型/输出绝对路径，迁移时需要修改。先准备上一轮源码环境并确认GPU空闲；完整命令与环境位于[README](README.md)。

## 下一步建议

优先尝试两条流的区域编译：让FSDP hook留在Python边界之外，单独捕获纯forward计算；mask缓存构建留在图外；MoE的断言与日志包装单独适配。不要直接删除PyTorch checkpoint/FSDP内部的disable，也不应为了完整图强行追踪通信状态管理。

本轮已经确认捕获原因及可行的成功对照，未把诊断修改部署到训练分支。下一阶段才应使用真实batch与全部层数，测量区域编译的速度、显存、重编译次数及数值影响。
