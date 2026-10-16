# RTX PRO 5000 单机 8 卡 DistMuon 测速更新

## 1. 测试配置与结果

| 项目 | 配置 |
|---|---|
| 硬件 | 8 × RTX PRO 5000 72GB，PCIe Gen5 x16，无 NVLink；2 × AMD EPYC 9575F，GPU 0–3 / 4–7 分属两个 NUMA；4 × ConnectX-8（8 × 400 Gb/s） |
| 软件 | 驱动 580.95.05；PyTorch 2.8.0+cu128；NCCL 2.27.3（默认设置，未设置任何 `NCCL_*` 变量） |
| 代码与配置 | 开源 lingbot-vla-v2 官方代码，`robotwin_dist_muon.yaml`，未改源码 |
| 数据 | RoboTwin 2.0 clean：50 任务、2,500 episodes、549,787 帧 |
| batch | 每卡 32 × 8 卡 × 梯度累积 1 = 256；开启激活重算 |
| 统计方式 | 训练 130 步，取 step 31–130 的平均 step 耗时 |

| 结果 | 数值 |
|---|---|
| 预热后 step 耗时 | **4.767 秒**（P95 4.794 秒）；另一台同型号机器独立复现 **4.778 秒** |
| 显存峰值 | 张量 53.9 GiB / 缓存池 69.2 GiB / 整卡 70.0 GiB |
| 5k steps 粗估 | 6.62 小时（平均 step × 5000，不含启动编译和 checkpoint 保存） |

## 2. 本地两台机器测速不一致的原因

本地两台机器硬件完全相同，但最初一台测得 5.567 秒/步，另一台 4.767 秒/步。对比 timeline 后，两台的计算时间基本相同，差距全部在 GPU 间通信上。

原因是**慢的那台在容器里没有启用 RDMA**：

- 同一 NUMA 内的 GPU 直接经 PCIe 通信；跨 NUMA 的 GPU 应该通过网卡用 RDMA 直接传输，不经过 CPU。
- 容器拿不到 RDMA 设备时，NCCL 不会报错，而是自动改为经 CPU 内存中转。训练照常进行，只是跨 NUMA 那一段变慢，整体通信都被拖慢。
- 给容器启用 RDMA 后，这台机器从 5.567 秒降到 4.778 秒，与另一台一致。

容器启用 RDMA 加了三个参数：

```bash
docker run ... --network=host --device=/dev/infiniband --ulimit memlock=-1 ...
```

- `--device=/dev/infiniband`：把 RDMA 设备映射进容器，不加时容器里看不到 RDMA 网卡。
- `--ulimit memlock=-1`：RDMA 需要锁定内存，docker 默认上限太小，会导致 NCCL 放弃 RDMA。
- `--network=host`：使用宿主机网络。我们的网卡是 RoCE（以太网上的 RDMA），依赖网口的 IP 地址建立连接。

## 3. 实验汇总

**NCCL 通信带宽**（启用 RDMA，1–2 GB 消息，busbw，GB/s）：

| NCCL 设置 | all-gather | reduce-scatter |
|---|---:|---:|
| 默认 | 25.8 | 25.6 |
| `NCCL_MIN_CTAS=8` | 26.0 | 26.2 |
| `NCCL_NTHREADS=128` | 23.0 | 17.7 |
| `NCCL_MIN_CTAS=8 NCCL_NTHREADS=128` | 24.9 | 23.5 |

启用 RDMA 时，默认设置已经用满链路，调整 NCCL 参数没有收益。

**训练速度**（DistMuon，step 31–130）：

| 跨 NUMA 通信 | NCCL 设置 | step 耗时 |
|---|---|---:|
| RDMA | 默认 | **4.767 / 4.778 秒** |
| 经 CPU 内存中转 | 默认 | 5.567 秒 |
| 经 CPU 内存中转 | `NCCL_MIN_CTAS=8` | 约 5.06 秒 |
| 经 CPU 内存中转 | `NCCL_MIN_CTAS=8 NCCL_NTHREADS=128` | 约 4.66 秒 |

经 CPU 内存中转的后两行为单次测量，仅供参考。可以看到：

- 跨 NUMA 通信是否走 RDMA，对 step 耗时影响约 15%。
- 没有 RDMA 时，NCCL 默认的通道数用不满经 CPU 中转的链路，多开通道（`NCCL_MIN_CTAS=8`）可以补回大部分差距。

## 4. 可以尝试的提速方法

以下方法都不改训练代码。**实际效果取决于机器的硬件配置**（网卡型号和数量、GPU 与网卡的 PCIe 拓扑、是否支持 RDMA 等），不一定能达到上面的速度，建议在目标机器上实测。

1. **确认当前的通信路径**：训练或通信测试时加上 `NCCL_DEBUG=INFO`。
   - 日志中有 `NET/IB : Using ...`，且跨 NUMA 的连接显示 `via NET/IB/.../GDRDMA`：已在使用 RDMA。
   - 跨 NUMA 的连接显示 `via SHM`：经 CPU 内存中转，没有用上 RDMA。
   - 同时用 `env | grep NCCL` 检查是否设置过 `NCCL_*` 变量，例如 `NCCL_NTHREADS` 调小会明显降低 reduce-scatter 带宽。
2. **有 RDMA 网卡但没用上**：给容器加上第 2 节的三个参数，并确认宿主机已加载 `nvidia_peermem` 模块（`lsmod | grep nvidia_peermem`）。
3. **没有 RDMA**：尝试 `NCCL_MIN_CTAS=8`，或再加 `NCCL_NTHREADS=128`，跑 130 步与默认设置对比 step 耗时。通道越多，NCCL 占用的 GPU 计算资源也越多，所以以训练的 step 耗时为准。已经启用 RDMA 的机器不建议调整这些参数。

更多训练侧的优化我们还在研究，有结论后再同步。
