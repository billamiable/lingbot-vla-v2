LingBot VLA 激活重算 / micro batch 对照实验，2026-11-08

主结果：aggregate.json；逐运行全部统计：summary_all.json。
计时器原始输入：runs/<name>/rank0..7.jsonl。
正式主比较只使用 alloc_ 前缀、各200步、每配置两次的8次运行。
默认分配器失败运行与复现运行单独保留，不参与主比较。

统计口径：
1. bench-repro：第11–30次完整更新的均值。
2. Qihan报告：第31–80次完整更新的中位数。
3. 本报告主指标：第31–200次完整更新的均值，保留所有慢步。
每一步耗时取8个rank最大值。global batch均为256。

复核已存日志（不占GPU）：
python3 summarize_study.py
python3 audit_study.py
python3 aggregate_study.py
python3 verify_final.py
/root/qihank/envs/lingbotvla/bin/python render_report.py
这些脚本的ROOT指向原实验绝对路径；在另一位置使用时先修改ROOT。

源码：main be969b8；opt 71a41bc。
源码目录及模型/数据不包含在归档内。optimized-vs-main.patch保存opt相对main的完整差异。
manifest.json记录完整commit，method_audit.json记录public bench-repro脚本SHA256。
harness为原样benchmark；study_harness添加输入指纹、数据等待和显存观测。
driver.py保存实际训练命令、环境和镜像运行顺序；新实验需改为全新ROOT并准备相应源码目录。
对已有实验根目录再次运行study_alloc会因目录已存在而拒绝覆盖。

verification.json检查8次完整运行、统一分配器、batch/GA/重算参数、输入指纹和源码状态。
输入指纹仅覆盖前3次更新；模型清单验证size/mtime，未对全部大权重重新做hash。
短程loss只用于健康检查，不能证明完整收敛或下游成功率等价。
