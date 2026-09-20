# 后续执行入口：本轮禁止启动

当前用户仅授权CPU准备。没有待补的标签或释义审核；实际GPU启动须等用户下一次明确指令。不得沿用旧轮次的GPU授权，不创建定时任务或后台等待。

收到新的明确启动指令后，在本轮新的launch-01之外创建不可变执行决定文件，记录用户原话、当前prepared-01/manifest.json的完整path/bytes/sha256、GPU_execution_authorized=true、GPU_time_constraint.confirmed=true、明确的deadline_unix或无截止null，以及允许卡号。绑定入口核对这些字段才会读取权重和查询GPU。文件只记录真实新指令，不能把本CPU准备消息视为启动许可。

使用现有Python环境，可先执行不会查询GPU的封存检查：

```sh
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/run_cross_term_mechanism_v1.py validate
```

以下入口只供后续授权后使用；当前不执行：

```sh
.conda/stage1-p0/bin/python scripts/review/launch_cross_term_mechanism_v1.py --decision FUTURE_AUTHORIZATION.json
```

启动器先验证授权和全部封存来源，禁止覆盖已有launch/run，然后寻找一张实时空闲、至少44,000MiB的卡并重新绑定。顺序固定为工程运行、CPU核对、正式运行、CPU核对、正常资源释放。被占用时45秒后检查；新授权有截止时每轮核对剩余窗口。CANCEL仅用于绑定前或两阶段之间；运行中用run-01/STOP，工作进程在提交单元边界安全暂停。失败和完成为终态，不自动重试，不终止其他人的进程。不要为实现普通业务调整更改已封存源码或门槛。

完成后按顺序执行CPU分析与独立复核：

```sh
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/run_cross_term_mechanism_v1.py analyze --run reviews/cross-term-mechanism-v1/run-01 --output reviews/cross-term-mechanism-v1/results-01
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/audit_cross_term_mechanism_results_v1.py --prepared reviews/cross-term-mechanism-v1/prepared-01 --run reviews/cross-term-mechanism-v1/run-01 --results reviews/cross-term-mechanism-v1/results-01 --output reviews/cross-term-mechanism-v1/result-audit-01.json
```

保存准确的本轮控制器与工作进程释放核对至process-release-check.json；检查图表和报告完整后再运行closeout_cross_term_mechanism_v1.py。它要求156格式端点、192控制、120跨条件效应和960分支证明均通过，并核对旧科学/网站选择器未变。准备包、原始结果、分析和来源分别封存。

此次CPU授权不包含网站发布。任何新科学结果均尚不存在。checks/synthetic-02仅为CPU合成向量生成的测试报告，不能当作实验结果或部署。
