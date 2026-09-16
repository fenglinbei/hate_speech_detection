# Q01 局部机制验证交付 v1

2026-09-15：CPU 准备已完成，正式输入和实现已封存。当前未加载 8B 权重，也未执行本实验的 GPU forward。24 项 CPU 测试、逐字节重建、独立来源／输入审计及四组历史 C/I 的高精度复算通过。GPU 验收是下一阶段。

| 交付 | 文件 |
|---|---|
| 正式协议及判读规则 | [PROTOCOL.md](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/PROTOCOL.md) |
| 96 个提示的逐位置表 | [positions.csv](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/positions.csv)、[positions.jsonl](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/positions.jsonl) |
| 双向干预与控制矩阵 | [完整矩阵](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/intervention-matrix.csv)、[紧凑矩阵](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/compact-matrix.csv) |
| 评分规范 | [scoring-spec.json](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/scoring-spec.json)；公式和解释见协议第 3、4、7、8 节 |
| hook 实现 | [q01_mechanism_hooks.py](/data/liaozijie/hate_speech_detection/src/diagnostics/q01_mechanism_hooks.py)、[评分与候选提名](/data/liaozijie/hate_speech_detection/src/diagnostics/q01_mechanism_scoring.py) |
| 验收与执行调度 | [12 遍调度](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/acceptance-schedule.json)、[执行器](/data/liaozijie/hate_speech_detection/src/diagnostics/q01_mechanism_execution.py) |
| CPU 验收 | [cpu-validation.json](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/q01-local-mechanism-v1/cpu-validation.json)、[测试日志](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/audits/cpu-tests-final.log) |
| 来源与完整实现快照 | [manifest.json](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/manifest.json)、[implementation-snapshot.json](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/frozen-01/implementation-snapshot.json) |

首轮保留 32 个主单元，四组、双向、两条示例臂及所有原标签／NCC／A/B 口径。主网格单遍 6,912 次候选评分，完整控制和 12 遍验收共 144,192 次候选评分、214,768 次评分 forward，donor 采集另计。完整调度不包括后续模块细化。

NCC 固定 recipient 背景的效应和目标都等于原均分，属于代数恒等。只有角色在背景中存在时才做完整重校准；query_hehe 没有背景对应位置。C/I 分别提名，最多两个单元；反向、未决、映射／探针异常及控制结果全部保留。机制 readiness 不自动改变。

## 本次时间安排

依据用户最新授权，2026-09-15 **08:30（北京时间）**开始检查四张 GPU，忙碌时每 30 分钟再查，满足空闲门槛便直接启动。16:45 停止接收新请求并保存检查点，16:55 为本任务进程的兜底退出期限，17:00 前释放 GPU。若窗口内未完成，保留暂停记录与分片，不跳过验收。

这里使用服务器上的独立定时执行进程；当前会话没有可调用的应用定时接口。其真实状态以 [窗口状态](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/window-20260915-01/state.json) 为准，配置和授权原文见 [窗口配置](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/window-20260915-01/config.json)。它保存运行、失败或完成回执；这不是应用内自动唤醒聊天的承诺。

GPU 执行路径为 `reviews/q01-local-mechanism-v1/run-01`，结果路径为同目录的 `results-01`。只有全部验收通过才生成 RESULTS.md、完整／紧凑比较表与 nomination.json；源分析参考在此前只做 hash 校验。全部旧材料、人工字段、结果与指针保持原状态。

如需取消已排定的服务器任务，在窗口目录创建名为 `CANCEL` 的空文件即可；进程在等待时每分钟检查，运行时每 5 秒检查，并只停止自己启动的运行。用户也可直接在任务中要求取消，由助手执行这一操作。

## CPU 检查与后续接续

以下命令从项目根目录执行，不会加载模型：

```bash
.conda/stage1-p0/bin/python scripts/review/prepare_q01_local_mechanism.py check
.conda/stage1-p0/bin/python scripts/review/run_q01_local_mechanism.py validate
```

定时进程已负责本窗口的启动，不能同时手动再开同一运行。检查点接续必须复用同一四卡列表、UUID、runtime、plan 与代码；只有 `paused` 状态允许续接，`failed` 和 `complete` 都拒绝新 forward。所有输入／代码更新另建版本，不能修改本次 pinned 文件。

冻结 plan ID：`q01-mechanism-9874868292f90400946a675d673662fe7500486b5e60b79479b9ae9de7576f30`。

冻结 manifest SHA256：`e9391ef24ff6cf136840040d21430890098dcf266a98c66e899cc111bb9fc3c0`。
