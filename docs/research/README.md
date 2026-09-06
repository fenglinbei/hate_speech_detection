# 研究文档

## 实验计划

- [`experiment-plans/general-model-ld-coverage-numerical-measurement-v1.md`](experiment-plans/general-model-ld-coverage-numerical-measurement-v1.md)：**合并词典 8B 完整 dev 已完成并通过独立审计**。Lnew=Lq∪Ld，全局去重、固定 ID 顺序，六核心条件及两 Lq 辅助参考；643 条、10,288 块及全部注册 CI 已验收。[结果报告](../../exps/causal_context/general_model_ld_coverage_v1/results/coverage-01/REPORT.md)与[运行证据](../../exps/causal_context/general_model_ld_coverage_v1/README.md)。
- [`experiment-plans/general-model-ld-coverage-replication-v1.md`](experiment-plans/general-model-ld-coverage-replication-v1.md)：**14B 已按用户要求于 22:36:06 暂停，等待加速方案决定**。所有所属进程退出、GPU 已释放；9/18 项预检已有封存证据，验证集候选顺序检查保留 289/384 块，完整 dev 未启动。不自动续跑或性能试跑，27B 未启动。[暂停记录及队列状态说明](../../exps/causal_context/general_model_ld_replication_queue_v1/USER_STOP_20260906_2236.md)、[14B 断点证据](../../exps/causal_context/general_model_ld_coverage_14b_v1/README.md)及[原午夜窗口授权](../../exps/causal_context/general_model_ld_replication_queue_v1/WINDOW_20260906_2400.md)。
- [`experiment-plans/general-model-ld-numerical-measurement-v3.md`](experiment-plans/general-model-ld-numerical-measurement-v3.md)：**当前后备数值运行注册，完整 dev 已完成**。既定真实 batch 1 备选沿用 FP32 与原 epsilon，适用预检及完整独立统计审计均通过。[正式科学结果](../../exps/causal_context/general_model_ld_numeric_v3/RESULTS.md)与[运行证据](../../exps/causal_context/general_model_ld_numeric_v3/README.md)。
- [`experiment-plans/general-model-ld-numerical-measurement-v2.md`](experiment-plans/general-model-ld-numerical-measurement-v2.md)：FP32 与目标位置投影修订。numeric-03 在 24 条 batch 4 验收失败，原始回执及[失败记录](../../exps/causal_context/general_model_ld_numeric_v2/README.md)保留，不用于放行 dev。
- [`experiment-plans/general-model-ld-numerical-measurement-v1.md`](experiment-plans/general-model-ld-numerical-measurement-v1.md)：数值测量的父注册，固定候选、配对 margin、统计口径及数据使用边界。[v1 失败记录](../../exps/causal_context/general_model_ld_numeric_v1/README.md)完整保留，不用于放行 dev。
- [`experiment-plans/general-model-ld-mechanism-protocol-v1.md`](experiment-plans/general-model-ld-mechanism-protocol-v1.md)：**当前正式研究协议**。固定词典下的通用模型机制研究，分类为主、宽松提取为副；明确 Qwen3-8B/14B、Qwen3.8-27B 分工、数据历史暴露、冻结后验证及 Q1–Q16 已确认决策。
- [`experiment-plans/general-model-ld-run-package-v1.md`](experiment-plans/general-model-ld-run-package-v1.md)：dev 运行包的构建、回放、8B 预检和运行入口；区分可执行开发条件与尚未开放的正式 test/内部干预阶段。
- [`experiment-plans/causal-context-six-stage-plan.md`](experiment-plans/causal-context-six-stage-plan.md)：原六阶段总体蓝图；当前研究对象、任务主次与阶段边界以新版机制协议为准。
- [`experiment-plans/stage1-p0-implementation.md`](experiment-plans/stage1-p0-implementation.md)：旧 Stage 1 P0 的四字段协议、train-only 数据边界及工程生命周期参考；其微调、category-free 资源与严格四字段门槛不直接适用于新版机制协议。
- [`experiment-plans/wp3-terminology-understanding-library.md`](experiment-plans/wp3-terminology-understanding-library.md)：WP3 无类别术语理解库的 A0–A5 试运行门控、解释/联网/人工补集、冻结发布命令与验收标准。

## 案例分析

- [`case-studies/lexicon-injection-boundary-cases.md`](case-studies/lexicon-injection-boundary-cases.md)：词典注入帮助划定 target–argument 混淆跨度边界的定性案例。
- [`case-studies/wp3-implementation-revisions-and-challenges.md`](case-studies/wp3-implementation-revisions-and-challenges.md)：WP3 从既有词典、旧 fit-only 群体损贬协议到 category-free 术语理解库的修订时间线、失败证据、当前状态与待讨论决策。
