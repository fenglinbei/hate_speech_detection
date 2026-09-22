# 研究文档

## 最新成果

- [2026-09-21 下午成果增量](results-20260921-afternoon/README.md)：新增六轮整段／词外／前后区域替换、无词典供体、正确示例混合及帮助保留实验；包括固定偏移对照、阶段报告、全部输入、实现与独立审计。
- [2026-09-21 成果归档](results-20260921/README.md)：新增 12 轮注意力、内容替换、“嘿嘿”机制与迁移、三词条扩展和“京巴”语境实验；汇总阶段结论、完整报告、冻结材料、审计及网站覆盖范围。
- [2026-09-18 成果归档](results-20260918/README.md)：跨词两轮行为实验、Qwen3-8B／14B 与 GLM-4-9B-Chat 三模型结果、完整表格、冻结输入及独立审计的阅读入口；附字节清单和只读校验命令。
- [正确参考信息的选择性利用：内部干预研究路线](experiment-plans/internal-reference-utilization-roadmap-20260918.md)：后续研究目标与路线讨论，尚未启动新的材料或内部干预实验。
- [2026-09-16 成果归档](results-20260916/README.md)：此前标注、输入对照及 Q01 内部干预结果。

## 实验计划

- [正确参考利用：GPU运行前准备已封存](experiment-plans/reference-routing-content-execution-v1/README.md)：2026-09-22用户整批采纳材料；固定17/18层 A/V、强基线、开发校准／确认隔离及双侧评价已实现，29项CPU测试通过。尚未启动GPU，实际数值资格及开发参数待后续运行。
- [正确参考利用：原材料审核稿](experiment-plans/reference-routing-content-materials-v1/README.md)：48条查询（24开发＋24确认）、108条示例及576条关系AI建议，附离线三栏工作台。原草案保留原始待审字段；新的整批采纳证据及执行输入另建版本，见上方入口。
- [正确参考的选择性利用：第18层 A/V 分解与双侧评价 v1](experiment-plans/reference-routing-content-v1/frozen-01/README.md)：**研究设计已冻结**。绑定已有六查询／两混合次序的 A/V 四格，规定24开发＋24确认材料、逐项关系审核、强基线及校准、G与修复／损害、双侧参考贡献和结果分流。新材料采纳、运行实现与 GPU 资格另行绑定；本次仅做 CPU 协议校验。
- [`experiment-plans/general-model-evidence-applicability-audit-controls-v1.md`](experiment-plans/general-model-evidence-applicability-audit-controls-v1.md)：**当前接续工作：证据适用性联合审核**。沿用冻结的 32 条 discovery；[规则 v2](annotation-guidelines/evidence-applicability-annotation-policy-v2.md)已于 2026-09-09 确认并上线，明确 group 对象和 others 范围，hate 规则沿用 v1。旧 AI 初稿与人审决定保留，受影响 group 等待人类重核。[执行记录](../../exps/causal_context/general_model_evidence_applicability_v1/README.md)登记工作台及验收进度；[交互设计](annotation-guidelines/evidence-applicability-assisted-review-design-v1.md)保留批准时的提案身份。
- [已采纳的 group 范围修订草案原字节](annotation-guidelines/evidence-applicability-group-scope-v2-draft1.md)：保留批准时文档；当前完整语义以已冻结 v2 为准。
- [`experiment-plans/general-model-ld-nolabel-paired-cases-v1.md`](experiment-plans/general-model-ld-nolabel-paired-cases-v1.md)：**逐查询配对分析已完成**。643 条 dev、32 条 discovery 与 16 条 reserve 选样已封存；[阶段结果](../../exps/causal_context/general_model_ld_nolabel_paired_cases_v1/RESULTS.md)。
- [`experiment-plans/general-model-ld-nolabel-v1.md`](experiment-plans/general-model-ld-nolabel-v1.md)：**Qwen3-8B 显式词典类别字段删除实验已完成**。六条件 7,716 个评分块及原四项配对差值通过封存核验；[完整 dev 结果](../../exps/causal_context/general_model_ld_nolabel_v1/RESULTS.md)。
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
