# A路线三模型：首批CPU审核包

状态：CPU设计与两家族小样已交付，待人工审核；非科学输入冻结、非执行冻结。请从[REVIEW.md](REVIEW.md)开始。

| 阅读入口 | 内容 |
|---|---|
| [REVIEW.md](REVIEW.md) | 八查询速览、S1–S6设计审核点、反馈范围 |
| [MATERIALS.md](MATERIALS.md) | 全部查询、四义项、十原示例、两N对照 |
| [RELATIONS.md](RELATIONS.md) | 120条逐项AI关系及指定范围；并有[速查TSV](relation-overview.tsv) |
| [MODELS.md](MODELS.md) | 8B、14B、GLM候选、权重身份、tokenizer事实与适配限制 |
| [RESEARCH-SPEC.md](RESEARCH-SPEC.md) | 四种竞争解释与本小样能回答的问题 |
| [CONDITIONS.md](CONDITIONS.md) | 15核心＋9辅助条件及伴随变化 |
| [PROMPTS.md](PROMPTS.md) | 192条件的完整system／user消息 |
| [ANALYSIS-AND-EXPOSURE.md](ANALYSIS-AND-EXPOSURE.md) | 依赖簇、开发／确认边界和统计事项 |
| [LEGACY-REPLICATION.md](LEGACY-REPLICATION.md) | 156个旧输入与未来三模型复验 |
| [EXECUTION-NEXT.md](EXECUTION-NEXT.md) | 审核后的扩充、适配和GPU准备步骤 |

机器文件包括材料、条件、576个表达式、全部关系、空白人工反馈、旧／新模型可见输入以及六份模型特定token化清单。24项原材料与120条关系的人审字段全部为空；AI建议的accepted/direct/none不能解释成人工采纳。

CPU检查结果：

- [结构检查](cpu-build-check.json)：192个新输入（120核心、72辅助），156个旧输入，120条关系，零新评分。
- [三模型tokenizer检查](cpu-tokenizer-audit.json)：1044次prompt重建、2088个候选边界；CUDA未初始化，未加载模型张量或执行前向。
- [独立实现检查](independent-cpu-check.json)：材料与条件映射、2560条关系呈现绑定、576个线性表达式、160项精确加性空交互核验及参考隔离；不是独立人工语义审核。
- [本地模型身份](cpu-model-inventory.json)与[上游核对](upstream-verification.json)：全部23个权重分片匹配官方公开哈希，8B同时匹配原冻结。GLM四个本地配套文件与当前官方commit不同，保留本地哈希及具体差异名单。

本包只有两个词项家族，保守登记为一个共享构造簇；不是独立确认集。没有访问原reserve/calibration/test，没有更改旧冻结、结果指针或人工字段，也没有恢复或停止任何GPU进程。

交付后manifest绑定本目录和四个新准备／审计脚本。写入型准备器会拒绝覆盖已有manifest的稿件。只读复核命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .conda/stage1-p0/bin/python scripts/review/check_cross_model_applicability_draft_v1.py
```

复核验证字节身份及已有CPU证据，不重新加载tokenizer、模型或执行任何GPU操作。后续修改应单独版本化。
