# 案例用途四步实施 v1

**后续执行已完成：** 用户再次授权空闲 GPU 后，已完成本页登记的全部 26 个条件及数值检查。[最新评分结果和用途表](../input-interventions-v1/README.md)。本页、当前准备指针和私有实施产物保留准备时点的状态与字节。

2026-09-12。用户明确同意“好的，可以按这四步开始实施”。已在既有用途草案上完成事实核对、用途建议落实、材料来源关联和具体操作记录，覆盖 **32 案例、64 任务、1,072 项材料**。优先 6 个案例形成 **12 份分任务协议、26 个完整输入条件**。当前指针见 [current.json](current.json)。

授权针对工作流程实施，不追认 64 条用途建议为逐项人工决定。原参考、案例用途选择、人审解释和旧版本保持原字节；本次操作方案与假设保留 assistant 作者身份，结果状态为尚未测试。

## 四步交付

| 步骤 | 实施结果 | 入口 |
| --- | --- | --- |
| 分任务更新事实 | 64 行保留原／审核标签、旧／新位型、六条件读数和政策身份 | [用途表 CSV](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/case_uses.csv)、[完整 JSONL](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/case_uses.jsonl) |
| 落实用途 | 多用途、准备顺序和具体协议回链；人工决定与分析建议分开 | [实施概览](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/IMPLEMENTATION.md) |
| 关联实际资源 | 32 个来源图，验证查询／示例／命中／entry 的一致性；逐候选计算 L_R、L_U 及共享来源 | [来源图](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/resource_graphs.jsonl)、[64 个任务的来源选项](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/source_options.jsonl) |
| 明确操作及待办 | 为优先 12 个任务指定资源 ID、操作前后版本、保持项、假设、替代解释、固定 foil 和后续检查 | [协议 CSV](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/interventions.csv)、[完整协议](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/interventions.jsonl)、[完整输入与 token 位置](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/contexts.jsonl) |

其余 52 个任务保留完整事实、用途、资源选项和按阶段划分的待办；本次没有将全队列扩为 GPU 运行名单。28 个任务仍无严格 U 候选，已有候选也未被自动整体选作 U。

## 优先案例的具体操作

所有协议采用 NoCat 环境，保持查询和任务指令。未指定的示例正文、答案、词典定义均沿用原输入。删除示例保留其他示例的相对顺序，并记录自然重编号的影响。

| 案例／任务 | 已准备操作 | 保持项及解释范围 |
| --- | --- | --- |
| 3169 / hate、group | 原 SD 与移除示例 3660 两条件 | 全部 6 个词条保持，包含 lex-0419；测量该示例包在原词典背景下的增量作用 |
| 5086 / hate | 示例 7248 有／无 × 其独有 lex-0073 有／无，共四条件 | 两因素独立；其他 9 个示例和 12 个词条保持。7248 话题直接、规则部分相关，登记为单独的相关示例来源分解 |
| 5086 / group | 仅将示例 127 的 `[LGBTQ, others]` 改为已确认的 `[LGBTQ]` | 正文、顺序及全部词典保持；解释为答案参考切换的总效应 |
| 541 / hate | 原 SD 与移除规则直接相关示例 826 两条件 | 固定两个词条和其余答案；显式类别字段对照另行准备 |
| 541 / group | 仅将示例 826 的 `[]` 改为已确认的 `[others]` | 保持正文及词典；不预设修订必然改善当前查询预测 |
| 6037 / hate、group | 屏蔽 `5230 → lex-0069` 的 `[32,34)` 命中边 | 已核验没有其他来源，因此词典由 3 条变为 2 条；10 条示例及原答案保持 |
| 1128 / hate、group | 固定词典，移除示例 387 | 稳定正确案例中的同类删除操作对照，未声称语义或位置匹配 |
| 4026 / hate、group | 固定词典，移除示例 6334 | 保留参考切换历史；作为删除一条 hate 示例的部分操作对照 |

**3169 的定义分支仍需另行准备。** lex-0419 的最终定义判定为 reasonable，且 `adopted_definition=null`。本次没有编造已采纳改写。查询中的原匹配虽被判为错义，在当前“固定词典、改变示例”对照中仍保持，以便单独测量示例操作的作用；定义或来源边改动需要独立条件。

**5086/hate 未放宽严格 U 规则。** 示例 7248 的原／审核 hate 均为 non-hate，但话题 direct、规则 partial；它不属于严格 U 分支。本次四条件明确针对有相关性的示例及其独有词典的来源贡献，不能汇入“可靠但不适用示例”的主结论。

**6037 的答案差异没有被同时修订。** 示例 5230 的 group 原答案与审核参考不同，该背景在边操作中固定并单独标记，避免同时改变词典边和示例答案。

六个 group 协议各自固定一个 foil，来自该任务封存 SD 分数中的最佳非审核候选；精确并列取最小 canonical ordinal。foil 已在新模型评分前登记，各条件不重新选择；完整 32 候选仍决定最终预测。hate 同时预设无 Gold 方向分数与原／审核 margin。

## 输入和验证

26 个输入已按现有 renderer 生成。12 个原条件的完整消息、提示文本、token 数和 token ID 哈希均与封存条件一致；所有 412 项提示／候选答案拼接边界通过，完整序列均符合历史容量限制。tokenizer 文件树身份与历史计划一致：`ff36aee198780ed161fe50302f0fb3ca54bd69076bba25cd150e3da17ad65809`。

本次使用完整提示的 token offset 记录查询位置和回答起点。3169 删除示例后，hate 从 742 缩短为 712 token，group 从 820 缩短为 785 token。其他操作的位置变化也逐条件保留；当前均为自然编辑，尚无固定槽位或等长语义控制。1128／4026 仅匹配操作类型、删除条数和被删除示例的 hate 答案，不能作为已经完成位置和语义匹配的对照。

7 项隔离测试覆盖共享来源保护、空词典因素、未知／重复资源、局部边删除、固定输入、单条答案替换和模板边界。另独立核对了全部 12 个协议的操作保持项。所有输出逐字节复算通过。最终输入准备运行禁用了模型计算框架，仅加载本地 tokenizer；没有模型权重或模型 forward。

[生成工具](../../../../scripts/review/implement_evidence_case_uses.py)与[测试](../../../../scripts/review/test_implement_evidence_case_uses.py)：

```bash
python -m unittest scripts.review.test_implement_evidence_case_uses -v
.conda/stage1-p0/bin/python scripts/review/implement_evidence_case_uses.py --tokenizer models/base/Qwen3-8B --output exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/case-uses-implementation-v1 --check
```

## 后续执行状态

本次四步的用途与操作记录已经落实。下一阶段先完成当前模型环境下的数值重放，再在同一次新运行中评分已固定的全部条件；历史分数只用于回归核验。当前没有记录 GPU 评分结果，也没有把文本/token 重建通过等同于模型数值复现。

结果须保留无效和反向效应；不能看到新分数后更换示例、foil 或假设。位置控制、激活采集无副作用检查和双向 patching 属于后续机制阶段，按独立待办保留。来源及输出身份见 [manifest](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/manifest.json)、[审计](../reviews/analysis-freeze-20260912/case-uses-implementation-v1/audit.json)。
