# 按 v2 分批补标：AI 草稿与疑点讨论

用户于 2026-09-11（Asia/Shanghai）要求重新标注当前未完成项，并进一步允许参考已有人工裁决，要求每批发现疑点后暂停讨论。

当前状态：**#4392 已明确改为 hate；已接受并启用“0→non-hate、1–4→hate、null→null”的本地暂行默认映射。第四批疑点已处理，第五批 30 条已逐条校准，7 条疑点待讨论；暂停于第五批，尚未进入第六批。**

- 最新只读快照中，118 项材料已完成人工确认；954 项未审核（28 查询、242 示例、283 关系、109 定义、292 位置义项）。
- 已暂存并逐条校准前五批共 150 项标签草稿；第五批 30 条已有 24 个 AI 严重度数值，6 条严重度待讨论，另含 2 条 group 疑点，合计 7 个待讨论对象。
- [第 1 批记录](batches/batch-01/discussion.md)：原有 15 项疑点已由用户逐项裁定。其余 15 项仍是 AI 草稿，不能因本批疑点已处理而自动记作人工确认。
- [第 2 批记录](batches/batch-02/discussion.md)：6 项疑点已由用户明确裁定，另 24 项仍为 AI 草稿；[本次裁决](discussions/batch-02/user-decisions-01.json)保留原理由与对象身份。
- [第 3 批记录](batches/batch-03/discussion.md)：4 项疑点已由用户逐项裁定，另 26 项仍为 AI 草稿；[本次裁决](discussions/batch-03/user-decisions-01.json)保留原理由及 #3281 的阈值说明，附完整 JSON/CSV。
- [第 4 批合并讨论表](batches/batch-04/review.md)：用户给出 8 条 group／严重度，见[类别裁决](discussions/batch-04/user-decisions-01.json)及[分数裁决](severity/batch-04/discussions/user-decisions-01.json)。#3845 修正为仅 Racism、3 级；#4744 修正为空集、0 级；#43 为 Region、0 级。用户随后[明确将 #4392 改为 hate](discussions/batch-04/user-decisions-02.json)，并启用分数默认映射；其他条目的分数映射不计作用户直接 hate 裁决。
- 已参考 42 项标签裁决及 13 个定义审核；[参照身份清单](provenance/human-reference.json)记录对象版本、哈希及来源快照，不复制完整人工会话。
- 首批参考前的 AI 草稿保存在 provenance 中；本次用户裁决前的草稿和导出另存于 [before-user-decisions](discussions/batch-01/before-user-decisions/labels-01.json)，保留原判断及其修改依据。
- [首轮用户裁决](discussions/batch-01/user-decisions-01.json)保存 14 项标签；[补充裁决](discussions/batch-01/user-decisions-02.json)保存 #1224 的 hate + Sexism。两份均绑定对象正文哈希、初始版本、v2 政策和来源快照。辅助选项、证据高亮和原答案对照仍是 AI 整理；不代表线上整项材料确认。
- [第一批核查记录](discussions/batch-01/followup.md)：#1224 前半句按用户回复作为行为选择的类比，不另归 others；#1509 按用户裁决取 hate + Sexism，“老g”尚无检索证实的释义，独立保留说明。
- [第二批校准记录](discussions/batch-02/followup.md)说明如何参照上述个案，保留校准前稿、来源哈希和字段差异；没有将新对象标成人审完成。
- 用户随后明确选择“调整后续口径：个人辱骂本身就应计入 others”。[增补规则确认](policies/personal-insult-others-v1.json)保存原回复；从第 3 批起，group 使用 `v2+personal-insult-others.1`，hate 沿用 v2。
- [第三批校准及规则应用](discussions/batch-03/followup.md)记录如何依据上述回复收敛 6 项共享问题。规则确认没有自动成为这些新样本的人工确认。
- [policy_context.json](policy_context.json)指定新口径从第 3 批生效；后续批次导出前须记录 `policy-application.json`。原输入清单、原 v2 文件及前两批既有记录保留其原版本；后续用户裁决文件使用该批的有效政策身份。JSON/CSV 均保留逐任务的规则版本及哈希。
- 前三批共登记 25 项明确的用户 hate/group 个案裁决，另外 65 项是 AI 草稿；共享规则确认不额外计作个案裁决。
- 第 4 批另登记 8 项 group 裁决、1 项明确 hate 裁决（#4392）；未显式给出的 hate 不计为用户直接裁决。标签导出支持按字段记录及分次补充，重复覆盖同一字段需明确更正。
- 用户回复“可以接受这版提案”，已接受攻击严重度 0–4／null 的试行方案。[接受记录](policies/attack-severity-v1.json)绑定原[讨论草案](discussions/attack-severity-proposal-v1.md)与[试行规则](../../../../../docs/research/annotation-guidelines/evidence-attack-severity-policy-v1.md)的哈希；原版约定严重度独立于 hate/group，该历史约定保留。用户随后以独立增补改为由严重度默认生成 hate；接受规则不等于逐条人工打分。
- [严重度首轮试标](severity/pilot-01/discussion.md)：从已讨论范围选择 30 条，24 条有 AI 数值；用户已明确判 #309＝1、#2750＝3、#2180＝2、#2800＝3，见[分数裁决](severity/pilot-01/discussions/user-decisions-01.json)。#1509、#2004 因隐语强度证据不足保留 null，不补猜具体词。4 个用户数值与 25 项 hate/group 裁决分开计数；不能互相当作另一字段的确认。
- [severity_context.json](severity_context.json)单独保存严重度规则与最初试标阶段的身份；[校准回执](severity/pilot-01/calibration-summary.json)登记首轮讨论已处理并转入第 4 批。原 `policy_context.json` 继续管理 hate/group，各任务版本独立保留。
- 第 4 批[严重度记录](severity/batch-04/ai_annotations.json)现有 8 条用户数值、22 条 AI 数值，均非 null。加上第五批后，本轮已有严重度记录 90 条：12 条明确用户数值、70 条 AI 数值、8 条 null；首轮 #1509、#2004 的严重度仍保留 null。group 与分数的最新裁决不覆盖严重度输入中保留的初始标签参照快照。
- 没有更新线上服务、写入真实审核会话、覆盖历史裁决或旧 v1 AI bundle。后续如接入工作台，需要独立 AI 来源层，并以届时最新人审状态排除已完成项。

- [暂行映射接受记录](policies/severity-hate-default-mapping-v1.json)保存用户原文及[独立增补规则](../../../../../docs/research/annotation-guidelines/evidence-severity-hate-default-mapping-v1.md)的身份；[hate_mapping_context.json](hate_mapping_context.json)指定当前派生层。原 severity_context、policy_context、v1 分级文件及最初输入清单均不回写。
- [当前映射总表](severity/hate-mapping-v1/README.md)覆盖已赋分的 90 条，提供 JSON/CSV。#1240 依现有 AI 严重度 1 映射为 hate，原用户 non-hate 及原政策仍保留；这不是新的用户分数或直接 hate 裁决。#1509、#2004 及第五批 6 个待定分数暂不生成映射结果，不擦除已有标签层。
- [第五批合并讨论表](batches/batch-05/review.md)：待讨论 #4810、#5113、#528、#5365、#5394、#5416、#5615。原文、候选分与理由齐备；24 个数值均是 AI 草稿，没有新增用户裁决。[校准说明](discussions/batch-05/followup.md)记录个案参照、词义检索及范围限制。
- 后续分析读取合并 JSON 的 `records[].values` 或合并 CSV；`label_annotation`、严重度的 `prior_label_reference` 和旧 `ai_annotations.json` 是独立参照层，保留原判断、来源与政策。映射结果不继承参照层的原答案疑错结论。group 仍保留各批有效政策，前两批不冒充已按第三批的 others 增补重核。

`input_manifest.json` 为初始输入冻结记录。其最初生成时尚未使用人工参照；后续获准参照的实际暴露以 `provenance/human-reference.json` 及各批 discussions 记录为准，不宣称盲标。954 项是初始快照中的线上未审核范围，不能扣除本地 AI 草稿或讨论记录后冒充实时线上完成数。
