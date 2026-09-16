# 本地资源协同复核

资源复核从定义开始，再检查词典命中，最后判断示例与查询的关系。沿用现有 evidence 工作台的定义判定枚举，不给词语本身标严重度；这里只维护本地协同记录。

- [当前定义裁决及偏好表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/preferences.md)
- [当前定义快照索引](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/definitions/current.json)
- [句子裁决偏好表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/preferences.md)
- [命中适配：当前结果、提问与偏好](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/hits/preferences.md) · [维护说明](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/hits/README.md)
- [示例与查询关系：当前结果与偏好](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/relations/preferences.md) · [维护说明](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/relations/README.md)

## 当前批次

定义第一批包含 40 个义项，按被待审命中引用的次数优先选择，涉及 161 处待审命中。选择依据和源文件哈希保存在 `definitions/batch-01/selection.json`。定义被引用不等于命中适配，也不计入命中进度。

定义第二批完成其余 69 个义项，涉及 72 处待审命中；10 项用户回答已保存，无待回复项。累计 109 项定义均有本地复核结果，其中 22 项有用户参与、87 项仅有 AI 复核。按原定义质量字段计，20 项为用户裁决、89 项为 AI 判定；“黑人喃”和“过男”的用户回答只补写或采纳定义，不另推定其质量枚举。批次汇总见 `definitions/batch-02/review.md` 和 `closeout.json`。

292 项命中适配和 283 项示例关系均已有本地复核结果。#5998“阿娜”两处由用户确认采用辱女义，#2445“靠过男人”中的“过男”按独立词边界为子串误命中。审查原资源时仍以冻结输入为对象；已采纳改写独立展示，不替换原文后再把原资源判为合理。

关系复核的 12 个校准问题和 1 个一致性补问已处理，283 组中 12 组有明确用户字段、271 组为 AI 复核；当前 20 个明确字段，含修订共 21 个字段事件。#5086/#127 的 group 规则由用户明确从部分改为直接，hate 规则仍部分。当前无待回复或关系字段未决；此前 37 组已确认关系只作参照。

命中第 1 批的 9 个校准问题已处理：8 项得到适配字段裁决，1 项按用户要求单列“女圈”词条。292 处中，11 处有明确适配字段，1 处有相关词条指示，其余 280 处为 AI 复核；累计 10 个来源适配字段、3 个查询适配字段。原义项来源适配不明 5 处，查询适配不明 4 处；这些不明记录与漏答分开。“女圈”新增词条与原 109 个冻结定义复核对象分别计数。句子联动复核及最新结果见 `hits/batch-01/post-decision-notes.json` 和 `sentence_context.json`。

`initial-ai-drafts.json` 保存最初建议；`ai-drafts.json` 是可修订的 AI 层。AI 修订前保留旧文件到 `draft-history/<sha256>.json`。`questions.json` 固定提问顺序、对象、原定义、改写建议和选项，它是绑定表而非逐字问题副本。

补充问题保存在 `additional-questions-NN.json`，按请求 ID 与问题序号共同绑定，避免不同请求的第 0 项混淆。第二批完整 AI 文件在用户回答到达后才落盘；来源说明明确保留这一时序，不声称它是回答前冻结的独立判断。该批问题绑定表同时保存原问题全文。

收到回答后，完整 `questionItemId`、`question`、`answer` 逐字保存为 `discussions/raw-replies-NN.json`；`user-decisions-NN.json` 引用其哈希及记录序号，并只提取用户实际裁定的字段。每个事件有唯一 `event_id`、`object_id`、`scope`、`values`、原定义 `source_sha256`、`question_item_id` 和逐字 `rationale`。原答不可用 AI 解释代替。

## 字段与复查

`definition_verdict` 沿用 `reasonable`、`too_narrow`、`other_problem`、`uncertain`。明确采纳完整改写时另存 `adopted_definition`；只提出修改原则时存 `rewrite_instruction`，不能把 AI 后续拟写的句子记为已批准原文。AI 的问题类型、证据片段和理由仍归 AI，不因确认一个选项而自动成为人工作答。

当前表逐条列出原定义、有效裁决、来源、用户原答、已采纳改写和 AI 建议，供下次复查。一般化规则仍需明确区分“用户规则”与“AI 个案归纳”。发现与先前案例存在冲突时，先指出具体字段和两例差别，再请求用户区分或修订；不得自行改写旧决定。

同一字段再次裁定必须在事件中用 `amends` 指向被替代的事件 ID，并给出 `amendment_reason`。旧字段和原答保留在 `human_field_history`。`uncertain` 表示已记录的未决判断，另列词义未决数量，不作为合理定义使用。

旧人工定义 13 项仅供校准，不计本轮新增。定义的原文质量判定、修改后的建议文本、实际命中的适配判断是不同对象；当前流程不改写冻结词库、实验输入或在线记录。

## 导出与验证

```bash
python scripts/review/export_definition_reviews.py
python scripts/review/export_definition_reviews.py --check
python -m unittest discover -s scripts/review -p test_definition_reviews.py
```

导出在独立文件锁下生成按内容哈希命名的 JSON、CSV、Markdown 快照，再更新 `current.json` 和本地 `definition_context.json`。表中“仅 AI 复核”“用户参与”“待回复”“尚未开始”分列，并另计质量裁决、完整改写与改写指示的来源。任何本地记录都不会自动增加在线人工确认数。
