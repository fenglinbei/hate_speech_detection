# 句子部分收尾

270句的三个标签字段均已补齐；本轮全部16条疑问回复已记录，无待讨论项。

| 项目 | 结果 |
| --- | --- |
| 句子 | 242示例＋28查询 |
| hate / non-hate | 151 / 119 |
| 严重度0 / 1 / 2 / 3 / 4 | 119 / 43 / 83 / 21 / 4 |
| 严重度来源 | 用户明确60；AI补标210 |
| 本轮新增明确字段 | 严重度11；group 7；直接hate 0 |
| 本轮范围 | 初始71句；规则补充后共73句 |

[完整JSON](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/current-sentences.json) · [完整CSV](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/current-sentences.csv) · [本轮全文与依据](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/review.md) · [规则偏好表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/preferences.md)

当前读数来自`records[].values`；字段来源见`field_provenance`、`current_field_sources`。辅助理由、攻击形式及证据位置为AI整理；它们不因用户只回复分数或类别而变成人工确认。当前汇总是本地AI辅助标注，不计作线上材料确认。

原9条类别复核最终均保留原group。新条件要求个人与种族攻击指向同一黑人受辱者，不能按同句共现合并others。#5423经用户明确复核增加others；#1746过宽AI合并撤回。#25、#2517按照个人辱骂口径补others，分别为AI补标和用户明确裁决。

原60条缺失严重度与2条null全部补齐。#1509、#2004现均为2级；老g的具体词义仍无适用外部来源，此次凭其余可见贬损定级，不声称隐语已查明。#3372明确0级，经映射由旧AI hate改为non-hate。

`selection.json`和`baseline.json`冻结初始71句与270句原视图；`policy-followup.json`记录后续同一对象条件及新增2条复核。原问答存于`discussions/`，每次AI稿变更留存于`draft-history/`，每版导出在`snapshots/`保留。已明确修订的人工字段通过事件ID连接旧值，不静默覆盖。

[验证](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/validation.json)覆盖270句、JSON/CSV一致性、字段来源、严重度映射、16条回复入库、冻结来源哈希和5项回归检查。[校准回执](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/sentence-completion-v1/calibration-summary.json)固定本轮采用的前后偏好快照。

只读复现：

```bash
python scripts/review/export_sentence_completion.py --check
python scripts/review/export_adjudication_preferences.py --check
```

旧批次与210条映射继续保留历史身份。后续工作使用上级`sentence_context.json`选定的当前层；资源适用性队列独立，尚未在本轮审核。
