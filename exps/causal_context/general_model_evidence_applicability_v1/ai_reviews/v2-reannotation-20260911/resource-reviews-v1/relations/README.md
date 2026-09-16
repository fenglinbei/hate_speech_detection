# 示例与查询关系复核

283 组均有本地协同复核结果：12 组有明确用户字段，271 组仅有 AI 复核。12 个校准问题和 1 个一致性补问均已回复，无待回复或关系字段未决项。此前 37 组线上已确认关系仅作参照，不计入这 283 组。

- [当前结果与规则偏好表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/relations/preferences.md)
- [JSON、CSV、Markdown 快照索引](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/relations/current.json)
- [起草与校准的逐字段改动](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/relations/batch-01/calibration-application.json)

沿用现有 evidence 工作台字段：`topic_hate`、`topic_group`、`rule_hate`、`rule_group` 分别取 `direct / partial / none / unclear`；`lexicon_risk` 取 `yes / no / unclear`。两段完整原文、证据字符区间和示例词条来源保留在导出中。hate/group 标签、词典命中、定义质量和关系判断是独立记录层。

用户明确字段共 20 个：hate 话题 6、group 话题 3、hate 规则 4、group 规则 4、词典风险 3。加上一次 group 规则修订，共有 21 个字段事件。#5086/#127 当前话题均直接、hate 规则部分、group 规则直接、词典风险未发现；先前 group 规则“部分”保留在历史中。

## 复用边界

偏好表含 12 条带实际原答来源的 AI 归纳。只共享类别、泛化、问句或网络平台不自动意味着有规则关联；需要定位具体主题、词义或判别结构。直接对应一个关键身份判断即可支持 group 规则直接，无需覆盖完整 group 集合；这不代表仅有标签交集就足够，也不代表所有身份话题都完全一致。

词典风险逐条追溯来源：实际词形不在查询中仍可不适配，即使示例的语义话题相关；同一多义词存在适配义项时，正常列出的其他义项不自动构成风险；定义整体质量问题与本组实际词义匹配分别记录。`introduced_entry_ids` 来自示例的词典命中，不证明该示例是词条唯一来源，也不证明词条已导致模型出错。

不以示例补齐查询中的省略内容。泛指计划、截图而没有可见的具体主题时，本轮依用户边界记“无”，这不等于证明两者实际上不同。#7959 原婚恋义仍保持命中层的“不明”；关系 #1160/#7959 由 AI 按缺乏可见关联记“无”，不改变词义裁决。

## 保存与修订

`batch-01/selection.json` 绑定全部 283 个授权对象、冻结输入、逐字问题和回答前完整初稿。`preparation.json` 另绑定句子、定义、命中及偏好快照。初稿 SHA-256 为 `d598e5351d666ef0167206a5de6439914d0faf903195d4a4213f8354fcc43570`；当前 AI 层保存在 `ai-drafts.json`，修改前的完整字节存入 `draft-history/<sha256>.json`。

用户逐字问题、答案和完整 questionItemId 保存在 `discussions/raw-replies-*.json`；明确字段事件通过哈希、对象、请求 ID、问题序号和实际字段范围绑定原答。重复裁决必须有 `amends` 和修订原因，不能按时间静默覆盖。一题涉及两个对象时，两个对象都答到才关闭；一致性补问须收到自己的回复，不能用旧字段自动关单。

AI 归纳、推广、证据和说明保持 AI 来源。即使用户接受某个字段，也不把其余字段、原句标签或所有相似配对记为人工作答。任何本地记录都不增加在线整项人工确认数；本轮未改线上会话、冻结实验输入或原始资源。

## 导出与检查

```bash
python scripts/review/export_relation_reviews.py
python scripts/review/export_relation_reviews.py --check
python -m unittest discover -s scripts/review -p test_relation_reviews.py
```

导出器在文件锁下创建不可变 JSON、CSV、Markdown 快照，然后更新 `current.json`、可读偏好表及本地 `relation_context.json`。每次导出核对授权对象、原文与证据区间、来源命中、初稿身份、实际回复范围和旧裁决历史。测试中的模拟事件仅在隔离临时目录或内存中使用。
