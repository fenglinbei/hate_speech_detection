# 裁决偏好表的维护

[当前可读表](preferences.md) · [当前 JSON](preferences.json) · [案例 CSV](cases.csv) · [当前快照指针](current.json)

该表服务于本地标注复核。人类决定来自已有 `user-decisions-*.json` 的实际 `scope`；规则文字与跨案例比较保留 AI 归纳来源。接受规则不等于确认全部示例或补齐未回答字段。

资源关系的独立偏好表见 [示例与查询关系规则及原答](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/relations/preferences.md)。其中的关系字段不混入本表的句子标签或严重度裁决；相似个案复查时同时读取对应任务的偏好。

每轮起草前读取当前表，在该轮校准记录中绑定 `current.json` 指向的不可变快照。每次收到裁决，先保存原回复和字段来源；本轮收齐后整理 `rule-cards.json` 和 `consistency-notes.json`，再刷新。相似案例只是复核候选，需比较对象、命题、立场、上下文及政策。若仍可能矛盾，向用户并列前后案例和具体字段，保留旧值等待说明，不自动覆盖。

刷新命令（在仓库根目录运行）：

```bash
python scripts/review/export_adjudication_preferences.py
python scripts/review/export_adjudication_preferences.py --check
```

脚本自动收录新增的明确字段事件、验证正文哈希和规则锚点的人工字段范围，列出未纳入规则卡的案例，以及同一案例同一字段出现多个历史值的候选冲突。跨案例的语义一致性仍需逐条比较，不声称自动检出全部冲突。机器标注、原标签和线上确认不会被本脚本用作人工裁决来源。

每次内容变化生成 `snapshots/<sha256>/`，保留 JSON、Markdown 和 CSV；重复运行内容相同时复用同一快照。旧轮次引用旧快照，当前入口更新不改变历史依据。若需要更改已确认个案，先按该批复核流程保存有明确来源的修订记录，不直接改写旧裁决文件。

最新合并优先级见 [group_priority_same_target_context.json](../group_priority_same_target_context.json)：仅同一黑人受辱者的种族与个人辱骂才合并；不同攻击对象保留 others。#5423 已经明确一致性复核改为 Racism、others，#1746 撤回过宽 AI 合并；当前句子结果见 [入口](../sentence_context.json)。

导出器同时收录 `sentence-completion-v1/discussions/user-decisions-*.json`。旧字段只有在新记录明确提供 `supersedes_event_ids` 与 `amendment_reason` 后，才从当前人工值中退为历史；不同值而无明确修订仍显示为冲突候选。不能按时间直接取最后一个值。相同字段复核、分数推导的 hate、AI 辅助解释分别记录，不增加未回答字段的人工确认。
