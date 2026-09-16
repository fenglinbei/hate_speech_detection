# v2（含个人辱骂增补）补标第 4 批：疑点已处理

本批 30 项，其中 0 项有待讨论疑点；0 项仍有 null 标签。已登记 8 项的用户标签字段裁决（hate 1 项、group 8 项）。尚未导入线上，没有新增线上整项确认。

已参考用户允许使用的既有人工裁决，来源身份见独立 reference 清单。原答案对照建议在本批标签写完后另行形成，不代表已经确认源数据错误。

[完整 JSON](ai_annotations.json) · [CSV](ai_annotations.csv)

本文件是标签参照层，保留显式裁决与 AI 标签草稿及其原政策。当前 hate 已改用严重度默认映射，请使用[合并讨论表](review.md)、[当前 JSON](review.json)或[当前 CSV](review.csv)。本层原答案对照状态不自动转为新映射下的源数据结论。

本批 group 采用用户确认的个人辱骂计入 others 增补规则；hate 沿用父版。

## 本次用户裁决

仅明确给出的标签字段及下列用户理由来自裁决；未明确给出的另一字段仍为 AI 草稿。辅助选项、证据高亮和原答案对照状态仍为 AI 整理，没有代替完整材料确认。

| 对象 | hate | group | 用户理由／采纳范围 |
| --- | --- | --- | --- |
| demo:3652 | 未明确裁决 | ["Racism", "Sexism"] | 带有一定讥讽性质 |
| demo:3660 | 未明确裁决 | ["Racism", "Sexism"] | 句内隐含女性身份 |
| demo:3742 | 未明确裁决 | ["Racism", "others"] | 轻微讥讽意 |
| demo:3845 | 未明确裁决 | ["Racism"] | 动物性别称谓用于人，已支持侮辱 |
| demo:3882 | 未明确裁决 | ["Region"] | 反歧视，不含主观攻击意 |
| demo:43 | 未明确裁决 | ["Region"] | 上下文不全，未包含贬损其实证据 |
| demo:4392 | hate | ["Racism", "Sexism", "others"] | 词义本身即有冒犯意，且不是对词本身的讨论；改为hate |
| demo:4744 | 未明确裁决 | [] | 上下文不全，但是句子本身不含贬损歧视义，若有上下文全的其他案例可改判 |

[用户裁决记录 1](../../discussions/batch-04/user-decisions-01.json)

[用户裁决记录 2](../../discussions/batch-04/user-decisions-02.json)

[补充核查与校准说明](../../discussions/batch-04/followup.md)

## 本批待讨论项

本批原有疑点已处理。未被用户明确裁定的其他记录仍是 AI 草稿。

