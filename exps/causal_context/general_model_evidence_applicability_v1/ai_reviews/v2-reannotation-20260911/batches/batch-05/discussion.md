# v2（含个人辱骂增补）补标第 5 批：疑点已处理

本批 30 项，其中 0 项有待讨论疑点；0 项仍有 null 标签。已登记 7 项的用户标签字段裁决（hate 7 项、group 7 项）。尚未导入线上，没有新增线上整项确认。

已参考用户允许使用的既有人工裁决，来源身份见独立 reference 清单。原答案对照建议在本批标签写完后另行形成，不代表已经确认源数据错误。

[完整 JSON](ai_annotations.json) · [CSV](ai_annotations.csv)

本文件是标签参照层，保留显式裁决与 AI 标签草稿及其原政策。当前 hate 已改用严重度默认映射，请使用[合并讨论表](review.md)、[当前 JSON](review.json)或[当前 CSV](review.csv)。本层原答案对照状态不自动转为新映射下的源数据结论。

本批 group 采用用户确认的个人辱骂计入 others 增补规则；hate 参照层默认沿用父版，用户显式裁决另有政策声明时按逐条记录保留。

## 本次用户裁决

仅明确给出的标签字段及下列用户理由来自裁决；未明确给出的另一字段仍为 AI 草稿。辅助选项、证据高亮和原答案对照状态仍为 AI 整理，没有代替完整材料确认。

| 对象 | hate | group | 用户理由／采纳范围 |
| --- | --- | --- | --- |
| demo:4810 | hate | ["Racism", "Sexism", "others"] | 美国佬为政治代理指控不参与讨论 |
| demo:5113 | non-hate | ["LGBTQ", "Sexism", "others"] | 仅为客观描述事件，主体是那个”男的“，并未直接攻击同性恋群体 |
| demo:528 | hate | ["Region", "Sexism"] | 使用明确辱称，且没证据表明是无害讨论 |
| demo:5365 | non-hate | ["Sexism", "others"] | 只在谈网络梗和生活选择 |
| demo:5394 | hate | ["LGBTQ", "others"] | 只是表达漠视、未认可伤害，没到达预定程度 |
| demo:5416 | hate | ["LGBTQ", "others"] | 归谬得到”男同最会传播艾滋病“带有污名化意，且整体带有阴阳怪气 |
| demo:5615 | hate | ["Racism"] |  |

[用户裁决记录 1](../../discussions/batch-05/user-decisions-01.json)

[补充核查与校准说明](../../discussions/batch-05/followup.md)

## 本批待讨论项

本批原有疑点已处理。未被用户明确裁定的其他记录仍是 AI 草稿。

