# v2（含个人辱骂增补）补标第 4 批：待讨论

本批 30 项，其中 1 项有待讨论疑点；0 项仍有 null 标签。已登记 8 项的用户标签字段裁决（hate 0 项、group 8 项）。尚未导入线上，没有新增线上整项确认。

已参考用户允许使用的既有人工裁决，来源身份见独立 reference 清单。原答案对照建议在本批标签写完后另行形成，不代表已经确认源数据错误。

[完整 JSON](ai_annotations.json) · [CSV](ai_annotations.csv)

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
| demo:4392 | 未明确裁决 | ["Racism", "Sexism", "others"] | 词义本身即有冒犯意，且不是对词本身的讨论 |
| demo:4744 | 未明确裁决 | [] | 上下文不全，但是句子本身不含贬损歧视义，若有上下文全的其他案例可改判 |

[用户裁决记录 1](../../discussions/batch-04/user-decisions-01.json)

[补充核查与校准说明](../../discussions/batch-04/followup.md)

## 本批待讨论项

### demo:4392

> debuff叠满了，本人西安上学维吾尔族极左普信男粉红爱国蛆😄

当前 AI 草稿：hate = non-hate；group = ["Racism", "Sexism", "others"]

疑点：个体辱骂的 hate 范围

用户确认 Racism、Sexism、others 与严重度 1，说明词义本身有冒犯意、并非讨论词本身。原 non-hate 草稿暂保留；已单独询问是否也改变 hate，不能仅由 1 级自动转成 hate。

