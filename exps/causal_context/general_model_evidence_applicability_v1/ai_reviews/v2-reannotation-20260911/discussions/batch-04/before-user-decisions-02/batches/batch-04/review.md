# 第 4 批：标签与攻击严重度

本批 30 条，1 条需要讨论。其中 hate/group 有 1 条疑点；严重度 30 条已有数值，0 条随语义或等级边界讨论，另 0 条证据不足保留待定。

本批 8 条有用户标签字段裁决（hate 0 条、group 8 条），用户严重度数值 8 条；其余各字段保留 AI 来源。没有新增线上整项材料确认。

严重度层的 prior_label_reference 保留赋分前的标签快照；本表的当前标签以 label_annotation 为准，两者差异显式保存，不覆盖旧快照。

[合并 JSON](review.json) · [合并 CSV](review.csv) · [标签记录](discussion.md) · [严重度记录](../../severity/batch-04/discussion.md)

严重度使用已接受的 0–4 级试行规则；#309＝1、#2750＝3、#2180＝2、#2800＝3 是用户给出的本轮校准锚点。

## 已登记的用户字段

仅下表标明的字段计作用户裁决；其余值保持 AI 来源。

| 对象 | hate | group | 严重度 | 用户裁决字段 |
| --- | --- | --- | --- | --- |
| demo:3652 | hate（AI） | Racism、Sexism | 1 | group, attack_severity |
| demo:3660 | hate（AI） | Racism、Sexism | 2 | group, attack_severity |
| demo:3742 | hate（AI） | Racism、others | 1 | group, attack_severity |
| demo:3845 | hate（AI） | Racism | 3 | group, attack_severity |
| demo:3882 | non-hate（AI） | Region | 0 | group, attack_severity |
| demo:43 | non-hate（AI） | Region | 0 | group, attack_severity |
| demo:4392 | non-hate（AI） | Racism、Sexism、others | 1 | group, attack_severity |
| demo:4744 | non-hate（AI） | 空 | 0 | group, attack_severity |

## 待讨论项目

### #4392

> debuff叠满了，本人西安上学维吾尔族极左普信男粉红爱国蛆😄

当前标签草稿：non-hate / Racism、Sexism、others。严重度：1。

用户确认 Racism、Sexism、others 与严重度 1，说明词义本身有冒犯意、并非讨论词本身。原 non-hate 草稿暂保留；已单独询问是否也改变 hate，不能仅由 1 级自动转成 hate。

用户判 1 级，说明词义本身具有冒犯意，且不是讨论词本身。该数值本身不确认 hate 字段；二分类另行对齐。

## 证据不足而暂不要求补猜

