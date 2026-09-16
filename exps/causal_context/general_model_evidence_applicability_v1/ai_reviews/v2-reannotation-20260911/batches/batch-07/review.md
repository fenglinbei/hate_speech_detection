# 第 7 批：标签与攻击严重度

本批 30 条，0 条需要讨论。其中 group 有 0 条疑点；严重度 30 条已有数值，0 条随语义或等级边界讨论，另 0 条证据不足保留待定。

本批 3 条有用户标签字段裁决（hate 0 条、group 3 条），用户严重度数值 5 条；其余语义判断为 AI 草稿，hate 另记分数映射来源。没有新增线上整项材料确认。

当前 hate 按用户暂行映射生成：0→non-hate，1–4→hate，null→null。JSON 的 records[].values 和 CSV 是当前结果；label_annotation 与 prior_label_reference 保留各阶段标签及来源。用户分数的映射也不新增一次直接 hate 裁决。

[合并 JSON](review.json) · [合并 CSV](review.csv) · [标签记录](discussion.md) · [严重度记录](../../severity/batch-07/discussion.md)

严重度使用已接受的 0–4 级试行规则；#309＝1、#2750＝3、#2180＝2、#2800＝3 是用户给出的本轮校准锚点。

## 已登记的用户字段

仅下表标明的字段计作用户直接裁决；分数映射另记来源，其余语义判断保持 AI 来源。

| 对象 | hate | group | 严重度 | 用户裁决字段 |
| --- | --- | --- | --- | --- |
| demo:6469 | hate（分数映射） | Racism（AI） | 1 | attack_severity |
| demo:6631 | non-hate（分数映射） | Racism、Region | 0 | group, attack_severity |
| demo:6989 | non-hate（分数映射） | 空 | 0 | group, attack_severity |
| demo:7189 | hate（分数映射） | LGBTQ、Sexism、others | 1 | group, attack_severity |
| demo:7232 | non-hate（分数映射） | Region、Sexism、others（AI） | 0 | attack_severity |

## 待讨论项目

## 证据不足而暂不要求补猜

