# 第 9 批：标签与攻击严重度

本批 30 条，0 条需要讨论。其中 group 有 0 条疑点；严重度 30 条已有数值，0 条随语义或等级边界讨论，另 0 条证据不足保留待定。

本批 8 条有用户标签字段裁决（hate 3 条、group 8 条），用户严重度数值 13 条；其余语义判断为 AI 草稿，hate 另记分数映射来源。没有新增线上整项材料确认。

当前 hate 按用户暂行映射生成：0→non-hate，1–4→hate，null→null。JSON 的 records[].values 和 CSV 是当前结果；label_annotation 与 prior_label_reference 保留各阶段标签及来源。用户分数的映射也不新增一次直接 hate 裁决。

[合并 JSON](review.json) · [合并 CSV](review.csv) · [标签记录](discussion.md) · [严重度记录](../../severity/batch-09/discussion.md)

严重度使用已接受的 0–4 级试行规则；#309＝1、#2750＝3、#2180＝2、#2800＝3 是用户给出的本轮校准锚点。

## 已登记的用户字段

仅下表标明的字段计作用户直接裁决；分数映射另记来源，其余语义判断保持 AI 来源。

| 对象 | hate | group | 严重度 | 用户裁决字段 |
| --- | --- | --- | --- | --- |
| demo:986 | hate（分数映射） | Region（AI） | 2 | attack_severity |
| demo:996 | non-hate（分数映射） | Racism、Region、Sexism（AI） | 0 | attack_severity |
| query:1160 | non-hate（分数映射） | 空 | 0 | group, attack_severity |
| query:1292 | non-hate（分数映射） | Racism | 0 | group, attack_severity |
| query:2011 | hate（另有明确裁决） | Racism | 2 | hate, group, attack_severity |
| query:2297 | hate（另有明确裁决） | LGBTQ | 3 | hate, group, attack_severity |
| query:3898 | hate（另有明确裁决） | Racism、Sexism | 3 | hate, group, attack_severity |
| query:4080 | non-hate（分数映射） | Racism、Region（AI） | 0 | attack_severity |
| query:4189 | non-hate（分数映射） | Sexism | 0 | group, attack_severity |
| query:5423 | hate（分数映射） | Racism | 2 | group, attack_severity |
| query:61 | hate（分数映射） | Sexism | 2 | group, attack_severity |
| query:7244 | hate（分数映射） | Region（AI） | 1 | attack_severity |
| query:7646 | hate（分数映射） | others（AI） | 1 | attack_severity |

## 待讨论项目

## 证据不足而暂不要求补猜

