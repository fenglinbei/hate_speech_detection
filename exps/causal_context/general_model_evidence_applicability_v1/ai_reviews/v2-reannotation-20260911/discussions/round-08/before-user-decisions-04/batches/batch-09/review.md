# 第 9 批：标签与攻击严重度

本批 30 条，3 条需要讨论。其中 group 有 1 条疑点；严重度 27 条已有数值，3 条随语义或等级边界讨论，另 0 条证据不足保留待定。

本批 7 条有用户标签字段裁决（hate 3 条、group 7 条），用户严重度数值 10 条；其余语义判断为 AI 草稿，hate 另记分数映射来源。没有新增线上整项材料确认。

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

## 待讨论项目

### #61

> 因为种族优势，中国人普遍都很小

当前标签草稿：待定 / 待定。严重度：待定。

种族优势提供种族评价线索，但“小”的比较维度缺失。可能是一般体型描述、轻度挖苦或男性生殖器刻板印象；不能直接由小字补出性别。与 #1815 的特定语境、#924 不额外加种族的边界比较后，严重度及完整 group 待定，倾向污名 2。

种族优势提供种族评价线索，但“小”的比较维度缺失。可能是一般体型描述、轻度挖苦或男性生殖器刻板印象；不能直接由小字补出性别。与 #1815 的特定语境、#924 不额外加种族的边界比较后，严重度及完整 group 待定，倾向污名 2。

严重度候选：[0, 1, 2]；倾向 2，倾向分未写入活动分值。

### #7244

> 知足吧，爱丁堡不比井盖、山越猴、京巴、九头鸟、东百太君、白完、胡建、陕蛋蛋、响马听起来洋气？

当前标签草稿：待定 / Region。严重度：待定。

group 本轮没有待讨论疑点；hate 由活动严重度映射。

用多个地域称呼作好听程度比较，外部用例支持爱丁堡指上海；不据未出现场景增 LGBTQ。待区分词语比较 0、实际调侃地域身份 1，或直接使用动物辱称 3；#5365 的谈梗、#528 的明确辱称并不支持仅靠词表决定。

严重度候选：[0, 1, 3]；倾向 1，倾向分未写入活动分值。

### #7646

> 我的好几个贴，都被删除了，会不会是这玩意举报的？

当前标签草稿：待定 / others。严重度：待定。

group 本轮没有待讨论疑点；hate 由活动严重度映射。

将可能的举报者称作这玩意，倾向轻度个人贬称 1／others；是否已达到明确侮辱 2 待校准。参照 #309 的轻度人格贬评，不仅凭物品指称升 3。没有依据新增举报者的性别或族群。

严重度候选：[1, 2]；倾向 1，倾向分未写入活动分值。

## 证据不足而暂不要求补猜

