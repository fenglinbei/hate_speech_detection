# 第 8 轮：补标讨论

共 30 条，来自冻结输入分块 9；当前 8 条待讨论。输入分块及原正文保持原身份。

本轮已登记用户直接 hate 1 条、group 3 条、严重度数值 5 条。另有 18 条 AI 数值、7 条待定分数。疑点处理不等于整批人工审核。

当前 hate 继续按 0→non-hate、1–4→hate、null→null 派生。单独给出的分数不额外计为直接 hate 或 group 裁决。没有新增线上材料确认。

[JSON](review.json) · [CSV](review.csv) · [范围与扩容依据](selection.json) · [词义查证与限制](sense-checks.json) · [校验](validation.json)

[本轮采用的裁决偏好快照](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/preferences/snapshots/67f52f0bb542bffb2eb511d72b045eabe01943a4e91ad6e76aa3f34bcec183f3/preferences.json) · [相似案例复核](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/discussions/round-08/preference-application.json)

## 待讨论的完整原文

### #2297

> 本质上根本就没有什么这个恋那个恋的，西方为了用身份政治来控制人民，以保证自己的剥削能够继续，才搞出了这些闹剧，把一群过街老鼠的怪癖放上大庭广众之下，让人们自发的结成两派，左手打右手，就没有合力去对付高高在上的资本财团，妙啊

当前严重度：3；group：待定。

把真实的人称为过街老鼠并认定其偏好是怪癖，拟 3；这不是仅批评资本。性少数是“这个恋那个恋”的候选所指，但句中未明说同性恋，完整 group 待定；不能将缺失身份一律装入 others。

### #3898

> 不是已经入狱了吗，话说回来，我心中反黑领导人是快手发展黑人祝福起家，领导一群肌肉黑人喃小弟的壮硕中国男人。当地黑社会一霸，并且每个黑人工资一个月只有一千块人民币。真正实现了心理生理双重奴役黑人喃。

当前严重度：待定；group：待定。

末句呈现奴役的认可语气，倾向严重身份支配 3，但“生理奴役”是否已认可身体伤害至 4 尚需校准；若实际是讽刺，也应先明确立场。Racism 明确；壮硕中国男人、黑社会角色是否独立参与身份评价，完整 group 待定。

候选分：[0, 3, 4]；倾向 3。倾向未写入活动值。

### #4080

> 我是河南的回族，简直互联网最底层🤣

当前严重度：待定；group：Racism、Region。

河南和回族是明确自我身份。倾向把互联网最底层理解成自嘲受歧视处境，取 0；但 #4392 的自称冒犯词被用户定 1，因此不能单凭自我描述排除攻击。待比较是否已认可身份贬低。

候选分：[0, 1]；倾向 0。倾向未写入活动值。

### #4189

> 😓早就听闻不少男的假冒女拳败坏我国大部分女性形象，今日一见，果真如此

当前严重度：待定；group：待定。

倾向将其看作批评假冒与败坏形象的行为，0／Sexism；也可能用女拳对具体对象作轻度贬评，1 且涉及 others。与 #6631／#6989 的行为批评边界和 #1158 的实际辱称用法比较后，严重度及完整 group 仍待定；我国仅为社会场景不另加 Region。

候选分：[0, 1]；倾向 0。倾向未写入活动值。

### #5423

> 专业护默开小号急的不行

当前严重度：待定；group：Racism、others。

将受话者称为护默并挖苦其开小号着急，Racism、others 拟成立：others 来自对具体护黑者的讥讽。程度倾向 2，但默是否在此形成动物化 3，还是仅轻度挖苦 1，待讨论。#5615 仅 Racism 未附理由，不视作取消个人辱骂规则；本句受话者是否个体也可修改。

候选分：[1, 2, 3]；倾向 2。倾向未写入活动值。

### #61

> 因为种族优势，中国人普遍都很小

当前严重度：待定；group：待定。

种族优势提供种族评价线索，但“小”的比较维度缺失。可能是一般体型描述、轻度挖苦或男性生殖器刻板印象；不能直接由小字补出性别。与 #1815 的特定语境、#924 不额外加种族的边界比较后，严重度及完整 group 待定，倾向污名 2。

候选分：[0, 1, 2]；倾向 2。倾向未写入活动值。

### #7244

> 知足吧，爱丁堡不比井盖、山越猴、京巴、九头鸟、东百太君、白完、胡建、陕蛋蛋、响马听起来洋气？

当前严重度：待定；group：Region。

用多个地域称呼作好听程度比较，外部用例支持爱丁堡指上海；不据未出现场景增 LGBTQ。待区分词语比较 0、实际调侃地域身份 1，或直接使用动物辱称 3；#5365 的谈梗、#528 的明确辱称并不支持仅靠词表决定。

候选分：[0, 1, 3]；倾向 1。倾向未写入活动值。

### #7646

> 我的好几个贴，都被删除了，会不会是这玩意举报的？

当前严重度：待定；group：others。

将可能的举报者称作这玩意，倾向轻度个人贬称 1／others；是否已达到明确侮辱 2 待校准。参照 #309 的轻度人格贬评，不仅凭物品指称升 3。没有依据新增举报者的性别或族群。

候选分：[1, 2]；倾向 1。倾向未写入活动值。

## 全部当前值

| 对象 | 严重度 | hate | group | 待讨论字段 |
| --- | --- | --- | --- | --- |
| #986 | 2 | hate | Region | — |
| #996 | 0 | non-hate | Racism、Region、Sexism | — |
| #1128 | 0 | non-hate | [] | — |
| #1160 | 0 | non-hate | [] | — |
| #1292 | 0 | non-hate | Racism | — |
| #1746 | 2 | hate | Racism、Sexism、others | — |
| #1900 | 0 | non-hate | Region | — |
| #2011 | 2 | hate | Racism | — |
| #2297 | 3 | hate | 待定 | group |
| #2802 | 2 | hate | Racism、Sexism | — |
| #3169 | 0 | non-hate | [] | — |
| #376 | 3 | hate | LGBTQ | — |
| #3898 | 待定 | 待定 | 待定 | group, attack_severity |
| #3919 | 0 | non-hate | others | — |
| #3950 | 2 | hate | Sexism、others | — |
| #4026 | 0 | non-hate | Region | — |
| #4080 | 待定 | 待定 | Racism、Region | attack_severity |
| #4137 | 0 | non-hate | LGBTQ、Sexism | — |
| #4189 | 待定 | 待定 | 待定 | group, attack_severity |
| #4450 | 2 | hate | Region | — |
| #4756 | 2 | hate | Racism | — |
| #5086 | 2 | hate | LGBTQ | — |
| #5423 | 待定 | 待定 | Racism、others | attack_severity |
| #554 | 0 | non-hate | Racism | — |
| #6037 | 2 | hate | Sexism | — |
| #61 | 待定 | 待定 | 待定 | group, attack_severity |
| #7244 | 待定 | 待定 | Region | attack_severity |
| #7285 | 0 | non-hate | Sexism | — |
| #7646 | 待定 | 待定 | others | attack_severity |
| #7960 | 2 | hate | others | — |
