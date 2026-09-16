# v2补标第 1 批：疑点已处理

本批 30 项，其中 0 项有待讨论疑点；0 项仍有 null 标签。已登记用户对 15 项的 hate/group 裁决。尚未导入线上，没有新增线上整项确认。

已参考用户允许使用的既有人工裁决，来源身份见独立 reference 清单。原答案对照建议在本批标签写完后另行形成，不代表已经确认源数据错误。

[完整 JSON](ai_annotations.json) · [CSV](ai_annotations.csv)

## 本次用户裁决

仅 hate/group 标签及下列用户理由来自明确裁决；辅助选项、证据高亮和原答案对照状态仍为 AI 整理，没有代替完整材料确认。

| 对象 | hate | group | 用户理由／采纳范围 |
| --- | --- | --- | --- |
| demo:1240 | non-hate | [] | 采纳建议：轻度个人贬称，不足以判 hate；无群体身份对象。 |
| demo:179 | hate | ["others"] | 采纳建议：对贴吧社群成员作性污名泛化。 |
| demo:1158 | hate | ["Sexism"] | 无论是上下文还是用词，都没有任何显然证据表明此处讨论的是”女性拳击“ |
| demo:100 | hate | ["Sexism"] | ”国内女权的德行“即带有贬损义，虽然程度不高，但足以纳入 |
| demo:1180 | non-hate | ["LGBTQ"] | 没有明显攻击意 |
| demo:1456 | non-hate | [] | 广东仅作地名 |
| demo:1509 | hate | ["Sexism"] | 虽老g意义暂时不明，但是没有证据显示这句话是对eg的非污名化讨论，可以检索一下老g的含义 |
| demo:1517 | hate | ["LGBTQ", "others"] | 提到同性恋的同时包含对个人的辱骂 |
| demo:159 | non-hate | ["Region"] | 没证据表明存在地域歧视 |
| demo:1729 | hate | ["Sexism"] | 白楠-白男 yy-意淫 lkd-烂裤裆，用于辱骂女性私生活不检点 |
| demo:1815 | hate | ["Sexism"] | 没有证据表明讨论的是男性生殖器以外的事物 |
| demo:1859 | non-hate | ["Sexism"] | 仅为话题讨论，不包含攻击 |
| demo:1915 | hate | ["Racism"] | BBC在此处没有任何证据表明指向英国广播公司以及其他专有名词，因此此处只能是对黑人群体的侮辱代称，直接导致仇恨+种族 |
| demo:2004 | hate | ["Region", "others"] | **并非被屏蔽，只是一种让人联想的写法，参考后面语句，没有证据表明此处不是辱骂意 |
| demo:1224 | hate | ["Sexism"] | 仅 Sexism：前半句只作行为选择的类比，不另标身份类别 |

[用户裁决记录 1](../../discussions/batch-01/user-decisions-01.json)

[用户裁决记录 2](../../discussions/batch-01/user-decisions-02.json)

[补充核查与校准说明](../../discussions/batch-01/followup.md)

## 本批待讨论项

本批原有疑点已处理。未被用户明确裁定的其他记录仍是 AI 草稿。

