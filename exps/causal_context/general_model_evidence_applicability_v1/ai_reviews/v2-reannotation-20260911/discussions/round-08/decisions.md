# 第八轮裁决汇总

13 条疑点全部收到并保存；其中用户直接 hate 3 条、group 8 条、严重度 13 条。其余 17 条数值仍为 AI 草稿。当前 hate 依暂行映射生成，不冒充额外人工裁决。

| 案例 | 用户回复 | 用户直接裁定字段 | 当前严重度／hate／group |
| --- | --- | --- | --- |
| #986 | 2 级：作者也认可了群体污名 | attack_severity | 2／hate／Region |
| #996 | 0 级：描述社会处境，没有主观攻击 | attack_severity | 0／non-hate／Racism、Region、Sexism |
| #1160 | 0 / [] | attack_severity, group | 0／non-hate／[] |
| #1292 | 0／Racism：仅提及，没有明确攻击 | attack_severity, group | 0／non-hate／Racism |
| #2011 | hate / 2 /  种族 | hate, attack_severity, group | 2／hate／Racism |
| #2297 | hate / 3 / LGBTQ | hate, attack_severity, group | 3／hate／LGBTQ |
| #3898 | hate / 3 / Racism、Sexism | hate, attack_severity, group | 3／hate／Racism、Sexism |
| #4080 | 0 级：自嘲处境，没有认可身份贬低 | attack_severity | 0／non-hate／Racism、Region |
| #4189 | 0／Sexism | attack_severity, group | 0／non-hate／Sexism |
| #5423 | 2 / Racism | attack_severity, group | 2／hate／Racism |
| #61 | 2 / Sexism 基本可判定为生殖器污名 | attack_severity, group | 2／hate／Sexism |
| #7244 | 1 级：借称呼轻度调侃地域身份 | attack_severity | 1／hate／Region |
| #7646 | 1 级：轻度个人贬称 | attack_severity | 1／hate／others |

分数单独回答的 #986、#996、#4080、#7244、#7646，其 group 仍为 AI 来源。#2011、#2297、#3898 的 hate 是明确给出的字段，其余 hate 为规则派生。

一致性复核得到明确新规则：“如果同时出现 种族+个人 那就优先仅种族”。这改变个人辱骂与 Racism 的合并优先级，不改严重度。本轮 #1746 的 AI 当前类别已由 Racism、Sexism、others 更新为 Racism、Sexism，原 AI 标签仍保留；未重写旧人工 group。

[用户原回复](consistency-reply-01.json) · [规则确认](../../policies/racism-personal-priority-v1.json) · [历史相关案例复核](../../group-priority-v1/README.md)

[30 条当前结果](review.md) · [本轮采用的偏好快照](preference-application.json) · [更新后的偏好表](../../preferences/preferences.md) · [最终校验](validation-after-decisions-01.json)
