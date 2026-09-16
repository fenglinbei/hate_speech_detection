# 当前严重度到 hate 的默认映射

用户已接受暂行映射：0→non-hate，1–4→hate，null→null。仅活动分值参与映射；旧标签及裁决保留各自政策。

共 210 条：1 条与现有标签层不同，14 条因分数待定无法映射。规则生成的 hate 不新增用户直接裁决。

[JSON](ai_annotations.json) · [CSV](ai_annotations.csv)

## 与现有标签层的差异

- demo:1240：旧值 non-hate（user_discussion）；分数 1 → 当前 hate（rule_mapping_from_ai_score）。
- demo:1509：旧值 hate（user_discussion）；分数 None → 当前 None（unresolved_severity）。
- demo:2004：旧值 hate（user_discussion）；分数 None → 当前 None（unresolved_severity）。
- demo:986：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- demo:996：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:1160：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:1292：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:2011：旧值 hate（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:3898：旧值 hate（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:4080：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:4189：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:5423：旧值 hate（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:61：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:7244：旧值 None（ai_note）；分数 None → 当前 None（unresolved_severity）。
- query:7646：旧值 hate（ai_note）；分数 None → 当前 None（unresolved_severity）。

#4392 的 non-hate 草稿已由用户明确改为 hate，裁决前稿另存于第四批归档；此处与其最新明确裁决对照。其他差异只来自默认映射，没有重写旧裁决。
