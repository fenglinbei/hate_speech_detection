# 攻击严重度试标

用户已接受 0–4 级攻击严重度及 null 待定方案，先完成已讨论材料的校准，再用于后续补标。确认原文及规则身份见 [接受记录](../policies/attack-severity-v1.json)，当前阶段见 [severity_context.json](../severity_context.json)。

首轮 [pilot-01](pilot-01/discussion.md) 共 30 条：25 条已有明确的用户 hate/group 裁决，另 5 条的 hate/group 为此前 AI 草稿。两种来源在参照字段中分别保留；它们都没有因规则接受或本次试标自动获得人工严重度分数。

首轮分布：0 级 7 条、1 级 8 条、2 级 9 条、3 级 3 条、4 级 1 条；另 2 条证据不足，活动分值为 null。其中 4 个分数已由用户逐项裁决，24 个数值仍为 AI 草稿。分布仅反映有目的选取的校准样本，不代表全部任务的严重度分布或标注可靠性。

- 用户已裁决：#309＝1、#2750＝3、#2180＝2、#2800＝3。原回复与分数字段单独见 [裁决记录](pilot-01/discussions/user-decisions-01.json)，[校准回执](pilot-01/calibration-summary.json)固定本阶段来源。接受这些分数不自动确认其 hate/group 或其他样本。
- 因证据不足暂留 null：#1509 的老g词义未明；#2004 的主动暗示写法 ** 无唯一强度含义，不能据此补猜具体脏词。
- #1240 的历史标签保留 non-hate、AI 分数保留 1；用户后续接受默认映射后，当前派生结果为 hate，来源仍为 AI 分数＋规则。
- #2598 对资本的抽象比喻取 0；#1517 的反仇恨立场与其独立个人辱骂分别考虑，取 2。

`pilot-01/inputs.jsonl` 是选中样本的正文及标签参照快照；`selection.json` 固定授权范围、版本与来源哈希；`annotations/pilot-01.json` 保存逐例整理的当前值。用户分数通过独立裁决文件确定来源，其他值保持 AI 来源；裁决前稿已归档。

导出命令：

```bash
python3 -B -S scripts/review/export_attack_severity_pilot.py --pilot 1
```

导出会验证规则及输入哈希、对象覆盖、分数与待定状态、候选值、原文引文和来源字段，生成 [JSON](pilot-01/ai_annotations.json)、[CSV](pilot-01/ai_annotations.csv) 与完整讨论材料。用户逐条分数裁决显式绑定政策、正文和对象版本；重复裁决需单独更正，不会被重导出降为 AI 草稿。普通标签批次使用 `--batch 4` 等参数导出同批严重度。

首轮讨论已完成，[第 4 批严重度](batch-04/discussion.md)已保存 8 条用户数值和 22 条 AI 数值，没有 null。用户最新 group 与严重度同步保存，严重度层的 `prior_label_reference` 保留原始标签快照；合并视图展示当前标签并标记两者差异。#4392 已由用户明确改为 hate，第四批疑点已处理。第五批 7 个疑点已由用户明确裁定，现有 7 条用户数值、23 条 AI 数值，全部已定级。第六批 6 个疑点也已逐条裁定，现有 6 条用户数值、24 条 AI 数值，没有本批待定项；本轮收尾于第六批。`severity_context.json` 保留最初接受规则时的阶段及身份；实际阶段进展按各次裁决与本目录记录判断。没有写入线上人审会话或实施线上规则迁移。


用户进一步回复“改为hate，且暂时先用默认非0严重度即是hate的映射”，已在[独立接受记录](../policies/severity-hate-default-mapping-v1.json)中保存。从现有分数及后续批次生成[当前 hate 映射层](hate-mapping-v1/README.md)：0→non-hate、1–4→hate、null→null。原分级规则及首轮校准回执保持原字节；新约定替代旧版中独立生成 hate 的条款，不反向调整分数或改变 group。

现在共 120 条严重度记录：25 条用户明确数值、93 条 AI 数值、2 条 null。映射不把用户分数再计成一次直接 hate 裁决。所有当前结果均保留分数来源、规则身份和历史标签差异；合并 JSON 的 `records[].values` 与 CSV 展示当前结果。

```bash
python3 -B -S scripts/review/export_attack_severity_pilot.py --batch 6
python3 -B -S scripts/review/export_v2_reannotation_review.py --batch 6
python3 -B -S scripts/review/export_severity_hate_mapping.py --pilot 1 --batch 4 --batch 5 --batch 6
```

第五批新增锚点：#4810＝2、#5113＝0、#528＝3、#5365＝0、#5394＝3、#5416＝2、#5615＝2，分别来自明确回复；其理由及使用边界见[第五批校准回执](batch-05/calibration-summary.json)。第六批新增 #5865＝0、#5900＝0、#5919＝1、#5998＝1、#6074＝0、#6263＝0；原回复见[分数裁决](batch-06/discussions/user-decisions-01.json)，[校准回执](batch-06/calibration-summary.json)保存最终引用。只有 #5865 与 #5998 同时获得明确 group 裁决，其余类别沿用 AI 草稿。
