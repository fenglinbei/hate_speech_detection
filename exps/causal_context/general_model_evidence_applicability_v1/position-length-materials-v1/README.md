# 第二批：位置交换与短／长材料

**后续状态：用户已回复“第二批也没有问题”。** [整体接受记录](../matched-materials-review-v1/batch-02-feedback-01.json)绑定本批 6 段正文；两批材料已合并形成[正式输入与评分方案冻结](../matched-input-freeze-v1/README.md)。本目录准备指针与以下发布时的“待审核”说明保留历史含义，本轮尚未运行 GPU。

**已准备 6 段新文本，输入检查通过，待核对语义。** [打开逐项材料](../reviews/analysis-freeze-20260912/position-length-materials-v1/prepared-01/MATERIALS.md)；[CSV 审核表](../reviews/analysis-freeze-20260912/position-length-materials-v1/prepared-01/materials.csv)；[当前指针](current.json)；[验证记录](validation.json)。

用户对第一批的“没什么问题”已作为[整体反馈](../matched-materials-review-v1/batch-01-feedback-01.json)独立保存，绑定原 8 段正文与材料包哈希。第一批准备文件和其中的逐字段来源保留；这次反馈不自动采纳第二批的新文本。

第二批继续使用 541/hate、3169/hate，每例新增一条位置交换用配对文本 P、目标示例短版 S 和长版 L。每条都有原文、查询、AI 标签及严重度、改写依据、实际 token 数和待核对字段。

| 案例 | P：交换用配对材料 | S/L：长短改写对象 | 交换位置 |
| --- | --- | --- | --- |
| 541/hate | 扩展原示例 3156 的中性个人经历 | 示例 826，保持反对由个人行为泛化到群体 | 第 5 ↔ 第 9 位 |
| 3169/hate | 原示例 6493 正文保留，句尾补省略标点 | 示例 3660，保持原指称、代称和贬损暗示 | 第 3 ↔ 第 2 位 |

## 位置实验

现有 20 条示例中没有满足等长、同标签且原／审核答案一致的直接配对。筛选记录见 [partner_screening.jsonl](../reviews/analysis-freeze-20260912/position-length-materials-v1/prepared-01/partner_screening.jsonl)。先将配对文本改到与目标完整块等长，再比较同一共同背景中的两种顺序：

- P0：目标和配对示例保持原来的顺序。
- P1：交换两条示例正文和各自答案，编号留在原槽位。

两种顺序的全部槽位边界、答案位置、各槽位标签、查询位置和总 token 数相同；两条正文之外的 token 完全一致。第一批的 A、C1、C2、D1、D2 都使用相同配对与相同交换，共 20 个位置条件。

配对文本改写本身的影响也另设比较：P0 减去第一批对应版本。这样可以把共同背景改变与 P1−P0 的顺序变化分别报告。541 的共同背景为 930 tokens，比第一批多 12；3169 为 743 tokens，比第一批多 1。

交换同时移动两条示例，结果描述二者的相对顺序与距离变化。3169 仅覆盖相邻交换，两例的交换距离也不同；这些结果不能单独证明目标示例的绝对位置效应。

## 长短实验

短／长版只改目标示例，保留原词典和其他示例，形成 4 个新条件。

| 案例 | 短版 S | 原始 A | 长版 L |
| --- | ---: | ---: | ---: |
| 541/hate | 907 tokens | 918 tokens | 938 tokens |
| 3169/hate | 736 tokens | 742 tokens | 749 tokens |

表中均为完整提示长度。答案内容不变，后续示例与查询的位置按长度差移动，位移已经核对。长短版本也改变措辞、重复和信息密度，因此只能检验长度与表达方式的敏感性，不能直接归因为纯长度。

## 产物与验证

本包共 36 个完整提示：12 个第一批参考、20 个位置条件、4 个长度条件；新增提示为 24 个。[预列比较表](../reviews/analysis-freeze-20260912/position-length-materials-v1/prepared-01/contrasts.jsonl)含 38 项位置、背景、语义及长短比较，尚未评分。所有分词检查过的 7 条草稿均保留，没有用新模型分数选材料。

5 项边界测试通过；12 个第一批提示逐字与逐 token 重放、10 对位置检查、10 对背景改写、4 项长度位移以及 72 个答案边界通过，输出逐字节复算一致。独立解析另核对交换正文、答案、未修改示例及全部 63 个绑定来源的哈希。

审核重点：文字是否自然，是否保持原对象、命题、立场和词义，标签与严重度是否适当。修改需保存新版本并重验 token。两批材料审核后统一冻结正式评分协议；当前未运行 GPU，未接入线上标注页面，也未开展激活采集或 patching。

```bash
.conda/stage1-p0/bin/python -m unittest scripts.review.test_prepare_evidence_position_length_materials -v
.conda/stage1-p0/bin/python scripts/review/prepare_evidence_position_length_materials.py --output exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/position-length-materials-v1/prepared-01 --check
```

本批指针独立于第一批、原分析参考、案例用途与已完成评测。准备输出目录拒绝覆盖。
