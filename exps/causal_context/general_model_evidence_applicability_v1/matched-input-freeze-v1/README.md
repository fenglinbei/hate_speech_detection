# 两批输入材料与评分方案冻结

**2026-09-13 后续执行已完成：** [36 条件的四口径结果与解读](../matched-input-results-v1/README.md)。全部评分及数值检查通过，4 张 GPU 已释放。本目录 `current.json` 和以下未运行说明保留冻结时的身份；完成状态由独立结果指针记录。

**两批材料均已获得用户整体认可，正式输入与分析方案已冻结。** 覆盖 541/hate、3169/hate 的 14 段新文本、36 个完整提示、72 个候选答案和 64 项预定比较。本轮完成 CPU 校验，尚未启动本轮 GPU 评分。

[冻结说明](../reviews/analysis-freeze-20260912/matched-input-freeze-v1/frozen-01/FREEZE.md)；[比较公式 CSV](../reviews/analysis-freeze-20260912/matched-input-freeze-v1/frozen-01/comparisons.csv)；[材料接受记录](../reviews/analysis-freeze-20260912/matched-input-freeze-v1/frozen-01/accepted_materials.jsonl)；[当前指针](current.json)；[验证回执](validation.json)。

## 接受范围与来源

第一批回复“第一批没什么问题，还有下一批吗？”绑定 8 段正文；第二批回复“第二批也没有问题”绑定另 6 段正文。[两次反馈](../matched-materials-review-v1/README.md)分别保留原话、正文哈希和材料包版本。认可用于已经说明的输入对照；原 AI 标签、严重度和解释保留作者来源，没有补造逐项独立人审或机制判断。

两批准备文件及其历史 `current.json` 保持原字节。当前正式输入身份以本目录指针为准，原稿中当时的“待审核”字段保留历史含义。原 64 条案例用途与分析参考、已完成的 26 条件自然编辑评测均保留。

## 固定比较

| 条件 | 数量 | 用途 |
| --- | ---: | --- |
| R-A / R-B | 4 | 原始与删除输入，绑定此前模型分数供重放 |
| R-C1 / R-C2 / R-D1 / R-D2 | 8 | 原位等长内容替换、语义保持改写 |
| P0 / P1 × A / C1 / C2 / D1 / D2 | 20 | 共同配对背景中的顺序交换 |
| S / L | 4 | 目标示例的短版与长版 |

64 项比较包括原第二批的 38 项，以及补齐的第一批 26 项；60 项为两条件差值，4 项为四条件交互。两个案例共享多个条件，不能将这些比较计作独立案例。

主口径沿用答案 token 对数概率之和，同时保留答案平均分和两种含 EOS 口径。连续量固定为 `score(non-hate) − score(hate)`。两例原参考和审核参考均为 non-hate，两套来源分别保存，原始评分封存且数值检查全部通过后才载入分析。

`epsilon = 0.0013427734375` 保持不变；两条件差值的数值未决范围为 ±0.002685546875，四条件交互为 ±0.00537109375。这些范围不是统计置信区间。无效、反向、措辞不一致和评分口径不一致的结果全部保留。

位置比较使用 P1−P0，配对文本改写另用 P0−R 衡量。交换同时移动两条示例；短长版本同时改变措辞、密度和后续 token 位置，因此不能分别归因为纯绝对位置或纯长度。

## 校验与接续

5 项隔离测试、36 个文本与 token 重放、72 个候选边界、逐字节重建及独立核验均通过。独立核验覆盖 12 个冻结文件与 104 个绑定来源，并确认新执行循环除实验名称和数量外与已验证循环一致；4 个旧条件的 16 条读数及 8 条比较通过分析兼容性检查，没有产生新模型结果。

```bash
.conda/stage1-p0/bin/python scripts/review/freeze_evidence_matched_inputs.py --check
.conda/stage1-p0/bin/python scripts/review/run_evidence_matched_inputs.py validate
```

下一阶段先重放 4 个历史条件，再评分全部 36 条件、72 候选，并完成重复、填充、前缀、候选顺序和 GPU 副本检查。沿用 4 个 float32 副本及原评分内核；8 个 pass 共 448 次候选评估，前缀检查内部可能包含更多 forward。执行入口已实现：

```bash
.conda/stage1-p0/bin/python scripts/review/run_evidence_matched_inputs.py run
```

默认读取本次 `frozen-01`，运行记录写入独立 `run-01`，全部检查通过后将 144 条条件读数和 256 条比较读数写入 `results-01`。冻结、已完成运行与结果目录拒绝覆盖；运行后可用 `check` 复核。材料变更或实现变更需保存新版本。当前没有激活采集或 patching 准备完成的结论。
