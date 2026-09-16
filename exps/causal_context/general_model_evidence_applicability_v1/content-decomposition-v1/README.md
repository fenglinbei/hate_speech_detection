# 内容拆分：GPU 前交付

当前状态：**材料、冻结输入和执行／分析实现已完成，CPU 验收通过，尚未运行 GPU。** 用户通知窗口后即可执行以下冻结计划。此前设计草案保留在 `design-draft-01`，现在以 [冻结入口](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/content-decomposition-v1/current.json) 为准。

[16 条完整措辞及 AI 核对说明](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/MATERIALS.md) · [逐条核对字段及来源](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/materials.json) · [GPU 输入](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/contexts.jsonl) · [运行计划](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/plan.json) · [比较系数](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/comparisons.jsonl)

## 这批输入回答什么

- **541/hate**：第 5 槽示例 826，篮球迷／异性恋主题 × 中性续文／明确反泛化规则，两套措辞，共 8 个条件。展示答案均为 non-hate。每套四格匹配人物、事件、两次类别提及；第二套规则避开“讨厌／群体”的查询措辞。每套有两项条件规则差、两项条件主题差和一项交互，共 10 项主比较。
- **3169/hate**：第 3 槽示例 3660，群体代称中的“嘿嘿／黑人”、普通笑声中的“嘿嘿／哈哈”，各两套措辞，共 8 个条件、4 项主比较。展示答案均为 hate。两种语境都有攻击；分别比较词形差，不估计攻击有无或跨两组的纯四格交互。

新正文共有 16 条。核对依据是既有政策／偏好快照，包括严重度 0→non-hate、1–4→hate 的当前映射，以及原 3660 和个人辱骂的案例锚点。新正文的具体 hate、严重度、自然性和因素判断由助手核对；用户授权连续完成本批准备和实现，**没有据此填写新的逐句人工裁决**。具体对象、命题、立场、局限与空白人工字段都保留在材料文件中。过去两批的接受只覆盖过去的文本。

查询、完整词典、其他九条示例、示例答案和顺序均取原 R 背景。只编辑目标示例正文；不重新检索，也不重算词典并集。

## 长度、位置与桥接

541 的 8 条新正文在原标签实查询下均为 **918 token**，保持原提示长度。3169 的 8 条均为 **743 token**，原提示为 742；群体配对两边共同加“们”，普通笑声使用明确个人辱骂的完整句子。与旧输入的共同复数化、措辞和位置差异进入桥接，不能从桥接差值中单独认定长度或某个字的作用。

所有 **14 项主比较 × 8 种编码／探针变体 = 112 项检查**通过：完整长度、目标正文／答案、其他示例、查询和生成起点的位置相同，目标正文外的 token 序列相同。内部语义 token 没有一一对齐。A/B 正反映射的位置一致；与原标签之间的说明及答案长度变化另存 [编码桥接](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/encoding-bridges.csv)。

[总输入桥接位置记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/input-bridges.jsonl) · [主比较匹配证明](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/matching-proofs.jsonl) · [完整位置表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-01/geometry.jsonl)

| 项目 | 数量 |
|---|---:|
| 新正文条件 | 16 |
| 历史锚点：每例 R-A、R-C1/C2、R-D1/D2 | 10 |
| 内容条件合计 | 26 |
| 原标签实查询／逐条件五探针／A/B 正反实查询 | 26／130／52 |
| 完整提示／候选 | 208／416 |
| 历史提示重放：锚点的全部编码及探针 | 80 |
| 主比较／总输入桥接／历史 D−C 对照 | 14／12／4 |
| GPU 评分轮次／累计候选评估 | 8／2816 |

这些条件和比较来自两个已暴露的发现案例，不是独立样本数。12 项桥接包括每套新规则／代称正文相对原始 A 和旧 D，以及新中性／笑声正文相对旧 C；4 项历史 D−C 单列复现。

## 评分与执行

沿用上轮的 Qwen3-8B、4 张卡（物理索引 0/1/2/3）的 FP32 副本、batch size 1、完整标签候选和全部数值约束。既有 FP32 内核与二分类适配器逐字不变，新实现通过独立入口使用它们。

继续并列：原标签总分及平均分（含 EOS 辅助口径）、逐条件五探针 NCC、A/B 正反映射。五探针仍为精确空串、单空格、`N/A`、`[MASK]`、`Lorem ipsum`；本批每个条件都有自己的探针提示并重新评分。NCC 导出原均分、背景和残差，保留五个单探针与五个留一结果。两种 A/B 映射分别报告。

基础 epsilon 仍为 **0.0013427734375**；两项差值的原始界为 2 epsilon、NCC 界为 4 epsilon，四项交互分别翻倍。八轮执行依次完成历史参考／重复、全量参考／重复、填充、前缀、候选顺序与物理副本变化检查。10 项原始数值门槛与 6 项派生门槛（每项 672 个读数）全部通过后封存。分析入口随后才解析原／审核查询参考。阈值、探针、映射不按结果选择；反向和未决结果完整导出。

**下面的 `run` 命令会加载模型，等待 GPU 窗口后再执行。本次交付未执行该命令。**

```bash
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/run_evidence_content_decomposition.py run
```

默认使用已冻结的 `frozen-01`，结果写入独立 `run-01`。中断后同命令可复用已核验的 SQLite 检查点；完整成功／失败终态均拒绝新 forward，保留原输出。数值失败应检查原因并采用新运行版本，不能放宽本计划门槛。

评分完成后的只读核验和分析：

```bash
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/run_evidence_content_decomposition.py check
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/run_evidence_content_decomposition.py analyze
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/analyze_evidence_content_decomposition.py --check
```

分析写入独立 `results-01`：分条件原分数与预测、NCC 背景／残差、全部 30 项比较的 16 种视图、300 条单探针／留一比较、按主比较／桥接／历史对照分组的结果说明。重复分析使用 `--check` 逐字节核验，不覆盖旧结果。

## 已完成的 CPU 验收

**15 项测试通过**：本批 7 项测试及此前 8 项校准回归测试。包括标签／因素与来源约束、完整输入匹配、篡改保护、条件效应与四项交互、反向／无效／映射敏感读数、分卡计划、八轮模拟执行、真实 SQLite 检查点复用、数值门槛、分析封存及结果字节重建。模拟执行使用合成分数；硬件及 GPU 数值验收仍待实际窗口。

冻结字节重建与 CLI `validate` 已通过。另一个核对脚本从旧完整提示独立恢复本批提示，验证 **154 个来源、208 个提示、416 个答案边界、112 项主比较变体和 80 份历史候选原始记录**。

[CPU 验收回执](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/audits/cpu-acceptance.json) · [独立输入核对](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/audits/independent-inputs.json)

以下命令仅在 CPU 重建／校验：

```bash
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/freeze_evidence_content_decomposition.py --check
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/run_evidence_content_decomposition.py validate
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/audit_evidence_content_decomposition.py
```

计划 ID：`evidence-content-decomposition-52df6d9e615ba79d7f48abc2432fac210d1adb1d183caf4d6182b8615648021a`。冻结清单 SHA256：`cf49b35baf350614c77e4e1eef1471a13e9bd1755570acfd62f6515c360cab88`。

既有参考、案例用途、人审源记录、材料／运行指针和设计草案均保留。本批没有启动模型、写入线上标注或建立 activation patching 准备资格；功能性查询对照留在后续独立阶段。
