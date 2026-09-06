# Qwen3-8B dev 分类指标补充

日期：2026-09-06（Asia/Shanghai）。来源：已完成的 `coverage-01`，全部 643 条 dev，8 个条件、2 个任务，共 10,288 个条件记录。

这是应用户要求新增的事后描述性分类汇总，不属于原注册主端点，也不沿用原主端点的独立审计通过声明。没有新增置信区间、模型运行、阈值选择或 test 评估。原封存报告、分析及分数未修改。

## 计算口径

- 以答案 token 总 logprob（`answer_sum`，不含 EOS）取最高分候选。所有记录的最高分均无并列。
- hate 在两个规范候选中选择；group 在五个标签的 32 个规范集合中选择。两个任务独立预测，不根据 hate 强制清空 group。
- Hate F1 以 hate 为正类；Hate Macro-F1 平均 hate/non-hate 两类。Gold 分布为 hate 390 条、non-hate 253 条。
- Group Micro-F1 汇总五类 TP/FP/FN；Macro-F1 等权平均五类 F1，零分母按 0 处理。集合完全匹配要求全部标签一致。
- Group Sample-F1 沿用项目口径：gold 和预测均为空集合时记 1。它与只统计正标签的 Micro-F1 不同。
- “两项同时正确”要求同一 query 的 hate 与完整 group 集合都正确，不是四元组抽取 F1，也不评价字段绑定。
- 所有分类指标均调用项目 `evaluate_predictions`；Accuracy/集合完全匹配、Micro-F1、Macro-F1 与 sklearn 交叉核验一致。
- 固定候选选择不等于自由生成；本次不测格式错误、抽取内容或端到端生成性能。

## 分类结果

下表单位为百分比。Lq 为当前 query 命中的词条；Lnew 为 query 与固定检索示例命中词条的去重并集；D 为固定的 10 条示例。PLnew、PD 为原协议的中性形态/近似长度对照。

| 条件 | Hate Accuracy | Hate F1 | Hate Macro-F1 | Group Micro-F1 | Group Macro-F1 | Group 完全匹配 | 两项同时正确 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 43.86 | 14.25 | 36.26 | 61.01 | 52.06 | 59.10 | 32.66 |
| CLq | 48.99 | 29.00 | 44.60 | 63.33 | 56.73 | 61.43 | 34.53 |
| CLnew | 46.97 | 22.32 | 41.03 | 65.44 | 59.00 | 64.54 | 37.17 |
| CD | 73.72 | 79.61 | 71.32 | 70.83 | 67.19 | 63.14 | 53.65 |
| CLqD | 70.92 | 77.17 | 68.56 | 72.26 | 67.83 | 63.30 | 52.72 |
| CLDnew | 62.36 | 62.31 | 62.36 | 72.24 | 67.98 | 64.07 | 43.70 |
| PLnew | 39.81 | 1.53 | 29.09 | 29.86 | 28.98 | 48.21 | 37.95 |
| PD | 47.28 | 25.17 | 42.23 | 55.72 | 45.88 | 51.48 | 24.42 |

D 的 hate Precision/Recall 为 75.17%/84.62%；Lnew+D 为 79.37%/51.28%。合并词典后 hate F1 的下降主要对应召回下降。D 的两项同时正确数为 345/643，Lq+D 为 339/643，Lnew+D 为 281/643。

Group 的两种词典+D 组合相较 D，Micro-F1 均提高约 1.4 个百分点；彼此点估计接近，没有据此证明稳定优势。Group 完全匹配最高为仅 Lnew（415/643），与 Lnew+D（412/643）仅差 3 条。没有一个组合在所有指标上都最好。

## Gold 概率辅助结果

NLL 越低越好；候选空间内的 Gold 质量不是经过校准的真实标签概率，也不等于分类准确率。

| 条件 | Hate Gold NLL | Group Gold NLL |
|---|---:|---:|
| C0 | 2.871 | 5.252 |
| CLq | 2.177 | 4.723 |
| CLnew | 3.495 | 4.851 |
| CD | 0.779 | 4.745 |
| CLqD | 0.814 | 5.033 |
| CLDnew | 1.017 | 4.383 |
| PLnew | 4.616 | 9.145 |
| PD | 2.537 | 6.102 |

Hate 的 D 在分类与 NLL 上均最好；Group 的 NLL 最优为 Lnew+D，但其完全匹配率并非最高。

## 文件与复算

- [summary.csv](summary.csv)：分类指标，比例以 0–1 保存，另含 hate Precision/Recall、Group Sample-F1 和联合正确条数。
- [metrics.json](metrics.json)：完整逐类 TP/FP/FN、Precision/Recall/F1、口径、来源 SHA-256 及 sklearn 核验状态。
- [gold_summary.csv](gold_summary.csv)：Top-1 正确数、百分比、平均 Gold NLL 和平均 Gold 质量。
- [复算脚本](../../audits/posthoc_classification_metrics.py)：只读已有 dev 分数及 Gold，CPU 计算；复算输出写入指定目录。

在仓库根目录执行：

```bash
.conda/stage1-p0/bin/python exps/causal_context/general_model_ld_coverage_v1/audits/posthoc_classification_metrics.py --output /tmp/qwen3-8b-classification-replay
```

脚本依赖项目 Python 环境及 sklearn。复算需要本地保留的计划、package、Gold、raw 分数及清单；这些实验运行产物按用户要求未上传 Git。脚本核验计划、Gold package 和 raw 分数哈希，不重新启动 GPU。迁移数据时需保持仓库内相对目录结构。
