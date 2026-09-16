# 原／审核双参考评估

同一冻结 discovery 子集、同一历史预测，只切换参考标签。本次为 CPU 描述性分析，没有模型 forward。

| 任务 | 查询总数 | 已确认 | 草稿 | 未审 | 未决 | 两套参考共同分母 | 排除 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hate | 32 | 32 | 0 | 0 | 0 | 32 | 0 |
| group | 32 | 32 | 0 | 0 | 0 | 32 | 0 |

## 主分类点值

| 任务/指标 | 条件 | 原参考 | 审核参考 |
| --- | --- | ---: | ---: |
| hate/macro_f1 | C0 | 37.64% | 49.09% |
| hate/macro_f1 | CLnew | 28.74% | 38.16% |
| hate/macro_f1 | CD | 60.00% | 65.32% |
| hate/macro_f1 | CLDnew | 37.50% | 32.73% |
| hate/macro_f1 | CLnewNoCat | 37.64% | 41.82% |
| hate/macro_f1 | CLDnewNoCat | 51.95% | 43.75% |
| group/micro_f1 | C0 | 60.00% | 59.65% |
| group/micro_f1 | CLnew | 71.70% | 70.00% |
| group/micro_f1 | CD | 61.29% | 57.97% |
| group/micro_f1 | CLDnew | 75.76% | 73.97% |
| group/micro_f1 | CLnewNoCat | 60.38% | 56.67% |
| group/micro_f1 | CLDnewNoCat | 74.19% | 66.67% |

完整的 exact accuracy、两类 F1 及逐标签计数见 metrics.csv 与 per_label_metrics.csv。

## 切换参考后的正确性变化（预测固定）

| 任务 | 条件 | 错→对 | 对→错 | 不变 |
| --- | --- | ---: | ---: | ---: |
| hate | C0 | 7 | 2 | 23 |
| hate | CLnew | 7 | 2 | 23 |
| hate | CD | 5 | 4 | 23 |
| hate | CLDnew | 4 | 5 | 23 |
| hate | CLnewNoCat | 6 | 3 | 23 |
| hate | CLDnewNoCat | 3 | 6 | 23 |
| group | C0 | 5 | 5 | 22 |
| group | CLnew | 5 | 7 | 20 |
| group | CD | 4 | 6 | 22 |
| group | CLDnew | 6 | 8 | 18 |
| group | CLnewNoCat | 4 | 5 | 23 |
| group | CLDnewNoCat | 4 | 9 | 19 |

## 条件转换（同一参考内）

| 任务 | 对比 | 参考 | 错→对 | 对→错 | 不变 |
| --- | --- | --- | ---: | ---: | ---: |
| hate | remove_with_D | original | 9 | 4 | 19 |
| hate | remove_with_D | reviewed | 8 | 5 | 19 |
| hate | remove_without_D | original | 2 | 0 | 30 |
| hate | remove_without_D | reviewed | 1 | 1 | 30 |
| hate | S_given_D | original | 6 | 9 | 17 |
| hate | S_given_D | reviewed | 4 | 11 | 17 |
| group | remove_with_D | original | 2 | 2 | 28 |
| group | remove_with_D | reviewed | 0 | 3 | 29 |
| group | remove_without_D | original | 1 | 6 | 25 |
| group | remove_without_D | reviewed | 0 | 4 | 28 |
| group | S_given_D | original | 6 | 3 | 23 |
| group | S_given_D | reviewed | 3 | 3 | 26 |

## 位型变化

| 任务 | 纳入数 | 位型变化数 |
| --- | ---: | ---: |
| hate | 32 | 9 |
| group | 32 | 15 |

四位次序固定为 C0、CLnewNoCat、CD、CLDnewNoCat；所有 16 种位型和 16×16 转换（含零计数）均导出。

## 连续读数与解释范围

已校验完整候选，按审核参考重新求 best-nongold margin；group 使用全部 32 候选。

计分口径固定为不含 EOS 的 answer_sum；精确并列取最小 canonical ordinal，计分口径敏感和原 epsilon 近并列标记保留。

原始 643 条总体表不改写。本子集按模型行为富集，未新增 bootstrap 或显著性检验，不估计总体误标率。

案例原标注状态、基础政策与材料补充政策分别保留。标签差异不自动证明源标注错误；资源删除还改变长度/位置。

case_masks 的模式仅描述行为变化，不是人工机制解释或输入实验放行。具体干预、假设、替代解释仍需单独冻结。
