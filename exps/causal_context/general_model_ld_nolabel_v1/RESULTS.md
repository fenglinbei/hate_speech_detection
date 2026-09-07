# Qwen3-8B 去词典类别字段：完整 dev 结果

本轮全部 643 条 dev、六条件、7,716 blocks 已完成。分类取不含 EOS 的答案总 logprob 最高分候选。
四个配对 F1 区间已从封存 raw 重新构建预测并直接重抽样复算；分类点值与 sklearn 一致。

## 分类指标

| 条件 | Hate Accuracy | Hate F1 | Hate Macro-F1 | Group Micro-F1 | Group Macro-F1 | 两项同时正确 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 43.86% | 14.25% | 36.26% | 61.01% | 52.06% | 32.66% |
| CLnew | 46.97% | 22.32% | 41.03% | 65.44% | 59.00% | 37.17% |
| CD | 73.72% | 79.61% | 71.32% | 70.83% | 67.19% | 53.65% |
| CLDnew | 62.36% | 62.31% | 62.36% | 72.24% | 67.98% | 43.70% |
| CLnewNoCat | 47.59% | 24.27% | 42.10% | 58.00% | 49.32% | 37.95% |
| CLDnewNoCat | 67.03% | 69.36% | 66.84% | 73.47% | 68.13% | 50.23% |

## 预注册主要差值

均为“去类别字段 − 保留类别字段”，单位为百分点。95% 区间为逐点描述性配对 bootstrap 区间。

| 对比 | 指标 | 差值 | 95% 区间 |
|---|---|---:|---|
| NoCat_without_D | hate macro_f1 | +1.07 | [-1.17, +3.28] |
| NoCat_without_D | group micro_f1 | -7.44 | [-11.54, -3.47] |
| NoCat_with_D | hate macro_f1 | +4.47 | [+1.79, +7.22] |
| NoCat_with_D | group micro_f1 | +1.23 | [-1.19, +3.55] |

## 解释边界

删除仅作用于词典的显式类别字段，定义及示例答案保持；输入因此缩短，定义仍可能透露类别。
这是已暴露 dev 上的固定候选分类结果，不是自由生成、test 泛化或内部机制识别。辅助读数不替代四个主要分类差值。

逐类与分层结果见 per_label.csv 和 classification_all_strata.json；Gold 见 gold.csv；所有原始辅助计分见两个 auxiliary.csv。

## 结果解读

D 表示冻结的 10 条示例；Lnew 表示新版词典；NoCat 仅删除词典显式类别字段。

1. 无 D 时，删除类别字段的 Hate Macro-F1 差值为 +1.07 pp，区间跨 0；Group Micro-F1 为 −7.44 pp，区间完全低于 0。本设置下，显式类别字段对 Group 分类有帮助。
2. 有 D 时，删除类别字段的 Hate Macro-F1 差值为 +4.47 pp，区间完全高于 0；Group Micro-F1 为 +1.23 pp，区间跨 0。支持有示例条件下 Hate 指标改善，尚不足以确认 Group 改善。
3. 六条件点值中，CD 的 Hate Macro-F1 最高（71.32%），CLDnewNoCat 的 Group Micro-F1 最高（73.47%）。删除类别后的 CLDnewNoCat 在 Hate 上仍低于 CD 约 4.48 pp，所以仅删除显式类别字段尚未消除加入词典时相对 CD 的全部点值差距。这是描述性比较，未新增区间或正式检验。
4. 两种 D 设置下的效果不同，但本轮没有预注册交互作用区间，不能由两组区间是否跨 0 直接断言交互作用成立。四个区间是逐点、未作多重比较校正的描述性区间。

## 完成与独立核验

- 18/18 预检通过；7,716/7,716 评分块、131,172/131,172 候选完整封存。
- 从原始候选分数重新提取预测，分类点值与 sklearn 核对一致。
- 直接按 query 重抽样 10,000 次（seed 42），独立复算全部四个主要 F1 区间，与流水线结果在 1e-12 精度内一致。
- plan、raw、analysis、Gold 来源与报告文件哈希校验通过；评分期间未加载 query Gold，未读取 test 内容。
- run 状态 complete，guard 状态 released，remaining_processes 为空。

复算回执：[report_manifest.json](results/nolabel-01/report_manifest.json)。CSV、分层和逐类指标位于 [结果目录](results/nolabel-01/)。
