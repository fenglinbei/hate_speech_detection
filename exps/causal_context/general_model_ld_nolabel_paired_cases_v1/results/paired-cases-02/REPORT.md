# Qwen3-8B NoCat：逐查询配对分析与机制候选集

已完成完整 dev 的 CPU 事后分析；新增统计均为探索性。未新增模型 forward，未访问 test，未修改原始产物。

## 输入与完成状态

643 queries、7,716 blocks、131,172 candidates；输入状态：complete。来源哈希、逐块预测/Gold及使用的读数已核对。原分类表和四个区间复现通过。

运行配置与实际源码快照见 `config.frozen.json`、`manifest.json` 和 `source/`。新增实现尚未单独提交时，以源码 SHA256 为执行版本身份。

## 六条件指标

| 条件 | Hate Macro-F1 (%) | Group Micro-F1 (%) | 两任务同时正确 |
| --- | --- | --- | --- |
| C0 | 36.26 | 61.01 | 210 |
| CLnew | 41.03 | 65.44 | 239 |
| CD | 71.32 | 70.83 | 345 |
| CLDnew | 62.36 | 72.24 | 281 |
| CLnewNoCat | 42.10 | 58.00 | 244 |
| CLDnewNoCat | 66.84 | 73.47 | 323 |

## 双向转换（完整 643 条）

下表 01 为错→对，10 为对→错；group 的对错指集合完全匹配，不等同于 Micro-F1 的逐查询贡献。

| 对比 | 任务 | 00 | 01 纠正 | 10 退化 | 11 |
| --- | --- | --- | --- | --- | --- |
| remove_with_D | hate | 185 | 57 | 27 | 374 |
| remove_with_D | group | 193 | 38 | 18 | 394 |
| remove_without_D | hate | 327 | 14 | 10 | 292 |
| remove_without_D | group | 191 | 37 | 60 | 355 |
| S_given_D | hate | 109 | 60 | 103 | 371 |
| S_given_D | group | 185 | 52 | 26 | 380 |
| S_vs_0 | hate | 317 | 44 | 20 | 262 |
| S_vs_0 | group | 174 | 89 | 77 | 303 |
| D_vs_0 | hate | 60 | 301 | 109 | 173 |
| D_vs_0 | group | 148 | 115 | 89 | 291 |
| D_given_S | hate | 145 | 192 | 67 | 239 |
| D_given_S | group | 129 | 122 | 82 | 310 |

## 四项新增探索性区间

单位为百分点；逐点、描述性、未作多重比较校正。配对查询 bootstrap 10,000 次，PCG64 seed 42，所有任务、条件与端点共享抽样；每次重算 TP/FP/FN 与 F1。

| 端点 | 任务 | 差值 (pp) | 95% 区间 (pp) |
| --- | --- | --- | --- |
| E_S_given_D_F1 | hate | -4.48 | [-8.26, -0.63] |
| E_S_given_D_F1 | group | 2.64 | [0.12, 5.19] |
| J_remove_by_D_F1 | hate | 3.41 | [-0.13, 7.04] |
| J_remove_by_D_F1 | group | 8.67 | [4.67, 12.82] |

E_S_given_D_F1 = F(SD) − F(D)。J_remove_by_D_F1 = [F(SD) − F(SGD)] − [F(S) − F(SG)]。

J 是聚合 F1 尺度的差中差；逐查询 gold_margin 的 I_S_D 另见 paired_readouts.csv，两者不是同一个量。

## 候选池与固定配额

| 候选桶 | 任务 | 行为候选 | gap 筛选后 | 其中计分口径敏感 |
| --- | --- | --- | --- | --- |
| H_rescue | hate | 50 | 50 | 21 |
| H_residual | hate | 89 | 88 | 8 |
| H_removal_harm | hate | 27 | 26 | 12 |
| G_category_support | group | 60 | 60 | 23 |
| G_category_harm | group | 37 | 37 | 17 |
| H_joint_only | hate | 9 | 9 | 5 |
| G_joint_only | group | 17 | 17 | 8 |
| Stable_correct | hate | 145 | 145 | 2 |
| Stable_correct | group | 216 | 216 | 16 |
| Stable_wrong | hate | 48 | 48 | 0 |
| Stable_wrong | group | 74 | 74 | 21 |

实际选取 discovery 32 条、reserve 16 条。配额不足不补齐；query/family 去重及选样重放通过。

| 主桶 | 侧 | 目标 | 实际 | 缺额 | 原因 |
| --- | --- | --- | --- | --- | --- |
| H_rescue | discovery | 4 | 4 | 0 | null |
| H_rescue | reserve | 2 | 2 | 0 | null |
| H_residual | discovery | 4 | 4 | 0 | null |
| H_residual | reserve | 2 | 2 | 0 | null |
| H_removal_harm | discovery | 4 | 4 | 0 | null |
| H_removal_harm | reserve | 2 | 2 | 0 | null |
| G_category_support | discovery | 4 | 4 | 0 | null |
| G_category_support | reserve | 2 | 2 | 0 | null |
| H_joint_only | discovery | 4 | 4 | 0 | null |
| H_joint_only | reserve | 2 | 2 | 0 | null |
| G_joint_only | discovery | 4 | 4 | 0 | null |
| G_joint_only | reserve | 2 | 2 | 0 | null |
| Stable_correct | discovery | 4 | 4 | 0 | null |
| Stable_correct | reserve | 2 | 2 | 0 | null |
| Stable_wrong | discovery | 4 | 4 | 0 | null |
| Stable_wrong | reserve | 2 | 2 | 0 | null |

完整成员资格允许重叠；主桶仅用于调度。G_category_harm 完整保留，但本版没有强制名额。每次排除和跳过见 cases/selection_events.jsonl。

## 数值边界与计分口径

| 任务 | 条件 | 精确并列 | gap ≤ 2ε | 四口径 argmax 有变化 |
| --- | --- | --- | --- | --- |
| hate | C0 | 0 | 0 | 8 |
| hate | CLnew | 0 | 1 | 14 |
| hate | CD | 0 | 1 | 25 |
| hate | CLDnew | 0 | 1 | 45 |
| hate | CLnewNoCat | 0 | 0 | 13 |
| hate | CLDnewNoCat | 0 | 0 | 49 |
| group | C0 | 0 | 0 | 30 |
| group | CLnew | 0 | 0 | 59 |
| group | CD | 0 | 1 | 14 |
| group | CLDnew | 0 | 0 | 17 |
| group | CLnewNoCat | 0 | 0 | 74 |
| group | CLDnewNoCat | 0 | 0 | 11 |

近并列保留在全量统计中，只限制对应候选类型所需的任务/条件。计分口径敏感性仅作标记，不更换 answer_sum 主口径。

## 留出侧与资源重叠

| 资源 | discovery 去重 ID 数 | reserve 去重 ID 数 | 共有 ID 数 | Jaccard |
| --- | --- | --- | --- | --- |
| union_ids | 119 | 69 | 43 | 0.296551724137931 |
| lq_ids | 28 | 11 | 3 | 0.08333333333333333 |
| ld_only_ids | 99 | 63 | 39 | 0.3170731707317073 |
| demo_ids | 280 | 149 | 26 | 0.06451612903225806 |

以上前两列指各侧去重后的集合大小，并非两侧互斥。reserve 仍来自已暴露 dev，且资格依赖行为结果；它仅保留给后续未参与路径定位的机制评估，不是新 test。

## 案例审阅与下一阶段

已导出 32 个定位案例的资源页、轨迹页和完整上下文 JSON；reserve 案例卡未导出。

审阅状态：AI-assisted-initial-12-complete-human-review-pending；人工审阅完成：False；AI 辅助初读：True。

首批审阅按主桶轮询的固定顺序，见 cases/initial_review_batch.json。人工审阅模板见 cases/review_template.csv。若有 AI 辅助初读，其记录单列，不冒充独立人工审阅。

优先解释 H_rescue 与 H_residual 的区别，保留 removal_harm 与 category_harm 反例；joint_only 同时保留证据相加跨阈值和信息依赖两种解释。下一阶段先设计输入对照与重复性验证，再另定激活干预协议。

## 解释边界与其余表格

NoCat 只删除显式类别字段；S 仍可带类别语义，D 仍含答案。S 来自查询与固定示例词条的并集，D 关闭时仍含示例贡献词条。删除同时改变长度和位置，当前结果不能分离这些因素。行为交互、正确预测或某种位型都不证明内部机制。

全部 16 位型（含零计数）、分层构成、逐标签转换、连续读数和长度/词条数的描述性关系分别见 tables/core_masks.csv、strata.csv、per_label_transitions.csv、continuous_summaries.csv、metadata_by_transition.csv。分层与连续读数没有临时增加区间。

案例全文、逐查询表及完整提示均保留本地；本次绑定的原始输入文件哈希在运行前后保持一致。
