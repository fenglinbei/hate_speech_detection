# 已登记输入干预：GPU 结果 v1

6 个预选 discovery 案例，12 份协议，26 个输入条件、412 个完整候选分数。所有条件在同一次新运行中评分。
12 个原条件完成历史数值重放；同 logits 的 float64 归一化、重复、填充、前缀、候选顺序和 GPU 副本检查均通过。

以下只报告自然输入操作的总效应。词条、示例、答案、foil 和假设均在新评分前登记；未建立等长位置控制或内部机制证据。

## 各条件结果

| 案例／任务 | 条件 | 预测 | 审核参考正确 | 审核 margin | 固定 foil margin | token 数 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 3169 / hate | demo_removed_dictionary_fixed | "hate" | 否 | -0.640865 | — | 712 |
| 3169 / hate | original | "hate" | 否 | -3.769180 | — | 742 |
| 3169 / group | demo_removed_dictionary_fixed | ["Racism"] | 否 | -17.810741 | -17.810741 | 785 |
| 3169 / group | original | ["Racism"] | 否 | -17.466990 | -17.466990 | 820 |
| 5086 / hate | T0_L0 | "hate" | 是 | +0.022371 | — | 1086 |
| 5086 / hate | T0_L1 | "hate" | 是 | +0.012417 | — | 1120 |
| 5086 / hate | T1_L0 | "hate" | 是 | +0.558306 | — | 1212 |
| 5086 / hate | T1_L1 | "non-hate" | 否 | -0.277863 | — | 1246 |
| 5086 / group | answer_reviewed | ["LGBTQ"] | 是 | +13.615681 | +13.615681 | 1345 |
| 5086 / group | original | ["LGBTQ","others"] | 否 | -7.807152 | -7.807152 | 1347 |
| 541 / hate | demo_removed_dictionary_fixed | "hate" | 否 | -0.403198 | — | 855 |
| 541 / hate | original | "non-hate" | 是 | +0.579376 | — | 918 |
| 541 / group | answer_reviewed | ["LGBTQ","Sexism"] | 是 | +4.014164 | +4.014164 | 1005 |
| 541 / group | original | ["LGBTQ","Sexism"] | 是 | +3.598564 | +3.598564 | 1004 |
| 6037 / hate | original | "non-hate" | 否 | -7.159782 | — | 547 |
| 6037 / hate | source_edge_masked | "non-hate" | 否 | -3.618416 | — | 502 |
| 6037 / group | original | ["Sexism"] | 是 | +2.338219 | +2.338219 | 621 |
| 6037 / group | source_edge_masked | ["Sexism"] | 是 | +4.095936 | +4.095936 | 576 |
| 1128 / hate | demo_removed_dictionary_fixed | "non-hate" | 是 | +15.267982 | — | 452 |
| 1128 / hate | original | "non-hate" | 是 | +14.820850 | — | 478 |
| 1128 / group | demo_removed_dictionary_fixed | [] | 是 | +21.384777 | +22.118214 | 522 |
| 1128 / group | original | [] | 是 | +20.223850 | +20.223850 | 549 |
| 4026 / hate | demo_removed_dictionary_fixed | "non-hate" | 是 | +2.409531 | — | 1112 |
| 4026 / hate | original | "non-hate" | 是 | +1.469704 | — | 1157 |
| 4026 / group | demo_removed_dictionary_fixed | ["Region"] | 是 | +14.006321 | +15.483625 | 1194 |
| 4026 / group | original | ["Region"] | 是 | +12.255768 | +12.255768 | 1242 |

审核 margin 为审核候选分数减该条件最佳其他候选分数。固定 foil margin 始终与封存协议中的同一候选比较。主预测采用完整候选空间的 answer_sum，精确并列取最小 canonical ordinal。

## 干预差值

| 案例／任务 | 对照（前者减后者） | 审核 margin 差值 | 固定 foil 差值 | 数值方向 |
| --- | --- | ---: | ---: | --- |
| 3169 / hate | demo_removed_dictionary_fixed_minus_baseline | +3.128315 | — | positive |
| 3169 / group | demo_removed_dictionary_fixed_minus_baseline | -0.343752 | -0.343752 | negative |
| 5086 / hate | T0_L0_minus_baseline | +0.300234 | — | positive |
| 5086 / hate | T0_L1_minus_baseline | +0.290279 | — | positive |
| 5086 / hate | T1_L0_minus_baseline | +0.836168 | — | positive |
| 5086 / hate | demo_given_L0 | +0.535934 | — | positive |
| 5086 / hate | demo_given_L1 | -0.290279 | — | negative |
| 5086 / hate | lexicon_given_T0 | -0.009954 | — | negative |
| 5086 / hate | lexicon_given_T1 | -0.836168 | — | negative |
| 5086 / hate | source_interaction | -0.826214 | — | negative |
| 5086 / group | answer_reviewed_minus_baseline | +21.422832 | +21.422832 | positive |
| 541 / hate | demo_removed_dictionary_fixed_minus_baseline | -0.982574 | — | negative |
| 541 / group | answer_reviewed_minus_baseline | +0.415600 | +0.415600 | positive |
| 6037 / hate | source_edge_masked_minus_baseline | +3.541367 | — | positive |
| 6037 / group | source_edge_masked_minus_baseline | +1.757717 | +1.757717 | positive |
| 1128 / hate | demo_removed_dictionary_fixed_minus_baseline | +0.447132 | — | positive |
| 1128 / group | demo_removed_dictionary_fixed_minus_baseline | +1.160927 | +1.894363 | positive |
| 4026 / hate | demo_removed_dictionary_fixed_minus_baseline | +0.939827 | — | positive |
| 4026 / group | demo_removed_dictionary_fixed_minus_baseline | +1.750553 | +3.227858 | positive |

epsilon 固定为 0.0013427734375。两条件 hate 差值使用 2 epsilon；group 的固定候选差／最佳其他候选 margin 使用保守的 4 epsilon。5086 的四项 hate 交互使用 4 epsilon。数值范围不表示统计置信区间。

## 范围及后续

相对于各自原条件，共 14 个编辑条件比较，5 个预测改变；按审核参考计 4 个由错变对、1 个由对变错。多个条件共享同一案例，这些计数不能当作独立样本或整体性能提升。

3169 仍需单独准备定义／来源对照；5086 的相关示例来源分解独立于严格 U 分支；541 的显式类别字段分支仍待准备；6037 本次仅处理指定来源边；1128／4026 仅作部分操作对照。

64 条用途记录已生成独立结果版本，其中 12 条附上本次评分并完成评分待办。人工用途选择、解释和原参考保持历史身份；本次模型结果不新增人工决定。后续位置控制、激活采集和双向 patching 仍未完成。

附表保留原／审核双参考、4 种分数口径、全候选分数及全部正向、反向和无变化结果。候选限制下的分数不能解释为校准后的现实概率。
