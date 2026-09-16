# 案例用途四步实施记录

32 个案例、64 个任务均完成事实、用途建议、材料关联和具体待办的更新。用户授权实施工作流程；用途和假设的作者身份仍保留，不追加人工标注。

| 案例/任务 | 操作 | 主要问题 | 条件数 | 输入准备 |
| --- | --- | --- | ---: | --- |
| 3169/hate | single_demo_removal_dictionary_fixed | 在词典内容完全固定时移除同形异义示例 3660，检验该示例包对错误方向分数的增量作用。 | 2 | text_and_tokens_verified |
| 3169/group | single_demo_removal_dictionary_fixed | 在词典内容完全固定时移除同形异义示例 3660，检验该示例包对错误方向分数的增量作用。 | 2 | text_and_tokens_verified |
| 5086/hate | related_demo_source_factor | 关联示例 7248 与其独有 txl 词条的增量作用可能不同，分别检验并报告双向或无效结果。 | 4 | text_and_tokens_verified |
| 5086/group | single_demo_answer_reference_change | 仅替换指定示例的任务答案，测量其对查询候选分数和集合预测的总效应；不预设改善。 | 2 | text_and_tokens_verified |
| 541/hate | single_demo_removal_dictionary_fixed | 固定词典后移除已确认规则直接相关的示例 826，检验其对 non-hate 支持分数的贡献。 | 2 | text_and_tokens_verified |
| 541/group | single_demo_answer_reference_change | 仅替换指定示例的任务答案，测量其对查询候选分数和集合预测的总效应；不预设改善。 | 2 | text_and_tokens_verified |
| 6037/hate | single_source_edge_mask | 屏蔽已确认的法国女性→国女误命中边，检验该词条在原示例包背景下的增量作用。 | 2 | text_and_tokens_verified |
| 6037/group | single_source_edge_mask | 屏蔽已确认的法国女性→国女误命中边，检验该词条在原示例包背景下的增量作用。 | 2 | text_and_tokens_verified |
| 1128/hate | single_demo_removal_dictionary_fixed | 对稳定正确案例施加相同的一条示例删除操作，记录一般扰动及方向变化。 | 2 | text_and_tokens_verified |
| 1128/group | single_demo_removal_dictionary_fixed | 对稳定正确案例施加相同的一条示例删除操作，记录一般扰动及方向变化。 | 2 | text_and_tokens_verified |
| 4026/hate | single_demo_removal_dictionary_fixed | 对稳定正确案例施加相同的一条示例删除操作，记录一般扰动及方向变化。 | 2 | text_and_tokens_verified |
| 4026/group | single_demo_removal_dictionary_fixed | 对稳定正确案例施加相同的一条示例删除操作，记录一般扰动及方向变化。 | 2 | text_and_tokens_verified |

## 范围与接续

5086/hate 的 7248 为话题直接、规则部分相关，采用独立的相关示例来源分解，不并入严格 U 分支。group 127 和 541/group 826 的答案修订单独构造；6037 只操作已核验的 5230→lex-0069 边。

3169 固定 lex-0419 及全部其他词典，只移除示例 3660。原定义合理且没有已采纳改写，释义改写仍为另一个待准备分支。

1128/4026 只作为删除一条 hate 示例且保持词典的部分操作对照；群体答案、内容及 token 长度未匹配，不称为完整位置或语义控制。

所有新条件保留无效和反向结果。后续须在当前模型环境重放基线，并在同一次新运行中评分全部条件。历史分数仅用于回归核对。自然编辑未控制位置；激活采集和 patching 尚未开始。

每个 group 协议的 foil 已从封存 SD 的最佳非审核候选固定，不随新条件重选。完整 32 候选仍是主决策。全部 64 个任务的资源选项与尚待落实的事项均保留。
