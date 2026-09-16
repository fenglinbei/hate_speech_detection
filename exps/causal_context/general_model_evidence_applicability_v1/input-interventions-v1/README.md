# 已登记输入干预：GPU 执行 v1

**已完成：26 条件、412 候选分数，全部数值检查通过，4 张 GPU 已释放。** [当前结果指针](current.json)、[完整结果表](../reviews/analysis-freeze-20260912/input-interventions-v1/results-v1/RESULTS.md)、[更新后的 64 条任务用途表](../reviews/analysis-freeze-20260912/input-interventions-v1/results-v1/case_uses.csv)、[12 份协议的执行结果](../reviews/analysis-freeze-20260912/input-interventions-v1/results-v1/protocol_results.jsonl)。

用户于 2026-09-12 明确授权“当前GPU空闲，可继续执行下一阶段”。本次执行前一版本已登记的 3169、5086、541、6037、1128、4026 六个 discovery 案例，合计 12 份协议、26 个输入条件。协议、示例选择、答案修订、来源边、固定 foil 和假设沿用[用途四步实施版本](../case-uses-implementation-v1/README.md)。

## 评分及验证

复用现有 NumericRunner、候选评分内核及四 GPU 副本执行器。模型为本地 Qwen3-8B，核验同一模型和 tokenizer 文件树；以已验证的 bfloat16 权重值精确拓宽到 float32，eager attention、batch=1、无 KV cache、关闭 TF32、种子 42、动态右侧填充。每条候选完整评分，无输入截断。主预测使用答案 token 对数概率之和，EOS 和长度归一化口径保留为辅助分析。

执行顺序在新评分前写入[本次计划](../reviews/analysis-freeze-20260912/input-interventions-v1/plan-v1/plan.json)：

1. 12 个原条件与历史完整候选分数比较，并验证同 logits 的 CPU float64 归一化参考及重复运行。
2. 26 个条件在一次新运行中评分，包含 12 个原条件和 14 个编辑条件。
3. 对完整 26 条件核对重复、额外 64 token 填充、无缓存答案前缀、候选顺序和 GPU 副本切换。
4. 全部数值检查通过并封存原始分数后，独立执行 CPU 分析和用途表更新。

误差阈值沿用历史值：reference/repeat 为 0.0001，其他数值检查为 epsilon=0.0013427734375。失败会停止并保留失败记录，不自动改精度、改变阈值或挑选新的条件。四个副本均保留实际物理 GPU UUID 和评分来源证明。

## 分析范围

原／审核参考只在分析阶段连接。完整 32 个 group 候选决定预测；每份 group 协议还沿用一个固定 foil，不能按新条件重新挑选。输出保留四种分数口径、五类 group 分数差、原／审核参考 margin、固定 foil margin 和全部无变化及反向结果。

5086/hate 四条件测量相关示例 7248 与其独有 lex-0073 的来源贡献，保留四个条件简单效应和四项交互；它不属于严格 U 分支。3169 本次固定词典而删除示例 3660，尚无已采纳定义改写。6037 本次仅屏蔽已登记来源边，保留示例原答案。1128／4026 是部分操作对照，不能消除语义和 token 位置差异。

结果只适用于这批预选案例和自然输入操作。多个条件共享同一查询，不能把编辑条件比较数当作独立样本，也不能将输入交互直接解释为内部机制。新用途表中的输入实验可用状态仅指已登记操作；人工用途、解释、参考版本及机制待办分别保留。

## 结果解读与用途

下表使用事先登记的 answer_sum 主口径。margin 为审核候选分数减该条件最佳其他候选分数；正值表示审核候选胜出。

| 案例 | 本轮发现 | 后续用途及限制 |
| --- | --- | --- |
| 3169 | 移除 3660 后 hate margin 从 −3.7692 升至 −0.6409，仍预测 hate；group 仍为 Racism，margin 反而下降 0.3438 | 保留 hate 示例增量作用候选；group 需另查共享词条和错义来源。删除该示例不足以纠正两项任务 |
| 5086 / hate | 7248 与 lex-0073 同时存在时预测 non-hate，其余三个条件均为 hate；主口径四项交互为 −0.8262 | 保留相关示例来源分解候选；仍独立于严格 U 分支。必须保留分数口径敏感性，见下文 |
| 5086 / group | 仅修订示例 127 的答案后，预测从 LGBTQ＋others 变为 LGBTQ，审核 margin 从 −7.8072 升至 +13.6157 | 明确保留答案参考切换实验价值；尚不能据此认定模型直接复制该示例答案 |
| 541 | 删除 826 后 hate 从正确 non-hate 变为错误 hate，margin 从 +0.5794 变为 −0.4032；单独修订 group 示例答案后预测仍正确 | 826 保留为有帮助的示例候选，不能因其他案例删除有益而统一删除；标签数量、长度和位置影响仍待区分 |
| 6037 | 屏蔽已确认误命中边后，hate margin 改善 3.5414 但仍错误；group 继续正确，margin 改善 1.7577 | 保留来源边诊断价值；去除这条误命中边未消除 hate 错误，也未破坏 group 正确预测 |
| 1128／4026 | 两项任务预测均保持正确，连续 margin 均变化 | 保留部分操作对照，并记录一般删除扰动；“预测不变”不等于“分数无变化” |

**5086/hate 的纠错依赖主评分口径。** 在 answer_mean 和 mean_with_eos 下，四个条件全部预测 non-hate，因此三次离散纠错不成立。answer_mean 的四项交互为 −0.002211，落在既定 4 epsilon=0.005371 的数值未决范围；不能概括为跨口径稳定交互。完整四口径结果见 [conditions.csv](../reviews/analysis-freeze-20260912/input-interventions-v1/results-v1/conditions.csv) 和 [contrasts.jsonl](../reviews/analysis-freeze-20260912/input-interventions-v1/results-v1/contrasts.jsonl)。

14 个编辑条件相对各自原条件共出现 5 次预测改变：4 次由错变对均来自 5086，1 次由对变错来自 541/hate。这是共享查询的条件比较计数，不能解释为整体性能提升或四个独立成功案例。

下一阶段先据此完善案例用途和具体对照：5086 区分答案口径、标签构成与来源交互；3169 保持 hate/group 分开并准备词典来源分支；541 保留有帮助的示例作为对照；6037 保留来源边诊断。位置／长度控制与语义对照尚未完成，激活采集和双向 patching 仍是独立后续阶段。这里的用途解释保留 assistant 作者身份，不回写人工解释字段。

## 验收

12 个历史原条件重放、新运行中的原条件复核、两次重复运行、候选顺序及 GPU 副本切换的最大差异均为 0。填充检查最大差异 0.0003356934，前缀检查 0.0003242493；均低于固定 epsilon。float64 归一化参考最大误差 0.0000111675，低于 0.0001。

11 项隔离测试通过，8 个评分 pass 的分片来源、完整候选、token 几何和数值比较已重新核验；分析逐字节复算通过。另从原始 token 分数独立复算 7,600 项标量检查，确认更新 12 条任务记录、其余 52 条记录不变，人工字段全部保持。详情见[验证回执](validation.json)。

## 工具

[GPU 执行入口](../../../../scripts/review/run_evidence_interventions.py)、[CPU 分析入口](../../../../scripts/review/analyze_evidence_interventions.py)。6 项执行边界测试与 5 项分析测试使用隔离的合成数据，不写入人工审核会话。

```bash
.conda/stage1-p0/bin/python -m unittest scripts.review.test_run_evidence_interventions scripts.review.test_analyze_evidence_interventions -v
.conda/stage1-p0/bin/python scripts/review/run_evidence_interventions.py check
.conda/stage1-p0/bin/python scripts/review/analyze_evidence_interventions.py --check
```

`run` 拒绝再次启动已完成或失败的运行。输入计划、评分回执、原始分数和分析结果位于独立的私有 `reviews/` 目录，旧参考冻结、双参考评估、用途草案与实施版本均保持原字节。
