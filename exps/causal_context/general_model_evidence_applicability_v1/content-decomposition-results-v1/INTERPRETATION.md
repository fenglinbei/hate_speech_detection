# 内容拆分结果与后续影响

本轮已完成。因用户通知只有两卡空闲，另建了物理 GPU 1、2 的执行版本；输入、参考、评分方法和数值门槛均沿用原冻结。2026-09-14 13:23 至 13:43（北京时间）完成八轮 GPU 执行，两个工作进程已退出；随后只读查询两卡利用率均为 0。

[完整结果表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/RESULTS.md) · [逐项比较及系数](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/comparison-summary.csv) · [全部口径](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/contrast-scores.csv) · [五探针／留一诊断](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/probe-contrasts.csv) · [原文及材料核对](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-02/MATERIALS.md)

## 3169：两类语境中都保留“嘿嘿”的相对推动作用

下表的差值均为 score(non-hate)−score(hate) 的条件差。群体配对比较“嘿嘿们”减“黑人们”，笑声配对比较“嘿嘿地笑”减“哈哈地笑”；负值表示前者相对更向 hate 移动。不同评分口径的量尺不能直接比较大小。

| 配对 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 |
|---|---:|---:|---:|---:|---:|
| 措辞 1／群体代称 | -1.732597 | -0.351333 | -0.367228 | -4.129208 | -3.540447 |
| 措辞 1／普通笑声 | -0.652332 | -0.162739 | -0.042335 | -2.972469 | -2.047405 |
| 措辞 2／群体代称 | -1.886978 | -0.383120 | -0.416110 | -4.272133 | -3.886108 |
| 措辞 2／普通笑声 | -0.861778 | -0.222180 | -0.182690 | -3.458389 | -1.992092 |

四组配对在上表五列及含 EOS 的辅助口径下均为负。与此前直接比较不同攻击内容相比，这次在每一配对中只替换所列称呼／笑声，外部 token 位置相同、hate 示例答案保持不变。因此结果与这两种受测语境中的共享词形关联相容。

**第一套普通笑声配对仍有背景探针敏感性。** 五探针 NCC 为 −0.042335，五个留一结果全为负；单用“一个空格”估计背景时为 +0.010276，超出该对照的数值未决界，其余四个单探针为负。不能写成所有探针都支持同一方向。

这不是无语境的“嘿嘿”因果定论：两类句子都有攻击，群体称呼改为显式称呼还会改变指代清晰度，两类攻击命题也不同。共同加入“们”相对旧材料的变化保留在桥接，不并入纯词形解释。

## 541：NCC 中的规则优势没有跨标签映射稳定保留

四项“规则加入”的条件差如下。T0 为篮球迷主题，T1 为异性恋主题。

| 规则对照 | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 |
|---|---:|---:|---:|---:|---:|
| 措辞 1／T0 | -0.076370 | -0.027630 | +0.311929 | +0.570354 | -0.217293 |
| 措辞 1／T1 | +0.031460 | +0.009576 | +0.205672 | +0.576180 | -0.180889 |
| 措辞 2／T0 | +0.014038 | +0.003529 | +0.267445 | -0.328114 | -0.349491 |
| 措辞 2／T1 | -0.172714 | -0.043134 | +0.181476 | -0.348286 | -0.624802 |

NCC 下四个规则差值均为正，而且五个单探针与五个留一方向一致。但 A/B 正映射只在第一套措辞中为正，第二套为负；反映射四项全为负。原总分和原均分也出现交叉方向。因而不能据此认定模型稳定利用了抽象反泛化规则。

主题差值同样受口径影响：原总分／均分四项全为负，NCC 四项全为正；A/B 的第二套“有规则”主题差在正反映射间反向。NCC 的两项主题×规则交互均为负（−0.106257、−0.085969），表示在这套校准读数中，规则的条件增量在 T1 下小于 T0；该交互也没有跨口径稳定保持。第一套交互在 A/B 正映射的平均分及含 EOS 平均分中为数值未决，辅助结果完整保留。

## 相对效应与分类分别看

两个查询的原参考和审核参考都是 non-hate。下表只列 8 个新正文条件的正确预测数；这是同一查询的多种提示，不能当作独立测试集准确率。

| 案例（每项共 8 条件） | 原总分 | 原均分 | NCC | A/B 正 | A/B 反 |
|---|---:|---:|---:|---:|---:|
| 541 | 8 | 8 | 0 | 0 | 8 |
| 3169 | 1 | 2 | 0 | 0 | 6 |

包含历史锚点的全部 26 个条件中，NCC 和 A/B 正映射都预测 hate。因此“扣除背景后仍有条件差值”不等于改善分类，更不能用这些输出修改人工参考或挑选一个表现较好的映射作为新主口径。

## 对下一步的影响

3169 现在更适合优先补功能性查询对照：另建普通笑声与明确群体指称／攻击的查询及独立参考，检查同样的输入变化是否具有与查询语义相符的选择性。这一步需要新的 ID、材料核对及冻结；当前结果不自动授权或证明内部机制。

541 应先解释措辞和映射之间的方向变化，再把它用作机制实验的稳定案例。可以继续使用本次已冻结的正向、反向和未决格子作为后续假设的依据，但不能只留下 NCC 的正向结果。所有人工机制字段和 activation patching 准备状态维持原状。

## 完整性与复现

- 26 内容条件：16 新正文、10 历史锚点；208 提示、416 候选；14 主比较、12 总输入桥接、4 历史 D−C 对照。
- 八轮共 2,816 次候选评估；80 个历史提示精确重放，重复、候选顺序和两卡互换最大误差均为 0。
- 填充最大误差 0.000141143798828125，前缀 0.00009918212890625；两者均低于未改动的 epsilon 0.0013427734375。
- 10 项原始数值门槛和 6 项派生门槛全部通过；每项派生门槛覆盖 672 个读数。
- 原准备 15 项 CPU 测试、两卡版本新增 4 项测试通过；175 个来源及全部输入的独立核对通过。
- 导出结果逐字节重建通过；50 位 Decimal 独立核对 2,210 个数值及 780 项对照方向，最大导出误差约 1.53e−15。

[执行版本](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/content-decomposition-v1/execution-two-gpu-01.json) · [GPU 执行记录](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/run-02/run_manifest.json) · [独立数值回执](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/audits/independent-results-two-gpu.json) · [GPU 退出回执](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/audits/gpu-release-two-card-01.json)

使用两卡入口的 `check` 可做只读复核，终态运行拒绝再次 forward。以下分析重建命令使用绝对路径；该冻结版本的分析 CLI 对显式相对路径不作自动展开。

```bash
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/run_evidence_content_execution_v2.py check
/data/liaozijie/hate_speech_detection/.conda/stage1-p0/bin/python /data/liaozijie/hate_speech_detection/scripts/review/analyze_evidence_content_decomposition.py --plan /data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/frozen-02 --run /data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/run-02 --output /data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02 --check
```

计划 ID：`evidence-content-decomposition-86b399a3e42c2d0641cfecde4eabacc8b4100bf306db3f970c95faa1d59b50e3`。原始分数 SHA256：`b469612fe8ca56c8b1f579181d31e5efe709c0092e02a5ad1d364b4a2e4eb94d`。

本说明为助手对已封存实验的分析，不新增逐句人工采纳、参考修订或机制裁决。旧四卡准备、所有材料／参考／用途／运行指针与人审源记录均保留。
