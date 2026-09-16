# 内容拆分：GPU 评测已完成

**仅使用物理 GPU 1、2，八轮评测及全部数值检查通过，GPU 工作进程已退出。** 输入、评分口径和数值门槛保持原冻结，采用独立两卡执行版本。

- [结果解读与后续影响](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/content-decomposition-results-v1/INTERPRETATION.md)
- [完整结果表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/RESULTS.md)
- [全部比较 CSV](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/comparison-summary.csv)
- [逐条件原分数与预测](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/condition-scores.csv)
- [NCC 原均分、背景及残差](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/ncc-conditions.csv)
- [五探针／留一比较](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/content-decomposition-v1/results-02/probe-contrasts.csv)

3169 四组词形配对的主方向跨原标签、NCC、A/B 保持负向，但第一套笑声配对有单空格探针反向例外。541 的规则差值在 NCC 下为正，跨措辞和标签映射未稳定保留。相对输入效应与分类表现分开报告，不将其提升为人审机制结论。

208 提示、416 候选、30 项比较；80 个历史提示精确重放，10 项原始数值门槛、6 项派生门槛及 2,210 个独立数值核对通过。旧准备和人审记录均保留。
