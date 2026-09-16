# 原标签、NCC 与 A/B 正反映射完成结果

当前结果由 `current.json` 选择。优先阅读 [结果解读](INTERPRETATION.md)，完整条件分数和全部 64 项对照位于指针选择的 `results-01/RESULTS.md` 与 JSONL/CSV 文件。

本轮完成全部 8 个 pass，10 项原始数值门槛、6 项 NCC 派生门槛及 4228 项独立数值复算通过。四张 GPU 已释放，旧指针和人审记录未变化。

3169 的配对内容负向效应跨三组保留；541 的配对内容效应在 NCC、A/B 正反映射下均为正。NCC 的 36 条条件预测全部为 hate，A/B 正反映射预测差异明显，因此保留三组并列结果，分别解读相对效应和分类结果。

## 复核

```bash
.conda/stage1-p0/bin/python scripts/review/run_evidence_label_calibration.py check
.conda/stage1-p0/bin/python scripts/review/analyze_evidence_label_calibration.py --check
python exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/label-calibration-v1/audits/independent_check.py inputs
python exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/label-calibration-v1/audits/independent_check.py results
```

准备入口为 `../label-calibration-v1/current.json`。旧 26 条件干预与 36 条件匹配输入结果继续保持原冻结状态。新改动需要新版本；本轮不新增人审机制结论或激活实验资格。
