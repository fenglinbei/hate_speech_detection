# 具体位置的词义适配

沿用 evidence 工作台的字段：`source_fit` 核对冻结原义项与来源位置，`query_fit` 核对该义项与当前查询。它们不自动更改定义质量、句子标签、严重度或示例关系。

第 1 批覆盖授权队列的全部 292 处命中。先检查 #5998 与 #2445，再扩展至 175 处；合并重复用法后扩展至全部队列。9 个校准问题绑定 12 个明确位置，其他位置的规则应用仍属于 AI。回答前已保存完整初稿 `batch-01/initial-ai-drafts.json`；起草时读取了已获准的人工参照，不宣称独立盲审。

- [当前结果及提问表](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/hits/preferences.md)
- [当前快照索引](/data/liaozijie/hate_speech_detection/exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/resource-reviews-v1/hits/current.json)

原始查询包保留来源全文、词典原文、精确 Unicode 字符位置及进入提示的来源元数据。已采纳改写只在 `adopted_definition_comparison_ai` 中作另一次比较，不替换原定义。黑人喃原释义为空的两处仍保留空值；新增释义的适配可明确，不能据此声称冻结提示中已经包含该释义。

用户完整答复保存为 `batch-01/discussions/raw-replies-NN.json`；实际提取的字段保存为 `user-decisions-NN.json`，按请求 ID、问题序号、原问题全文及源记录哈希绑定。只答 `source_fit` 不计为 `query_fit` 的确认，多位置问题不因只答一处就确认其他位置。修改既有人工字段需要明确更正理由及 `amends`；旧值保留。

规则确认不等于受影响位置均获人工确认。相似位置的 AI 应用、旧 AI 修订及偏好归纳分别记录。待回复数量与保留不明数量分开：已经答复“无法判断”仍可完成该字段的人工记录。

用户若选择另外的处理方式，保存为 `question-outcomes-NN.json`，引用具体回复与独立成果。例如 #4137 要求把“女圈”单列词条，相关指示可结束该适配讨论，但不会伪造原“女拳”条目的人工 `query_fit`。当前新增词条见 `../lexicon-additions/current.json`；其定义措辞为 AI 按用户指示整理，原话单独保留。原词条对查询不适配仍为 AI 按实际词形规则的应用。

`source-rule-application.json` 记录独立词边界和核心所指规则对 12、21 个候选位置的复查；`query-literal-form-application.json` 记录词形规则对 41 个 AI 查询适配值的修订。完整初稿和历次 AI 文件均保留。

```bash
python scripts/review/export_hit_reviews.py
python scripts/review/export_hit_reviews.py --check
python -m unittest discover -s scripts/review -p test_hit_reviews.py
```

导出只写本地命中目录和 `hit_context.json`。JSON、CSV、Markdown 快照按内容哈希命名；不写在线人审会话。自动化验证使用内存测试与只读构建，不向真实人工记录增加答复。

本批校准及联动复核已完成，见 `batch-01/closeout.json` 与 `validation.json`。#5998 改为 Sexism、others，保留 1 级；#4137 在明确词义后仍判 0 级。后续队列为 283 项示例关系，已保留不明的命中不自动重计为漏标。
