# 2026-09-16 成果归档

本次按用户确认的范围同步尚未推送的代码、文档、最终标注、实验结果表和复核报告。下列入口选择各实验的完成结果；同目录中保留的准备、暂停、失败和中期记录仍描述各自的历史时点。

| 成果 | 阅读入口 | 结果与复核材料 |
| --- | --- | --- |
| Q01 六模块细化 | [完整解读](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-module-refinement-v1/interpretation-01/INTERPRETATION.md) | [完成结果选择器](../../../exps/causal_context/general_model_evidence_applicability_v1/q01-module-refinement-results-v1/current.json)、[全部结果表](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-module-refinement-v1/results-01/)、[全视图及例外表](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-module-refinement-v1/interpretation-01/)、[审计](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-module-refinement-v1/audits/) |
| Q01 整块定位 | [完整解读](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/interpretation-01/INTERPRETATION.md) | [完成结果选择器](../../../exps/causal_context/general_model_evidence_applicability_v1/q01-local-mechanism-results-v1/current.json)、[全部结果表](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/results-01/)、[审计](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/q01-local-mechanism-v1/audits/) |
| 8 条功能查询、两个阶段 | [结果与解读](../../../exps/causal_context/general_model_evidence_applicability_v1/functional-query-results-v1/README.md) | [冻结输入、两阶段结果及验收](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/functional-query-diagnostics-v1/execution-01/) |
| 词条依赖、限定说明 | [结果与解读](../../../exps/causal_context/general_model_evidence_applicability_v1/lexicon-scope-results-v1/README.md) | [冻结输入及结果](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/lexicon-scope-v1/) |
| 内容拆分与标签校准 | [内容拆分](../../../exps/causal_context/general_model_evidence_applicability_v1/content-decomposition-results-v1/README.md)、[原标签／NCC／A-B](../../../exps/causal_context/general_model_evidence_applicability_v1/label-calibration-results-v1/README.md) | [冻结输入、结果及审计](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260914/) |
| 标注、参考与前期输入干预 | [最终材料标注](../../../exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911/final-results-v1/README.md)、[双参考评估](../../../exps/causal_context/general_model_evidence_applicability_v1/dual-reference-v1/README.md) | [参考冻结、用途、匹配材料与干预结果](../../../exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/) |
| 前期数值、覆盖与去标签实验 | [数值实验](../../../exps/causal_context/general_model_ld_numeric_v3/RESULTS.md)、[覆盖实验](../../../exps/causal_context/general_model_ld_coverage_v1/RESULTS.md)、[去标签实验](../../../exps/causal_context/general_model_ld_nolabel_v1/RESULTS.md) | 同步此前遗漏的结果表、CPU 审计和运行元数据；14B／27B 等历史记录保留原完成或未完成状态 |

Q01 两轮均已完成全部 12 个 pass。整块定位选择 block 34 / pre_answer；模块细化分别选择 attention 35（C）和 MLP 34（I）。必须同时阅读全视图、例外和标签表：这是一个已暴露查询的局部干预结果，方向转移不等于稳定标签修复，`mechanism_ready` 仍为 `false`。本次归档没有新增 GPU 运行、人工裁决或结果解释。

## 归档范围

- 保留代码、协议、最终标注及其来源历史、结果表、解读、冻结输入、执行清单与复核回执。`reviews/` 中的这些成果通过明确文件清单加入 Git，原忽略规则仍保留。
- 原始逐次评分、capture 记录、检查点数据库、大型历史生成输入、GPU 运行日志、模型与缓存留在本地。私密权威会话、登录凭据、运行锁和取消控制文件不进入本次提交。
- 原始冻结文件和选择器按原字节保存。旧状态、空人工字段和哈希均不改写；`.gitattributes` 禁用成果目录的文本换行转换。
- 这是成果归档，不是完整运行备份。部分冻结清单引用本地原始分数、环境或模型；完整数值重放仍需这些未归档依赖。元数据中的原机器绝对路径保留，可按仓库根目录对应阅读，不应把它们直接当作另一台机器上的可执行路径。

## 字节检查

[`manifest.json`](manifest.json) 记录本次新增或更新成果文件的路径、字节数和 SHA256，不重复收录本目录的归档说明与检查程序，也不代表整个既有仓库的清单。Git 提交绑定本目录文件。运行以下只读检查可验证清单中的文件：

```bash
python docs/research/results-20260916/verify.py
```

[`verification.json`](verification.json) 记录本次归档检查及相关测试结果。这些是交付检查；既有科学结果的数值审计以各实验原回执为准。
