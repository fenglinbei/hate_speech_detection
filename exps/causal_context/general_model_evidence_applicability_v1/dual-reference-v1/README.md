# 原／审核双参考评估 v1

2026-09-12。实现、测试和正式评估已完成。固定 32 条 discovery、64 个任务参考和 6 个历史条件，复现全部 384 组预测并校验 6,528 个候选分数；原参考与审核参考的分母均为每任务 32 条。所有记录已确认，未审、草稿、未决和排除数均为 0。

[完整评估报告](../reviews/analysis-freeze-20260912/dual-reference-v1/REPORT.md)给出分类指标、正确性转换和四条件位型变化。[当前评估指针](current.json)绑定结果 manifest，SHA-256 为 `256fb6bdb36ae1f39a6a2eb8a9ecd413c022a5821fccc646b85059bd9a390df7`。[验证回执](validation.json)记录测试和复算结果。

## 结果与解读

| 项目 | hate | group |
| --- | ---: | ---: |
| 可用参考／任务总数 | 32/32 | 32/32 |
| 相对原 Gold 的标签差异 | 9 | 18 |
| 四条件正确性位型变化 | 9 | 15 |
| 审核参考下，CD → CLDnewNoCat 错→对 | 4 | 3 |
| 审核参考下，CD → CLDnewNoCat 对→错 | 11 | 3 |

位型使用 `(C0, CLnewNoCat, CD, CLDnewNoCat)`，1 表示该参考下预测正确。标签改变不一定改变位型：group 可以在两套参考下都预测错误。因此，18 个标签差异与 15 个位型变化并不矛盾。

例如 #3683/hate 按用户明确确认的案例级 non-hate 参考，从 `0011` 变为 `1100`，改变了该例对资源效果的描述。#3919/group 的最终参考为有效空集合 `[]`；#3950/group 为 `[Sexism]`；#4615/group 为 `[Racism, Sexism]`。四项案例／材料差异均沿用已冻结的显式优先级，两层原值仍可回查。

上述条件转换是子集内的离散描述。该队列按模型行为富集，不用于推断总体误标率或显著性；原始 643 条总体结果保持原样。标签差异也不能独自排除政策调整对结果的影响。材料与案例政策身份分别保留，不统一伪写为一个新版政策。

## 实现与产物

[命令入口](../../../../scripts/review/evaluate_evidence_references.py)、[计算模块](../../../../src/diagnostics/general_model_evidence_evaluation.py)和[固定配置](../../../../config/stage1/general_model_evidence_dual_reference_v1.json)只使用 Python 标准库及现有的纯计算候选工具，不加载模型或 tokenizer。实际代码文本、哈希、Git 身份、配置与 50 个源文件哈希封存于输出。

| 文件 | 用途 |
| --- | --- |
| [blocks.jsonl](../reviews/analysis-freeze-20260912/dual-reference-v1/blocks.jsonl) | 384 组固定预测、两套正确性、原／新 margin、候选分数、提示身份及计分敏感标记 |
| [metrics.csv](../reviews/analysis-freeze-20260912/dual-reference-v1/metrics.csv) | 每任务、条件、参考的 exact accuracy、macro F1、micro F1 |
| [per_label_metrics.csv](../reviews/analysis-freeze-20260912/dual-reference-v1/per_label_metrics.csv) | 固定标签空间内的 TP、FP、FN、support 与 F1 |
| [reference_change_counts.csv](../reviews/analysis-freeze-20260912/dual-reference-v1/reference_change_counts.csv) | 预测固定时，切换参考产生的正确性转换 |
| [case_masks.csv](../reviews/analysis-freeze-20260912/dual-reference-v1/case_masks.csv) / [JSONL](../reviews/analysis-freeze-20260912/dual-reference-v1/case_masks.jsonl) | 64 个任务的原／新四位型、标签、原 bucket 和行为标记 |
| [core_mask_counts.csv](../reviews/analysis-freeze-20260912/dual-reference-v1/core_mask_counts.csv) / [转换表](../reviews/analysis-freeze-20260912/dual-reference-v1/core_mask_transitions.csv) | 全部 16 种位型及 16×16 转换，零计数保留 |
| [case_pairs.jsonl](../reviews/analysis-freeze-20260912/dual-reference-v1/case_pairs.jsonl) / [转换计数](../reviews/analysis-freeze-20260912/dual-reference-v1/transition_counts.csv) | 六种条件对比在两套参考下的正确性转换和 margin 差 |
| [margin_interactions.jsonl](../reviews/analysis-freeze-20260912/dual-reference-v1/margin_interactions.jsonl) | 两套参考的四条件 margin 与 SD−S−D+0 描述性读数 |
| [coverage.csv](../reviews/analysis-freeze-20260912/dual-reference-v1/coverage.csv) | 每任务确认、草稿、未审、未决与共同分母，含逐例排除原因 |
| [manifest.json](../reviews/analysis-freeze-20260912/dual-reference-v1/manifest.json) / [audit.json](../reviews/analysis-freeze-20260912/dual-reference-v1/audit.json) | 文件身份、输入校验、执行身份与本次范围 |

主决策保持不含 EOS 的 `answer_sum`，精确并列取最小 canonical ordinal，近并列使用历史 epsilon。审核后的 group margin 从全部 32 个候选重新取 Gold 与最佳非 Gold 之差；没有对旧 margin 直接翻符号。

原分数文件若确实缺失，只有显式 `--allow-missing-scores` 才导出离散结果，重算的连续分数为 null、状态为 `candidate_scores_missing`。原分数存在却哈希不符、候选不完整、顺序错误、预测或上下文不匹配时直接失败，不能降级掩盖错误。原卡片 margin 仅在明确命名的 `original_margin_archived` 字段保留。

## 验证与复算

17 项测试通过，涵盖完整候选、新 group margin、并列、空集合与未决、共同分母、来源和预测漂移、重复／缺行、结果不可覆盖及篡改检查。真实 32 条框架的缺分数模式已验证：离散指标与位型表完全一致，384 个重算 margin 均为 null；损坏分数仍拒绝。另用现有环境中的 scikit-learn 1.8.0 独立核对了 24 行指标、72 个标量和 84 个逐标签混淆表。scikit-learn 仅用于独立验证，不是评估程序的运行依赖。

在仓库根目录运行，均无需 GPU：

```bash
python -m unittest src.tests.test_general_model_evidence_evaluation scripts.review.test_freeze_evidence_analysis -v
python scripts/review/evaluate_evidence_references.py validate
python scripts/review/evaluate_evidence_references.py check --output exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/dual-reference-v1
```

`check` 已确认全部生成表逐字节重建一致。`evaluate --output <新目录>` 可生成独立副本，拒绝覆盖既有评估或参考冻结目录；输出只能在本实验被忽略的 `reviews/` 树下。若人工参考或代码修订，应建立新版本，不改本次封存结果。

## 工作收束与 GPU 状态

第一步的材料审核、案例裁决、参考冻结和双参考评估交付现已齐备。[参考冻结 v1](../analysis-reference-v1/README.md)的冻结包及其 `current.json` 指针保持原字节，其中评估未完成的状态描述的是冻结时点；本目录单独登记后续评估完成状态。

用户已通知服务器 GPU 可用并授权启动测试和评测。本次实际评分资料完整，评估在 CPU 完成，没有发起新的 GPU 推理。此实现只切换评估参考；后续改变示例或词典的输入实验仍需形成具体干预、保持项、假设和替代解释。当前冻结的 64 条输入实验资格仍为 false，不能把自动生成的行为标记当作人工机制解释或已冻结的输入实验计划。
