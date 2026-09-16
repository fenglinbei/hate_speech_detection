# 案例裁决检查与分析参考冻结 v1

**接续状态（2026-09-12）：[双参考评估 v1 已完成](../dual-reference-v1/README.md)。** 本文与冻结文件中的“下一步／未执行”保留参考冻结时点的含义；后续评估使用独立目录与指针，未改写本冻结。

2026-09-12。已读取 digitalocean-sgp 的最新权威 evidence 会话，完成检查并冻结分析参考。32/32 案例确认，64/64 任务可用于参考标签分析；材料确认 1,072/1,072，无过期依赖、无 group 待重核。此次只读生产，未改写原 Gold、材料决定、案例决定或模型输入。

[冻结指针](current.json)绑定会话 revision `47c6ab2652111fc7b20e6f6824a1e1ea5879380a830fddfccbcbbeb3ba0d5419`，精确字节 SHA-256 `86af6ea6296f59980f6135b5ffc45571f73a0cd29b8bf9d029ba37d41d46f5f0`。冻结 manifest SHA-256：`4e6ef030d6dfdd8d90087f3e32160de41f58a8c2a2d47905eddf99d60fd7a3af`。

## 检查结果

| 项目 | 结果 |
| --- | --- |
| 案例及任务 | 32 案例、64 个任务均 confirmed/resolved；均选择 reference_analysis |
| 材料 | query 32、demo 280、definition 122、hit 318、relation 320 |
| 相对原 Gold 的标签差异 | hate 9/32；group 18/32 |
| 页面原标注状态 | hate：accepted 23、suspected_error 9；group：accepted 14、suspected_error 18 |
| 依赖 | 所有材料和案例快照均未过期，材料快照哈希与来源身份校验通过 |
| 解释及案例备注 | 32 条 explanation_choice=defer，案例 note 均为空；不虚构解释或修改理由 |
| 输入实验资格 | 此次未冻结具体干预、假设及替代解释，因此未放行输入实验 |

9/18 是任务标签差异计数，不能将用户选择的 suspected_error 自动升级成已排除规则变化的源数据错误率。原审核状态保留原值；分析参考可用与原错误归因是否充分是独立字段。材料证据/理由会回链，但不冒充案例层新增的修改理由。

## 已明确解决的两层标签差异

本次检查发现四项案例标签与材料标签不同，用户明确回答“全部以案例级最终裁决为准”。只为本次分析参考登记优先级，不改两层历史。

| 查询/任务 | 原材料标签 | 最终案例参考 |
| --- | --- | --- |
| 3683 / hate | null（未决） | non-hate |
| 3919 / group | [others] | [] |
| 3950 / group | [Sexism, others] | [Sexism] |
| 4615 / group | [Sexism] | [Racism, Sexism] |

显式回复与问题范围单独保存在 [reconciliation.json](../reviews/analysis-freeze-20260912/reconciliation.json)，绑定本次权威会话字节身份。未追加推测的语义理由，也未对其他字段扩大确认范围。

## 冻结交付

人工记录和派生明细均放在被忽略的 reviews 树，当前指针及本文只登记状态和身份。

| 文件 | 内容 |
| --- | --- |
| [gold_overlay.jsonl](../reviews/analysis-freeze-20260912/frozen-v1/gold_overlay.jsonl) | 64 条查询—任务的原标签、最终标签、材料标签、审核状态、确认身份及政策来源 |
| [adjudications.jsonl](../reviews/analysis-freeze-20260912/frozen-v1/adjudications.jsonl) | 64 条裁决、材料证据回链、案例快照身份和四项差异的显式处理依据 |
| [eligibility.jsonl](../reviews/analysis-freeze-20260912/frozen-v1/eligibility.jsonl) | 64 条逐任务可用性、原 bucket、使用选择和后续门槛；新行为分型留待双参考评估 |
| [reference_summary.csv](../reviews/analysis-freeze-20260912/frozen-v1/reference_summary.csv) | 可直接查阅的原/审核标签与可用性表 |
| [material_reviews.jsonl](../reviews/analysis-freeze-20260912/frozen-v1/material_reviews.jsonl) | 全部 1,072 项，包含此前 118 项及本轮 954 项；保留原始字段来源、版本及最终采纳记录 |
| [policy_provenance.json](../reviews/analysis-freeze-20260912/frozen-v1/policy_provenance.json) | 基础政策及最终采纳时绑定的补充政策正文/身份 |
| [finalizations.json](../reviews/analysis-freeze-20260912/frozen-v1/finalizations.json) | 批量采纳授权、逐字段来源、18 项采纳释义及女圈独立补充的历史 |
| [audit.json](../reviews/analysis-freeze-20260912/frozen-v1/audit.json) / [manifest.json](../reviews/analysis-freeze-20260912/frozen-v1/manifest.json) | 覆盖、差异检查、完整输出哈希和执行源码身份 |

案例 `task_reviews` 仍引用基础 hate v1/group v2；部分材料 `final_annotation` 另绑定严重度映射、个人侮辱 others 和同一对象优先等补充政策。导出分别保留 `native_case_policy` 与 `query_material_policy`，不把同一个基础政策版本冒充全体最终标签的统一语义来源。完整政策与原始审核快照一起回链。

## 验证与接续

冻结结束后再次只读核对远端会话 SHA-256，与下载快照完全一致；本次没有遗漏检查期间的新保存，也未修改权威记录。

[冻结工具](../../../../scripts/review/freeze_evidence_analysis.py)只在临时副本上调用支持 finalization 的现有 reader，检查后生成独立目录；拒绝覆盖既有冻结包。权威快照有 remote-authority 标记，防止误启本地 writer。5 项资格边界测试通过；冻结包逐字节重建及所有输出哈希校验通过。所有 64/64/64/1,072 条输出的 ID 唯一。

从仓库根目录复算检查（需要被忽略的私人快照）：

```bash
python scripts/review/freeze_evidence_analysis.py --session exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/authoritative-session.json --expected-sha256 86af6ea6296f59980f6135b5ffc45571f73a0cd29b8bf9d029ba37d41d46f5f0 --reconciliation exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/reconciliation.json --output exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/frozen-v1 --check
```

本轮完成的是“冻结分析用参考与可用性清单”。下一步可对本冻结的 64 个任务，用相同历史预测做原/审核双参考评估；本次未执行该评估，未运行模型。需要进入输入实验的案例仍须另行冻结具体操作、假设与替代解释。后续人审变更应生成新冻结版本并保留本版，不能覆盖此目录。
