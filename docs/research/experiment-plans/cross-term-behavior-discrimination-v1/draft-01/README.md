# 第二轮行为辨别实验草案

日期：2026-09-17。状态：**已构造，等待人工审核；尚未冻结执行，也未运行新增条件。**

本次交付对应用户同意的下一项工作：补齐第一轮异词示例的覆盖，检验不适用词条的影响是否需要同词示例才能改变，并检查反对辱称文本上的作用。设计是在第一轮结果之后提出的探索性补充。

从 [REVIEW.md](REVIEW.md) 开始审核即可。它列出需要确认的四项设计判断、全部新增关系及重点边界；机器可读人工字段均为空。正文沿用上一轮逐条关系表的审核方式，完整来源／目标和依据另见 [relation-review.md](relation-review.md)。现有线上工作台及其历史会话不属于本交付的写入对象。

## 输入规模

| 组成 | 空词典 | 真实词典L | 合计 |
|---|---:|---:|---:|
| 无示例、同词A、同词B：12查询×3背景 | 36个旧输入 | 36个旧输入 | 72个旧输入 |
| 异词A、异词B：G2/G4/J2/J4/B2/B4 | 12个旧输入 | 12个新增输入 | 24 |
| 异词A、异词B：G1/G3/J1/J3/B1/B3 | 12个新增输入 | 12个新增输入 | 24 |
| 核心分析合计 | 60 | 60 | 120＝84旧＋36新 |

旧实验全部120个输入和252个比较保持历史身份。其中36个N条件不加入本轮异词组合，但其结果及解释完整保留。合并两轮共有156个不同输入；不是156个独立案例。

新增科学输入只有36个，执行时所需的历史桥接、数值资格检查等额外forward不在这个科学条件数里。本交付不指定运行窗口、GPU或执行器，不把历史误差界自动用于新输入。

## 审核内容

- S1：上述36个新增条件，以及保留全部查询的设计范围。
- S2：12条新L→异词示例关系和24条新D→查询语义关系，特别是垃圾／废物的候选部分语义联系。
- S3：24条新D→查询规则关系的具体范围、可迁移部分与限制。
- S4：三类问题分别解释：分类修复、L效应的交互、同词与异词整包差异。

新关系共36条：12条L→demo、24条D→query；这是关系数，不等同于需复审36条材料正文。旧84条已采纳关系和12条L→query外部引用复用，新旧共有132条关系。旧11条AI范围观察保持原身份；本轮不按规则等级作二元因果分组。

## 可核查材料

| 文件 | 用途 |
|---|---|
| [REVIEW.md](REVIEW.md) | 人工审核主入口，S1–S4与逐条建议 |
| [INTERPRETATION-PLAN.md](INTERPRETATION-PLAN.md) | 竞争解释、公式和预定结果分岔 |
| [PROMPTS.md](PROMPTS.md) | 36个新增条件的完整user正文与固定system链接 |
| [new-conditions.tsv](new-conditions.tsv) | 新增条件ID、L/D来源、顺序、长度与哈希 |
| [condition-coverage.tsv](condition-coverage.tsv) | 全部120个核心条件及新旧身份 |
| [relations.json](relations.json) | 36条新关系、原文片段、限制和AI来源 |
| [relation-review.md](relation-review.md) | 同一批关系的展开阅读版 |
| [relation-appearances.json](relation-appearances.json) | 每个条件的资料实际出现状态与L→D→query关联 |
| [model-inputs.jsonl](model-inputs.jsonl) | 新36条完整模型输入及token IDs，无查询参考／关系建议 |
| [analysis-plan.json](analysis-plan.json) | 独立分析文件，含原采纳参考和168项比较提案 |
| [historical-bridges.json](historical-bridges.json) | 84个历史核心输入的原请求／评分记录身份；尚未执行新GPU桥接 |
| [review-decisions.json](review-decisions.json) | 全空人工审核模板，不能误读为已采纳 |
| [sources.json](sources.json) | 本草案实际使用的原始文件哈希 |

## CPU核验

使用本地tokenizer核对全部120个核心输入；84个旧输入须与已运行版本的完整消息、chat字节、token IDs和答案边界一致。新增36条关系通过冻结schema、原文片段和身份检查；所有新人工字段保持空值。核对168项公式、资料呈现与历史记录关联。检查不加载模型权重，不进行forward。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .conda/stage1-p0/bin/python scripts/review/prepare_cross_term_behavior_discrimination_v1.py check
```

人工反馈应另建记录并绑定本草案manifest；不会通过修改本草案的空模板冒充原始裁决。接受材料和关系之后再准备独立执行绑定，已完成的第一轮run保持终态。
