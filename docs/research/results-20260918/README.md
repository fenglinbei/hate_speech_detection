# 2026-09-18 成果归档

本次按用户要求同步最新结果、文档及对应代码，接续 [2026-09-16 归档](../results-20260916/README.md)。下列完成结果选择器是阅读入口；各准备、暂停和失败记录保留其历史状态，冻结输入、人工采纳来源和原始哈希均按原字节归档。

| 成果 | 完成状态与解释 | 完整结果及审计 |
| --- | --- | --- |
| 三模型跨词实验 | [完成结果选择器](../experiment-plans/cross-model-applicability-results-v1/current.json)、[完整解释](../../../reviews/cross-model-applicability-execution-v1/interpretation-01/INTERPRETATION.md) | [跨模型全部表格](../../../reviews/cross-model-applicability-execution-v1/comparison-01/)、[8B](../../../reviews/cross-model-applicability-execution-v1/results-qwen3-8b-01/)、[14B](../../../reviews/cross-model-applicability-execution-v1/results-qwen3-14b-01/)、[GLM](../../../reviews/cross-model-applicability-execution-v1/results-glm4-9b-chat-01/)、[独立审计](../../../reviews/cross-model-applicability-execution-v1/audits/)、[最终收尾](../../../reviews/cross-model-applicability-execution-v1/final-closeout-01/) |
| 跨词第二轮行为实验 | [完成结果选择器](../experiment-plans/cross-term-behavior-results-v1/current.json)、[完整解释](../../../reviews/cross-term-behavior-execution-v1/interpretation-01/INTERPRETATION.md) | [全部结果](../../../reviews/cross-term-behavior-execution-v1/results-01/)、[执行与独立复核](../../../reviews/cross-term-behavior-execution-v1/launch-01/)、[CPU 审计](../../../reviews/cross-term-behavior-execution-v1/audits/) |
| 跨词首轮 next-token 实验 | [完成结果选择器](../experiment-plans/cross-term-next-token-v1/current.json)、[完整解释](../../../reviews/cross-term-next-token-v1/interpretation-01/INTERPRETATION.md) | [全部结果](../../../reviews/cross-term-next-token-v1/results-01/)、[独立审计](../../../reviews/cross-term-next-token-v1/audits/) |
| 材料、规则与后续路线 | [任务与主评分冻结](../experiment-plans/task-applicability-scoring-v1/frozen-01/README.md)、[三模型科学输入冻结](../experiment-plans/cross-model-applicability-v1/frozen-01/README.md) | [内部干预研究路线讨论](../experiment-plans/internal-reference-utilization-roadmap-20260918.md)；材料草稿、反馈、采纳与执行修订均保留 |

三模型均通过既定数值与输出格式门槛。两组自然语境主材料中，每个模型的 192 个条件均与采纳参考一致；208 项主比较中，93 项三模型方向一致，115 项存在相反方向。因此，连续评分调制不能直接解释为已证明的分类损害或修复。另两组解释性补充材料单独报告，保留所有错误与模型差异。材料仍属一个保守依赖簇中的开发性证据，不是独立确认。

全部 12 条历史查询、36 条外部 N 条件及旧 8B 物理评分身份和个别误差界均保留。GLM 的旧边界诊断、UTF-8 读取失败、新运行及两项独立修订均保留；它们不构成可以重启历史运行的授权。更完整的结论与边界以各原始解释报告和全部结果表为准。

## 归档范围

- 纳入研究文档、源代码、冻结输入、结果表、解释报告、数值验收摘要、运行元数据与独立审计回执。
- 沿用前次成果归档范围：原始逐次评分、全词表向量、模型权重、检查点、缓存、运行日志、临时锁和控制文件留在本地。私密会话与凭据不进入提交。
- 这是成果归档，不是完整运行备份。原始清单仍可能引用上述本地依赖；完整数值重放需要原运行数据、模型与环境。归档校验只验证已纳入文件的字节及其冻结引用，不冒充一次新的数值实验。
- 原文中的机器绝对路径保持不变；可按仓库根目录对应定位文件。上表使用相对路径供远端阅读，不改写冻结文件内部的路径或链接。
- 本次没有新增模型前向、人工裁决、科学分析或内部干预，也没有续跑已结束的实验。

## 字节检查

[`manifest.json`](manifest.json) 记录本次新增或更新文件的路径、字节数与 SHA256，不包含本归档目录自身，也不是整个仓库的总清单。Git 提交绑定本目录文件。`.gitattributes` 为新增冻结材料与对应源码禁用换行转换。

运行只读检查：

```bash
python docs/research/results-20260918/verify.py
```

检查程序复用前次归档的实现。[`verification.json`](verification.json) 记录此次文件格式、冻结引用与提交字节的核验结果。各实验原有科学数值审计仍以表中链接的原回执为准。
