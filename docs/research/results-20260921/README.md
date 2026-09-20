# 2026-09-21 成果归档

本次接续 [2026-09-18 归档](../results-20260918/README.md)，整理自提交 `04c77ccf0cc378c6c7ae34a3b1816faa0a662c20` 以来完成的 12 轮实验，以及材料审核、CPU 分析、实现代码和审计记录。研究已经从“模型看向哪里”推进到“替换哪些内部状态会改变输出”；目前支持若干位置和分支的条件性作用，尚未得到跨语境、跨词条的通用修复方法。

## 完成结果的阅读入口

各行的“完成记录”指向 `results-current.json`；同目录的准备选择器和旧状态文件保留其历史身份。下列运行均已结束，归档不构成重启授权。表内层号均从 0 起算。

| 实验 | 材料与主要结果 | 报告和全部结果 | 完成记录 |
| --- | --- | --- | --- |
| 1. 案例注意力观察 | #541、#3169，各 6 种输入条件；记录答案前及辅助位置的逐层注意力。高关注位置为后续替换提供候选，不能直接作为贡献排序。 | [报告](../../../reviews/case-attention-v1/report-01/REPORT.md) · [结果](../../../reviews/case-attention-v1/results-01/) | [完成记录](../experiment-plans/case-attention-v1/results-current.json) |
| 2. 内容替换 | 44 项材料、88 个输入。#3169 的普通义替换能修复输出；#541 的示例答案替换出现关注转移，但未形成稳定修复。 | [修订报告](../../../reviews/case-content-replacement-v1/report-02/REPORT.md) · [结果](../../../reviews/case-content-replacement-v1/results-01/) | [完成记录](../experiment-plans/case-content-replacement-v1/results-current.json) |
| 3. “嘿嘿”释义与语境 | Q01–Q03 × 原释义／普通义／双义，9 个输入。普通义修复 Q01，却使 Q02 误判；双义未修复 Q01。 | [报告](../../../reviews/hehe-sense-context-v1/report-01/REPORT.md) · [结果](../../../reviews/hehe-sense-context-v1/results-01/) | [完成记录](../experiment-plans/hehe-sense-context-v1/results-current.json) |
| 4. 双义呈现与逐层读数 | 3 条查询 × 6 条件，18 个输入。调整双义顺序、合并成一句均未修复 Q01；保留全部 36 层注意力、MLP 与归一化读数。 | [报告](../../../reviews/hehe-presentation-mechanism-v1/report-01/REPORT.md) · [结果](../../../reviews/hehe-presentation-mechanism-v1/results-01/) | [完成记录](../experiment-plans/hehe-presentation-mechanism-v1/results-current.json) |
| 5. “嘿嘿”位置与层替换 | Q01/Q02 的原释义与普通义，扫描全部 36 层；比较目标词和等 token 数前置位置，保留双方向。目标词附近部分层可改变标签，前置对照未翻转。 | [报告](../../../reviews/hehe-focal-patching-v1/report-01/REPORT.md) · [结果](../../../reviews/hehe-focal-patching-v1/results-01/) | [完成记录](../experiment-plans/hehe-focal-patching-v1/results-current.json) |
| 6. 第17层替换后的传播 | 固定目标词第17层替换后，追踪答案前的全部 36 层状态。第23层 MLP、第26层注意力、第28层 MLP 出现与最终分数变化同向的增量，也保留后续抵消和归一化影响。 | [报告](../../../reviews/hehe-bridge-v1/report-01/REPORT.md) · [结果](../../../reviews/hehe-bridge-v1/results-01/) | [完成记录](../experiment-plans/hehe-bridge-v1/results-current.json) |
| 7. 单分支恢复 | 第17层目标词替换后，分别将答案前第26层注意力或第28层 MLP 恢复为接收条件原本输出；8 个方向均减弱上游替换效应。 | [修订报告](../../../reviews/hehe-branch-restore-v1/report-02/REPORT.md) · [结果](../../../reviews/hehe-branch-restore-v1/results-01/) | [完成记录](../experiment-plans/hehe-branch-restore-v1/results-current.json) |
| 8. 两处联合恢复 | 同时恢复第26层注意力与第28层 MLP；四个方向削弱上游效应约 33.0%–56.8%，均强于单处恢复，仍有明显剩余效应。 | [报告与结果](../../../reviews/hehe-joint-restore-v1/results-01/REPORT.md) | [完成记录](../experiment-plans/hehe-joint-restore-v1/results-current.json) |
| 9. 新“嘿嘿”案例迁移 | T01–T04，8 个释义条件输入；沿用第17层替换和第26／28层恢复。T01/T02 复现普通笑声受贬义释义影响，但固定替换仅修复其中 T02；T03/T04 的边界结果保留。 | [解读](../../../reviews/hehe-transfer-v1/interpretation-01/REPORT.md) · [完整报告](../../../reviews/hehe-transfer-v1/results-01/REPORT.md) | [完成记录](../experiment-plans/hehe-transfer-v1/results-current.json) |
| 10. 三词条机制扩展 | 京巴／垃圾／公交车，共 12 条查询 × 无词典／贬义／普通义，36 个输入。固定干预规则仅新增 J01 的正确判断，总体由 8/12 到 9/12；其余反例不剔除。 | [解读](../../../reviews/cross-term-mechanism-v1/interpretation-01/REPORT.md) · [完整报告](../../../reviews/cross-term-mechanism-v1/results-01/REPORT.md) | [完成记录](../experiment-plans/cross-term-mechanism-v1/results-current.json) |
| 11. 六条新“京巴”语境 | J05–J10，18 个输入。宠物语境延续分数变化，但没有新增标签修复；反对辱称的 J07/J08 仍误判。J08/J10 的目标词状态相同，相同替换产生不同最终效应。 | [解读](../../../reviews/jingba-context-v1/interpretation-01/REPORT.md) · [完整报告](../../../reviews/jingba-context-v1/results-01/REPORT.md) | [完成记录](../experiment-plans/jingba-context-v1/results-current.json) |
| 12. “京巴”第26层注意力恢复 | 保留 J05–J10，在第17层目标词替换后恢复答案前第26层注意力；9/12 个方向减弱效应、3/12 增强，所有标签不变。 | [修订解读](../../../reviews/jingba-attn-restore-v1/interpretation-02/REPORT.md) · [完整报告](../../../reviews/jingba-attn-restore-v1/results-01/REPORT.md) | [完成记录](../experiment-plans/jingba-attn-restore-v1/results-current.json) |

另收录[条件差距、每层新增差距与位置替换的 CPU 对照分析](../../../reviews/hehe-gap-patch-interpretation-v1/analysis-02/REPORT.md)。它整合已有结果，没有增加一次 GPU 实验。

## 当前可以支持什么

**释义会改变判断，但适合某一语境的改动可能损害另一语境。** Q01 是参考答案“无”的普通笑声用法，Q02 是参考答案“有”的攻击语境。将“嘿嘿”换成普通义能纠正 Q01，却使 Q02 出错；双义呈现的几种调整也没有同时解决这两个问题。这是继续区分“词条含义”和“全文语境”的直接理由。

**局部状态和后续分支均能影响分数，但作用不等于完整解释。** 激活替换是将一种释义条件下某位置的内部状态放入另一条件，观察最终“有／无”评分怎样改变。第17层目标词替换及第26／28层恢复支持这些位置在特定干预下的作用；恢复效应不可以相加成独立贡献，也没有消除全部上游效应。答案方向投影则是用最终输出头读取中间状态，观察其偏向；它与注意力权重、最终标签、干预效应分别报告，不能互相替代。

**迁移结果保留了机制线索，也明确限制了通用修复的主张。** 新“嘿嘿”材料与部分“京巴”语境出现同类分数响应，但固定规则没有普遍修复错误。垃圾、公交车以及反对辱称的错误案例都保留在结果中。T03 按用户决定沿用原标签“有”；G03 按本轮审核采纳“无”，源标签仍单独保留。人工构造材料、历史暴露和相邻位置对照不匹配状态范数等限制见各报告，不把这些材料称作独立确认。

**J08/J10 的相对变化需要与绝对量同时阅读。** 第26层恢复使二者干预效应差距缩小约 44%，但两方向绝对缩小量仅约 0.194、0.395 分，且标签均未改变。它支持全文后续部分会影响相同局部改动的处理，尚不足以支持把该分支作为主要修复手段；词后文本、长度和位置也同时不同，不能单独归因于立场理解。

## 材料、复核与版本

- 三批新材料的候选、人工采纳与来源分别保存在[“嘿嘿”迁移材料](../../../reviews/hehe-transfer-candidates-v1/)、[三词条扩展材料](../../../reviews/cross-term-mechanism-candidates-v1/)、[六条“京巴”材料](../../../reviews/jingba-context-candidates-v1/)。早期替换和释义材料随各实验的准备目录归档。
- 单分支恢复的原报告生成步骤曾因复制范围检查失败而停止，后经 CPU 修订完成；跨词条机制的独立审计也保留原失败记录和严格比较器修订。相关恢复没有重启 GPU 或放松原运行门槛；以完成记录指向的最终审计、报告为准。
- “京巴”第26层恢复的 `interpretation-02` 修正了旧解读中一处 J08 方向说明；数值、图表及封存科学结果没有改变。历史版本按原字节保留。
- 本次仅整理已完成成果。“整段查询第17层替换”、逐头干预和新的通用修复实验仍是待决定的路线，没有新增数据或执行结果。

## 网站与仓库的覆盖范围

[网站部署选择器](../../../deploy/case_attention/digitalocean-sgp/current.json)仍指向 `incremental-20260919-04`，包含注意力、内容替换、释义语境、双义呈现和目标词替换的结果。后续传播、分支恢复、迁移和“京巴”实验的报告已纳入本次仓库归档，尚未增量部署到网站。本次 Git 同步没有修改线上服务。

归档包含对应查看器源码、部署脚本和部署验证回执。大型逐 token 注意力显示文件及重复托管构建包留在本地，因此仓库中的部分交互 HTML 需要本地显示数据；远端阅读优先使用本页链接的报告、静态图和汇总表。

## 归档范围与字节检查

纳入研究文档、源代码、完整冻结输入、审核与采纳来源、结果汇总表、图表、解释报告、数值验收摘要、运行元数据、CPU 检查和独立审计回执。未纳入原始逐次评分及全词表向量、激活数组、密集注意力显示数据、模型与环境、缓存、运行日志、临时锁／控制文件、重复托管包、合成测试的批量产物、私密会话或凭据。它们仍保留在本地；CPU 合成检查的回执不是科学结果。

这是成果归档，不是完整运行备份。完整数值重放仍需要原运行数据、模型与环境。冻结文件中的机器绝对路径保持原样，可按仓库根目录映射；本页的相对链接用于远端导航。本次没有新增模型前向、人工裁决，也没有改写已封存的科学文件。

[`manifest.json`](manifest.json) 记录此次归档文件的路径、字节数、SHA256 和排除类别，不是整个仓库的总清单。本目录自身由 Git 提交绑定，避免清单自引用；`.gitattributes` 禁止相关冻结文件发生换行转换。

```bash
python docs/research/results-20260921/verify.py
```

上述只读程序复用上次归档实现。[`verification.json`](verification.json) 记录文件格式、冻结引用、隐私模式扫描、导航与暂存字节的检查。它验证归档交付，没有重复运行各实验的科学数值审计；科学审计以原始回执为准。

[`historical-references.json`](historical-references.json) 单独列出 11 处仍指向旧字节的历史引用及对应说明：封存前开发快照、旧网站版本、初始 UI 包和已修复的清单自引用错误。这些记录按原样保留，不能替代最终冻结或完成凭据；本次没有为了让旧引用匹配而改写历史文件。该文件还记录 5 处认证负例测试造成的扫描误报，测试要求返回 HTTP 401，实际凭据不在归档中。
