# 2026-09-21 下午成果增量

本次接续[同日上一份归档](../results-20260921/README.md)，汇总提交 `4c02aa79e884f22b0f2c4536084f830b37541436` 之后完成的六轮实验，共3510次真实前向。研究从“整段与局部替换为什么不同”推进到“无需测试答案的固定局部规则，能否保留正确示例的帮助”。目前尚未证明内部规则优于简单分数偏移，也尚无独立材料上的通用修复证据。

优先阅读[四步自主推进的阶段报告](../../../reviews/autonomous-reference-progress-20260921/closeout-01/REPORT.md)。其中四轮为2430次前向；加上此前整段查询486次、词外单独594次，构成本次六轮归档。各轮均已完成、正常释放GPU并通过独立数值复核；本次Git同步没有新增模型前向。

## 六轮实验入口

表内层号从0起算。解读与完整报告分开保存，后续发现不回写旧报告。

| 实验 | 新增问题／操作 | 前向 | 解读与完整结果 | 完成记录／冻结输入 |
|---|---|---:|---|---|
| 整段查询替换 | 第17层整段查询W，对照目标词U和前置P | 486 | [解读](../../../reviews/jingba-query-scope-v1/interpretation-01/REPORT.md) · [完整结果](../../../reviews/jingba-query-scope-v1/results-01/REPORT.md) | [完成](../experiment-plans/jingba-query-scope-v1/results-current.json) · [输入](../../../reviews/jingba-query-scope-v1/prepared-01/ALL-PROMPTS.md) |
| 词外单独替换 | 增加C＝查询除京巴外的位置，检查组合差 | 594 | [解读](../../../reviews/jingba-query-complement-v1/interpretation-01/REPORT.md) · [完整结果](../../../reviews/jingba-query-complement-v1/results-01/REPORT.md) | [完成](../experiment-plans/jingba-query-complement-v1/results-current.json) · [输入](../../../reviews/jingba-query-complement-v1/prepared-01/ALL-PROMPTS.md) |
| 前后区域分解 | 查询前文B、京巴U、后文A的完整组合 | 1026 | [解读](../../../reviews/jingba-query-regions-v1/interpretation-01/REPORT.md) · [完整结果](../../../reviews/jingba-query-regions-v1/results-01/REPORT.md) | [完成](../experiment-plans/jingba-query-regions-v1/results-current.json) · [输入](../../../reviews/jingba-query-regions-v1/prepared-01/ALL-PROMPTS.md) |
| 无词典供体 | 十二条跨词条材料，固定第17层D00→D01目标词／前置替换 | 756 | [解读](../../../reviews/dictionary-free-donor-v1/interpretation-01/REPORT.md) · [完整结果](../../../reviews/dictionary-free-donor-v1/results-01/REPORT.md) | [完成](../experiment-plans/dictionary-free-donor-v1/results-current.json) · [输入](../../../reviews/dictionary-free-donor-v1/prepared-01/ALL-PROMPTS.md) |
| 正确示例混合 | 六条查询，无示例／两单组／四条示例两种次序 | 270 | [解读](../../../reviews/jingba-mixed-demos-v1/interpretation-01/REPORT.md) · [完整结果](../../../reviews/jingba-mixed-demos-v1/results-01/REPORT.md) | [完成](../experiment-plans/jingba-mixed-demos-v1/results-current.json) · [输入](../../../reviews/jingba-mixed-demos-v1/prepared-01/ALL-PROMPTS.md) |
| 示例帮助保留 | 两种混合次序下，使用无参考供体固定替换京巴／前置位置 | 378 | [解读](../../../reviews/jingba-demo-donor-v1/interpretation-01/REPORT.md) · [完整结果](../../../reviews/jingba-demo-donor-v1/results-01/REPORT.md) | [完成](../experiment-plans/jingba-demo-donor-v1/results-current.json) · [输入](../../../reviews/jingba-demo-donor-v1/prepared-01/ALL-PROMPTS.md) |

## 阶段发现与边界

- 位置分解把J06的组合差主要缩小到目标词与后文的条件依赖，但J05方向不同；没有新增分类修复，也没有定位到单一词、头或自然通路。
- 无词典供体无需人工普通义：十二条正确数8/12→9/12，修复J01、保住J03；逐条预测与普通义供体、已有固定+7对照相同。
- 正确示例混合修复J08，两种次序均4/6→5/6，J07仍错。J08其中一种次序仅+0.016，不能称稳健解决了反对辱称识别；四条与两条比较还改变数量及长度。
- 无参考局部替换保住J08的示例帮助并改善其分数，J07更接近正确一侧但仍错；两种次序仍5/6。示例仍能通过其他位置起作用，不能把局部替换等同于删除全部参考信息。
- 在这六条开发材料上，混合示例加此前固定+7得到6/6，内部方法尚无分类优势。这不构成+7的通用有效性证明，也不应据此继续调参后再宣称独立验证。

[统一偏移的CPU诊断](../../../reviews/autonomous-reference-progress-20260921/score-offset-02/REPORT.md)另列使用参考标签的事后排序分析；其阈值区间仅作解释，不是可部署规则。[最终RMS读出诊断](../../../reviews/autonomous-reference-progress-20260921/readout-01/REPORT.md)是代数检查，不是新增内部干预。[下一步最小比较草案](../../../reviews/autonomous-reference-progress-20260921/closeout-01/NEXT-STEP.md)建议先冻结原生／局部规则／+7，在新材料比较修复、损害与成本，再决定逐头定位。

全部材料均为已暴露开发材料或其既有组合；示例用法分组不等于人工审核的完整适用关系。不同条件、方向和工程重复不能当作独立样本。主计划仍需有益／有害正确参考的对照、关系审核及独立任务验证。

## 归档与复核

归档包括六轮冻结输入、协议、实现与测试源码、最终结果JSON／TSV、全部静态图、独立科学审计、释放凭据、解释报告和CPU诊断。执行均已终止；完成记录不构成重启授权。

沿用以前的成果归档范围：模型、环境、大型逐前向向量／状态／格式记录、日志、锁和合成测试输出留在本地；合成测试说明与审计回执保留并明确标为非科学结果。因此仓库是可阅读、可追溯的成果归档，完整数值重放仍需本地原运行数据和模型。冻结清单中的机器绝对路径不改写；本页提供远端相对链接。

[`manifest.json`](manifest.json)固定本增量的路径、字节数和SHA256；本目录由Git提交绑定，避免自引用。`.gitattributes`保护冻结字节不被换行转换。只读校验：

```bash
python docs/research/results-20260921-afternoon/verify.py
```

[`verification.json`](verification.json)记录格式、引用、导航、凭据模式和暂存字节检查；它不代替原科学审计。[`historical-references.json`](historical-references.json)列出封存前开发快照的旧字节引用及证据，原文件保持原样。原成果归档目录保留，根研究索引与AGENTS继续作为可更新入口；核对旧归档时应使用其对应Git提交。

本次仅同步Git仓库。网站仍保留原部署版本；未重启研究任务或修改已封存的结果。
