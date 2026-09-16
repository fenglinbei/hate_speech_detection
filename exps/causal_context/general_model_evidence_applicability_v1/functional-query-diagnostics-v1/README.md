# 3169：查询适用性功能诊断草案

**执行完成（2026-09-15）：** 用户整批采纳AI标签及建议后，两个四卡阶段均已完成。[采纳记录](feedback-01.json)、[执行协议](EXECUTION-01.md)与[两个正式冻结入口](execution-01.json)单独保存，原草案及其current指针不改写。[结果入口](../functional-query-results-v1/README.md)及[结果解读](../functional-query-results-v1/INTERPRETATION.md)包含全部主口径、反向、未决和探针诊断。7项CPU测试、两阶段字节重建、独立输入／数值核对及全部GPU门槛通过，80个共同提示精确重放，四卡已释放。以下文字保留材料草案交付时的状态。

本轮按用户要求准备材料。`current.json` 选择 draft-01，状态为**待人审草案**，不代表正式输入冻结或GPU运行授权。新查询没有原数据Gold；人工适用性、标签和正文采纳均保持空值。

| 交付 | 内容 |
|---|---|
| [8条查询草案](draft-01/QUERIES.md) | 四类语境各两条，完整正文、AI建议、文本哈希及CPU长度 |
| [分开审核表](draft-01/REVIEW.md) | 义项适用性与作者hate标签分别记录，含立场、对象、依据及未决原因 |
| [紧凑输入矩阵](draft-01/MATRIX.md) | 第一阶段96个等长主条件＋32个删除条件；4个旧查询锚点另计 |
| [预先判读标准](draft-01/INTERPRETATION-PLAN.md) | C/I、查询区分度、跨查询依赖、NCC抵消及成功／失败边界 |

第二阶段限定释义补充预先列出64个条件，仅用F1示例；与第一阶段分开。E1/E2位置移动条件不在本批重复。所有新查询都测试两套示例的嘿嘿／哈哈配对，查询本身的“嘿嘿”不替换。

Q07/Q08没有直接写明黑人，具体所指是独立审核点；hate标签可以与义项适用性有不同的可用状态。A/B两套查询还存在显式所指、叙述人称和攻击形式差异，分别报告，不称为完整的纯语义因子实验。

## 可填写与机器文件

- [适用性CSV](draft-01/applicability-review.csv)、[标签CSV](draft-01/label-review.csv)：各8行，AI列与空的人审列分开，均带完整正文及哈希。
- [完整矩阵CSV](draft-01/input-matrix.csv)、[预定比较JSONL](draft-01/comparisons.jsonl)。
- [无参考标签的查询输入](draft-01/query-inputs.jsonl)、[材料来源及AI依据](draft-01/materials-source.json)、[CPU检查](draft-01/cpu-audit.json)。
- [完整提示预览](../reviews/functional-query-diagnostics-v1/draft-01/prompt-previews.jsonl)、[条件与提示绑定](../reviews/functional-query-diagnostics-v1/draft-01/prompt-bindings.jsonl)：708个去重提示、1416个候选边界，包含两阶段预览及历史锚点。它们是CPU预览，不是现有GPU执行器可直接运行的冻结计划。

沿用现有三栏工作台的字段含义，本批以本地表交付；没有部署新页面或写回原32案例的正式会话。审核反馈应另建有来源的版本，不直接覆盖本草案中的空人工字段。修改文本建立后继草案并重新绑定依赖。

## CPU复核

在仓库根目录执行：

```bash
.conda/stage1-p0/bin/python scripts/review/prepare_evidence_functional_queries.py --check
```

该命令仅使用本地CPU tokenizer重建字节，验证父输入、词条匹配位置、删除位移、候选边界和跨查询NCC背景抵消。没有模型权重加载、模型forward或run子命令；正式GPU计划、评分调度和数值验收属于后续实现。

旧词条实验入口、原／审核参考、原人工记录以及已完成结果保持原身份。本轮主问题是固定资料对这些查询的作用是否依赖语境；不会因某个标签翻转就宣称找到内部义项机制。
