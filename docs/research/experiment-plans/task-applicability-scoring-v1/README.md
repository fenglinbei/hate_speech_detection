# 任务规则、适用关系与主评分：已冻结入口

用户已授权完成本阶段收尾。[current.json](current.json) 选择 [frozen-01 协议](frozen-01/README.md)，状态为 `protocol_freeze_complete`。

- [模型实际任务说明](frozen-01/model-task.txt)：完整 0–4 级锚点，0→无、1–4→有，最终仅输出一个字。
- [人工规则与优先关系](frozen-01/human-rules.md)：沿用既有政策来源，人工未决不映射为无。
- [适用关系字段](frozen-01/relations.md)与 [JSON Schema](frozen-01/relation-record.schema.json)：资料正确性、词典两类适配、示例两维关联、指定词形重合及判题相关性分别记录。
- [主评分规格](frozen-01/scoring.md)：同一下一 token 位置的无减有分数差、固定零阈值、平局与数值未决、配对效应及参考一致性。
- [CPU 验证回执](audits/freeze-01-cpu.json)：6 条关系格式样例、6 项错误记录拒绝检查、8 个输入条件／16 次候选边界检查，以及合成分数和平局、误差传播检查通过。

本次第一步的协议交付完成。具体科学材料、实际人工裁决、执行矩阵和 GPU 数值资格属于下一阶段。本包没有模型运行结果，格式样例不计为已审核实验材料。

此前 discussion 文档和 task-contract-review v1/v2 保留其形成时的历史状态；新实验采用本入口，不再把这些历史稿的“待确认”描述当作当前状态。后续修改新建冻结版本，保留本包和来源哈希。
