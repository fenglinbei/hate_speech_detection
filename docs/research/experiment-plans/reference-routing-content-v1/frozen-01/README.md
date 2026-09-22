# 正确参考的选择性利用：读取权重与内容协议 v1

协议身份：`reference-routing-content/v1`。**研究设计冻结；新材料与执行包尚未冻结。** 授权是用户在当前任务中的要求：“可以开始包含‘第18层 A/V 分解、材料与关系审核、强基线、双侧评价和结果分流标准’的下一份协议冻结”。本次完成这一协议阶段，不启动模型推理。

研究问题是：**同一查询的无参考局部状态，能否减少参考造成的不当影响，同时保留有益影响；其中第18层读取权重与 Value 内容分别起什么作用？** 方法收益和内部解释分别评价，不能用其中一项替代另一项。

所有层号均为 **zero-based**：在第17层完整 decoder block 输出替换目标词状态，在紧接着的第18层分解注意力；按从1计数，它们分别是第18、19个 block。A 表示 softmax 后的注意力权重，V 表示 Value 内容。

## 两个阶段

| 阶段 | 固定范围 | 主要回答 |
|---|---|---|
| A：已有材料的机制诊断 | J05–J10，M00/MPS/MSP 的18个原样提示；两混合次序共12个比较 | 在 U 背景中交叉组合原生／U 的 A、V，区分读取、内容与交互 |
| B：新材料的受控评价 | 24条开发、24条确认；四种语用情形；确认含12条新词项查询 | 在强基线、关系审核和配对参考条件下，评价正确方向推进、分类收益、损害和参考贡献 |

A 阶段结果已经历旧研究的材料暴露，只能作为诊断；B 阶段的“确认”指冻结后首次观察该批模型结果，不代表独立自然语料或总体泛化。确认材料及其示例不能用于开发校准。每条查询的两个次序、八个单示例替换对照是重复测量，不增加独立查询数。

## 冻结的文件

| 文件 | 内容 |
|---|---|
| [mechanism.md](mechanism.md) | 第18层 A/V 四格、位置、结构预测、自身替换与复现要求 |
| [materials-and-relations.md](materials-and-relations.md) | 48个材料槽位、开发／确认隔离、材料与关系审核、固定四示例配对设计 |
| [evaluation.md](evaluation.md) | U、强基线、G、修复／损害、双侧参考贡献、成本和统计单位 |
| [decision-rules.md](decision-rules.md) | 数值、机制、任务和双侧评价的独立分流标准 |
| [protocol.json](protocol.json) | 机器可读常量、层号、参数预算、阈值与完成状态 |
| [stage-a-index.json](stage-a-index.json) / [stage-a-jobs.json](stage-a-jobs.json) | 精确绑定18条已有输入和114个注册评分配置；不等同于114次实际 forward |
| [material-slots.json](material-slots.json) / [review-contract.json](review-contract.json) | 新材料构成及待审字段；没有伪造的文本、人工标签或关系裁决 |
| [validate_contract.py](validate_contract.py) | 只读 CPU 校验、配对矩阵与代数检查；不加载模型权重 |
| [manifest.json](manifest.json) | 包文件、依赖协议、已有输入和实现来源的 SHA-256 |

任务、候选 token、人工政策和关系字段继承 [task-applicability-scoring/v1](../../task-applicability-scoring-v1/frozen-01/README.md)。G 继承本任务已确认的 [报告约定](../../../../../reviews/correct-direction-gain-v1/analysis-01/METRIC.md)。本包新增对比规则；不改变这些父规则。

## 当前完成边界

- 已完成：五项研究设计及参数／判断规则冻结，已有 A 阶段输入与评分配置绑定，新材料槽位与审核契约，CPU 只读验证和文件清单。
- 接下来：按冻结槽位构造并审核新材料；A 阶段另建运行实现，验证 A/V hook、原生／U 重建和工程误差；B 阶段绑定完整输入、参考、开发参数及执行配置。
- 本包没有新增人工采纳，没有实现可直接启动的 GPU 控制器，没有运行真实模型。此前16:00窗口已结束，旧 COMPLETE 实验不得重启。新的输入或执行冻结必须引用本清单，不能原位补写本包。

两阶段可以分别准备；A 的结果不得改变本版 B 的固定 U 规则、确认材料构成、阈值或基线参数预算。若结果提示另一方案，应另立探索版本，保留本版结果和失败。

复核命令（仓库根目录）：

```bash
python docs/research/experiment-plans/reference-routing-content-v1/frozen-01/validate_contract.py
```

回执位于 [audits/freeze-01-cpu.json](../audits/freeze-01-cpu.json)。CPU 合同检查不代表真实 A/V 张量、模型格式输出或 GPU 数值已经验收。
