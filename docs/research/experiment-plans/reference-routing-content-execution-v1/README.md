# 正确参考利用：GPU 运行前准备

2026-09-22 状态：**材料已整批采纳，CPU 执行准备已封存，未启动 GPU。**

用户审核原材料稿后授权 GPU 运行前准备。新采纳记录绑定48条查询、108条示例与576条关系建议；原稿及原规范协议不变。整批采纳不等于逐条新增审核或最小充分跨度标注。

- [执行包与分阶段预算](../../../../reviews/reference-routing-content-v1/prepared-01/README.md)
- [材料采纳依据](../../../../reviews/reference-routing-content-materials-v1/adopted-01/decision.json)
- [CPU 准备清单](../../../../reviews/reference-routing-content-v1/prepared-01/CPU-READINESS.json)
- [独立输入审计](../../../../reviews/reference-routing-content-v1/audits/prepared-01-cpu.json)
- [文件与历史选择器审核](../../../../reviews/reference-routing-content-v1/audits/prepared-01-pins.json)
- [CPU 开发记录](../../../../reviews/reference-routing-content-v1/checks/development-note.md)
- [规范协议](../reference-routing-content-v1/frozen-01/README.md)

已实现固定17层目标词替换、第18层 A/V 四格、QAS 与强基线、G／分类／双侧评价、开发参数锁定和确认入口限制。11项原生随机 CPU 模型检查、6项运行调度检查、8项评价检查与4项端到端检查通过。真实模型工程资格和开发参数尚未产生。

| 阶段 | 评分配置 | 通常前向 | 最大前向 |
|---|---:|---:|---:|
| A：旧六查询 A/V | 114 | 930 | 1614 |
| B：24条开发 | 1272 | 11136 | 18768 |
| B：24条确认 | 648或960 | 6144或8640 | 10032或14400 |

确认阶段只使用开发集锁定的一个 QAS 分支。预算包括数值重复、前缀、自身重建与格式检查，不将重复提示当作独立样本。新执行必须另有绑定清单的 GPU 指令；前一日的已完成窗口不会被延长。
