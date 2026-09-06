# General-model L/D numerical measurement v3

日期：2026-09-05（Asia/Shanghai）。状态：**numeric-04 完整 dev-b1、原统计分析、独立全量审计及报告导出均已完成**。643 条 query、7,716 个 block、131,172 个候选全部封存；query gold 仅在 raw 封存后关联，未读取 test。四个 GPU worker 已退出。

- [正式结果解读](RESULTS.md)：hate 对 D 的强正向响应与负非加性；group 的主总分方向必须结合长度归一化及固定集合大小敏感性解释。
- [完整数值报告与表格](results/numeric-04/REPORT.md)、[独立全量审计](audits/numeric-04-full-dev/audit.json)和[报告封存清单](results/numeric-04/report_manifest.json)。192 个端点在六个主分层的全部 CI 已独立复算，最大端点差为 8.88e-15。

- [后备协议](../../../docs/research/experiment-plans/general-model-ld-numerical-measurement-v3.md)与[精简执行配置](../../../config/stage1/general_model_ld_numeric_v3.json)。科学、输入和实际 FP32 runtime 全部继承父注册。
- [冻结计划定位器](plan_ref.json)：`gmlnum3-790a36db6bddb6f6898ecef5781deb6e3c02946ba6bebf93b41204c8876d6e17`，计划文件 SHA256 为 `1fa729179d38f9cd8c8d0afd28e758945b8c1b6a5f226aeb12f7e3279d378cfa`。
- [numeric-04 运行状态](runs/numeric-04/run_manifest.json)。实际四卡 runtime SHA256 为 `983ead5bc67e89e1c330a1ce7f136c6f7da7f8cfe3453282009b6cb9e614572e`，与原 numeric-03 校准来源完全一致。
- [冻结实现独立复审](audits/independent-fallback-review.json)未发现阻断问题；真实 GPU 后备数值验收及其独立复算现已完成。
- [正式后备预检回执](runs/numeric-04/preflight/preflight_report.json)已完整通过：原 8 条最大差 `0.000316619873046875`，24 条最大差 `0.0010089874267578125`；两组重复、候选执行顺序和跨卡读数差均为零。两组 CPU 算术参考分别约 `9.60e-6`、`9.63e-6`。
- 上述回执 SHA256 为 `0c7e35e96ff6686b555de74c923390749b16aa9b89ecba5c4347081343b0404b`。原 `E8=0.00067138671875` 与 `epsilon=0.0013427734375` 未重算；本版四类适用挑战通过不等于原 batch 4 六挑战通过。
- [计划独立审计](audits/plan-independent-engineering.json)、[原 8 条独立复算](audits/numeric-04-preflight-stage1.json)及[完整后备预检独立复算](audits/numeric-04-preflight-final.json)全部通过。最终复算核对完整 12 pass 和 158 个证据文件；未读取完整 dev 的部分分数或 gold。
- [联合测试回执](audits/frozen-code-tests.xml)：311 项测试、339 个 subtest 全部通过。包含原实现回归、后备流程、独立审计及报告导出测试；CPU 环境一次不可用 CUDA 探测警告不作真实 GPU 运行结论。
- [numeric-03 失败](../general_model_ld_numeric_v2/README.md)保留，不改判、不拼接旧分数。原 epsilon=0.0013427734375 继续使用，不重新校准。
- 新预检从头覆盖原 8+24 条：batch 1 重复/CPU 算术参考，以及额外 padding、完整逐前缀、候选执行顺序和跨物理卡四类适用挑战。24 条已参与原 batch 4 验收，不称为新未见集。
- 仅后备全部验收通过才自动扩展 dev-b1；后备再失败不自动搜索其他 profile。完整 raw 封存后才读取 gold、独立审计并发布科学结果。

全部 7,716 block、131,172 候选已封存并复核；47,040 项条件/配对汇总及完整辅助读数均通过独立审计。终态为 complete，目标范围内的全量科学结果已发布；这不等于内部机制、其他模型或 test 阶段已经完成。
