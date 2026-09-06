# General-model L/D numerical measurement v2

日期：2026-09-05（Asia/Shanghai）。当前状态：`numeric-03` 在 24 条验收的 batch 4 比较中失败并停止，完整 dev 未启动、query gold 未读取，四个 GPU worker 已退出。`numeric-02` 在原 8 条首遍阶段主动中断，部分分数不复用。尚未发布完整 dev 科学结果。

## 正式失败回执

- [预检报告](runs/numeric-03/preflight/preflight_report.json)及[终态运行回执](runs/numeric-03/run_manifest.json)均已封存，失败为 `validation-batch-compatibility-gate`。报告 SHA256 为 `d8175e894e007b1ff3facc8132de3215576d459b0b05c92ad9645a4188f8c648`。
- 原 8 条的六类挑战全部通过：`E8=0.00067138671875`，由注册公式冻结 `epsilon=0.0013427734375`。跨物理卡、相同 batch 4 几何的全部读数差为零。
- 24 条两次 batch 1 完全一致，同 logits CPU 参考最大差 `0.000009625574207916543`；batch 4 最大差 `0.001621246337890625`，超过冻结 epsilon。其余五类 24 条挑战未执行，不能称预检通过。
- 最大差位于 `844:group:CD/group-24/answer_sum`。独立审计发现超阈值的是三个 record 的答案总分及含 EOS 总分，共六个读数；token、均分、EOS 和 margin 未单独超阈值。这不豁免已注册的候选总分门槛。
- [独立失败审计](audits/numeric-03-preflight-engineering-failure.json)核验已完成的 11 个 pass、差异算术及哈希链。[输入暴露审计](audits/numeric-03-exposure-design-audit.json)只检查 gold-free 输入，不包含科学效应。
- 后续按父注册已经列明的真实 batch 1 备选另建 [v3 计划与运行](../general_model_ld_numeric_v3/README.md)。不改变本次失败、不提高 epsilon、不复用旧分数拼接主产物。24 条已暴露于本次 batch 4 验收，后备重验不称为全新未见验证集。

## 数值修订

v1 失败记录保持不变。四个原 regression context 的对照显示，BF16 形状差异可以在早期 Transformer 层出现，单独关闭低精度归约或提高输出层精度不足以修复。全 FP32 对照的答案总分最大跨形状差为 `0.00054168701171875`，被探查的单个前缀 token 最大差为 `0.000522613525390625`。这只是开发诊断，不是正式预检通过证明。

- [原 BF16 诊断](diagnostics/bf16-original/manifest.json)
- [关闭 BF16 低精度归约](diagnostics/bf16-no-reduced/manifest.json)
- [BF16 Transformer + FP32 输出层](diagnostics/bf16-head-fp32/manifest.json)
- [全 FP32 对照](diagnostics/fp32/manifest.json)

新实现采用全 FP32、TF32 禁用、eager attention、无 KV-cache，仍为全序列 teacher forcing，仅在答案和 EOS 预测位置进行全词表输出投影。其实际精度身份与父 BF16 自由生成运行分开记录；不宣称逐值等价。

## 冻结与运行

- [修订协议](../../../docs/research/experiment-plans/general-model-ld-numerical-measurement-v2.md)；[配置](../../../config/stage1/general_model_ld_numeric_v2.json)。
- [当前 plan 定位器](plan_ref.json)，ID `gmlnum2-9442ed0ab175496139742e93998dc6641c56ad8df07ad9f653f37c418fd35209`。早期单卡 plan `gmlnum2-c75cb6d9fdd515f4f824feade2a7eea2ad64ca87844201d59202d1b600a5d8a4` 及源码快照保留。
- [numeric-03 运行状态](runs/numeric-03/run_manifest.json)：四张物理 L20 的相同 FP32 副本按原子 batch 组数据并行，现已失败退出。每个候选记录实际物理 GPU，跨卡挑战由真实 UUID 及相同批次几何核验。
- [numeric-02 中断记录](audits/numeric-02-interruption.json)：已原子提交 74 block、928 候选，均来自原 regression 集。停止原因是发现两候选 hate 的置换公式退化为空操作；没有读 gold、没有执行额外 24 条或完整 dev。这些部分分数不用于放行，也不混入新运行。
- 修正后的正式预检涵盖原 8 条与额外 24 条的完整矩阵，除重复/同 logits CPU 参考外，还有 batch 4、hate 尾批、额外 64 padding、完整逐前缀、任务特定成员置换和跨物理 GPU 六类挑战。E8/epsilon 不放宽原 0.005 上限。
- 科学读数和数据范围继承父注册：hate 两候选、group 32 集合、六条件、7,716 block、131,172 候选。query gold 只在全量 raw 封存后加入，不访问 test。

实施测试已通过 220 项及 254 个 subtest，涵盖父版本回归、新内核、诊断脚本、分片生命周期及新运行门槛。独立只读复审未发现明确的计分源码错误，但这些检查不能替代真实 GPU 数值验收；numeric-03 的数值失败以本页正式回执为准。CPU 测试中的不可用 CUDA 探测警告与该真实数值失败不是同一问题；GPU 诊断四份运行均已完成。

前缀参考产物中的 `sequence_tokens` 为完整候选的逻辑长度；实际运行是不同长度的独立前缀。其 `batch_members` 记录逻辑候选覆盖，不代表同时执行的 GPU batch 行。实际为无 padding、batch 1、每个唯一前缀一次前向。

本运行没有完整 dev raw 或科学分析。全量独立审计及报告工具已用合成数据测试，但工具可用不代表科学测量已经完成。后备运行实际通过并封存 raw 后，才可复算候选分数、margin、八个 contrast、分层汇总和 CI，并发布 EOS 与集合大小附表。
