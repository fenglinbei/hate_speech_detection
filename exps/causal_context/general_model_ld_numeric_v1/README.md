# General-model L/D numerical measurement v1

日期：2026-09-05（Asia/Shanghai）。本轮已完成正式注册、实现、测试、全候选输入核验和 regression 数值预检；**batch 4 兼容性未通过，已按冻结规则停止自动扩展**。

## 注册与实现

- [正式数值协议](../../../docs/research/experiment-plans/general-model-ld-numerical-measurement-v1.md)；[数值配置](../../../config/stage1/general_model_ld_numeric_v1.json)。
- [CLI](../../../scripts/stage1/general_model_numeric.py)；独立的 [运行组件](../../../src/diagnostics/general_model_numeric.py)、[计分内核](../../../src/diagnostics/general_model_numeric_kernel.py)、[分析组件](../../../src/diagnostics/general_model_numeric_analysis.py)。
- 输入 package：`gmlpkg-98926ff494901fc7ea09b41e0f40548ec4ccc8ab3713726361b7ab45c1f01fc4`。原 v2 的 13 个源码哈希及 payload 均保持一致。
- 数值 plan：`gmlnum-9d42aaf7255a7cf66da07eeb7d6d03b6ca709006e6d3aff9f4a11ff034a7a2f1`；[定位器](plan_ref.json)。独立封存了配置、协议、候选目录、完整样本顺序和数值源码快照。
- 643 条 dev、两任务、六条件，共 7,716 个计划 block、131,172 个候选；全部 prompt/答案分词边界、序列长度和候选身份检查通过。此计数是计划覆盖，不表示已经完成全量 GPU 计分。
- 测试：本轮 44 项数值测试和 119 项现有 v2 相关回归测试，共 163 项通过。独立 [plan 工程审计](audits/plan-independent-engineering.json)的 30 项静态检查通过。

## 实际预检结果

运行目录：[numeric-01](runs/numeric-01/run_manifest.json)；正式判定：[preflight_report.json](runs/numeric-01/preflight/preflight_report.json)。进程以预期的 gate-failure 状态退出，非未处理异常。

| 检查 | 实测 | 判定 |
| --- | --- | --- |
| 原 8 条完整矩阵 | 每遍 96 block、1,632 候选；batch 1 两遍、batch 4 一遍均完整 | 通过 |
| batch 1 重复最大绝对差 | `0` | 通过，阈值 `1e-4` |
| 同次 BF16 logits 的 CPU float64 参考最大误差 | `1.1473908990922155e-5` | 通过，阈值 `1e-4` |
| batch 4 最大绝对差 `E8` | `3.9062767028808594` | **未通过**，上限 `0.005` |
| hate 两候选尾批最大绝对差 | `1.3002471923828125` | **未通过** |
| 额外 24 条 validation | 未执行 | 未验收 |
| 完整 dev、主效应/交互及 gold 辅助分析 | 未执行 | 未发布 |

本轮总共执行 4,908 个候选序列：三遍 regression 主帧共 4,896 个，六个 hate 尾批 block 共 12 个。没有新增 query，没有读取 query gold 进行计分或分析，没有读取 test，也没有扩展提取或自由生成任务。

最大 batch 差对应 `7648:group:CLD` 的 `group-27` 候选答案总分。完整逐项差异保存在 [batch 差异](runs/numeric-01/preflight/regression-batch-differences.json)和[尾批差异](runs/numeric-01/preflight/regression-tail-differences.json)。不能将这项工程差异当成 L/D 科学效应，或仅凭它确定具体后端根因。

独立 [全预检误差审计](audits/numeric-01-preflight-batch-audit.json)复现了全部四个 pass 的误差和通过/失败判定；同 prompt、候选和目标 token 的对应关系正确，未发现求和或 EOS 错配。差异也影响主读数：288 个条件级主 margin 中，276 个的 batch 差超过 `0.005`；hate 最大为 `1.5472107`，group 最大为 `1.7666854`。这里比较的是同一条件在不同计算形状下的误差，不是跨条件 L/D 效应。

公式给出的 epsilon 上限值为 `0.005`，但 `E8` 本身未通过前置门槛，因此该值不是通过验收的计算误差保证，不能用于放行全量运行。

独立 [首遍数值审计](audits/numeric-01-regression-b1-r0.json)核验了 96 个 block、1,632 个候选、13,440 个答案 token 及 22,752 个数值读数。独立 SciPy/math 复算得到同一 CPU 参考最大误差；主 margin 参考误差最大为 `6.919391967841193e-6`。原始全词表 logits 按注册不保存，离线审计验证的是已存参考值与聚合的一致性。

前向计时单列：batch 1 第二遍 `188.3721 s`，batch 4 主帧 `164.6534 s`，约 `1.14x` 前向加速；这不包含加载、身份核验、分词、归一化、CPU 参考或写盘，且加速不能抵消数值门槛失败。

运行结束后已用 `nvidia-smi` 核实 GPU 0/1/2/3 均为 `0 MiB`、利用率 `0%`。

## 后续边界

本次自动执行到此停止。离线检查还发现 batch 1 的同 prompt、同答案前缀会随完整候选序列长度变化而产生分数漂移。因此，两遍完全重复并不证明跨候选长度的比较可靠，不能直接将既定 batch 1 备选用于完整 dev。后续应先定位并处理前向计分的形状依赖，再按新运行登记完成适用的 24 条基线验收。不能复用这份失败回执放行 dev，也不能拼接 batch 1 与 batch 4 分数、放宽阈值或回写本次失败记录。

独立 [前缀与形状审计](audits/numeric-01-prefix-geometry-audit.json)测得 batch 1 同前缀跨序列长度的最大 logprob 差为 `2.25`。在四个 pass 中，固定 context、答案前缀、padding 后长度和实际 batch size 的组内差为 `0`。[本地内核诊断](audits/numeric-01-shape-diagnostic.json)的 CPU 小型 Qwen3 检查未发现因果 mask、位置编号或右侧 padding 的明显错误；这不能代替实际 8B/CUDA 后端的根因定位，也不能证明修改 padding 就已解决问题。

重新校验当前 plan 的命令：

```bash
.conda/stage1-p0/bin/python scripts/stage1/general_model_numeric.py validate \
  --plan exps/causal_context/general_model_ld_numeric_v1/plan_ref.json
```

该命令只核验身份，不启动 GPU。已完成或失败运行目录不覆盖；恢复普通中断时同一 plan、模型、环境和 batch 身份下的完整 block 才可复用。
