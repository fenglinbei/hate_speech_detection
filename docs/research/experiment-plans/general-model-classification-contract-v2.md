# 通用模型分类输出与解析契约 v2

修订日期：2026-09-05。上位研究边界沿用[正式研究协议](general-model-ld-mechanism-protocol-v1.md)；修订依据与历史失败见 [v1 运行说明](general-model-ld-run-package-v1.md#生成预检)。本文件是观察 dev 预检错误后的工程政策修订，不是首次生成前的事前注册，也不将原有冻结规则称为实现 bug。

## 修订边界

v1 的 hate 指令已要求输出 JSON 字符串，解析器也已允许有限格式恢复。本次只调整分类结果的恢复边界和审计记录，不修改研究对象、任务标签或语义判断标准。

- 保持分类 prompt、L/D 材料、检索、示例 ID 与顺序、上下文渲染、token 预算、权重、tokenizer 和解码设置不变。
- hate 仍为独立二分类；group 仍为独立多标签分类，不能从一项结果推导另一项。
- 辅助提取解析政策不变，原 64 条提取帧不扩展、不重选。
- 不按 query、实验条件、模型身份或 gold 设置解析特例，不通过查看分类分数挑选规则。
- 本次 v2 不启动完整 dev、test、跨模型复现或内部干预实验；生成预检通过也不等于正式机制实验就绪。

新增配置位于 `config/stage1/general_model_ld_run_v2.json`，新产物根目录为 `exps/causal_context/general_model_ld_v2`。构建应核对 v1/v2 的共同 query 上下文和 token IDs 一致；修改解析政策和预检帧不能悄然改变模型输入。

## 规范输出

规范形式与现有 prompt 一致。判断是否规范时允许 JSON 自身合法的空白，不以唯一字节序列作为格式标准。

| 任务 | 规范形式 | 标签约束 |
| --- | --- | --- |
| hate | 整段输出为 JSON 字符串，例如 `"hate"` | 仅 `hate`、`non-hate`，不得推断或改写标签。 |
| group | 整段输出为 JSON 数组，例如 `["Region","Sexism"]` | 元素均为 `Racism`、`Region`、`LGBTQ`、`Sexism`、`others` 中的唯一标签；合法空集合为 `[]`。 |

group 数组顺序不影响类别集合，不因顺序不同认定恢复。缺答、非法标签或解析失败不能转成 `[]`；`non-hate` 不是 group 任务的类别。

## 有限恢复

恢复只读取整段模型输出及其任务类型，不查看 gold 或输入文本。每次恢复必须能唯一确定一个已出现的合法答案，并留下固定规则标识。

| 输出形式 | v2 处理 |
| --- | --- |
| hate 裸标签 `hate`、`non-hate` | 保留已有接受能力，标记为恢复，不计作规范 JSON 字符串。 |
| group 裸标签或逗号分隔标签 | 保留已有的合法标签校验与去重要求，标记为恢复。 |
| 完整单字段 JSON 对象 | hate 字段白名单为 `hate/label/hateful/result`；group 为 `group/groups/labels/targeted_group/result`。新增 `result`，其余保留既有规则；字段值仍须通过对应任务校验。 |
| 整段 fenced JSON | 保留完整代码围栏恢复能力；围栏内只允许一个完整 JSON 值，并通过相同的类型、标签与字段白名单校验。围栏外不得有解释文本。 |
| hate 完整单标签花括号形式 | 只允许整段为 `{"hate"}` 或 `{"non-hate"}` 及分隔符周围的空白。两类对称处理，标记为恢复；这种写法仍然不是合法 JSON。 |
| 已有的其他合法非规范表示 | 仅保留原分类规则已允许且答案唯一的类型转换，例如 group 的合法标签字符串；一律记录为恢复，不混入规范格式率。 |

分类 v2 关闭从自然语言或其他外层文本中搜索 `embedded-json` 的行为，辅助提取维持原规则。此次修订同时明确拒绝边界，并非只放宽当前出现的两个错误模式。

以下情形继续判无效：多个答案、多个对象字段、重复 JSON 键、未知字段、未知或重复类别、非有限 JSON 常量、解释文本中的标签、额外前后缀、残缺输出、相互冲突的答案，以及不在明确恢复范围内的非法 JSON。不能扫描全文寻找 `hate` 子串，不能把 `non-hate` 拆成 `hate`，不能根据某个答案看似合理而从冲突结果中择一。

不使用模型二次纠错、语义猜测、默认类别或按 gold 回填。没有截断的格式偏离可以恢复，不代表实际发生 length 截断时可以豁免运行门槛。

## 记录与评分

保留未经修改的原始文本、输出 token IDs、上下文身份及解析版本。分类解析结果新增或明确以下记录：

| 字段 | 含义 |
| --- | --- |
| `valid` | 规范解析或有限恢复后，是否获得一个合法、无歧义的分类值。 |
| `strict_valid` | 是否直接符合本任务规范输出；恢复得到的答案为 false。 |
| `format_status` | `canonical/recovered/invalid/missing` 四类，区分规范、恢复、无效和缺答。 |
| `recovery_rule` | 实际采用的固定恢复规则；规范、无效或缺答不假称采用恢复。 |

每个评测分组同时报告规范格式率、恢复率、最终无效率（含缺答）和单列的缺答率，分母均为完整预定帧。解析后的有效率由规范与恢复两类组成；它衡量可计分性，不是语义正确率。

主分类指标继续使用合法解析或恢复后的值，并将无效、缺答保留在固定分母中。另报固定分母的 `strict_metrics`：只有 `strict_valid=true` 的输出作为有效预测，其余按无效预测处理，不能只在规范输出子集上计算分数。两套指标采用相同任务、标签与评分口径，用于检查结论是否依赖格式恢复；不能把恢复带来的差异自动解释为模型分类能力或 L/D 机制的改变。

预检描述性评分只使用每组重复生成的第一遍，第二遍验证 token 级确定性，不把两次生成当作两份独立样本。辅助提取继续按原字段级政策评分，不借用新增分类 `strict_metrics` 改写提取结论。

## 新预检帧

v2 保留原 8 条 dev query 作为回归帧，并在产生 v2 模型输出前固定另 24 条、与原 8 条不重叠的 dev query。新增帧按 hate 标签、group 数量（0/1/至少 2）及词典命中分层轮取；层内按独立命名空间 `preflight-validation/v2`、seed 42 和 query ID 的 SHA-256 排序。规则物化 ID 与哈希，不能运行后因结果不佳换样本。新增 query 的角色是额外工程验证，不称为独立、未暴露或确认性数据；已有 dev 使用和历史暴露仍然存在。

| cohort | query 数 | 任务 | 条件与重复 | 生成次数 |
| --- | ---: | --- | --- | ---: |
| regression | 原 8 条 | hate、group、extraction | 6 核心条件，每组 2 遍 | 288 |
| validation | 额外固定 24 条 | hate、group | 同上 | 576 |
| 合计 | 32 条不同 query | 按上述帧 | `C0/CL/CD/CLD/PL/PD` | 864 |

原 64 条辅助提取帧保持不变；额外 24 条不因进入分类预检而自动加入提取帧。

验收继续要求记录帧完整、生成 logits 有限，以及同一输入两次输出 token IDs 完全一致。分类输出如以 length 停止，即使文本碰巧可解析也记为无效；辅助提取的 length 停止继续单列诊断，不增加统一完整提取门槛。分类有效率仍以 95% 为门槛，按 `cohort × task × condition` 分别核验：

- regression 每格 8 条，需 8/8 有效。
- validation 每格 24 条，需至少 23/24 有效。
- 不跨 cohort、任务或条件合并后替代局部门槛，不用规范格式率替代有效解析率，也不把 F1 提升加入验收条件。

这些小样本通过门槛仅表示本次工程验收通过，不证明总体有效率的统计下界已达到 95%。若仍失败，保留结果，重新明确下一版本的调整范围，不降低本版本阈值，也不无限追加样本直到平均数达标。

## 版本与留档

v1 包、预检原始输出、评分和 `passed=false` 回执不修改。原 13 个来源文件已归档在 `exps/causal_context/general_model_ld_v1/source_snapshots/preflight-01-v1`，并与旧包记录的哈希核对一致；该归档用于解释和复现旧解析政策。

v2 使用新配置、新源码哈希、新 package ID 和新运行目录。可用 v2 重解析旧 raw 输出作为诊断，必须标明原生成包与新解析版本；这种重放不等于新生成预检，不能据此将 v1 判为通过，也不能代替 v2 的 864 次预定生成。

本文件初始编写时，v2 GPU 预检尚未执行，未记录任何通过结论。后续实际结果须由绑定新包的运行 manifest、原始输出、预检报告和独立复算证据支持；GPU 使用前重新核验空闲状态。完整 dev 和正式 test 的授权边界不因本契约修订而改变。

## 执行入口

以下命令从仓库根目录运行。运行和评分输出目录必须是新路径，不覆盖已有回执；已执行的命令与实际结果见后续验收记录。

```bash
env PYTHONPATH=src .conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py build-dev --config config/stage1/general_model_ld_run_v2.json --output-root exps/causal_context/general_model_ld_v2
env PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 .conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py run-preflight --package exps/causal_context/general_model_ld_v2/package_ref.json --device cuda:0 --output exps/causal_context/general_model_ld_v2/runs/preflight-02
env PYTHONPATH=src .conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py evaluate --package exps/causal_context/general_model_ld_v2/package_ref.json --run exps/causal_context/general_model_ld_v2/runs/preflight-02 --output exps/causal_context/general_model_ld_v2/evaluations/preflight-02.json
```

## 本轮验收

2026-09-05 已构建 v2 并完成 `preflight-02` 的全部 864 次生成。实际生成验收为 **passed=true**，没有启动完整 dev 或 test。包 ID 为 `gmlpkg-98926ff494901fc7ea09b41e0f40548ec4ccc8ab3713726361b7ab45c1f01fc4`。

证据入口：[包引用](../../../exps/causal_context/general_model_ld_v2/package_ref.json)、[运行 manifest](../../../exps/causal_context/general_model_ld_v2/runs/preflight-02/run_manifest.json)、[原始输出](../../../exps/causal_context/general_model_ld_v2/runs/preflight-02/predictions.jsonl)、[预检报告](../../../exps/causal_context/general_model_ld_v2/runs/preflight-02/preflight_report.json)、[评分](../../../exps/causal_context/general_model_ld_v2/evaluations/preflight-02.json)、[独立审计](../../../exps/causal_context/general_model_ld_v2/audits/preflight-02.json)。

| 检查 | 实际结果 |
| --- | --- |
| [单元测试](../../../exps/causal_context/general_model_ld_v2/validation/unit-tests.json) | 119 项通过；测试时源码哈希与新包完全一致。 |
| [输入等价核验](../../../exps/causal_context/general_model_ld_v2/validation/input-equivalence.json) | 全部 18,388 条上下文文件与 v1 字节级一致；fit/dev 投影、检索结果、词典匹配、模型清单及协议快照也完全一致。 |
| 预检帧 | 原 8 条回归、额外 24 条验证互斥；辅助提取 64 条及其顺序不变。 |
| 生成与确定性 | 864/864 完成，432/432 组重复 token 一致，生成 logits 全部有限，0 次 length 停止。 |
| 独立复核 | 包与运行哈希、完整计划/帧、报告复算、864 条 tokenizer 解码以及新旧源码快照均通过。原 8 条对应的 288 次生成，其 raw、token IDs、上下文及停止状态与 v1 完全相同。 |
| 分类验收 | 24 个 cohort-task-condition 格全部达标：23 格有效率 100%，`validation:group:CL` 为 23/24，即 95.83%。 |
| 评分范围 | 只计第一遍；18 个聚合任务/条件格与 30 个 cohort/任务/条件格，24 个分类 cohort 格均附严格口径指标。 |
| 运行环境 | 与 v1 的 runtime identity 完全相同；GPU 0 上运行，结束后已核实四张 GPU 显存占用均为 0。 |
| 生成用量 | 共 9,432 个输出 tokens；单次生成耗时合计 588.52 秒，不包含构建、核验和模型加载时间。 |

第一遍分类状态按完整帧记录如下。此表汇总用于展示格式构成，**不替代逐条件验收**。

| cohort / 任务 | 规范 | 恢复 | 无效（含缺答） | 总数 |
| --- | ---: | ---: | ---: | ---: |
| regression / hate | 43 | 5 | 0 | 48 |
| regression / group | 48 | 0 | 0 | 48 |
| validation / hate | 125 | 19 | 0 | 144 |
| validation / group | 143 | 0 | 1 | 144 |

上述分类输出均无缺答。`2281:group:CL` 输出 `["Racism", "Sexism", "Others"]`，因 `Others` 不属于精确标签空间（合法值为 `others`）记为 `unknown_group_label`。两遍均保留此无效结果，不新增大小写映射，不删除样本，也不改变 95% 门槛；对应验证格仍以 23/24 达标。不能据格式诊断推断其语义标签是否正确。

[旧输出解析重放](../../../exps/causal_context/general_model_ld_v2/diagnostics/preflight-01-parser-replay.json)仅为诊断：旧 192 条分类输出中，184 条原有效标签不变，8 条原无效输出按 v2 恢复；旧 96 条辅助提取解析逐条完全相同。旧 `preflight-01` 的失败回执不变，这份重放没有新生成验收效力。

新包中的 `generation_preflight_passed=false` 是不可变的构建时快照，不随后续运行回写；本次实际生成状态以运行目录的预检报告为准。`formal_test_ready=false` 继续有效。候选答案 margin、其他数值检查和后续机制实验注册仍待完成，不能将本次生成工程验收视为完整机制实验已经完成。
