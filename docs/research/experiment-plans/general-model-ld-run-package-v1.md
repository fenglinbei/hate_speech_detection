# 通用模型 L/D 机制 dev 运行包 v1

构建日期：2026-09-04；生成预检验收更新：2026-09-05。主协议：[正式研究协议](general-model-ld-mechanism-protocol-v1.md)。

后续分类解析政策见 [v2 输出与解析契约](general-model-classification-contract-v2.md)。该修订使用新包和新预检，本页 v1 历史结果及失败回执保持不变。

## 当前范围

本包将已确认研究边界实现为可验证的 dev 上下文与本地 8B 运行入口。**它不是正式 test 冻结包，构建状态与 GPU 预检结果分别记录。** CLI 不提供 test 子命令，不下载模型、不调用外部服务。已执行的 GPU 预检见下方“生成预检”节。

- 配置：[`general_model_ld_run_v1.json`](../../../config/stage1/general_model_ld_run_v1.json)。
- CLI：[`general_model_ld.py`](../../../scripts/stage1/general_model_ld.py)。
- 当前包定位器：[`package_ref.json`](../../../exps/causal_context/general_model_ld_v1/package_ref.json)。每个包由 payload 哈希获得独立 `gmlpkg-*` ID，不覆盖旧包。
- 校验源数据/partition/词典身份，固定代码哈希、环境版本、模型与 tokenizer 文件清单，以及协议快照。
- train 只用于核对 partition 和提取 5,165 条 fit；616 calibration 不进入示例池、任务投影或评测。只读取 train/dev 指定 payload 成员，不读取 test 文件。

## 数据与上下文

D 使用全部合法 fit，包括多四元组和合法 null 字段；相同标准化内容同 gold 取最小数字 ID，冲突 gold 整簇排除并记录。查询同 ID/同内容排除。BGE-large-zh-v1.5 本地 CPU FP32、16 线程、batch 32、CLS L2 归一化，最长 512 tokens，截断单列记录。缓存绑定模型全文件树、语料及环境，不复用旧单四元组缓存。

每个 dev 查询固定 10 个示例，类别配额按 `non-hate/Region/Racism/Sexism/LGBTQ/others = 4/1/1/2/1/1`，候选倍数 10、最低相似度 0。多类别入多个候选桶，最终 ID 全局去重并按已冻结 selector 补位和排序。示例选择不使用查询 gold；同一查询的任务和条件从同一清单派生。

构建时发现旧候选倍数 3 会因多类别重叠耗尽 `others` 桶，故在任何新分类/提取输出产生前调整为固定倍数 10。对至少存在 10 个合格候选的类别桶，即使其他配额已选走最多 9 个 ID，仍可补位；若实际合格池仍不足则显式失败，不降低配额或删除查询。此次调整是构建规则修复，不是根据模型分类表现挑选检索策略。

| 任务 | dev 样本 | 条件 |
| --- | ---: | --- |
| hate 二分类 | 643 | 14 个开发条件 |
| group 多标签 | 643 | 同上 |
| 辅助提取 | 64 | 6 个核心条件 |
| 生成预检 | 上述帧中事前选定的 8 条 | 3 任务 × 6 核心条件 × 2 次，共 288 次生成 |

核心条件：`C0/CL/CD/CLD/PL/PD`。另外物化 `L-Definition/L-Category/L-CategorySwap/D-Input/D-Schema/D-LabelShuffle/D-Order/LD-CategorySwap`。定义替换、语义改写、立场反转、语义冲突、实例删除加回及 activation patching 尚未实现，不能从已有条件自动推出相应机制结论。

预检帧按 hate、group 数量、词典命中分层轮取，层内 SHA-256 排序；辅助提取帧包含预检样本并按独立 SHA-256 顺序补足，seed 42。均在新模型输出前固定。此处只是 dev 工程采样，不是约 300 条 test 子集的注册。

PL/PD 用固定自然中性文本替换内容，保留资源单元布局、示例数量和任务输出形态，不保留真实标签或 group 标签基数。完整注入块 token 差须在 `max(8, ceil(2% × 原长度))` 内。它们是近似形态/长度控制，并不精确保持每个内容单元的长度、可学习标签分布或信息量。无词典命中按空资源退化保留；不可实施的类别交换、无实际变化的 shuffle 与构造失败分别记录，禁止静默删除这些记录。

每条 context 保存 messages、实际 chat-template 文本、prompt/token-ID 哈希、长度及渲染 trace。所有条件保留查询，预算为 prompt + 对应最大输出 ≤ 8,192，输出预留为 hate 64、group 128、extraction 512；超长不裁剪。所选帧含任何 `control_valid=false` 都会拒绝执行，包括不可实施或无实际变化的干预，不只是构造失败。`primary_context_ready=true` 只表明工程构造和长度就绪；默认 14 条件能否全部执行须同时查看 `all_interventions_valid`。

## 运行命令

在项目根目录使用 `.conda/stage1-p0/bin/python`：

```bash
.conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py build-dev
.conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py validate --replay
.conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py inspect
```

下面列出 GPU 预检及后续 dev 入口。`preflight-01` 已执行并保留失败回执，不得覆盖；`run-dev` 尚未执行，当前失败回执也不能用于放行该命令。重新预检须使用新运行目录。

```bash
.conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py run-preflight \
  --device cuda:0 --output exps/causal_context/general_model_ld_v1/runs/preflight-01
.conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py run-dev \
  --device cuda:0 --preflight exps/causal_context/general_model_ld_v1/runs/preflight-01 \
  --output exps/causal_context/general_model_ld_v1/runs/dev-01
.conda/stage1-p0/bin/python scripts/stage1/general_model_ld.py evaluate \
  --run exps/causal_context/general_model_ld_v1/runs/dev-01 \
  --output exps/causal_context/general_model_ld_v1/evaluations/dev-01.json
```

运行目录不得已存在；失败保留部分输出和 `failed` manifest，不冒充完整帧。当前不提供原地续跑，修复后使用新目录。dev 可用 `--tasks`、`--conditions` 按完整任务/条件分批，不能传入任意查询子集。正式 dev 生成须有相同 package、GPU、环境、权重、tokenizer 和运行配置的成功预检回执。

8B 运行固定本地 safetensors、BF16、batch 1、eager attention、greedy、seed 42、关闭 thinking；用已有 source lease 在模型加载前后重新核验完整模型/分词器身份。预检验证生成 logits 有限、两次输出 token 相同、每个分类任务/核心条件的有效解析率 ≥ 0.95，不以 F1 提升为门槛。候选答案 margin、不同批形状和内部干预的数值校验仍待补充。

评分保留缺答与解析失败。hate 以 macro-F1/accuracy 为主；group 以逐样本 F1/exact match 为主，固定五标签 micro/macro-F1 为辅。合法 `[]` 与缺答区分。提取按字段独立匹配，target/argument 另报已有 similarity-v1 最优匹配分数和覆盖率，**当前没有绑定/tuple F1 评分**。`evaluate` 目前只输出逐任务、逐条件的 dev 描述性指标，配对翻转、来源效应及 L×D 交互尚待实现；不生成确认性 p 值或内部因果结论。

## 构建验收

本轮已物化的包 ID：`gmlpkg-dd4d0dd7c1d52e14d87c19ca5c041149feb8adb4f49fc8e2bcc4e554c941b1b2`。读取入口仍为上面的 `package_ref.json`；[完整回放校验结果](../../../exps/causal_context/general_model_ld_v1/validation.json)和[缓存重建一致性结果](../../../exps/causal_context/general_model_ld_v1/rebuild_check.json)绑定同一包 ID。

| 验收项 | 实际结果 |
| --- | --- |
| fit 示例池 | 5,165 条全保留，其中多四元组 775 条、多类别 585 条；无重复/冲突内容簇被删除。 |
| dev 检索 | 643 条查询全部选满 10 个示例；既定类别配额、全局去重、同内容排除通过。 |
| 词典命中 | 223 条 dev 有命中、420 条无命中；无命中不从评测帧删除。 |
| 上下文 | 18,388 条：hate 9,002、group 9,002、辅助提取 384。 |
| 长度 | 最长 prompt 2,058 tokens；加上任务输出预留后最大 2,570，低于 8,192。 |
| 控制质量 | 0 个无效对照、0 个构造失败、0 条超长；本帧全部 14 个分类条件可供后续生成。 |
| 检索编码截断 | fit 0 条、dev 0 条。 |
| 自动校验 | 118 项测试通过；全部上下文的配置/源码/来源/投影及 prompt/token 序列回放通过；缓存重建获得完全相同的包 ID。 |
| 模型状态（构建时） | 8B 本地权重/分词器清单已绑定，权重文件树与既有 base 来源一致，尚未生成预检；14B、27B 模型来源仍待绑定。 |
| 数据使用（构建时） | 未读取 test；未进行分类/提取 GPU 推理，也未实施语义材料盲审。 |

当前 payload 合计 181,994,049 字节。构建快照中 `primary_context_ready`、`primary_core_ready` 和 `all_interventions_valid` 为 true；`generation_preflight_passed`、`formal_test_ready` 为 false。这表示开发输入已可供预检，不表示完整机制实验已经获准执行。

## 生成预检

`preflight-01` 已完成全部 288 次调用，但**未通过冻结的解析有效率门槛**。使用 GPU 0 的 NVIDIA L20，原 8B 权重、BF16、greedy、batch 1、eager attention 和非 thinking 设置均未更改；运行结束后已核实四张 GPU 的显存占用均为 0。没有读取 test，也没有继续启动全量 dev。

证据：[运行 manifest](../../../exps/causal_context/general_model_ld_v1/runs/preflight-01/run_manifest.json)、[原始预测](../../../exps/causal_context/general_model_ld_v1/runs/preflight-01/predictions.jsonl)、[预检判定](../../../exps/causal_context/general_model_ld_v1/runs/preflight-01/preflight_report.json)、[独立复算审计](../../../exps/causal_context/general_model_ld_v1/audits/preflight-01.json)、[描述性评分](../../../exps/causal_context/general_model_ld_v1/evaluations/preflight-01.json)。

| 验收项 | 结果 |
| --- | --- |
| 覆盖 | 8 query × 3 task × 6 condition × 2 遍 = 288 次；144 组重复完整。 |
| 确定性、数值 | 144/144 组输出 token 完全一致；生成 logits 全部有限。 |
| 输出长度 | 0 次 length 截断，全部正常 EOS。 |
| hate 解析（第一遍） | C0/PL/PD 均 8/8；CL/CD 各 7/8；CLD 为 6/8。 |
| group 解析（第一遍） | 六个条件均 8/8。 |
| 辅助提取（第一遍） | 宽松解析有效 48/48：直接解析 5、格式恢复 42、部分字段 1；不表示字段正确或完整。 |
| 完整性 | 文件哈希、预定记录顺序、报告复算及 token IDs 解码得到原始文本均通过。 |
| 耗时范围 | 各调用记录的 elapsed 合计约 386.0 秒，不含运行包验证与模型加载；两遍共生成 6,374 tokens。 |

门槛按每个分类任务、每个条件分别要求有效率 ≥95%；本帧每格只有 8 条，因此实际需达到 8/8，不能用跨条件总体有效率代替。`run_manifest.status=complete` 只代表运行完成，`preflight_report.passed=false` 才是本次验收结论。

四条不同 prompt 的无效输出涉及两个 query，两遍共出现八次，均没有拒答、截断或 token 不稳定：

| query 与条件 | 原始输出 | 当前规则下的原因 |
| --- | --- | --- |
| `7648:hate:CL`、`7648:hate:CLD` | `{"result": "hate"}` | JSON 合法，但 `result` 不在冻结的分类字段白名单中。 |
| `884:hate:CD`、`884:hate:CLD` | `{"hate"}` | 不符合 JSON 语法，也不是允许的裸标签。 |

失败原因是生成格式与当前解析契约不一致，不是本轮已经证明这些标签在语义上判断错误。既有源码、提示、门槛、运行包及本次回执均不回写。任何输出约定或恢复规则修订须形成新版本，并通过新预检，不能对这次输出事后改规则后宣布原预检通过。

评分只统计第一遍，18 个任务/条件格各 N=8，保留缺答与部分字段。这些是工程开发小样本上的描述性结果，不用于宣称 F1 稳定提升或 L/D 因果机制成立。候选 margin、内部干预、14B/27B 与 test 均未在本轮验收。

## 正式实验前

1. 明确并修订分类输出/解析契约，物化新版本后重新完成 8B 生成预检；继续补齐候选 margin 和其他数值检查，再锁定最终 dev 政策。
2. 绑定 14B 与 27B 的真实模型来源/后端，按各自 tokenizer 回放相同示例清单和核心条件；不得复用 8B token 预算证明跨模型可运行。
3. 实现按 query 配对的翻转、来源效应及 L×D 交互分析；注册 test 统计比较、置信区间/多重比较、约 300 条子集的确切样本规则及失败处理，随后才开放新 test 物化。
4. 注册并完成必要语义干预材料的有限盲审；在稳定 dev 行为配对上开发实例归因和 8B 内部干预。

工程构建结果以各包的 `readiness.json` 为准，它记录不可变的构建时状态，不随后续生成更新。后续运行状态和预检结果分别记录在运行目录的 `run_manifest.json`、`preflight_report.json`；不得回写已封装的 package。`formal_test_ready` 在本开发入口始终为 false。
