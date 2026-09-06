# 通用模型 L/D 数值测量正式注册 v1

日期：2026-09-05（Asia/Shanghai）。注册标识：`general-model-ld-numerical-measurement/v1`。

**状态：Q1–Q24 已确认，用户已授权立即实施、运行数值预检，并在全部门槛通过后自动扩展完整 dev；实际运行结果待验收。** 本文件固定数值测量口径，运行状态由新产物目录的 manifest、预检报告及分析验收回执记录，不能用本页授权状态代替通过结论。

上位边界见[通用模型正式研究协议](general-model-ld-mechanism-protocol-v1.md)。输入、任务投影和提示沿用[分类输出契约 v2](general-model-classification-contract-v2.md)对应的冻结运行包。本次数值注册发生在已有 dev 开发、词典修订及生成预检之后，属于后续数值测量前的注册，不声称全部研究政策均在首次接触 dev 前确定。

## 1. 研究对象与执行范围

主问题是：**L 和 D 将模型对类别的相对偏好推向哪里，幅度多大，二者组合是否表现为非加性。** 判断更接近 gold 的程度作为辅助读数，效果方向或正收益不构成工程通过门槛。

| 项目 | 冻结范围 |
| --- | --- |
| 模型 | 本地 `Qwen/Qwen3-8B`，与 v2 包相同权重、tokenizer、chat template 和非 thinking 设置。 |
| 任务 | 独立 hate 二分类与 group 多标签分类。 |
| 条件 | `C0/CL/CD/CLD/PL/PD` 六个核心条件。 |
| 数值预检 | 沿用 v2 原 8 条 regression 和额外 24 条 validation，两个 cohort 不重叠，不重新按结果选样。 |
| 完整 dev | 全部 643 条 query，任务和条件内完整配对。 |
| 本次未纳入 | 14B、27B、test、calibration 评测、辅助提取、自由生成扩展、额外干预条件及内部干预。 |

数据和上下文从已有包读取。fit 示例身份和顺序、词典、检索、提示、任务 gold 投影及六条件内容沿用原冻结结果。查询 gold 只可在原始数值产物封存后的分析阶段关联；不能进入候选生成、GPU 计分、候选筛选、批次安排或数值门槛。已有预检帧的选择曾使用 gold 分层，沿用这些 ID 不等于运行时用 gold 选择答案。

精确绑定：

- 配置：[`general_model_ld_numeric_v1.json`](../../../config/stage1/general_model_ld_numeric_v1.json)。
- 输入包：[`package_ref.json`](../../../exps/causal_context/general_model_ld_v2/package_ref.json)，package ID 为 `gmlpkg-98926ff494901fc7ea09b41e0f40548ec4ccc8ab3713726361b7ab45c1f01fc4`。
- 生成预检依赖：[`preflight-02`](../../../exps/causal_context/general_model_ld_v2/runs/preflight-02/preflight_report.json)，须核验其包身份和已通过状态。
- 新产物根：`exps/causal_context/general_model_ld_numeric_v1`。本协议、配置、数值实现及数值运行建立独立身份和源码哈希；v2 包绑定的原源码及 payload 保持其冻结身份。

## 2. 候选空间与计分边界

hate 固定两个完整 JSON 字符串，顺序为 `"hate"`、`"non-hate"`。比较完整答案的条件 logprob，不能把标签当成单个 token。当前 tokenizer 下两者长度分别为 3 和 5，具体 token IDs 须进入候选 manifest 并逐条验证。

group 的标签顺序固定为 `Racism, Region, LGBTQ, Sexism, others`，候选为该五标签的全部 32 个子集。每个子集只使用一种规范、无额外空白的 JSON 数组，内部标签按上述顺序排列；空集合为 `[]`。候选 ID 按 bitmask `0..31` 排序，位 `i` 对应上述第 `i` 个标签，最低位对应 `Racism`。完整枚举不使用 gold，不按候选长度或生成结果筛选。

输入保留现有实际 chat-template 文本及 assistant 非 thinking 前缀。每个候选都要验证联合编码中 prompt token IDs 是完整前缀，答案起点明确、答案非空，且 prompt、答案和 EOS 总长度不超过 8,192。发现 token 边界变化、非法候选或超长即失败，不静默裁剪或改变序列化。

记固定 query `q`、条件 `c` 的 prompt 为 `P`，候选答案 tokens 为 `y_1..y_T`：

```text
ell_t = log p(y_t | P, y_1..y_(t-1))
s(q,c,y) = sum_t ell_t
u(q,c,y) = s(q,c,y) / T
e(q,c,y) = log p(EOS | P,y)
s_eos(q,c,y) = s(q,c,y) + e(q,c,y)
u_eos(q,c,y) = s_eos(q,c,y) / (T+1)
```

主分数为 `s`，只包含答案 token，总分不计 prompt 或 EOS。`u`、EOS 单项 `e`、含 EOS 总分 `s_eos` 和含 EOS token 均值 `u_eos`（代码字段 `mean_with_eos`）保留为辅助，并纳入对应数值检查。EOS token ID 从冻结 tokenizer 绑定，当前 Qwen3-8B 为 151645；PAD 为 151643，两者不可互换。EOS 贡献属于结束偏好，不能事后选择是否纳入主分数。

## 3. 类别偏好与配对比较

hate 主读数为 `m_hate = s(hate) - s(non-hate)`，正值表示在两种规范答案中偏向 hate。

group 对每个标签 `g`，分别汇总包含和不包含该标签的 16 个候选：

```text
m_g(q,c) = logsumexp_{S:g in S} s(q,c,S)
           - logsumexp_{S:g not in S} s(q,c,S)
pi(S | q,c,candidate-space) = exp(s(S) - logsumexp_{S'} s(S'))
pi_g = sum_{S:g in S} pi(S)
m_g = log(pi_g / (1-pi_g))
```

五个 group 标签可共现，其边际质量不要求相加为 1。这里的质量或概率只在固定规范候选空间内归一化，不是自由生成总体上的标签概率，也不是已经校准的真实正确率。

以下比较对每个 query、任务和相应标签的同一种 margin 计算，再汇总；不得先改变样本组成或混合不同任务的 raw margin 尺度。

| 层级 | 比较 | 定义 |
| --- | --- | --- |
| 主 | L | `m(CL)-m(C0)` |
| 主 | D | `m(CD)-m(C0)` |
| 主 | LD | `m(CLD)-m(C0)` |
| 主 | L×D | `m(CLD)-m(CL)-m(CD)+m(C0)` |
| 辅 | L\|D | `m(CLD)-m(CD)` |
| 辅 | D\|L | `m(CLD)-m(CL)` |
| 辅 | 词典相对形态对照 | `m(CL)-m(PL)` |
| 辅 | 示例相对形态对照 | `m(CD)-m(PD)` |

L×D 表示在注册 margin 尺度上的非加性。PL/PD 是已有近似形态和长度控制，没有额外联合 placebo；这些比较不能独立确立内部因果路径、语义机制或排除所有位置和长度影响。

## 4. 主总体与描述性不确定性

主总体是完整 643 条 dev query。每条 query 权重相同，主汇总为 query 级配对 margin 变化的算术均值；另报中位数、分位数及数值可分辨的方向计数。保留无词典命中和 group 空集查询，不按解析成功、效果方向或正确性删除记录。

分别报告 lex-hit/no-hit 切片，以及 gold 集合大小 `0/1/≥2` 切片。当前 dev 的 lex-hit/no-hit 为 223/420；gold 大小 `0/1/2/3/4/5` 为 `252/328/56/7/0/0`。这些是固定帧的描述，不改变主总体权重。大小 2 和 3 可单列附表，大小 3 的 7 条不能支撑稳定的主分层推断。

对 query 做联合、成对 bootstrap，固定 10,000 次、seed 42，报告 percentile 95% CI。CI 范围为四种候选计分口径下，hate 与五个 group 标签 margin 的八项配对比较；其他 gold、集合大小等辅助读数保留逐 query 条件值、配对变化及描述性均值和分位数，不额外增加 CI。每次重抽取保留同一 query 的全部条件、任务与 CI 读数，不能把条件、候选、标签或重复前向当成独立样本；固定分层内的区间在该层内重抽取 query。置信区间是逐项描述性区间，未作同时覆盖或多重比较校正，不报告确认性 p 值，也不能以“区间不跨零”代替预注册的确认性检验。

dev 已参与开发；bootstrap 表达该帧经验分布下的抽样不确定性，不能修复历史暴露、标注误差或选择偏差。数值误差带与 bootstrap 区间分别报告，一个描述计算可分辨度，一个描述 query 间变异。

## 5. 辅助 gold 与敏感性读数

raw 完整封存后再关联已有冻结 gold。hate 取自身投影；group 取全部合法四元组类别并集并去除 non-hate，不从另一任务推断。gold 仅用于以下辅助诊断：候选中 gold 的 rank、候选限制质量及 NLL、gold 相对最佳非 gold 的 margin；group 另报将 gold 集合逐一增删五个标签后的五个 `s(gold)-s(toggle_g(gold))` margin。辅助 gold 诊断不参与候选空间、GPU 分数计算或预检门槛。

每个候选保存逐 token 分数，使以下检查无需增加 GPU forward：

- 用答案 token 均值、含 EOS 总分、含 EOS token 均值替代答案总分，重算对应偏好与配对变化；另报 EOS 单项贡献。
- 在主候选限制分布下报告集合大小质量 `P(k)`、期望集合大小、候选分布 entropy，以及各大小层的分数和质量。
- 另将每个大小层的证据除以该层候选数 `C(5,k)`，即 `a_k = logsumexp_{|S|=k}(s(S)) - log C(5,k)`；在六个大小层上对 `a_k` 归一化得到 `P_equal_k(k)`，报告其期望集合大小和 entropy。它为每个大小层采用相同先验权重，作为去除层内候选数量影响的敏感性，和原始 `P(k)` 同时保留，不替代主候选空间。
- group 标签在固定大小 `k=1..4` 内分别汇总包含/不包含候选，得到条件 margin。`k=0` 或 `k=5` 缺少一侧，记为 NA，不伪造有限 margin。
- 固定大小内增加包含/不包含两侧的 logmeanexp 计数归一化，作为绝对水平敏感性。该修正与 logsumexp margin 相差常数，因此同一大小层的配对变化和交互不变。
- 同时报 gold 集合大小分层，观察查询组成对读数的影响。

总 logprob 对答案长度、标签拼写、固定序列化次序及集合大小敏感。候选大小 `0..5` 各有 `1/5/10/10/5/1` 个集合，聚合质量还涉及候选数量。gold 大小又与 hate、词典命中等变量混杂：当前大小 0 全为 non-hate，大小至少 2 全为 hate，大小 1 仅一条 non-hate。因此这些分层不能独立识别“集合大小机制”。首版不增加反序或全排列候选敏感性，排序依赖作为明确限制随结果保留。

## 6. 数值实现与 batch 几何

运行固定本地模型、BF16、eager attention、`eval`、seed 42、关闭 thinking，完整序列 teacher forcing，首版 `use_cache=false`。采用 attention mask 和一致的因果位置；右侧 padding 不计分。主算术以 FP32 对目标 token 的 logit 与全词表 logsumexp 求差，分块处理以控制内存，不保存全词表 logits。

batch 1 为数值基准。batch 4 的几何固定如下：

| 任务 | batch 4 组成 |
| --- | --- |
| group | 同一 query-task-condition 原子 block 的 32 个候选按固定顺序形成 8 批，每批 4 个。 |
| hate | 同一 cohort、task、condition 内相邻两个 context，每个 context 两个候选，组成一批 4 个。 |

执行顺序固定为任务、条件、数值 query ID、候选顺序，cohort 从该冻结顺序保序过滤；不跨 cohort、任务或条件混合，不临时按长度重排。最终单数 context 尾批保留两个候选，单列该实际 batch 几何。预检中，每个 cohort、每个条件取该 cohort 冻结排序的最后一个 hate context，额外执行一次两个候选的 size 2 尾批，与同一 context 的 batch 1 比较；其全部注册读数差纳入该 cohort 的 batch 最大误差，regression 尾批差也参与 `E8`。该验证不增加 query 或改变候选。不得静默丢弃完整 dev 的第 643 条。所有候选都保存 batch 身份、大小、padding 后长度和位置，以便复算。

CPU 参考与汇总使用固定 4 个 PyTorch CPU 线程，线程数记录在数值运行身份中。

记录 batch 1/4 的 CUDA 同步前向 wall time、归一化耗时、CPU 参考耗时、候选吞吐和峰值显存；模型加载和身份核验开销单列，避免将其混入前向加速比。性能改善不能覆盖任何数值失败，也不能因观察到某个类别更有利而选择 batch。

## 7. 预检误差与门槛

预检先运行固定 regression 8 条，再运行独立 validation 24 条。每个 cohort 的两个任务和六条件均须包含全部候选；batch 1 至少两遍，batch 4 至少一遍，并执行上述 hate 尾批检查。单次完整 32-query pass 为 `32×6×(2+32)=6,528` 个候选、384 个原子 block；两个 cohort 的额外尾批合计 12 个 block、24 个候选前向。重复前向不增加独立 query 数。

同一个 BF16 forward 得到的 logits 上，用 CPU float64 计算参考目标-token logprob，核验 FP32 计分算术。此检查只验证相同 logits 上的算术，不证明 BF16 模型与 FP32 模型等价，也不把另一次模型前向误差混入算术参考。

注册绝对误差作用域包括：答案逐 token logprob、EOS logprob、答案总分、答案均值、含 EOS 总分、含 EOS 均值，以及每条件注册的 margin（四种候选计分口径均包括）。候选身份、token IDs、prompt 边界、帧覆盖和运行身份要求精确一致，不能用数值容差豁免。

1. batch 1 两次重复的每个注册数值差、CPU float64 算术参考误差均须不超过 `1e-4`。基准必须 100% 完整、有限、稳定。
2. 在 regression 8 条上，取全部注册读数 batch 4 及注册 size 2 尾批相对 batch 1 的最大绝对差为 `E8`，要求 `E8 ≤ 5e-3`。
3. 只用这 8 条确定 `epsilon = min(5e-3, max(1e-4, 2×E8))`，随后冻结。
4. 额外 24 条的每个注册读数（含尾批）必须全部满足 `abs(accelerated-batch1) ≤ epsilon`；不能重新用这 24 条估算或放宽 epsilon。
5. 所有候选身份、token 边界、有限性、重复、参考、完整性及 batch 等价门槛均为 100%。任一缺失或失败即阻止扩展；效果方向、accuracy、F1、与自由生成的一致率均不作为门槛。

两个 cohort 按顺序执行；若 8 条阶段出现任一门槛失败，立即保存失败回执并停止，额外 24 条记为“未执行/未验收”。不能将已有完整的小帧结果或未执行的验证帧称为全部预检通过。

主配对方向保存未舍入的原始符号和数值，同时传播每条件 margin 误差界：两条件差的绝对值不超过 `2×epsilon`、四项 L×D 交互的绝对值不超过 `4×epsilon` 时，方向标为“数值未分辨”。落入误差带不代表零效应，跨越误差带也不自动代表统计或科学上有意义。报告不能把原始正负号当作已经稳定判别的方向。

## 8. 原始产物、原子提交与续跑

原子单位为一个 `query × task × condition` 的全部候选。group 为 32 条，hate 为 2 条；一个 GPU batch 可承载两个 hate block，但每个 block 必须独立满足完整性。保存临时结果后校验完整候选集、有限值、身份与哈希，再原子提交。未完成 block 不计入覆盖，不伪装为完整结果。

raw 保存候选 ID 与规范 JSON、答案 token IDs 与逐 token logprob、EOS ID 与 logprob、答案总分/均值/含 EOS 总分/含 EOS 均值、query/task/condition/cohort/repetition、context/prompt/hash、package/model/tokenizer、数值源码/配置/协议、环境与实际 batch 身份。全词表 logits 不写入 raw；查询 gold 不写入 GPU 评分产物。

续跑只允许恢复同一运行身份下已验证的完整 block。续跑前重新验证配置、源码、输入包、模型、tokenizer、GPU/后端/环境和 batch 几何；身份漂移即拒绝合并。已有完整 block 不覆盖；部分 block 重新计算时须保留失败或中断证据。完成顺序不能改变最终冻结计划和覆盖判定。

所有原始记录封存后生成完整性、哈希及计划覆盖回执。gold 关联、汇总、bootstrap 和报告在 raw 封存后执行，并绑定其哈希；分析产物不能回写候选分数。预检回执和完整 dev 回执分别留档。

## 9. 授权执行梯级与验收

本次用户授权的执行顺序如下：

1. 实施独立数值协议、配置、CLI、计分和分析组件；完成必要单元测试、固定候选/边界检查及 v2 依赖身份核验。
2. 核验 GPU 可用状态及本地模型来源，运行 regression 8 条数值检查并冻结 epsilon；全部通过后运行额外 24 条验证。
3. 仅当 batch 4 的完整数值预检全部通过，自动以注册 batch 4 几何扩展到 643 条完整 dev。
4. 完整 dev 必须覆盖 `643×2×6=7,716` 个原子 block、`643×6×34=131,172` 个候选；100% 完整封存后才发布 gold 辅助分析和汇总报告。
5. 任一门槛失败，保留失败阶段、最大误差、失败项及未执行范围，停止自动扩展。不得修改阈值、换样本、删读数或降低覆盖率后将原失败改为通过。

batch 4 失败时，既定备选是另建完整 batch 1 运行；不得把两种 batch 的结果拼成一个主产物。Q24 的执行梯级优先：**本次自动流程遇失败即停止并留下 fallback 入口，不在失败后自动继续完整 dev。** 后续实际执行 fallback 使用新 run、新回执和适用的数值基准，保留原 batch 4 失败记录。

本注册的完成证据是实现与测量验收、可追踪的完整 dev 数值产物及受限的描述性分析。它不等于正式 test 已开放，不证明词典或示例有正收益，也不独立完成跨模型、语义材料或内部机制验证。

## 10. 决策索引

| 访谈项 | 冻结决定 |
| --- | --- |
| Q1–Q2 | 类别偏好方向为主问题；同时完成 hate 和 group。 |
| Q3–Q5 | group 32 集合、五标签汇总；答案总 logprob 和配对 margin 为主；EOS 及 token 均值为辅。 |
| Q6–Q8 | 单一固定顺序规范 JSON；边际 log-odds；候选大小和 gold 大小均作敏感性。 |
| Q9–Q11 | 完整 dev 六核心条件；不增排序对照；验证 batch 加速。 |
| Q12–Q14 | 8 条定容差、24 条验证；传播数值未分辨带；基准与完整性 100% 门槛。 |
| Q15–Q18 | gold 辅助诊断、固定配对比较族、描述性 bootstrap、集合大小敏感性。 |
| Q19–Q21 | 可复核 raw 边界；原子提交/身份一致续跑/独立 fallback；643 条等权主总体与固定分层。 |
| Q22–Q24 | 明确误差作用域与参考阈值；冻结 batch 4 几何且不使用 KV-cache；全部通过后自动完整 dev，失败即停止。 |

运行结果在本页注册时尚未产生；最终状态须读取对应新 run 的验收回执。
