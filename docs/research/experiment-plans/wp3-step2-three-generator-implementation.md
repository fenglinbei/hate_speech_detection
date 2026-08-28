# WP3 Step 2：三个候选生成器实施计划

> **状态：IMPLEMENTATION STARTED / MODEL RUN NOT AUTHORIZED**
>
> 日期：2026-08-27（Asia/Shanghai）
>
> 上游手册：`wp3-terminology-evidence-handbook/v1.0`
>
> 手册 SHA-256：`f5a2c5c2141b87074f42fd8502e2886e2913ff309becd8c7358a25f7072a7c57`

## 1. 目标与本轮边界

Step 2 只解决一个问题：从冻结 fit 原文中高召回地提出“可能需要解释的完整表达 mention”。它不
决定条目是否进入 A/B/C/R，不生成正式释义，也不发布 `L_policy`。

首批只实现三个互补生成器：

1. `G1 rewrite_alignment`：最小改写与差异对齐；
2. `G2 direct_mention`：按冻结手册直接提出最小术语 mention；
3. `G3 form_rule`：音形、缩写、混写、Unicode、分隔与 emoji 形式规则。

冻结边界：

- 输入只来自 5,165 条 fit 记录的 `id/content` 投影；
- 不向生成器提供任务字段、旧词典类别、下游预测、dev/test 或 outcome statistics；
- 三个生成器只产生 proposal，不能自动发布、自动分层或自动写入正式词典；
- 所有最终 offset 由代码在 immutable raw content 中解析，拒绝模型给出的数值 offset；
- 保留嵌套、重叠、同面多次出现和生成器分歧，不以交集删除候选；
- 不恢复全量中文 n-gram；G3 只在明确形式触发或冻结 reference index 命中时提出候选；
- 本轮可以实现和测试离线代码，但不调用模型、不联网、不产生付费请求。

现有 A0、A1、80-case、49-case 与 240-case 产物继续只作开发案例和历史审计。新路线使用独立
`wp3-candidate-generators/v2` 契约、配置、checkpoint 与 artifact root，不修改或复用旧批准状态。

## 2. 共用数据契约

### 2.1 Public task

模型任务的完整可见对象固定为：

```json
{
  "task_id": "...",
  "blind_alias": "...",
  "content": "原始文本"
}
```

`task_id/blind_alias` 只用于重放和盲化；prompt 的语义输入只有 `content`。进入 request 前递归扫描
禁止字段，request capture 也必须证明模型可见字段集合没有漂移。

### 2.2 Generator observation

三个生成器先输出统一 observation，而不是直接合并为 entry：

```text
observation_id
record_id / content_sha256
surface / start / end / occurrence_ordinal
generator / generator_variant / mechanism
replacement（可空）
requires_context（可空）
rationale
source（provider/model/prompt/response hash，或 rule version）
```

`surface == content[start:end]` 是硬不变量。`replacement` 只是候选解码，不是已确认 canonical form。

### 2.3 Candidate mention union

按 `(record_id, content_sha256, start, end, surface)` 精确并集：

- 同一 occurrence 的多个 observation 合成一个 candidate；
- 保存全部 `observation_ids/generators/generator_variants/mechanisms`；
- 保存所有不同 replacement hypotheses，不作多数投票；
- `requires_context` 只汇总票数，不决定 B；
- 不合并边界不同的嵌套或重叠 span；
- 每个 candidate 初始均为 `unreviewed`，只能由后续人工操作转为
  `accept/trim/expand/split/reject/defer`。

## 3. G1：最小改写—差异对齐生成器

### 3.1 拆成两个 pass

`surface_decode` 只处理：

- 拼音首字母、同音/近音；
- 字形替代、拆字、混写；
- 数字、符号、emoji 编码；
- 全半角、分隔或其他非标准形式。

`lexical_pragmatic` 只处理：

- 网络俚语、固定搭配；
- 非组合隐喻；
- 不依赖任务类别、但需要特定语用知识才能理解的表达。

拆分可避免一个 prompt 把所有普通负面词都改写为“特殊术语”。

### 3.2 响应与代码校验

模型必须返回完整 `rewritten_text` 和逐项 edits：

```json
{
  "rewritten_text": "...",
  "edits": [
    {
      "source_surface": "J生虫",
      "occurrence_ordinal": 1,
      "replacement": "寄生虫",
      "mechanism": "mixed_script",
      "requires_context": false,
      "reason": "局部混写需要解码"
    }
  ],
  "record_reason": "..."
}
```

规范化器执行：

1. 用 `source_surface + occurrence_ordinal` 在 raw content 中解析 offset；
2. 拒绝重复、重叠、空 replacement、原样 replacement 和超长 span；
3. 按解析出的 edits 由左到右重建句子；
4. 重建结果必须逐字等于 `rewritten_text`；
5. 每个合法 edit 转成一个 G1 observation。

模型不能通过改写整句、删除上下文或偷偷加入未声明编辑来制造候选。结构或对齐失败是显式失败，
不能当作空结果。

### 3.3 默认 provider 路线

开发 pilot 默认由本地 Qwen 对两个 pass 各运行一次。是否在完整 fit 上为 G1 增加第二 provider，
只根据 guideline development set 的互补召回与错误类型决定；冻结后不得根据 sealed 结果临时追加。

## 4. G2：直接术语 mention 提取器

G2 使用冻结手册的通俗六问、span 规则和正反最小对照，要求模型只回答：

> 原文中哪些最小连续表达，如果不了解网络语、缩写、编码形式、固定搭配或特殊语用，就可能无法
> 正确理解？

响应中只提交复制自原文的 `surface + occurrence_ordinal`、mechanism、context flag 和简短理由。
代码解析 offset，并拒绝数值 offset、整句越界、重复 occurrence 与非精确 surface。

开发 pilot 默认分别运行 Qwen 与 DeepSeek，候选取并集。两者一致只作为 provenance 特征，不是
自动 accept；两者分歧进入高优先级人工队列。完整 fit 是否保留两路模型，必须在 sealed 运行前
冻结，不能因为其中一路结果更符合预期而事后选择。

## 5. G3：形式规则生成器

G3 是纯离线 labeling function，只负责召回，不决定含义。首版规则分为：

1. `mixed_script`：Latin 与汉字、Latin 与数字等异常混写连续块；
2. `unicode_nfkc`：全角、兼容字符等 NFKC 后发生变化的最小连续块；
3. `emoji`：emoji 或 ZWJ 序列；
4. `known_variant`：冻结 form reference 中逐字出现的变体；
5. `pinyin_initials`：Latin token 与冻结 reference 的拼音首字母一致；
6. `phonetic_variant`：纯汉字 span 与 reference 的无声调拼音序列相同或满足冻结距离门；
7. `separator_insertion`：去掉冻结分隔符后与 reference form 一致；
8. `orthographic_variant`：冻结视觉/字形变体映射命中。

`form_reference` 只保存 canonical、拼音序列、首字母和有来源的形式变体，不保存任务类别或 verdict。
纯汉字音近检测可以在内部按 reference 的 syllable 长度扫描窗口，但只有匹配 reference 的窗口才
输出；不得把所有窗口写入候选池。

当前环境没有已冻结的拼音 backend。因此首个工程切片实现 backend-independent 接口、混写/NFKC/
emoji/reference 精确规则和可注入 romanizer 的音近规则；正式启用拼音前需固定依赖版本、完整资源
hash、声调和距离策略。backend 缺失时必须显式报告该 rule family 为 disabled，不能假装运行成功。

## 6. Prompt、provider 与 checkpoint 策略

- G1 两个 pass 和 G2 使用不同 prompt version 与独立 response schema；
- temperature 固定为 0，输出只接受 JSON object；
- 每次请求绑定 handbook SHA、prompt SHA、provider、requested/returned model、参数和 content hash；
- checkpoint 以 `(contract_sha256, generator_variant, provider, task_id)` 为主键；
- success、empty、parse_failure、transport_failure 分开记录；失败不能转为空；
- 重试次数和 physical attempt cap 在模型运行授权前冻结；
- raw response 受限保存，公开 observation 只保存安全 hash 与规范化字段；
- DeepSeek 等外部 provider 只接收 public task；本地 Qwen 也使用相同 content-only 投影。

## 7. 开发、冻结与 sealed 评估顺序

### S2.0：离线工程骨架

- 冻结 common schema、config 与 handbook binding；
- 实现 G1/G2 request builder 和严格 response normalizer；
- 实现 G3 离线规则与三路精确并集；
- 用合成案例覆盖重复 surface、嵌套、substring、重叠 edit 和禁止字段；
- 不调用任何模型。

### S2.1：guideline development pilot

- 旧 240-case、A1 200-case、80-case、49-case 只作开发案例；
- 人工按冻结手册给出完整 mention 与 A/B/C/R 后续去向；
- 只在本阶段修 prompt、mechanism enum、G3 规则和解析器；
- 记录每个生成器的独有召回、边界错误、碎片率和解析失败率。

### S2.2：冻结新 sealed mention benchmark

- 从 fit 依据文本属性抽取 240–600 条，不看任务标签；
- 分层覆盖混写、异常字符、长文本、多个潜在表达、模型历史分歧与随机基线；
- 保留隐藏重复项检查人工稳定性；
- 人工完整标注后冻结，不再用于修改本版本 prompt 或规则；
- 若 gate 失败，生成新 v2.x 契约和新的 sealed frame，不能在原 sealed 上调参后重报。

### S2.3：模型运行授权与生成器冻结

运行前提交：任务数、每 provider 请求数、最大 physical attempts、外发范围、预计费用、checkpoint
路径和恢复规则。获得明确授权后才调用模型。正式运行契约至少冻结：

- G1/G2 provider matrix；
- prompt/response schema/hash；
- G3 reference/backend/hash；
- 最大 spans、最大长度和失败策略；
- union policy 与人工 review frame。

### S2.4：全 fit 候选生成

- 对 5,165 条 fit 运行冻结 G1/G2/G3；
- 输出 generator observations、exact union candidates 和失败清单；
- 候选按生成器覆盖、分歧和形式特征排序，仅供人工调度；
- 不设固定候选数或正式词条数；
- 人工按手册执行六种操作和 A/B/C/R 分层，R 记录永久保留。

## 8. Step 2 验收门

冻结门至少要求：

| 指标 | 初始门槛 |
|---|---|
| sealed mention recall（union） | `>= 0.90` |
| sealed exact-boundary F1（union） | `>= 0.85` |
| raw offset 可重放率 | `100%` |
| 模型可见禁止字段 | `0` |
| 未显式记录的 generator failure | `0` |
| 全量 n-gram observation | `0` |
| candidate 自动发布或自动分层 | `0` |

还必须分别报告 G1、G2、G3 的 recall、precision、unique true positives、边界错误类型，以及去掉任一
生成器后的 union recall。总 union 通过不能掩盖某个生成器完全无贡献或产生不可接受噪声。

## 9. 首个工程切片的交付

本轮启动 Step 2 时创建：

- `src/build_lex/terminology_candidate_generators.py`：公共契约、G1/G2 normalizer、G3 和 union；
- `schemas/wp3_candidate_observation_v1.schema.json`；
- `schemas/wp3_candidate_mention_v1.schema.json`；
- `config/stage1/wp3_candidate_generators_v1.json`；
- `src/tests/test_wp3_candidate_generators.py`；
- 本实施计划。

本切片不创建正式 artifact，不写 refs，不触碰旧 A0/A1 目录。后续先完成 S2.1 开发复核与 S2.2
sealed frame，再提交模型运行预算和授权包。

## 10. 当前启动状态

截至 2026-08-27，S2.0 的首个离线工程切片已经落地：

- 手册 v1.0 的版本与 SHA 已写入 config，并在加载时对本地文件重新计算；
- G1 两个 request/response 契约和“edits 必须逐字重建 rewrite”校验已实现；
- G2 content-only request 与 exact occurrence normalizer 已实现；
- G3 已实现混写、NFKC、emoji、reference、拼音首字母、分隔插入和可注入 romanizer 的音近候选；
- 三路 observation 可按 exact occurrence 并集，并保留嵌套、重叠、分歧与全部 replacement hypotheses；
- observation/candidate JSON Schema 已落地；
- 11 项离线回归测试通过，包含重复 occurrence、模型伪 offset、重叠 rewrite、`J生虫` 混写边界、
  `txl` 缩写、`幕刃` 注入式拼音匹配、`easy/easy girl` 嵌套保留和任务字段隔离。

尚未实现或尚未授权：provider HTTP runner、resumable checkpoint、正式 form reference、冻结拼音 backend、
development/sealed review package、全 fit 运行和任何 artifact/ref 发布。下一工程切片是 S2.1/S2.2 的
只读抽样与人工 review frame；模型运行必须等待单独的任务量、外发范围和费用授权。
