# WP3 S2.1b：当前生成器诊断正式修订

> **状态：ACCEPTED / OFFLINE SUCCESSOR IMPLEMENTED / MODEL CALLS GATED**
>
> 生效日期：2026-08-30（Asia/Shanghai）
>
> 协议版本：`wp3-s21b-current-generator-diagnostic/v1`
>
> 范围：`development-only / non-sealed / non-scientific`

> **Implementation amendment（2026-08-30）：** 本文保留历史决策与 locked raw-gold 边界；G3 form
> lifecycle、prompt v2、runner/checkpoint/vault/evaluator 已在独立 successor 中实现。当前尚无模型调用，
> 详见 [`wp3-s21b-g3-runner-implementation-handoff.md`](wp3-s21b-g3-runner-implementation-handoff.md)。

> **Provider amendment（2026-08-30）：** 本文 §5 中的 Qwen development grid 已在任何模型执行前由
> `wp3-s21b-provider-plan/v2` 替代；raw-gold 与本文其余生命周期不变。当前模型、共同参数和 2,544-slot
> grid 以 [`wp3-s21b-provider-plan-v2.md`](wp3-s21b-provider-plan-v2.md) 为准。

## 1. 修订决定

S2.1 阶段 A 已在任何提案揭示前完成并锁定 424 条 raw-only mention 标注。原阶段 B 的 6,306
条提案由历史 A1、历史双模型与首版 G3 混合构成，不能代表当前 G1/G2/G3。因此从本修订起：

1. 原 6,306 条队列状态改为 `superseded-not-completion-gate`；不再要求穷尽人工决定。
2. 原 frame、raw lock、session、导出包和远端服务必须原样保留，不删除、不覆盖、不改 ID。
3. 已锁定的 424 条标注独立固化为 `wp3-s21-locked-raw-gold/v1`；gold 范围只包括
   mention occurrence 与边界，A/B/C 路由继续只是 development hypothesis。
4. 新阶段 B 只诊断当前 G1/G2/G3 development run；历史 A1/双模型仅进入 archival baseline。
5. 当前 G1/G2/G3 先在同一 424 条 development set 上迭代并冻结；其指标只能称 development
   diagnostics。真正无偏 gate 仍由完全不重叠的 S2.2 提供。

这里的“当前生成器完成”不是先跑 5,165 条 full fit。正确顺序是 424-case baseline → 诊断与迭代
→ development freeze → S2.2 sealed gate → 获批后才允许 full-fit execution。

## 2. 修订依据

以 2026-08-30 最终锁定的 424-case、356-mention raw artifact 为基准，旧 6,306 proposal union 的纯几何
关系为：

| 关系 | 数量 |
|---|---:|
| exact boundary | 218 |
| 与任一 gold mention 完全无重叠 | 5,571 |
| proposal 包含一个 gold | 279 |
| proposal 位于一个 gold 内 | 65 |
| proposal 包含多个 gold | 9 |
| 其他部分重叠 | 164 |

因此旧 union 的 exact-boundary precision 为 `218/6306 = 3.46%`，exact recall 为
`218/356 = 61.24%`，无重叠项占 `88.34%`。这些数值足以证明旧队列适合作为历史错误基线，
但不适合作为当前生成器的穷尽人工完成门。当前 v1 lifecycle 将 post-reveal raw amendment 数固定为
0；若以后确需修订，必须先制定新协议/schema revision，再重算所有数值，不得复用本 artifact。

## 3. 不可变旧生命周期

以下 active frame 继续有效：

- frame ID：`wp3devframe-0a28b5b91dff61db5d087a9dc64c095b01bf85fc64552fed5db05ea17fb47dab`
- payload manifest SHA：`8610262616ad993c7414b290e02daa74c4c8123cee6f4e6afd66cab5fe52415a`
- legacy proposal count：`6,306`

旧 frame 已绑定下列文件 SHA。本修订只新增 v2 文件，禁止修改这些字节：

| 绑定对象 | SHA-256 |
|---|---|
| `src/build_lex/terminology_candidate_review.py` | `2456d9a1a4eb58c6f37aba30be77b912557e0f9df3159dc45bd8fca308d07003` |
| `src/build_lex/terminology_candidate_generators.py` | `66a11209e615bcd6be90977276665118b7f155f32ecf1388b4ae08ea7b37f1bf` |
| `config/stage1/wp3_candidate_generators_v1.json` | `585cc46ceb212f49a4c58893594cebee33e440586d534de904900464c402733b` |
| raw annotation v1 schema | `9d578713e4ff9f77e957d2fa8d1a0a3b59af69b84d623e4c715c460d7d46e569` |
| diagnostic decision v1 schema | `2fe95c3673e569e4f0c8ff5530b39d7ccee31002cb1e461506cb35966ba8afe0` |

禁止原地修改旧 `diagnostic_proposals.json`、private source map、frame ref 或 session；禁止清空旧
decision、回退 phase 或用旧 finalizer 跳过提案。旧 `finalize-dev-gold` 只保留为历史 lifecycle
工具，不再代表 S2.1b 完成。

## 4. S2.1a：locked raw gold

新 artifact：

- schema：`wp3-s21-locked-raw-gold/v1`
- schema `$id`：`https://local.invalid/schemas/wp3_s21_locked_raw_gold_v1.schema.json`
- schema canonical SHA：`cab8287f2510d14c38484ceec218268fc5428e1889ea76957be84b4ae3426bb8`
- kind：`wp3-s21-locked-raw-gold`
- ID：`wp3rawgold-<canonical SHA256>`
- 默认 ref：`wp3_candidate_generators_v2/refs/s21b_locked_raw_gold_ref.json`

构建门：

- session 必须绑定上述 frame/payload，处于 `diagnostic` 且尚未 legacy-finalize；
- 424/424 raw annotations 必须为 `confirmed`；
- raw-lock SHA 必须与当前 raw annotations 完全一致；
- surface、ordinal、offset、mention ID、content SHA 与正式 fit 原文 100% 重放；
- 所有 amendment 逐项校验，v1 只允许 pre-reveal raw amendment，post-reveal 数必须为 0；
- 禁止字段扫描为 0；
- 不要求任何旧 diagnostic decision；
- 构建前后旧 session 字节必须完全相同。

artifact 自带 protocol 与 JSON Schema 副本，使用临时目录、payload manifest、内容寻址 ID 和原子发布。
它包含 `annotations.json`、reviewer declaration、protocol、schema 和 payload manifest；不复制旧
diagnostic decisions，也不回写旧 session 的 `finalized_gold_id`。protocol 对 schema SHA 做实现内硬绑定，
所有锁定计数拒绝 bool/float 类型旁路；output/ref 不能别名到旧 session、frame、frame ref 或 artifact 内。

CLI：

```bash
/usr/bin/python3 scripts/stage1/wp3_candidate_review_v2.py \
  freeze-locked-raw-gold \
  --reviewer-id liaozijie \
  --session-file /path/to/locked/session.json

/usr/bin/python3 scripts/stage1/wp3_candidate_review_v2.py \
  validate-locked-raw-gold \
  --require-current-sources \
  --require-current-validator
```

2026-08-30 已发布并独立验证的 active raw gold：

- raw-gold ID：`wp3rawgold-41f16aabde6fd253485d853319ad949c1507a4caf29ba0f658e7ea28a3d3a7b0`
- payload manifest SHA：`35b655724079678cc5831a944aed6294c179d74470f4d76c22b7af79ab500364`
- validator implementation SHA：`ce4cb18045a90dccf163cdd9324c8ff4e5bbc43723cf8fcaaef4f22609f8df3e`
- source session revision：`269e41c6a1cbef97954a99031f1c378c38e74164a24fe55b2798b54f57746823`
- raw annotations SHA：`1d668c0fccbebad0776393456ab99486aff48034d3779a020f34524e652a5029`
- counts：424 cases、356 mentions、228 empty records、15 raw amendments、0 post-reveal raw amendments、
  0 legacy diagnostic decisions；
- 路由快照：A 112、B 221、C 23；路由仍不构成正式 evidence tier。

早期 `wp3rawgold-8d090…` 产物因协议、schema 与 validator 已收紧，仅保留为归档快照；active ref 不再
指向它，也不得用当前 validator 声称有效。

## 5. S2.1b：current G1/G2/G3 run

current run 只能接收旧 frame/正式 fit 的 `task_id + blind_alias + content` 投影，不能读取 raw gold、
provisional route、历史 proposal、任务标签或人工 notes。run artifact 与 raw-gold artifact 是两个独立
依赖，只在自动评估和 Phase-B-v2 builder 中汇合。

以下 424-case provider grid 是未执行的 v1 历史计划，状态为 `superseded-before-execution`：

| 生成器 | development 路线 | logical tasks |
|---|---|---:|
| G1 | Qwen，`surface_decode` + `lexical_pragmatic` | 848 |
| G2 | Qwen + DeepSeek，provider union/no vote | 848 |
| G3 | offline deterministic rules | 0 model tasks |
| 合计 | 4 个模型任务/record | 1,696 |

v2 的正式 runner 仍必须在任何调用授权前实现并冻结：

- prompt/version/prompt SHA、response schema SHA、handbook/config/implementation SHA；
- provider、requested/returned model、transport origin、参数、thinking 配置；
- `(contract, generator_variant, provider, task_id)` checkpoint 主键；
- success、empty、parse_failure、transport_failure 的显式终态；
- retry policy、physical-attempt cap、timeout、concurrency、外发范围与费用上限；
- request/content/response hash、provider request ID、finish reason 与 usage；
- private raw response vault 与公开 sanitized observation artifact 分离；
- 2,544 个 v2 logical slot 完整或逐项记录失败，failure 绝不能折算为空结果。

当前协议中的 model/network/paid/publication authorization 全部为 `false`。本修订不授权或执行任何
模型调用。

## 6. Phase-B-v2 人工队列

新 frame 必须依赖 locked raw gold 和完整 current-run artifact，并使用新的 content-addressed ID、private
source map、session、finalizer、report 与将来的独立 UI/deploy/state 目录。历史 A1 和历史双模型不得
进入新 completion gate。

full union 的 recall、precision、exact F1、unique TP、ablation 和几何边界错误由程序对 locked gold
全量计算；人工队列不再盲目穷举 full union。确定性队列包含：

1. 全部 missed gold；
2. 全部与 gold 重叠但边界不一致的 current proposals；
3. 每个 generator/provider 的 unique exact TP；
4. 对与 gold 完全无重叠的 FP，按 generator set、provider、mechanism、长度和形式特征固定 seed 分层抽样。

proposal item 继续使用 `accept/trim/expand/split/reject/defer`。missed-gold 必须使用独立 item type 和
原因契约，不能伪装成 proposal 操作。finalize 要求队列 100% confirmed、`defer=0`、full run slot grid
完整；最终指标必须来自 full run，不得使用 sampled queue 充当总体分母。

## 7. G1 v1 历史方案与仍适用的阻断

本节的 provider/参数已由 provider plan v2 取代；下面保留的是 v1 离线契约及仍需处理的 prompt、
normalizer 与 runner 风险。

G1 已实现的离线契约：

- 两 pass：形式解码与词汇/语用改写；
- content-only request，`temperature=0/top_p=1/max_tokens=1024/json_object`；
- 返回完整 rewrite 和最多 8 个声明 edit；
- 代码以 `surface + occurrence_ordinal` 解析 offset，拒绝重复、重叠、原样 replacement 和伪 offset；
- 逐字重建必须等于模型的 rewritten text；
- 每个合法 edit 转成 exact-occurrence observation，再进入不投票 union。

正式运行前阻断：

- 没有 provider runner、checkpoint、failure ledger、raw response artifact 或 task-grid gate；
- model source 由 caller 自报，尚未绑定 task/request/prompt/handbook/config、requested/returned model；
- prompt 只有 version 字符串，未冻结 prompt SHA 和独立 response schema；
- config loader 尚未深验 provider plan、enabled、max edits；
- prompt 未明确“最多 8 项”，normalizer 却在第 9 项 fail；没有 overflow 契约；
- 语义等价、最小边界与普通词排除仍需 locked gold 评估，alignment 只能证明改写可重建。

## 8. G2 v1 历史方案与仍适用的阻断

本节的 Qwen provider slot 已在执行前被 GLM-5.3-Flash 取代；旧契约只作为迁移记录。

G2 已实现的离线契约：

- content-only direct mention prompt；最多 8 个 surface occurrence；
- 历史计划为 Qwen 与 DeepSeek 两路 development provider，按 exact occurrence 并集、不投票；
- response 只含 surface、ordinal、mechanism、requires_context、reason；
- offset 由代码解析，拒绝重复 occurrence、未知 mechanism、伪数值 offset；
- agreement 只作 provenance，不自动 accept 或分配 A/B/C。

正式运行前阻断：

- 同样缺 provider runner/checkpoint/raw envelope/completeness gate；
- loader 实际只锁了 aggregation，尚未锁 provider matrix、enabled 与 max mentions；
- 任意 provider/model/prompt version 目前可以由 caller 自报并通过 normalizer；
- 没有独立 provider-response schema、strict envelope parser 或 returned-model 校验；
- `requires_context` 没有可重复的判定定义，不能作为 B 层证据；
- 缺原文 prompt-injection 防护与 provider-specific thinking 参数；
- 任意 provider 子集都能形成合法 union，无法证明 848 个 G2 slot 完整；
- reason 文本尚未执行 decision-language 清洗，empty `record_reason` 也未 fail closed。

新 G2 与旧双模型不是同一口径：旧 prompt 只找贬损身份称谓/词干，新 G2 覆盖全部非透明术语并有
occurrence/boundary 契约。但 v2 仍需在 locked 424 gold 上分别报告 GLM、DeepSeek 与 union 的指标，
不能仅因 prompt 更新就假定质量改善。

## 9. 后续门槛

下一工程切片是 current-run ingress/runner/checkpoint 与完整性 gate，不是模型调用。完成新增测试并提交
2,544 logical slots 的外发、重试、费用和恢复授权包后，才能请求 development 模型运行授权。

在 current-run、Phase-B-v2、development freeze 完成前：

- 不构建 S2.2；
- 不运行 5,165 full fit；
- 不写 `lexicon_ref.json`；
- 不修改现有远端 legacy service/session；
- 不将 development 指标声称为 scientific 或 sealed 证据。

## 10. 实现验证记录

2026-08-30 最终发布前验证：

- candidate generator、旧 review lifecycle 与新 v2 lifecycle：31/31 通过，其中 v2 13/13；
- loopback review server：6 项通过、3 项因可选依赖未安装而跳过；
- UI core：11/11 通过；
- Playwright E2E 未执行，原因是当前 Node.js `12.22.9`，而现有 Playwright 要求 Node.js `>=20`；
- schema Draft 2020-12 自检、Python compile、JSON parse 与 diff whitespace 检查通过；
- active frame 严格重放通过，五个不可变依赖 SHA 与 §3 完全一致；
- active raw-gold 使用 current sources/current validator 独立验证通过；
- 本地锁定 session 与远端 `/var/lib/hsd-wp3-review/session.json` SHA 均为
  `13895c8f4076b48fd2fe49a49da7247bcaa54809168cb37518c78f2c7935f8c9`，未被修订流程写回。
