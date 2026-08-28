# Stage 1 P0：WP3 正式付费 build 准备包

> [!CAUTION]
> **本准备包已退役，仅作旧“群体损贬词典”attempt 1/2 的历史与对账记录，不得按本文命令
> 启动新 build。** 主实验资源已改为无类别
> `terminology-understanding-library/v1`；候选 frame、LLM response schema、entry ID、
> renderer、config/source hashes 与成本估算均已变化。新正式执行必须先生成独立的新协议
> readiness、只读 preflight、verification receipt 和外发/计费授权。当前状态见
> [Stage 1 P0 执行状态](stage1-p0-execution-status.md)。
>
> 快照：2026-08-24。
>
> 历史状态：**RETIRED / DO NOT EXECUTE**。attempt 1 在任何 HTTP、target mkdir
> 与 billing ledger 之前 fail-closed；attempt 2 在完成 321/1000 candidate 后因一次 Tavily
> TLS transport EOF fail-closed，且这 321 个 candidate 在旧协议下不可恢复。用户已报告新的
> Tavily key2 可用额度为 3100 次；新 provider-slot checkpoint 与重试预算正在做最终离线验证。
> 本轮没有读取任何 key 值、没有发起 HTTP、没有产生新计费，也没有取得本轮正式 live
> build 授权。`lexicon_ref.json` 与 formal target 仍不存在。

## 1. 旧协议原计划执行量

计划仍选取 1000 个 fit-only candidate。每个 candidate 冻结 3 个 Tavily logical slots
（`query_1/query_2/query_3`）和 3 个 DeepSeek logical slots
（`context_judge/web_evidence_judge/final_lexicon_judge`）；每个 slot 最多 3 个 physical
attempt。全局 physical-attempt cap 会先于每次发送原子占位，因此 per-slot 理论上限不会绕过
provider 总预算。

| 项目 | 名义执行量 | 新协议 physical-attempt cap | 当前额度/余额口径 | 无故障计划余量 |
|---|---:|---:|---:|---:|
| Tavily | `1000 × 3 = 3000` logical requests | **3100** | 用户报告 key2 为 **3100** | **100 attempts** |
| DeepSeek context judge | 1000 logical calls | 计入 9000 总 cap | 按账户余额计费 | — |
| DeepSeek Web judge | 1000 logical calls | 计入 9000 总 cap | 按账户余额计费 | — |
| DeepSeek final judge | 1000 logical calls | 计入 9000 总 cap | 按账户余额计费 | — |
| DeepSeek 合计 | **3000 logical calls** | **9000** | 按账户余额计费 | 最多 6000 retry attempts |
| DeepSeek output token | 典型约 0.427M | 3.072M；含全部重试 9.216M | — | — |

第二轮 8-case live smoke 为 `24/24` Tavily、`24/24` DeepSeek、0 retry、8/8
completed，人工复核 75/80，结论为“GO for controlled WP3 preparation”；它没有授权或
执行正式 build。当前配置在 smoke 后新增 Web 字段上限，因此当前源码/config 与第二轮
smoke 不是逐 hash 相同协议；候选 dry-run 的 8 个 probe 排名与过滤结论保持不变。

attempt 2 使用旧 key/旧协议：964 次 Tavily attempt（963 成功、1 TLS transport 失败）和
963 次成功 DeepSeek attempt。旧协议没有可恢复的 provider-slot checkpoint，因此不得把其
321 个已完成 candidate、legacy cache 或 partial capture 导入新一轮；原受限 billing ledger
继续原样保留，仅用于对账。新的 key2 是独立的用户报告额度，不能用 attempt 2 的 964 次消耗
去扣减；正式启动前仍需在 Tavily dashboard 复核 key2 的即时可用额度。

新协议不再强制 `resume=false`。中断后以同一冻结 intent 打开 checkpoint，复用已成功的
provider slots；未知是否已送达 provider 的 pending attempt 会先转为 `ambiguous` 并计入
physical cap，然后只有在 per-slot 与 provider 全局余量都允许时才重试。3100 是 Tavily
物理请求硬上限，不是“完整重跑”额度：如果 100 次 reserve 被耗尽，系统必须在下一次 HTTP
前 fail-closed。

## 2. DeepSeek V4 Flash token 与费用估算

价格快照来自 DeepSeek 官方
[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/)；2026-08-24
页面当前列出的 `deepseek-v4-flash` 统一单价（USD / 1M tokens）为：

| input cache hit | input cache miss | output |
|---:|---:|---:|
| $0.0028 | $0.14 | $0.28 |

本次正式启动门于 2026-08-24 再次读取官方页面；页面已不再列峰谷双价。价格会变化，
每次启动前仍须重新读取官方页面；实际账单以 API 返回 usage 与供应商账单为准。
DeepSeek 官方也说明 API response 的 usage 是准确计数依据，字符换算只适合粗估，见
[Token & Token Usage](https://api-docs.deepseek.com/quick_start/token_usage/)。

离线估算使用官方 `DeepSeek-V4-Flash-0731` tokenizer：

- Hugging Face revision：`7872f01b1d1fe23eabc4c98b48bffcef5a386062`；
- `tokenizer.json` SHA-256：
  `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`；
- `transformers==4.55.4`；
- 使用 `encoding_dsv4.encode_messages(..., thinking_mode="chat")` 后由官方 fast
  tokenizer 计数；
- fit records 5165、raw candidates 120566、selected candidates 1000；
- 输出按第二轮 8-case JSON 响应循环投影，并取较保守的 pretty JSON + EOS 结果。

典型情景的阶段分解：

| stage | input tokens | projected output tokens |
|---|---:|---:|
| context judge | 1.018M | 0.117M |
| Web evidence judge | 1.890M | 0.117M |
| final judge | 2.172M | 0.192M |
| **合计** | **5.080M** | **0.427M** |

各阶段数值独立显示到 0.001M；合计使用未舍入值 0.427125M，因此显示值相加会有
0.001M 的舍入差。

典型 Web 分布来自第二轮 smoke 的 `[3,5,4,3,0,2,0,1]`，平均 2.25 条 evidence / candidate。
费用公式（token 数以百万计）：

```text
费用 = cache_hit × 0.0028 + cache_miss × 0.14 + output × 0.28
```

| 输入情景 | input | output | 全 miss |
|---|---:|---:|---:|
| 无 Web evidence | 3.535M | 0.427M | $0.61 |
| **smoke 典型 2.25 条** | **5.080M** | **0.427M** | **$0.83** |
| 3 条、320 字摘要、短 URL | 6.463M | 0.427M | $1.02 |
| 9 条、320 字摘要、短 URL | 12.302M | 0.427M | $1.84 |

典型输入若 50% cache hit，估算为 $0.48；全部 hit 为 $0.13。不过 DeepSeek context
caching 是 best-effort，相同公共 prefix 才有机会命中，不能把命中率当作预算保证，见
[Context Caching](https://api-docs.deepseek.com/guides/kv_cache)。主预算因此使用全 miss。

补充压力情景：

- 典型 input/output、每阶段均尝试 3 次：约 $2.49；
- 典型 input、每次均用满 1024 output tokens，正常 3000 calls：约 $1.57；
- 典型单轮 input 同比放大到约 15.240M、9000 attempts 且全部用满 output：
  约 $4.71。

这些是估算包络，不是严格美元硬上限。当前 Web evidence 已具有代码级有限边界
（每候选最多 9 条；title 180、snippet 320、URL 2048、source 80 字符），provider
completion usage 也不得超过每次 1024 tokens；但发送前尚未用官方 tokenizer 实施
`max_prompt_tokens` fail-closed gate。因此不得把上表任一金额描述为数学 hard cap。
按当前统一价，建议 DeepSeek 可用余额至少 $5；若希望给上述 $4.71 压力估算保留至少
50% 运营余量，可按整数金额向上准备 $8。这里的 $5/$8 是余额建议，不是可由当前代码强制的美元
stop-loss；正式授权仍应同时写明请求上限和可接受的估算金额。

## 3. 已完成的付费前安全收口

- 修复 `python -m` 把入口文件作为 `__main__` 重复加载造成的 builder 对象身份不一致：
  formal handoff 现在显式解析 canonical module callable，`train_only` 的精确对象身份、
  module/name 与源码 hash 门禁均保持原样；两条入口回归已进入冻结 receipt 命令清单。
- `candidate_settings.use_jieba=false`，与两轮 smoke 实际使用的无-jieba 路径一致，
  消除冻结环境未声明可选依赖时的静默行为漂移。
- Web evidence 在进入 direct-match、identity、LLM prompt 和 capture 前执行字符上限；
  超长 URL 置空，不生成带省略号的伪 URL。
- formal capability、artifact ID、manifest 和 provenance 同时绑定
  `formal_checkpoint.py`、`train_only.py`、`stage1_preflight.py`、`web_search.py` 及主 LLM
  builder 的源码 hash；
  preflight 后任一源码漂移都会在 mkdir/HTTP 前 hard-fail。
- 新 checkpoint policy 固定为 `provider-slot-checkpoint/v1`，并绑定 authorization、
  config/data/train frame、候选顺序、protocol code 与 provider budgets；不允许把 attempt 2
  或另一冻结 intent 的状态混入本轮。
- checkpoint 使用稳定的 owner-only 单 writer lock；目录强制 mode `0700`、文件强制 mode
  `0600`。每次状态更新使用临时文件、file `fsync`、原子 `replace` 与父目录 `fsync`，恢复时
  会深验 manifest、连续 attempt 序列、slot success 和 candidate commit chain。
- 每个 candidate 精确映射 3 个 Tavily slots 和 3 个 DeepSeek slots。发送前先持久化
  attempt reservation；进程中断留下的 pending/reserved attempt 在恢复时转为
  `ambiguous`，仍计入 Tavily 3100 或 DeepSeek 9000 的 physical-attempt cap，只能在剩余
  per-slot/provider budget 内重试。已成功 slot 直接复用，不重复发送。
- 成功和带 provider JSON 的失败响应均保留 `model` 与 usage；正式审计要求实际返回
  model 属于 `deepseek-v4-flash` family，并复核 prompt/cache-hit/cache-miss/completion/
  total 五项 token 与合计关系。
- build 成功时 provider usage 汇总进入 evidence audit、artifact ID、manifest 与
  provenance；发布 validator 会从 raw captures 重新计算。
- 正式 target 新增发布 `debug_tavily_attempts.jsonl`，只包含 provider/budget scope、
  rank/slot/attempt、status、HTTP/error/retryable、result count 与 dispatched 标记等安全字段；
  不发布 term、query、request payload、response content 或 credential。validator 会把它与
  logical search capture 和 evidence audit 交叉复算。
- checkpoint 是私有 runtime state，不是 scientific artifact；成功发布后仍保留，以便审计
  attempt 计数和防止同一 intent 被误重跑。旧 attempt 2 的 billing ledger 也继续原样保留，
  但它不是 checkpoint，不能恢复或发布。

## 4. 当前离线收口快照

- 历史 config schema 为 `stage1-train-only-lexicon-config/v2`；当前主实验已改为
  `stage1-train-only-terminology-library-config/v1`，所以下列参数全部只描述旧协议：
  `resume=true`，Tavily retry policy 为 `tavily-transient/v1`，budget scope 为
  `stage1-p0-wp3-formal-full-tavily-key2/v1`、cap 3100，DeepSeek cap 9000；
- data：`data-1e2fdc07cf916a7faad2a9f6586b0809ff825883d296527ef6b25ea68ab45843`；
- train partition：
  `tpart-dc73828edb28d36b4da72268914bea6b0ee94c2fa0e2bd37762829e6523eeb19`；
- fit records：5165；raw candidates：120566；eligible：1000；
- `lexicon_ref.json`：不存在；formal target、临时 build 目录、atomic ref temp：均不存在；
- attempt 1：0 Tavily、0 DeepSeek、0 计费；
- attempt 2：321 个 candidate 完成三个 judgement stage；第 322 个在首个 Tavily
  transport attempt 处 fail-closed；
- attempt 2 Tavily：964 attempts，963 成功、1 失败；DeepSeek：963 attempts，全部成功，
  `context_judge/web_evidence_judge/final_lexicon_judge` 各 321；
- provider model：963/963 均为 `deepseek-v4-flash`；usage 为 prompt 1,447,855
  （cache hit 334,080 / miss 1,113,775）、completion 136,471、total 1,584,326；
- attempt 2 DeepSeek 费用：按 2026-08-24 价格快照估算 `$0.195075804`，约 `$0.1951`；
  实际账单以供应商为准；
- 唯一保留文件：`.formal-billing-ledger.u9xzizl9.jsonl`，mode 600、1927 行、357327 bytes，
  JSONL/schema/content-key containment 检查通过，SHA-256
  `49b44039455cdcb04eff68a820f72f11357ff5ab8694f20fe0def0fc243c0d8e`。该 ledger
  不含 term/query/prompt/response/content/credential/error，只供对账，不能 resume 或发布。
- 新 key2 的 3100 次额度来自用户报告；本轮只读 preflight 已从 `.env` 解析凭据存在性，
  仅报告 `env-file` 来源而没有显示明文，也没有调用 Tavily/DeepSeek、没有产生计费；
- 当前改动仅为离线协议加固和测试。新的 canonical config/source hashes、实际 preflight
  结果与完整测试结果必须由最终冻结流程产生，本文不预写未知值。

### Stage 1 全树 verification receipt

- 旧 artifact `vrec-fecaf3a16f66c66352644fb7da3a367a99fba8de7fc5049ae38c305b0dcf093a`
  只证明 attempt 2 当时的旧 source/config tree；checkpoint、retry、budget、capture 与测试清单
  已变化，因此它对当前树**已失效**，不能作为新 live build 的 gate。
- 新 verification receipt 尚未生成。必须等最终离线测试全部通过后由主流程运行并独立
  validate，再把真实 artifact/hash/command count 回填本文；在此之前不得写占位 hash，
  也不得启动 live build。

## 5. 新正式启动 gate 与命令

attempt 2 已在负责人基于 attempt 1 为 0-HTTP/0-计费作明确确认后执行。失败原因是
Tavily transport 层 TLS EOF；DeepSeek 未失败。它运行于旧的不可恢复协议，故那 321 条仍
不得从 billing ledger、legacy cache 或手工拼接 JSONL 恢复。新 checkpoint 只服务于使用
新 config/source/authorization 创建的新 intent，不追认 attempt 2。

待最终测试、新 receipt、额度复核与新 live 授权全部通过后，正式命令为：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .conda/stage1-p0/bin/python \
  -m build_lex.llm_lexicon_builder \
  --dataset full \
  --config config/stage1/lexicon_train_only.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --env-file .env \
  --split train \
  --write-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json
```

不得加入 `--no-resume`；正式 config 与 wrapper 要求 `resume=true`，并只接受冻结的
`provider-slot-checkpoint/v1`。

启动前必须同时满足：

1. 最终离线测试与 checkpoint 崩溃/恢复、单 writer、预算及 capture validator 测试全部通过；
2. 新 verification receipt 已生成且独立 validate，且绑定当前 config/source/schema inventory；
3. Tavily dashboard 显示 key2 即时可用额度不少于 3100，DeepSeek 可用余额满足已接受的估算；
4. 重新运行只读 preflight 并确认 blockers `[]`；该步会读取白名单 key 值用于存在性检查，
   但不会 HTTP，也不会输出 key 值/hash；
5. 负责人针对本轮新 intent 明确授权语料派生内容外发、Tavily 最多 3100 physical attempts、
   DeepSeek 最多 9000 physical attempts 及对应计费。此前 attempt 2 或 smoke 的授权不沿用。

不得 `source .env`，不得临时改为 no-web/cache/no-resume，失败后不得把 partial ledger、
checkpoint 或 engineering smoke 提升为正式 lexicon。只有 process exit 0、ref 发布且独立
`validate_lexicon_ref` PASS 后才可进入 WP4；成功后 checkpoint 继续保留，不随 publication
清理。
