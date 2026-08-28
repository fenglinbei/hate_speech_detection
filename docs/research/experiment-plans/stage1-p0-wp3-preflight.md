# Stage 1 P0：WP3 正式 fit-only 无类别术语理解库离线预检

> **已被试运行门控流程取代。** 本页记录的是旧的 web-first builder/付费预检历史，
> 不得再作为正式建库入口，也不得据此直接发起联网或计费调用。当前唯一实施入口见
> [`wp3-terminology-understanding-library.md`](wp3-terminology-understanding-library.md)，
> 全量 Qwen 扫描必须持有有效的 `span-pilot-decision/v1` 人工 `PASS`。

> 状态更新：2026-08-26。当前为 **PRE-LIVE HOLD / NEW PROTOCOL PREFLIGHT PENDING**。attempt 1
> 在任何 HTTP/计费前 fail-closed；attempt 2 在 321/1000 后因 Tavily TLS transport EOF
> fail-closed，旧 321 个 candidate 不可恢复。主实验资源现已从群体损贬词典改为
> `terminology-understanding-library/v1`；本轮只做离线迁移与测试，尚未对新协议运行仓库级
> preflight、未发起 HTTP、未产生新计费，也未取得本轮 live 授权。旧 preflight/receipt
> 不能覆盖当前新协议。

本预检是正式 fit-only 术语理解库构建的强制前置门。它只读取冻结 config、`.env`
中白名单变量名对应的值，以及 `data_ref` + `train_partition_ref` lineage；不会联网、不会挖候选、不会创建
target/cache 目录、不会打开 SQLite，也不会写 report/artifact。stdout 只报告凭据
来源或缺失状态，不输出凭据值或凭据 hash。

## 当前结论

当前 `config/stage1/lexicon_train_only.json` 使用
`stage1-train-only-terminology-library-config/v1`，canonical config hash 为
`e95d01ec695402299fe61230fb8c97c582c3f54d8d5e4b3fcc9f0dcfa447e811`，并冻结：

- `required_artifact_kind=train-partition`，且只消费 `lexicon_source_partition=fit`；
- `excluded_partition=calibration`、`calibration_contribution_count=0`；
- data/partition 使用 exact portable dependency binding；
- `resource_role=terminology-understanding-library/v1`；候选 objective 为
  `terminology-value/v1`，source policy 为 `content-only-no-task-labels/v1`；
- `min_hate_count_for_llm=0`，候选排序/过滤不使用 hate count、category purity 或 target
  annotation；`keep_all_content_ngrams=true`，高置信 singleton 不在 LLM 前被丢弃；
- LLM 与最终 entry 禁止 `category/categories/targeted_group/hateful`；公开语义字段只有
  term/definition/usage_notes/ambiguity_notes/variants；
- `cache_enabled=false`；
- `cache_path=null`；
- web 每个 candidate 固定 3 个 logical slots、每次最多 3 个结果；Tavily transport 每个
  slot 最多 2 次额外重试，但所有 physical attempts 共用 cap 3100；
- web evidence 在进入 direct-match/LLM 前限制为 title 180、snippet 320、URL 2048、
  source 80 字符；
- LLM 每个 candidate 有 3 个 judgement stage，每个 stage 初次请求加 2 次重试；
- `resume=true`、`resume_require_config_match=true`，checkpoint policy 为
  `provider-slot-checkpoint/v1`，pending/reserved attempt 的恢复策略为
  `count-and-retry-within-budget/v1`；
- DeepSeek physical-attempt cap 为 9000；
- `use_jieba=false`，不依赖冻结环境中未声明的可选分词器；
- `max_candidates=1000`。

因此旧的 Git-tracked 共享 SQLite cache 已不再是当前阻断项，也不得重新启用或读取。
旧 canonical config hash 属于 attempt 2 的历史协议；当前 source 尚未生成最终
verification receipt。稳定且仍需在最终 preflight
重放的 lineage 为：

- data `data-1e2fdc07cf916a7faad2a9f6586b0809ff825883d296527ef6b25ea68ab45843`；
- train partition `tpart-dc73828edb28d36b4da72268914bea6b0ee94c2fa0e2bd37762829e6523eeb19`；
- 5165 条 fit record；
- cache policy `disabled`；
- config 中 Tavily physical cap 仍为 3100、budget scope 为
  `stage1-p0-wp3-formal-full-tavily-key2/v1`；实际可用额度必须重新在 provider dashboard 核验。

attempt 2 共记录 964 次 Tavily attempt（963 成功、1 TLS transport 失败）与 963 次成功
DeepSeek attempt；这些消耗属于旧 key/旧 intent，不能从 key2 的用户报告额度扣减，也不能
导入新 checkpoint。原 `.formal-billing-ledger.u9xzizl9.jsonl` 继续原样保留用于对账，
但不能 resume 或发布。旧 verification receipt
`vrec-fecaf3a16f66c66352644fb7da3a367a99fba8de7fc5049ae38c305b0dcf093a` 因当前
checkpoint/retry/budget/capture/config/source 变化而失效；新 receipt 要等最终测试通过后生成。
若研究负责人改用 no-web 策略，仍必须先形成新的版本化 config/hash，并重新执行全部
preflight；不得在当前 search-api 配置下把 LLM judgement 冒充 web evidence。详细请求、
旧 token 与费用估算见已退役的
[WP3 正式付费 build 准备包](stage1-p0-wp3-paid-build-readiness.md)，不能直接授权新库。

## 运行命令

在仓库根目录、Stage 1 conda 环境中运行：

```bash
PYTHONPATH=src .conda/stage1-p0/bin/python scripts/stage1/preflight_lexicon.py \
  --config config/stage1/lexicon_train_only.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --env-file .env
```

退出码：

- `0`：所有离线门通过；
- `2`：阻断，stdout 给出不含密钥值和密钥 hash 的安全 JSON。

当前名义执行量是 3000 个 Tavily logical slots 和 3000 个 DeepSeek logical slots。
Tavily physical-attempt cap 为 3100，因此正常 3000 次之外最多有 100 次 transport retry
余量；DeepSeek physical-attempt cap 为 9000。9000 的计算为
`1000 candidates × 3 stages × (1 initial + 2 retries)`；这是失败重试全部耗尽时的
attempt upper bound，不代表期望调用量。发送前必须在 checkpoint 中持久化 reservation；
若进程在返回结果落盘前中断，该 pending/reserved attempt 在恢复时转为 `ambiguous` 并照常
占用 cap，只有 per-slot 和 provider 总余量都允许时才重试。此前 live 授权已由 attempt 2
消费；新一轮必须重新取得外发/计费授权。

## Formal capability 与不可绕过绑定

独立 CLI preflight 仅供只读诊断。正式
`python -m build_lex.llm_lexicon_builder --data-ref ... --train-partition-ref ...` 会在同一进程内重新执行
preflight，并得到一次性、不可序列化、只能消费一次的 capability。builder 必须在
任何 target `mkdir`、候选挖掘、cache 构造和 HTTP 之前消费它，并校验以下精确
绑定仍未漂移：

- `dataset=full`、CLI `split=train`，但 lexicon source partition 精确为 `fit`；
- config canonical 内容及 hash；
- `data_ref`、data target、data build ID、payload manifest；
- `train_partition_ref`、partition target/ID/payload、其精确 data dependency；
- 5165 条 fit record/ID frame 及其 hash，以及 616 条 calibration 的完整排除证明；
- 主 LLM builder code hash；
- `formal_checkpoint.py`、`train_only.py`、`stage1_preflight.py`、`web_search.py` 的完整
  formal protocol source hash。

伪造 capability、跨进程复用、二次消费、注入替代 builder/client，或 preflight
与 build 之间发生任一 config/data/partition/code 漂移都会 hard-fail。正式 CLI 不做通用
环境变量展开，也不把递归默认值静默合并进正式配置。

正式 checkpoint 绑定同一 authorization/config/code/data/candidate frame 与 provider caps；
使用稳定的单 writer lock，目录/文件权限分别为 `0700`/`0600`，更新采用 atomic replace，
并对文件及父目录执行 `fsync`。每个 candidate 精确拥有 3 个 Tavily 与 3 个 DeepSeek
slots；已成功 slot 和连续 candidate commit 可在同一 intent 下复用。成功 publication 后
checkpoint 仍保留，但 checkpoint 本身不是 scientific artifact。

正式 target 还必须保存并在独立 validator 中复算候选、web、judgement、接受与
拒绝证据，包括 raw candidate/web/judgement/rejected 审计文件及规范化 search/LLM
capture；新增的 `debug_tavily_attempts.jsonl` 只发布安全 attempt 元数据，并与 logical
search capture、physical/retry/ambiguous counts 交叉复算。任何缺失、额外文件或 hash
漂移都会使 target 无效。
DeepSeek capture 还必须提供实际返回 model 与 prompt/cache-hit/cache-miss/completion/
total usage；validator 会核验模型 family、token 合计与每次 1024 output-token 预算。
旧 attempt 2 的紧凑 billing ledger 只保留计费元数据，不保留 term/query/prompt/response
content/credential，且不能作为新 checkpoint。

## 正式 live build 启动前必须满足

- config schema、字段集合、backend、HTTPS endpoint、认证绑定、
  `deepseek-v4-flash`、`max_tokens=1024`、JSON/stream/temperature、重试与
  failure policy 均与冻结协议一致；
- config 不含明文 key/secret，所有必需凭据非空；cache 命中不能绕过凭据门；
- `data_ref` locator、payload manifest、train/split hashes、ID frame 和
  train/dev/test 隔离全部通过严格 validator；
- `train_partition_ref` 从同一 data 深验并重放为 5165 fit / 616 calibration；
  content-hash cluster 不跨区，fit/calibration 全覆盖互斥；
- candidate、support examples、LLM inputs 只来自 fit，calibration contribution 精确为 0；
- preflight 与 build 使用同一份已验证内存 config 和一次性 capability；
- 单 writer、intent/config match、pending→ambiguous、原子写与 `fsync`、成功后 checkpoint
  保留、safe Tavily attempt capture 的最终离线测试全部通过；
- 新 verification receipt 已生成并独立 validate；旧 `vrec-feca…` 不得复用；
- Tavily dashboard 已复核 key2 即时可用额度至少 3100；
- 名义 3000 Tavily / 3000 DeepSeek logical slots、Tavily 最多 3100 与 DeepSeek 最多
  9000 physical attempts，以及完整语料派生内容外发和计费，已获本轮独立 live 授权。

本轮已运行上面的只读 preflight：它从 `.env` 解析凭据存在性但只输出 `env-file` 来源，
没有显示 key 值/hash，也不会发起 HTTP。最新代码、配置与 receipt 稳定后，正式启动前仍应
再运行一次并保存输出。若需要 provider 连通性测试，应另设显式、可计费且限定
请求数的 live-probe 步骤；静态测试或 preflight PASS 都不能视为 live 授权。
