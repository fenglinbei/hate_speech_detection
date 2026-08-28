# Stage 1 P0 执行状态与继续推进手册

> 执行快照：2026-08-26
>
> 设计与验收规格：[Stage 1 P0 实施规格](stage1-p0-implementation.md)
>
> 人工复核操作细则：[Stage 1 P0 人类裁决交接](stage1-p0-human-review-handoff.md)

## 1. 当前结论

P0 的代码、生命周期、engineering smoke 与阻断式校验已经落地；WP1 人工裁决、正式 data freeze 和 train partition 已于 2026-08-23 闭环，但正式 P0 **尚未达到 Definition of Done**。2026-08-26 已把正式 $L$ 从“群体损贬主词典”改为无类别 `terminology-understanding-library/v1`：entry schema 不再允许 category/categories，候选发现与排序只读 fit utterance content，模型可见 block 与 semantic retrieval text 统一为 `category-free-terminology-evidence/v1`。条目提供 definition、usage_notes、ambiguity_notes 和 variants，帮助模型理解术语，但不提供 targeted_group/hateful 结论；V2 prompt 继续要求这两个任务字段独立判断。该变更使所有既有 WP3 付费 attempt、verification receipt 和 context/generation smoke 只保留为历史记录，对当前 source/config tree 已失效。正式 Qwen3-8B 训练没有启动，任何 engineering/legacy 产物都不得当作性能或因果结果。

所有现有 smoke 产物均为 engineering/nonformal、`scientific_eligible=false`；模型/context
smoke 为 dev-only，旧 WP3 lexicon smoke 则由 fit-only train 派生。正式链路仍被以下事项阻断：

1. 30 条 group–hate 的 D14 首轮盲审已封存为 10 条严格一致 auto accepted + 20 条人类二轮 corrected；另有 4 条 field-type 人类 corrected。最终 34 行、reviewer declaration 与盲化声明均已签署并深验；不应重复调用 GLM 5.3 或 DeepSeek v4 Flash。
2. data audit `daudit-68c8…` 与正式 data artifact `data-1e2f…5843` 均通过原始 std source、sealed blind composition、34 条裁决、split 与 payload 深度重放。正式 split 为 5781/643/1605。
3. full-information-isolated train partition `tpart-dc73…eb19` 已发布并重放为 5165 fit / 616 calibration；全覆盖、互斥且 normalized-content 跨区交集为 0。calibration 从训练 loss、demo pool、lexicon 全部隔离，presentation 固定。
4. 新正式 fit-only 术语理解库配置已显式 `cache_enabled=false, cache_path=null`，resource role 为 `terminology-understanding-library/v1`，canonical config hash 为 `e95d01ec695402299fe61230fb8c97c582c3f54d8d5e4b3fcc9f0dcfa447e811`。候选保留所有合理 content n-gram，高置信单例可进入判断，不使用 hate count/category purity。旧 attempt 1/2（包括 321 个已判断 candidate、964 次 Tavily 与 963 次 DeepSeek 记录）属于已退役的群体损贬协议，只能用于对账，不能恢复或导入新库。新协议尚未运行正式 preflight、没有 live 授权，也无正式 ref/target；额度必须重新从 provider dashboard 确认。
5. 正式 dev context 尚未生成，因而约 400 条正式 CF cohort 尚不能 proposal、GLM/DeepSeek 首轮盲审和人类二轮复核。`cfp-03b8…` 是 20-query engineering proposal，未调用外部模型，不能替代正式 cohort。
6. legacy checkpoint 的 BF16 margin batch calibration 未通过 hard cap；系统按设计拒绝发布 `margin_ref.json`，所以也没有 `analysis_ref.json`。

旧 receipt `vrec-feca…093a` 曾对 attempt 2 的历史 tree 深验通过，但当前 category-free
terminology builder/prompt/retrieval renderer、checkpoint、config 与测试清单已变化，不能继续作为 engineering verification
gate。新 receipt 尚未生成；必须等最终离线测试通过后由主流程运行并独立 validate，且不得
预写未知 artifact/hash。

test 数据仍保持封存：已验证 sealed boundary 与非正式 synthetic 接口，旧 3-row formal fixture 已退役；真实 frozen sealed integration 尚未运行，也没有构建或查看正式 test context/control/CF/prediction。

## 2. WP0–WP10 实际进度

| WP | 已完成 | 尚未完成或限制 |
|---|---|---|
| WP0 决策冻结 | `decision_register.json` 已冻结 seeds `{42,43,44}`、Qwen3-8B、最多 5 epochs、full-information-isolated 名义 10% hash calibration early stop、新 conda 环境、双高参数模型盲审和默认策略；calibration 的 loss/demo/lexicon 全隔离与 fixed presentation 是强制不变量。 | 后续若改变 partition、隔离或 presentation 规则，必须产生新的 partition/config/evidence/plan/schedule IDs。 |
| WP1 数据、split 与 partition | immutable audit、D14 blind review、20 + 4 人工裁决、签署 declaration、正式 data 与 train partition 均已生成并深验。34 行为 10 accepted / 24 corrected；split 5781/643/1605；partition 5165 fit / 616 calibration，全覆盖互斥且 normalized-content 跨区为 0。 | WP1 已闭环。99 个 substring observations 仍是 informational-only warning，不是 annotation blocker。 |
| WP2 四字段协议 | canonical compact JSON array、四字段 normalizer/serializer、strict/recover parser、prompt 与 round-trip/异常测试已实现；正式 data ref 已可供下游消费。 | 后续 artifact 仍须显式绑定正式 data lineage。 |
| WP3 fit-only 术语理解库 | builder、完整 raw/normalized evidence captures、证据/provenance、config 一致性和泄漏校验已迁移到 category-free 协议；formal build 使用同进程一次性 capability，并把 config/data/partition/fit、主 builder 与 protocol source 精确绑定。候选 source policy 为 `content-only-no-task-labels/v1`，排序、过滤和 substring suppression 均不读取 hate/category；LLM 三阶段响应与最终 entry 禁止任务类别字段，条目 ID 为基于可见语义的 `lex:v2`。`provider-slot-checkpoint/v1` 的物理预算和持久化保护仍保留。 | `lexicon_ref.json` 不存在。旧 attempt 2 的 321/1000、billing ledger、preflight 和授权不能恢复/发布为新库。当前新配置 hash 为 `e95d01…e811`；仍需完整回归、新只读 preflight、新 receipt、provider 额度复核和本轮 live 授权。 |
| WP4 检索与 selector | 结构化 hit、稳定排序、全局去重、quota、fit-only/self-overlap 防护与重放校验已实现；术语检索统一使用 `evidence_kind=terminology` 与 category-free rendered block。正式 score evidence 保存 canonical base64 little-endian float64 C-order bytes；本地 BGE 使用完整 regular-file inventory，SentenceTransformer constructor 与全部 encode/replay 均处于 verified lease 内。 | 正式 data/partition 已就绪；检索 bundle 仅等待新 terminology `lexicon_ref`。 |
| WP5 context 生命周期 | prepare/build/render/validate、一次 CLD budget finalize、四条件纯函数投影、train/dev policy lineage 与 `seal-test` 已实现。context ID 精确绑定 system/user/example prompt bytes、prompt/context/retrieval renderer code 与 thinking mode；formal tokenizer/scorer 均冻结全树、`local_files_only=true`、`trust_remote_code=false`，完整 build/validate operation 都在 lease 内，formal caller injection 被拒绝。当前 20-query engineering context 已重建为 `ctx-2ba0…`。 | 正式 train/dev context 仅等待 WP3；旧 3-row“formal”sealed fixture 已退役，真实 frozen sealed integration 当前未执行。 |
| WP6 环境、模型与训练计划 | 新 conda 环境、Qwen3-8B base ref、正式 partition、legacy model ref、seeds `{42,43,44}` 与 calibration-only early stop recipe 已冻结。training evidence 现必须嵌入 `base_model_ref`，evidence/plan/schedule base lineage exact；evidence 和 schedule 的全部 tokenizer replay 位于 registered full-tree lease 内，正式 schedule 不接受 tokenizer-root。DeepSpeed logical file、plan resolved object 与 hash 三者一致后，runtime 只注入独立 verified dict，marker hash 防对象漂移与 cwd 同名文件。工程 plan/registry 已重建为 `tpl-1eeb…` / `mreg-c273…`。 | formal evidence/plan/schedule 与六个正式 model slots 尚未生成或训练；source lease 是 POSIX 非特权边界，不声称抵御特权 mount/raw-device 攻击。 |
| WP7 placebo controls | PL/PD 独立 lifecycle、配额/长度/来源校验和 sealed builder 已实现；control ID 冻结 tokenizer full-tree 与 constructor policy，constructor、token count、matching、render/replay 全程持 lease，formal CLI 不需且禁止 tokenizer override。当前 `ctl-b250…` 的 PL/PD 各 20 条可用。 | 正式 control ref 等待正式 context。 |
| WP8 自由生成评测 | strict per-query evaluator、tuple assignment、字段指标、flip/statistics 和 immutable evaluation lifecycle 已实现；formal vLLM 强制 V1、保存原始 token IDs/stop reason并独立 decode。当前 `gen-968a…` 在同一命令内对 120-row ordered frame 完整执行恰好 2 次，共 240 次 inference，逐记录/raw output/token/stop projection 全等、0 mismatch；`eval-4bda…` 精确消费该 generation。 | 62/120 条输出以 length stop 结束，strict format rate 仅为 0.15–0.40；该链为 `scientific_eligible=false` 的 legacy engineering 诊断，只证明 plumbing 与确定性，不支持模型性能或因果结论。 |
| WP9 CF 与 margin | proposal/review/finalize 的两阶段盲审生命周期、GLM/DeepSeek 独立调用、盲化 human queue、teacher-forcing field margin scorer 与 synthetic tests 已实现。D14 validator 锁定 provider/model、seed 42、threshold 0.8、QC 0.1、完整 prompt/request/response hash；科学/formal CF 只允许内置 live HTTP transport。当前 20-query engineering proposal 已重建为 `cfp-03b8…`，含 25 gold tuples / 329 candidates。 | smoke proposal 未调用外部模型、未 final；正式约 400 cohort 等待正式 context 后由双模型首轮与人类二轮盲审。legacy BF16 calibration 阻断 margin 发布。 |
| WP10 smoke 与冻结 | sealed boundary 已做递归 metadata-only 防护；P0 validator 要求 evaluation 精确消费 generation、generation/margin 共用 registry/context/control、analysis 精确包含 evaluation/margin refs。当前 2-repetition GPU generation 与下游 evaluation 仍是可验证的 engineering 产物，model-registry 的 22 个节点被精确覆盖一次。 | `vrec-feca…` 仅对应 checkpoint 改动前的历史 tree，现已失效；新 receipt 待 WP3 最终测试后生成。两份全局 engineering/formal 报告也仍是历史 `FAIL=0` 快照；正式下游 refs 缺失，不得把旧 receipt、定向测试或 engineering smoke 等同于正式 DoD。 |

## 3. 当前工程 locator 与 lineage 状态

下表中的 locator 均位于 `exps/causal_context/stage1_p0/refs/`。data、train partition、
environment 与 base 是正式前置；标为 current/smoke/legacy 的行只构成 engineering 链，
不是正式 Stage 1 结果。

| 产物 | Artifact ID | 说明 |
|---|---|---|
| environment | `env-900ecb3c683f3dde0dc729045e05d36070524cf97570e3a4f1dca2274b86f514` | `.conda/stage1-p0`，Python 3.11 / Torch 2.6 CUDA 12.4 lineage。 |
| Qwen3-8B base | `mdl-b1900d60ec438799f065d2a7fbb34bd86a1610d69d9ec77b4f8fdbcd4de1da27` | 正式训练所需 base model 的 immutable file-tree ref；尚不是 Stage 1 checkpoint。 |
| data audit | `daudit-68c8c24b3ee84d3376a8d00345e868df289c4eb11a1b80a0242dadaa9acdf76e` | 34 blocking issues、99 warnings；已从实际 std source 深度重建 inventory/split/issues/templates/report/provenance/rubric 与 payload。 |
| data blind review | `dreview-f628521d554eeb86f76a7c7a58e004cd5f075b67c97bea3a0042c52832411a27` | `refs/data_blind_review_ref.json` 指向的 30 条 group–hate D14 双模型首轮不可变快照；payload manifest hash 为 `2f677eabb334d65d380b560b4208d655ff5dc691a7d65aca76cfcf32d8bcda48`。10 条自动共识、20 条盲化人审队列；provider/model/prompt/request/response hash 已深验，不需再次调用 API。 |
| data | `data-1e2fdc07cf916a7faad2a9f6586b0809ff825883d296527ef6b25ea68ab45843` | payload `7cadd0929c59481e82b3e0a1aff01830109dcca692860b73a44b287b1c6ce85d`；34 条裁决、签署声明、sealed composition 与原始 source 均已深验，split 5781/643/1605。 |
| train partition | `tpart-dc73828edb28d36b4da72268914bea6b0ee94c2fa0e2bd37762829e6523eeb19` | payload `c98edd5243b1c7153d583937a20444d5d7b1eca4a2d4f100e3c884672e7c788c`；5165 fit / 616 calibration，完整覆盖且互斥，normalized-content 跨区为 0。 |
| legacy model | `mdl-15eab32ae9614acdcaf004e76c10a8be929d13eda218964348d033a2e666cc0f` | `M_legacy/smoke`，`legacy-smoke-only`，`scientific_eligible=false`。 |
| current legacy smoke context/control | `ctx-2ba0fde1035b2fbdc2f36684914b8245825fd6c724fd265e1490435bd9fd1b0f` / `ctl-b2501efd67b842c4f967b20afa1580eaa03317fd9a86b90c25ec8e860d4619cb` | 20 条 dev engineering context 与 PL/PD；payload 分别为 `346f8da83ec57f6bf933309e0844f733b7dfd327203bff146d96fb02489324e8` / `8296adf0588c8a52121728c73229a5ad69fd2dc3fdf643e7fd01ffd77bae440d`。context ID 已绑定 prompt/renderer identity，control 已绑定 tokenizer full-tree lease；均 `scientific_eligible=false`。 |
| current smoke plan/registry | `tpl-1eebd1cce2d30230045f10f2aaf505b68d40867b4da7c6179b7115f2d3b18a2d` / `mreg-c273f3562fdfeb62aa41da8d54a6af3616dd9d3092c147201e8954c4c2917bab` | payload 分别为 `95060c7d6c055ff904adcc2614efd9647d99fd4245e98924013c53b344e7b4e5` / `424fcef707941555ddf81205cf2ace1f121ca756b85875b237ad32aace829dba`；仅绑定 `M_legacy/smoke`，scope=`engineering-smoke`，均 `scientific_eligible=false`。 |
| current generation | `gen-968a5f3c6ea3e22304ffdc7ae0e5148664c39f6e2f20bb3ff2ed143fd4e4fb4d` | `generation_run_ref.json` 与 `legacy_smoke_generation_ref.json` 均指向该 target；payload `c64a51c5ecc9b12f6533ce5bb302f56916bafd9de6669067855e9461e51fa28d`。20 queries × 6 conditions = 120 rows，同一 publication 恰好执行 2 遍、共 240 次真实 inference；逐记录/raw output/token/stop projection 全等，0 mismatch。 |
| current free evaluation | `eval-4bda81c2895c496d9fc1be9c354d95fb208d5b58482bd8355f78d986b1f75802` | `evaluation_ref.json` 精确消费 current generation；payload `234cb17b03896dd7c9b2cab5aa720ae79850728fbda04d6d1f56d7ecfeac2642`。62/120 length-stop、58/120 EOS；仅为 engineering 诊断。 |
| current smoke CF proposal | `cfp-03b8219e17fa62290a778791ada045bf15fe0d6aac2df0ca78b31a7a24391fac` | payload `d4d112d6bb399b2e370a1fe8722bc231e5ead402a6d79c456a2a0bd736ce2754`；20 queries / 25 gold tuples / 329 candidates，其中 50 条 group/hate 自动候选、279 条 target/argument 候选需 review。未调用 API，不能作为正式约 400 cohort。 |
| historical verification receipt | `vrec-fecaf3a16f66c66352644fb7da3a367a99fba8de7fc5049ae38c305b0dcf093a` | 只绑定 attempt 2 时的旧 tree；新 checkpoint/retry/budget/capture/config 改动后已失效，不能作为当前 build gate。新 receipt 待最终测试后生成，artifact/hash 尚未产生。 |

current generation 已满足同一 `run-hf` publication 内
`--determinism-repetitions 2`、完整 frame 两遍全等并写入 `determinism.json` 的要求；
旧 `gen-5df047...`、`gen-c69018...`、`gen-4c44...` 及其 sidecar/迁移 fixture 仅保留为
历史诊断，不能替代或覆盖 current ref。

current evaluation 的 strict format rate（C0/CL/CD/CLD/PL/PD）为
`0.30/0.15/0.40/0.35/0.25/0.20`；hard/soft tuple F1 分别为
`0/0`、`0/0`、`0.242424/0.303030`、`0.1875/0.25`、`0/0`、
`0.068966/0.206897`。再结合 62/120 length-stop，该结果只说明旧三字段模型的
engineering 失败路径；它不是正式模型效果比较，也不能进入论文结论。

## 4. 阻断式验收快照

当前没有覆盖最新 category-free terminology prompt/retrieval 与 WP3 source/config tree 的有效 receipt。`vrec-feca…` 是旧协议下的历史 receipt，已经失效；下列两份报告又是在正式 data/partition 与更早 receipt
`vrec-a6f4…` 发布后生成的**历史快照**，同样不能作为当前 tree 的冻结证据：

- `reports/p0_validation_report.json`：engineering-smoke 为
  `PASS=20 / BLOCKED=1 / PENDING=8 / FAIL=0`，report hash
  `ac1d2887a43cedac4bd3a1652a5dc9c391f2777c499960a0b7e0a9f464517292`。唯一 BLOCKED
  是旧 engineering `counterfactual.blind_review`；PENDING 为 lexicon、training
  evidence/schedule、CF review/final、margin/analysis/downstream chain。
- `reports/p0_formal_readiness_report.json`：formal-readiness 为
  `PASS=12 / BLOCKED=0 / PENDING=18 / FAIL=0`，report hash
  `bbc9070562cf46e5855f602d67ebb7b633b9ca0247ab8e7c713858ccba495d33`。18 个 PENDING
  从 lexicon 开始，覆盖 formal train/dev context、training、controls、CF、registry、generation/evaluation/margin/analysis
  与完整 lineage/chain。

当时两条 validator 命令均以退出码 1 表达 `ready=false`（engineering=`BLOCKED`，
formal=`PENDING`），不是执行失败；两份历史报告都满足 `FAIL=0`。正式 WP3 付费 build 的
当前 gate 是最终离线测试、最新只读 preflight、新 verification receipt、key2 额度复核与本轮
live 授权。进入 WP4 或报告新的全局 P0 状态前，必须先生成并 validate 新 receipt，再据此
重建两份报告。

## 5. margin calibration 阻断

legacy checkpoint 的本地 HF scorer 在相同冻结 pair 上完成了 token、attention mask、position ID 和字段 span 对齐，但 BF16 的 batch-vs-single 数值一致性没有通过冻结 hard cap：

- SDPA 最大 `|Δmargin| = 0.02544900611974299`，推导所需 tolerance 为 `0.050898012239485979`，超过 hard cap `0.005`；
- eager 最大差异为 `0.06211581058534055`；
- FP32 对照分别约为 `4.90e-6` 与 `5.26e-6`，通过 `1e-5` 检查。

因此这是 legacy BF16/sequence-shape kernel drift 的工程诊断。当前正确行为是保持 gate 阻断，不提高 tolerance、不伪造 margin artifact。正式 Qwen3 checkpoint、正式 CF ref 就绪后应按同一冻结 profile 重新校准；只有通过时才允许创建 `margin_ref.json` 和下游 `analysis_ref.json`。

## 6. 现在应执行的命令

以下命令已按当前 CLI `--help` 核对。默认从仓库根目录执行：

```bash
export PYTHONPATH=src
STAGE1_PY=.conda/stage1-p0/bin/python
```

### 6.1 已完成：20 + 4 条人工复核与 data freeze

以下命令作为本轮 provenance/replay 记录保留；两个人审文件已经完成，不要重新进入裁决或重新生成 declaration：

- `exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl`：20 行；
- `exps/causal_context/stage1_p0/review_inputs/field_type_human_adjudication.jsonl`：4 行，必须 `corrected`。

填写完成后依次执行：

```bash
$STAGE1_PY scripts/stage1/run_blind_review.py merge-human \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json \
  --human-completed exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl \
  --output exps/causal_context/stage1_p0/review_inputs/group_hate_adjudication.merged.jsonl

$STAGE1_PY scripts/stage1/build_contexts.py merge-data-adjudication \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --input exps/causal_context/stage1_p0/review_inputs/group_hate_adjudication.merged.jsonl \
  --input exps/causal_context/stage1_p0/review_inputs/field_type_human_adjudication.jsonl \
  --reviewer-id dual-blind-panel-v1 \
  --output exps/causal_context/stage1_p0/review_inputs/data_adjudication.jsonl

$STAGE1_PY scripts/stage1/build_contexts.py prepare-data-declaration \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/data_adjudication.jsonl \
  --reviewer-id dual-blind-panel-v1 \
  --write-template exps/causal_context/stage1_p0/review_inputs/data_reviewer_declaration.json
```

人类签署者已核对 hash 并确认 attestation。本轮 finalize 使用 audit target 内冻结配置：

```bash
DATA_AUDIT_CONFIG=exps/causal_context/stage1_p0/data_audits/daudit-68c8c24b3ee84d3376a8d00345e868df289c4eb11a1b80a0242dadaa9acdf76e/config.resolved.json

$STAGE1_PY scripts/stage1/build_contexts.py finalize-data \
  --config "$DATA_AUDIT_CONFIG" \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --data-blind-review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/data_adjudication.jsonl \
  --reviewer-declaration exps/causal_context/stage1_p0/review_inputs/data_reviewer_declaration.json \
  --write-ref exps/causal_context/stage1_p0/refs/data_ref.json

$STAGE1_PY scripts/stage1/build_contexts.py validate-data \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json
```

`finalize-data --config` 是 audit-lineage 输入，必须与 audit target 的
`config.resolved.json` canonical 全对象相等。当前 source context config 已新增下游专用
`tokenizer_revision`，不得回灌既有 audit；后续 context 命令仍使用当前 source config。

任一命令非零退出时立即停止，不得跳过 validator 或手工伪造 ref。

### 6.2 已完成：immutable train partition

以下为已执行且通过重放的命令：

```bash
$STAGE1_PY scripts/stage1/build_train_partition.py build \
  --config config/stage1/train_calibration_partition.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json

$STAGE1_PY scripts/stage1/build_train_partition.py validate \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json
```

validator 必须从正式 data 重放 `CRLF→LF` content-hash clusters、每簇最小 canonical
数字 ID、`salt=stage1-train-calibration-v1`、`hash mod 10000` 与 `<1000` threshold，
并得到 5165 fit / 616 calibration、全覆盖互斥且 normalized-content 跨区为 0。

### 6.3 确认正式术语理解库 evidence 与请求预算

当前配置使用 Tavily search API 与 DeepSeek judgement，并已显式冻结
`cache_enabled=false, cache_path=null`；旧共享 SQLite 不会被读取或写入。config schema 为
`stage1-train-only-terminology-library-config/v1`，resource role 为
`terminology-understanding-library/v1`，canonical config hash 为
`e95d01ec695402299fe61230fb8c97c582c3f54d8d5e4b3fcc9f0dcfa447e811`。它固定
`resume=true`、`provider-slot-checkpoint/v1`、Tavily transport retry 与 physical attempt
cap 3100、DeepSeek physical attempt cap 9000。

attempt 2 的 1000-candidate 群体损贬 build 授权已经消费；该轮在 321/1000 后 fail-closed，
原受限 billing ledger 继续保留但不能 resume，新 checkpoint 也不得导入这些 321 条。新协议
改变了候选来源、排序目标、LLM response schema、entry ID 和 renderer，因此必须从空
checkpoint 开始。当前尚未针对新协议执行仓库级只读 preflight，未向 Tavily/DeepSeek 发出
新 HTTP、未产生新计费，也未取得本轮 live 授权。

新 checkpoint 对每个 candidate 固定 3 个 Tavily 与 3 个 DeepSeek slots，以 owner-only
单 writer、目录/文件 `0700/0600`、atomic replace + file/parent-dir `fsync` 持久化。
provider 发送前先 reserve；崩溃遗留的 pending/reserved attempt 在恢复时转为 ambiguous
并计入全局 cap，只有 per-slot 与 provider 余量都允许时才重试。成功 slots/candidate commits
复用，成功 publication 后 checkpoint 仍保留；正式 target 发布并校验安全的
`debug_tavily_attempts.jsonl`。

不得把 DeepSeek judgement 描述成 Tavily 搜索证据，也不得临时重启 legacy cache。
当前 data/partition 已就绪。下面的 preflight 只做本地读取与校验，不发起网络请求；它会从
`.env` 的白名单字段读取 key 值作存在性检查，但不会输出 key 值/hash。本轮尚未执行这一步，
必须等最终离线测试通过后再运行。若选择 no-web，则必须先形成另一份版本化配置并重新冻结，
不得临时替换当前 `--config`：

```bash
$STAGE1_PY scripts/stage1/preflight_lexicon.py \
  --config config/stage1/lexicon_train_only.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --env-file .env
```

只有该命令退出 `0`、新 receipt 独立 validate、provider dashboard 确认即时可用额度
足以覆盖冻结预算，并取得本轮明确外发/计费授权后，才能运行正式 build；无需且不得
`source .env`。builder 会在
同一进程内重跑 preflight 并消费一次性 capability，在任何 mkdir/候选/cache/HTTP
之前校验 config/data/partition/ref/fit、主 builder、`formal_checkpoint.py` 与其余 formal
protocol source 精确绑定。
旧价格估算与请求体来自已退役协议，不能作为新库的预算授权。应先按新候选 frame 重新生成
价格/额度快照；[WP3 正式付费 build 准备包](stage1-p0-wp3-paid-build-readiness.md)当前仅作
历史记录。获得独立授权后才可运行：

```bash
$STAGE1_PY -m build_lex.llm_lexicon_builder \
  --dataset full \
  --config config/stage1/lexicon_train_only.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --env-file .env \
  --split train \
  --write-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json
```

不得追加 `--no-resume`；正式 config/wrapper 要求 checkpoint resume。新授权应明确覆盖
Tavily 最多 3100、DeepSeek 最多 9000 physical attempts、完整语料派生内容外发与对应计费。

详细门禁见 [WP3 正式词典离线预检](stage1-p0-wp3-preflight.md)；其中仍使用
`lexicon` 作为内部 artifact-kind/CLI 兼容名，不代表科学资源仍是群体损贬词典。

### 6.4 构建 train/dev context，再冻结 evidence/plan/schedule/control

术语理解库验证通过后，先分别计算 train 与 dev retrieval，再各自发布 immutable context。
两者都必须绑定同一 `train_partition_ref`，demo scorer 只接收 5165 条 fit source；
5781 条 train query 中的 616 条 calibration 仍会有自己的固定 eval prompt，但不能成为
任何 query 的 demo：

```bash
$STAGE1_PY scripts/stage1/build_contexts.py prepare-retrieval \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --lexicon-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json \
  --config config/stage1/context_factorial.json \
  --split train \
  --embedding-model models/base/bge-large-zh-v1.5 \
  --device cuda:0 \
  --output exps/causal_context/stage1_p0/prepared/retrieval.train.json

$STAGE1_PY scripts/stage1/build_contexts.py build \
  --config config/stage1/context_factorial.json \
  --prepared-bundle exps/causal_context/stage1_p0/prepared/retrieval.train.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --lexicon-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/train_context_ref.json

$STAGE1_PY scripts/stage1/build_contexts.py prepare-retrieval \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --lexicon-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json \
  --config config/stage1/context_factorial.json \
  --split dev \
  --embedding-model models/base/bge-large-zh-v1.5 \
  --device cuda:0 \
  --output exps/causal_context/stage1_p0/prepared/retrieval.dev.json

$STAGE1_PY scripts/stage1/build_contexts.py build \
  --config config/stage1/context_factorial.json \
  --prepared-bundle exps/causal_context/stage1_p0/prepared/retrieval.dev.json \
  --data-ref exps/causal_context/stage1_p0/refs/data_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --lexicon-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/dev_context_ref.json

$STAGE1_PY scripts/stage1/build_contexts.py validate \
  --context-ref exps/causal_context/stage1_p0/refs/train_context_ref.json

$STAGE1_PY scripts/stage1/build_contexts.py validate \
  --context-ref exps/causal_context/stage1_p0/refs/dev_context_ref.json
```

两份 formal context 均以不带 tokenizer override 的 `build_contexts.py validate` 通过后，
按 train context → evidence → plan → schedule 的单向顺序执行。formal builder 会从
`config.budget.tokenizer_path` 解析并冻结 tokenizer 全树，caller-supplied tokenizer/replayer
会 hard-fail；当前已有的
`environment_ref.json` 与 `base_model_ref.json` 继续按各自 validator 深验，不重建或
改写：

```bash
$STAGE1_PY scripts/stage1/build_training_schedules.py build-evidence \
  --context-ref exps/causal_context/stage1_p0/refs/train_context_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --base-model-ref exps/causal_context/stage1_p0/refs/base_model_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/training_evidence_ref.json

$STAGE1_PY scripts/stage1/build_training_schedules.py validate-evidence \
  --training-evidence-ref exps/causal_context/stage1_p0/refs/training_evidence_ref.json

$STAGE1_PY scripts/stage1/build_training_plan.py freeze \
  --source-spec exps/specs/stage1_context_factorial.json \
  --scope formal \
  --context-ref exps/causal_context/stage1_p0/refs/train_context_ref.json \
  --training-evidence-ref exps/causal_context/stage1_p0/refs/training_evidence_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --base-model-ref exps/causal_context/stage1_p0/refs/base_model_ref.json \
  --environment-ref exps/causal_context/stage1_p0/refs/environment_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/training_plan_ref.json

$STAGE1_PY scripts/stage1/build_training_plan.py validate \
  --training-plan-ref exps/causal_context/stage1_p0/refs/training_plan_ref.json

$STAGE1_PY scripts/stage1/build_training_schedules.py build-schedule \
  --training-plan-ref exps/causal_context/stage1_p0/refs/training_plan_ref.json \
  --training-evidence-ref exps/causal_context/stage1_p0/refs/training_evidence_ref.json \
  --train-partition-ref exps/causal_context/stage1_p0/refs/train_partition_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/schedule_ref.json

$STAGE1_PY scripts/stage1/build_training_schedules.py validate-schedule \
  --schedule-ref exps/causal_context/stage1_p0/refs/schedule_ref.json

$STAGE1_PY scripts/stage1/build_controls.py build-controls \
  --config config/stage1/context_factorial.json \
  --context-ref exps/causal_context/stage1_p0/refs/dev_context_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/control_ref.json \
  --split dev

$STAGE1_PY scripts/stage1/build_controls.py validate-controls \
  --control-ref exps/causal_context/stage1_p0/refs/control_ref.json \
  --context-ref exps/causal_context/stage1_p0/refs/dev_context_ref.json
```

每一步完成后先运行同一 CLI 的 `validate-*`/`validate` 子命令。正式 schedule 必须
从 evidence 内嵌的 registered base/tokenizer lineage 重放，不接受 tokenizer-root；正式
control 同样从 context/config 解析 tokenizer，不需且禁止显式 tokenizer override。schedule
必须证明 fit-only loss，且所有 calibration 行均 `presentation_epoch=1`；不得先训练再补
partition/base lineage。

### 6.5 正式 context 后执行约 400 条 CF 双盲流程

只有正式 `dev_context_ref.json` 已验证后才执行；不要把 `smoke_cf_proposal_ref.json` 传入本流程：
下列 CLI 固定使用 `live-d14`，不提供 synthetic transport 开关；provider/model 环境覆盖若
不仍精确解析为冻结的 GLM 5.3 与 DeepSeek v4 Flash，会在首个请求前阻断。

```bash
$STAGE1_PY scripts/stage1/build_counterfactuals.py propose-cf \
  --config config/stage1/context_factorial.json \
  --foil-policy config/stage1/cf_foil_policy.json \
  --review-rubric config/stage1/cf_review_rubric.md \
  --context-ref exps/causal_context/stage1_p0/refs/dev_context_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/cf_proposal_ref.json \
  --split dev

$STAGE1_PY scripts/stage1/run_blind_review.py preflight-cf \
  --proposal-ref exps/causal_context/stage1_p0/refs/cf_proposal_ref.json \
  --policy config/stage1/blind_review.json \
  --env-file .env

$STAGE1_PY scripts/stage1/run_blind_review.py cf \
  --proposal-ref exps/causal_context/stage1_p0/refs/cf_proposal_ref.json \
  --policy config/stage1/blind_review.json \
  --env-file .env \
  --output-dir exps/causal_context/stage1_p0/blind_reviews/cf_dev \
  --write-ref exps/causal_context/stage1_p0/refs/cf_blind_review_ref.json

$STAGE1_PY scripts/stage1/run_blind_review.py export-cf-human \
  --review-ref exps/causal_context/stage1_p0/refs/cf_blind_review_ref.json \
  --output exps/causal_context/stage1_p0/review_inputs/dev_cf_human_review.jsonl \
  --packet-output exps/causal_context/stage1_p0/review_inputs/dev_cf_human_packets.jsonl
```

人类完成冻结队列后：

```bash
$STAGE1_PY scripts/stage1/run_blind_review.py merge-cf-human \
  --review-ref exps/causal_context/stage1_p0/refs/cf_blind_review_ref.json \
  --human-completed exps/causal_context/stage1_p0/review_inputs/dev_cf_human_review.jsonl \
  --output exps/causal_context/stage1_p0/review_inputs/dev_cf_review.jsonl \
  --declaration-output exps/causal_context/stage1_p0/review_inputs/dev_cf_reviewer_declaration.json
```

核对并手工把 CF declaration 的 `attestation_confirmed` 改为 `true` 后：

```bash
$STAGE1_PY scripts/stage1/build_counterfactuals.py finalize-cf \
  --proposal-ref exps/causal_context/stage1_p0/refs/cf_proposal_ref.json \
  --blind-review-ref exps/causal_context/stage1_p0/refs/cf_blind_review_ref.json \
  --review-file exps/causal_context/stage1_p0/review_inputs/dev_cf_review.jsonl \
  --reviewer-declaration exps/causal_context/stage1_p0/review_inputs/dev_cf_reviewer_declaration.json \
  --write-review-ref exps/causal_context/stage1_p0/refs/cf_review_ref.json \
  --write-ref exps/causal_context/stage1_p0/refs/cf_ref.json

$STAGE1_PY scripts/stage1/build_counterfactuals.py validate-cf \
  --cf-ref exps/causal_context/stage1_p0/refs/cf_ref.json
```

`finalize-cf` 会深验该 blind-review target，并要求 review JSONL 恰好等于
`auto_review.jsonl` 的逐行冻结副本与 `human_queue.jsonl` 全量人工完成行的并集。
直接从 proposal 手填整份 review、改写自动共识行、漏掉人审队列或引用 sibling
proposal 的 blind-review ref 都会 hard-fail。

### 6.6 重放或验证已完成的 engineering chain、receipt 与阻断报告

当前 `ctx-2ba0… / ctl-b250… / tpl-1eeb… / mreg-c273…` 已由 `gen-968a…` 完整消费，
一次命令对冻结 frame 执行恰好两遍且 0 mismatch，`eval-4bda…` 也已发布。下面命令仅用于
上游 identity 变化后的重新构建，或对 current refs 做验证；不得为了取得不同输出而重复运行。
任一 repetition 不一致都会在 ref 写入前终止：

```bash
$STAGE1_PY scripts/stage1/generate.py run-hf \
  --profile config/stage1/generation_legacy_smoke.json \
  --context-ref exps/causal_context/stage1_p0/refs/legacy_smoke_context_ref.json \
  --control-ref exps/causal_context/stage1_p0/refs/legacy_smoke_control_diagnostic_ref.json \
  --model-registry-ref exps/causal_context/stage1_p0/refs/smoke_model_registry_ref.json \
  --model-key M_legacy/smoke \
  --target-root exps/causal_context/stage1_p0/generations \
  --write-ref exps/causal_context/stage1_p0/refs/generation_run_ref.json \
  --determinism-repetitions 2 \
  --engineering \
  --execute-real-inference

$STAGE1_PY scripts/stage1/generate.py validate \
  --generation-ref exps/causal_context/stage1_p0/refs/generation_run_ref.json

$STAGE1_PY scripts/stage1/evaluate.py free \
  --generation-run-ref exps/causal_context/stage1_p0/refs/generation_run_ref.json \
  --evaluation-profile config/stage1/evaluation_strict.json \
  --write-ref exps/causal_context/stage1_p0/refs/evaluation_ref.json \
  --split dev
```

所有 `src`、`scripts/stage1`、`config/stage1`、schemas 与环境规格稳定后，运行冻结
verification manifest；运行期间不得修改 inventory 内文件。任一命令非零、超时或
inventory 漂移都不会发布 receipt：

```bash
$STAGE1_PY scripts/stage1/run_test_receipt.py run \
  --workspace-root . \
  --jobs 4

$STAGE1_PY scripts/stage1/run_test_receipt.py validate \
  --workspace-root . \
  --ref exps/causal_context/stage1_p0/refs/verification_receipt_ref.json

$STAGE1_PY scripts/stage1/validate_p0.py \
  --mode engineering-smoke \
  --write-report exps/causal_context/stage1_p0/reports/p0_validation_report.json

$STAGE1_PY scripts/stage1/validate_p0.py \
  --mode formal-readiness \
  --write-report exps/causal_context/stage1_p0/reports/p0_formal_readiness_report.json
```

旧 `vrec-feca…` 只证明 checkpoint 改动前的历史 tree，当前已失效。等待 WP3 最终离线测试
完成后，主流程必须运行上面的 receipt 命令并独立 validate，再把真实 artifact/hash 与命令
结果写回文档；当前不得预写未知新 receipt。只要
`lexicon_ref/formal context/formal CF/margin/analysis` 仍缺失，
报告返回非零并包含 `BLOCKED/PENDING` 就是正确行为；receipt 只证明冻结测试清单全部
通过，不会把外部/人工阻断变为 PASS。不得通过删除 requirement 或伪造 ref 让报告“变绿”。

## 7. 正式推进顺序

WP3 category-free 术语库最终离线测试 → 生成并独立 validate 新 verification receipt →
Tavily/DeepSeek dashboard 复核余额并重算新协议预算 → 取得冻结 physical-attempt、语料派生
内容外发与计费授权 → 运行只读 preflight → 从空 checkpoint 构建 fit-only terminology
`lexicon_ref` → train/dev fit-only retrieval/context → base-bound training evidence/plan/schedule → dev controls/约 400 条 CF 双模型首轮与人类二轮 → seed-42 pilot (`M_LD`、`M_drop`) → seeds 43/44 → formal registry → generation/evaluation/margin/factorial → 更新 verification receipt 与阻断报告 → 冻结政策 → 一次 sealed test。

其中任何 build/finalize/validate 非零退出都阻断后续步骤。calibration 的 full-information isolation 与 fixed presentation 已冻结；本轮正式 partition 已由签署后的 data ref 构建并通过重放。GLM/DeepSeek/Tavily 凭证只能从环境解析，不得写入 ref、target、日志、文档或提交历史；provider 回显和异常文本同样先经 secret containment，不能原样持久化。正式训练目前未启动；只有全部上游 refs 和 runtime preflight 通过后才可启动。真实 GPU backend smoke 只能标为 engineering、不得发布正式 ref，任何 engineering 输出都不能升级为科学结论。
