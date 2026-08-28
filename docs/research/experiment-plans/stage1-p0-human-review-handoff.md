# Stage 1 P0 人类裁决交接（已完成快照）

本交接只处理 WP1 的 34 个阻断异常。30 条 group–hate 已由 GLM 5.3 与
DeepSeek v4 Flash 完成相互独立的首轮盲审；首轮结果已封存为：

- locator：`exps/causal_context/stage1_p0/refs/data_blind_review_ref.json`；
- artifact ID：`dreview-f628521d554eeb86f76a7c7a58e004cd5f075b67c97bea3a0042c52832411a27`；
- payload manifest SHA-256：`2f677eabb334d65d380b560b4208d655ff5dc691a7d65aca76cfcf32d8bcda48`。

两模型原始 decision projection 在 15/30 条上相同；经置信度门槛、失败处理与冻结的
QC 抽检策略后，只有其中 10 条严格结果可在 merge 时自动消费，20 条分歧/失败/低置信
或抽检项进入人类二轮。另有 4 条 field-type 结构异常只由人类处理。因此，人类实际填写
20 + 4 行，最终 issue frame 和 `data_adjudication.jsonl` 恰为 34 行。

本轮交接已于 2026-08-23 完成：20 条 group–hate 与 4 条 field-type 人工队列均为
`corrected`，加上 10 条 sealed auto `accepted`，最终为 10 accepted / 24 corrected。
签署者已确认 `attestation_confirmed=true`、`saw_condition_outputs=false`、
`saw_model_scores=false`。以下操作说明作为 provenance/replay 手册保留；不要重新裁决、
重新生成 declaration 或重复调用两个模型。

既有 30 条 artifact 已按 D14 对两套 provider/model policy、完整 system/user prompt、
request body/hash、requested/returned model、安全化 raw response 与逐行 projection 做过
深度重放并通过。二轮人审只消费 sealed auto/queue；**不要重新调用 GLM 5.3 或
DeepSeek v4 Flash，也不要用新模型响应覆盖该 target**。若 `seal_data_review.py validate`
失败，应停止并报告 lineage/target 问题，而不是重新发起第一轮。

请勿修改
`exps/causal_context/stage1_p0/data_audits/`、
`exps/causal_context/stage1_p0/blind_reviews/` 或
`exps/causal_context/stage1_p0/data_blind_reviews/` 下的产物。二轮盲审只消费
`refs/data_blind_review_ref.json` 指向的内容寻址快照，不再直接依赖可变的原始
`dblind-*` 目录。

人工裁决应使用下面的本地网页或 `scripts/stage1/review_data_adjudication.py`，不要直接
编辑任何人审或最终 JSONL。两者都逐条确认、支持断点续审，并通过同一冻结 validator
后原子写入；启动和提交时都会绑定 audit、rubric、packet 与 review queue。界面不会展示
任一模型票据、置信度、分歧/低置信队列原因或 legacy coupling 启发式字段，也不会生成
或签署 reviewer declaration。

### 推荐：本地 HTML 裁决台

在仓库根目录启动只绑定 loopback 的本地服务：

```bash
.conda/stage1-p0/bin/python tools/stage1_adjudication_ui/server.py \
  --workspace-root .
```

终端会打印地址，默认是 `http://127.0.0.1:8765/`。页面只接收经过 allowlist 的安全投影，
浏览器拿不到完整 packet、queue reason、legacy hint 或任何模型结果。每条“核对并确认”后，
后端生成 UTC 时间、运行权威语义校验并原子写入现有 20 行或 4 行文件；两个标签页并发提交
会通过 revision 检查拒绝旧版本覆盖。已确认条目在网页内锁定；读取、提交和导出都会重验
冻结依赖。24/24 完成后可下载绑定当前两套 revision 的两份 JSONL 与 hash manifest 结果包，
但最终 34 行合并和 declaration 签署仍必须按第 3 节显式执行。

若 8765 端口占用，可传 `--port 0` 自动选择空闲端口。`Ctrl+C` 停止服务。详细说明见
`tools/stage1_adjudication_ui/README.md`。

### 终端交互备选

终端工具要求真实 TTY；它同样不会自动生成或签署 reviewer declaration。

先设置当前 shell 的解释器变量：

```bash
export PYTHONPATH=src
STAGE1_PY=.conda/stage1-p0/bin/python
```

## 1. 二轮 group–hate 盲审（20 条）

- 可编辑裁决表：
  `exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl`
- 无模型投票的上下文包：
  `exps/causal_context/stage1_p0/review_inputs/group_hate_human_packets.jsonl`
- 规则：`config/stage1/data_adjudication_rubric.md`

逐行填写 `decision`、`edits`、`reason_code`、`reason` 和 `reviewed_at`。
保留 `issue_id`、`issue_kind`、`data_audit_id`、字段集合及
`reviewer_id=dual-blind-panel-v1` 不变。`accepted` 的 `edits` 必须为空；
`corrected` 必须只编辑 packet 中的 `allowed_edit_paths`。

先检查状态，再进入二轮盲审：

```bash
$STAGE1_PY scripts/stage1/seal_data_review.py validate \
  --review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json \
  --workspace-root .

$STAGE1_PY scripts/stage1/review_data_adjudication.py status \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --packet-file exps/causal_context/stage1_p0/review_inputs/group_hate_human_packets.jsonl \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl \
  --review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json

$STAGE1_PY scripts/stage1/review_data_adjudication.py review \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --packet-file exps/causal_context/stage1_p0/review_inputs/group_hate_human_packets.jsonl \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl \
  --review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json

$STAGE1_PY scripts/stage1/review_data_adjudication.py validate \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --packet-file exps/causal_context/stage1_p0/review_inputs/group_hate_human_packets.jsonl \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl \
  --review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json
```

`validate` 在仍有合法 pending 行时退出码为 `3`，文件或裁决不合法时为 `2`，
全部完成且通过时才为 `0`。

完成后把 sealed ref 中 10 条模型严格一致结果与 20 条人类结果合并。不要从 raw
模型目录手工复制票据或结论：

```bash
$STAGE1_PY scripts/stage1/run_blind_review.py merge-human \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --review-ref exps/causal_context/stage1_p0/refs/data_blind_review_ref.json \
  --human-completed exps/causal_context/stage1_p0/review_inputs/group_hate_human_adjudication.jsonl \
  --output exps/causal_context/stage1_p0/review_inputs/group_hate_adjudication.merged.jsonl
```

## 2. 本地结构异常裁决（4 条）

- 可编辑裁决表：
  `exps/causal_context/stage1_p0/review_inputs/field_type_human_adjudication.jsonl`
- 上下文包：
  `exps/causal_context/stage1_p0/review_inputs/field_type_human_packets.jsonl`

这 4 条尚未发送到任何外部模型。它们都不能 `accepted`，必须
`corrected`；按语义将数值字段改为明确字符串或 JSON `null`，只能使用
packet 给出的路径。字符串值需写成 JSON 字符串，例如 `"414"`，不能写成
数值 `414`。

执行本地结构异常复核并严格校验：

```bash
$STAGE1_PY scripts/stage1/review_data_adjudication.py review \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --packet-file exps/causal_context/stage1_p0/review_inputs/field_type_human_packets.jsonl \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/field_type_human_adjudication.jsonl

$STAGE1_PY scripts/stage1/review_data_adjudication.py validate \
  --audit-ref exps/causal_context/stage1_p0/refs/data_audit_ref.json \
  --packet-file exps/causal_context/stage1_p0/review_inputs/field_type_human_packets.jsonl \
  --adjudication-file exps/causal_context/stage1_p0/review_inputs/field_type_human_adjudication.jsonl
```

人类仍负责语义判断。工具只构造允许的 edit，拒绝 no-op、非法 group/hateful
值、非规范字符串、reason-code/value 不一致、部分填写行、并发覆盖和输入漂移。

## 3. 合并、签署与冻结 data ref

两部分完成后，构造精确 34 行的最终文件：

```bash
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

人类签署者已核对声明中的 hash，并把 `attestation_confirmed` 从 `false` 改为
`true`。以下为本轮实际使用的可重放冻结命令：

`finalize-data` 不信任合并文件名：它会重新深验 sealed blind artifact，并逐行确认
30 条 group-hate 等于 sealed auto 行加 frozen human queue 的完成行；4 条
field-type 则必须来自 audit 冻结的 human-only queue。任何 auto 行改写、queue
漏项或 scope 替换都会 hard-fail。

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

`finalize-data` 的 `--config` 是 audit-lineage 输入，必须与 `data_audit_ref` 所指 target
内冻结的完整 `config.resolved.json` canonical 相等。当前
`config/stage1/context_factorial.json` 已增加供下游 context 使用的
`tokenizer_revision`，不得用它重放既有 audit，也无需因此重建 audit；后续
retrieval/context 仍使用当前 source config。

本轮已发布并深验正式 data artifact
`data-1e2fdc07cf916a7faad2a9f6586b0809ff825883d296527ef6b25ea68ab45843`
（payload `7cadd0929c59481e82b3e0a1aff01830109dcca692860b73a44b287b1c6ce85d`，
split 5781/643/1605），随后发布并重放 train partition
`tpart-dc73828edb28d36b4da72268914bea6b0ee94c2fa0e2bd37762829e6523eeb19`
（payload `c98edd5243b1c7153d583937a20444d5d7b1eca4a2d4f100e3c884672e7c788c`，
5165 fit / 616 calibration）。人工交接已结束；后续入口是 WP3 只读 preflight 与
evidence/预算裁决。不得从人工工作区或 adjudication 文件另行划分 fit/calibration。
