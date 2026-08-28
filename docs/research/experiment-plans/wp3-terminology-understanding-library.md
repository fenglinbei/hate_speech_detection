# WP3 无类别术语理解库：试运行门控实施手册

## 1. 实施状态与不可变边界

本仓库已实现以下正式链路：

`A0 全量规则普查 → A1 小样本调试 → A2 锁定验证 → 人工 PASS → A4 全量扫描 → A5 全量抽检 → 解释门控 → 人工清零 → 冻结 → Stage 1 发布`

正式资源角色是 `terminology-understanding-library/v1`。公开条目只有：

```text
term, definition, usage_notes, ambiguity_notes, variants
```

任何任务标签、仇恨类别、目标群体或分类结论都不得进入模型输入或公开术语条目。主实验把检索结果标记为可由 verifier 修正或拒绝的“理解参考”；正式推理不联网。

当前只完成了实现、单元测试和当前冻结契约下的完整本地 A0 验证，尚未调用 Qwen、DeepSeek 或联网服务，也没有生成可解锁全量扫描的人工 `PASS`。因此不能把实现完成误写为实验通过。

## 2. 代码与产物入口

- CLI：`scripts/stage1/terminology_span_pipeline.py`
- span/A0–A5：`src/build_lex/terminology_span_pipeline.py`
- 解释、联网、人工合并与发布：`src/build_lex/terminology_resolution.py`
- 冻结配置：`config/stage1/terminology_span_pipeline.json`
- 离线 span 复审页：`tools/terminology_span_review/`
- 解释 Gold schema：`schemas/terminology_resolution_gold_v1.schema.json`
- 人工解释 schema：`schemas/human_terminology_resolution_v1.schema.json`
- 三次失败后的人工 span 终态 schema：`schemas/terminology_span_exception_resolution_v1.schema.json`
- 回归测试：`src/tests/test_stage1_terminology_span_pipeline.py`

所有正式命令都应从仓库根目录运行：

```bash
WP3_PYTHON=.conda/stage1-p0/bin/python
WP3_CLI=scripts/stage1/terminology_span_pipeline.py
```

下文中的 `<...>` 必须替换为上一条命令实际输出的内容寻址路径，不能手工猜测 ID。

## 3. A0：全量低成本普查

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" span-census
```

该命令只读取冻结 fit 分区的 `content`，不调用模型，不读取任务标签，也不使用 dev/test。它固定 200 条 A1、100 条 A2 和 60 个隐藏复测关系，并输出：

- `records.jsonl`：每条记录的规则候选与统计特征；
- `rule_terms.jsonl`：频率、PMI、左右边界熵和字符特征；
- `pilot.public.json`：模型可见的 content-only 任务；
- `pilot.audit.json`：私有抽样层与盲审顺序；
- `census.meta.json`：数据、配置、抽样覆盖与所有载荷哈希。

规则穷举中文 2–8 字片段时，真实数据可能不存在 `rule_empty` 记录。系统会把该层记入 `unavailable_sampling_strata`，并将实际回退分配写入内容寻址身份；不得伪造一个“空结果层”。

## 4. A1：先小跑，再完成 200 条调试

可以先跑少量记录验证本地服务、JSON 和偏移解析：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" span-pilot-tune \
  --census-dir <CENSUS_DIR> \
  --checkpoint <A1_CHECKPOINT.sqlite3> \
  --limit 20
```

确认无误后去掉 `--limit`。同一 checkpoint 会复用已完成的精确请求：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" span-pilot-tune \
  --census-dir <CENSUS_DIR> \
  --checkpoint <A1_CHECKPOINT.sqlite3>
```

此时只生成 A1 调试复审包，不能提前暴露 A2：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" review-span-pilot \
  --phase tune \
  --census-dir <CENSUS_DIR> \
  --checkpoint <A1_CHECKPOINT.sqlite3>
```

把命令生成的 `.zip` 解压到独立评审目录，再打开其中的 `index.html`；不要把 output root 一并交给评审者。候选来源、阶段和复测关系只保存在协调者权限为 `0600` 的同级 `.private` 映射中，不进入评审 zip。对 200 条完整原文做穷尽标注。候选只是提示；人工必须能删除、改边界并主动新增 span。A1 可根据 `missed_span`、`too_wide`、`too_narrow`、`ordinary_phrase`、`sentence_fragment`、`wrong_occurrence` 等错误修改提示词、规则或解析。

只要提示词、模型参数、规则、偏移/排序实现或配置发生变化，就必须使用新契约重新跑完整 A0 和 A1。A0 身份与 Qwen checkpoint 同时绑定 span 核心实现文件 SHA-256，旧 census、checkpoint 和批准会因实现契约不一致而拒绝复用。A1 结果不作为锁定性能结论。

## 5. A2：冻结后运行 100 条锁定记录

在研究记录中先声明提示词、模型、参数、规则、偏移算法和排序特征已冻结，再运行：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" span-pilot-validate \
  --census-dir <CENSUS_DIR> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3> \
  --tune-review-package <A1_TUNE_REVIEW_DIR> \
  --tune-annotations <A1_TUNE_ANNOTATIONS.json>
```

该命令会先验证 A1 的 200 条人工复审完整、只含 tune 记录，并与当前 Qwen 契约一致；否则在任何 A2 模型调用前阻断。如果冻结后的 A1 使用另一个 checkpoint，必须先在这个 checkpoint 中重新完成 `span-pilot-tune` 和 A1 复审；最终复审包要求 300 个任务均有成功结果或明确失败记录。

然后生成 300 个原始任务加 60 个隐藏复测的盲审包：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" review-span-pilot \
  --phase all \
  --census-dir <CENSUS_DIR> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3>
```

完成人工复审并导出 annotations 后计算门控和 `terminology-gold/v1`：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" calibrate-gates \
  --census-dir <CENSUS_DIR> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3> \
  --review-package <PILOT_REVIEW_DIR> \
  --annotations <PILOT_ANNOTATIONS.json> \
  --gold-output <TERMINOLOGY_GOLD.json> \
  --output <SPAN_GATE.json>
```

返回值只能是：

- `PASS`：全部预注册阈值通过；
- `FAIL`：观察到的点估计违反硬阈值；
- `INCONCLUSIVE`：主要是有效 Gold 或自动提升分母不足，不能自动放宽阈值。

若为 `INCONCLUSIVE`，按连续的 100 条锁定块扩充：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" span-pilot-extend \
  --block 1 --census-dir <CENSUS_DIR> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3> \
  --pilot-gate <INCONCLUSIVE_SPAN_GATE.json>

PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" review-span-extension \
  --block 1 --census-dir <CENSUS_DIR> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3>
```

重新校准时追加：

```text
--extension-review <BLOCK_1_REVIEW_DIR> <BLOCK_1_ANNOTATIONS.json>
```

扩展命令只接受与当前配置和 Qwen 契约一致的 `INCONCLUSIVE` gate，并自动要求下一个连续 block；原因必须是锁定 A2 的 Gold、recall 或自动错误率置信区间分母不足。A1 ranker 类别不足、人工复测不一致或模型失败不能靠增加 A2 修复。扩展块必须从 1 开始连续、彼此及原 300 条互斥。隐藏复测复用原 Qwen 输出，不重复计入模型指标。

## 6. 人工批准与 A4 全量扫描

只有 gate 自身为 `PASS` 时，研究负责人才能签署：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" approve-span-pilot \
  --census-dir <CENSUS_DIR> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3> \
  --review-package <PILOT_REVIEW_DIR> \
  --annotations <PILOT_ANNOTATIONS.json> \
  --gold-output <TERMINOLOGY_GOLD.json> \
  --decision PASS \
  --reviewer-id <REVIEWER_ID> \
  --notes "已检查典型正确、漏检和边界冲突" \
  --output <SPAN_PILOT_DECISION.json>
```

全量命令会重新校验模型契约、配置、census、gate 和人工决定的哈希；没有有效 `PASS` 时直接拒绝：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" span-discover-full \
  --census-dir <CENSUS_DIR> \
  --pilot-decision <SPAN_PILOT_DECISION.json> \
  --checkpoint <FROZEN_CHECKPOINT.sqlite3>
```

可先加 `--limit` 做诊断，但带 limit 的运行不会物化正式 span frame。正式运行必须覆盖全部 5,165 条。失败不能当作空结果；三次失败后，只能按 exception-resolution schema 完成人工 span，并在原命令追加 `--exception-resolution <EXCEPTION_RESOLUTION.json>` 形成 `exception_resolved` 终态，再重新运行。最终每条记录只能是 `success`、`empty` 或 `exception_resolved`。

全量 frame 保留规则与 Qwen 并集、嵌套和重叠 occurrence，使用冻结的 L2/liblinear 排序器，最多输出 1,000 个待解析术语。无 occurrence 达到自动阈值的候选不会消失，而是进入人工 span 分支。

## 7. A5：全量结果抽检

先生成 200 个 occurrence 的盲审包：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" audit-full-spans \
  --span-frame <SPAN_FRAME_DIR>
```

完成人工 annotations 后评估：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" audit-full-spans \
  --span-frame <SPAN_FRAME_DIR> \
  --review-package <FULL_AUDIT_REVIEW_DIR> \
  --annotations <FULL_AUDIT_ANNOTATIONS.json> \
  --decision PASS \
  --output <FULL_SPAN_AUDIT_DECISION.json>
```

系统严格要求 200 条，重新计算边界错误率及 Clopper–Pearson 95% 上界。上界大于 5% 时不能签 `PASS`，解释阶段也会拒绝启动。

## 8. 解释门控、联网与人工清零

先冻结解释阶段的完整执行契约；该步骤不调用模型或联网，但会重算 BGE 文件树、校验 libpsl，并绑定 Qwen/DeepSeek 请求模型、四份提示词、联网配置和适配器代码：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" freeze-resolution-contract \
  --output <RESOLUTION_EXECUTION_CONTRACT.json>
```

`terminology-resolution-gold/v1` 是后续匿名 A/B 与单页证据审核形成的校准输入，必须把上述 `contract_sha256` 写入其 `execution_contract_sha256` 字段。`auto_accept` 必须来自同一冻结执行契约下的自动分支判定，人工只填写 `human_acceptable` 和 notes；不能让评审者为了通过门控手改自动决定。文件契约见 schema。

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" calibrate-resolution-gates \
  --span-frame <SPAN_FRAME_DIR> \
  --gold <RESOLUTION_GOLD.json> \
  --execution-contract <RESOLUTION_EXECUTION_CONTRACT.json> \
  --output <RESOLUTION_GATE.json>
```

某分支只有在自动通过样本的实质错误率 95% 上界不超过 5% 时才会启用；分母不足时分支保持关闭，候选进入下一分支或人工队列。

正式解释运行：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" run-resolution \
  --span-frame <SPAN_FRAME_DIR> \
  --full-audit-decision <FULL_SPAN_AUDIT_DECISION.json> \
  --resolution-gate <RESOLUTION_GATE.json> \
  --checkpoint <RESOLUTION_CHECKPOINT.sqlite3>
```

执行策略如下：

1. Qwen 生成五字段解释；结构错误会在 checkpoint 内最多重试三次。
2. DeepSeek 独立判断上下文支持、冲突并给出独立定义；BGE 只提供一致性特征。
3. 结构、上下文、冲突、相似度或校准任一不通过，就进入联网分支。
4. 每术语固定最多 3 个查询、3 个去重页面。可信源表为空时必须有两个不同 eTLD+1；否则不能自动通过。eTLD+1 由冻结 SHA-1 的系统 `libpsl` 数据计算，public-suffix 数据漂移时 fail-closed。
5. Qwen 根据证据重写，DeepSeek 只能引用允许的 evidence ID 检查支持关系。
6. 失败项进入精确人工补集；自动条目稳定抽取 10% 做 QC。

`--limit` 仅产生显式不完整的诊断 artifact，不能 merge 或 finalize。

根据 `human_terminology_queue.jsonl` 和 `auto_qc_sample.jsonl` 填写 `human-terminology-resolution/v1`。如果某条自动路线的 QC 样本存在实质错误，系统关闭对应路线并要求对该路线全部条目补齐人工决定；未出错的另一条路线不受牵连。否则只要求人工队列。决策集合不精确或仍有未决项时，`merge-human` 会拒绝：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" merge-human \
  --resolution-dir <RESOLUTION_DIR> \
  --human <HUMAN_RESOLUTION.json> \
  --output <MERGED_RESOLUTION.json>
```

## 9. 冻结、验证与主实验发布

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" finalize-library \
  --resolution-dir <RESOLUTION_DIR> \
  --merged <MERGED_RESOLUTION.json>

PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" validate-library \
  --library-dir <TERMINOLOGY_LIBRARY_DIR>
```

最后发布为 Stage 1 当前检索器可消费的标准 lexicon locator：

```bash
PYTHONPATH=src "$WP3_PYTHON" "$WP3_CLI" publish-stage1-library \
  --library-dir <TERMINOLOGY_LIBRARY_DIR> \
  --span-frame <SPAN_FRAME_DIR> \
  --pilot-decision <SPAN_PILOT_DECISION.json> \
  --full-audit-decision <FULL_SPAN_AUDIT_DECISION.json> \
  --resolution-gate <RESOLUTION_GATE.json> \
  --resolution-dir <RESOLUTION_DIR> \
  --merged <MERGED_RESOLUTION.json> \
  --human <HUMAN_RESOLUTION.json> \
  --write-ref exps/causal_context/stage1_p0/refs/lexicon_ref.json
```

发布器会深度重验：fit 数据的 5,165 条原文与 frame 是否逐条一致、试运行/全量抽检/解释 gate 是否绑定、人工补集是否精确、未决项是否为零、公开字段是否恰好五个，以及 `lex:v2` 身份能否重算。dev/test 只能在冻结后检索现有条目，不能创建或修改术语。

## 10. 验收命令

正式状态推进顺序固定为：

```text
span-census → span-pilot-tune → review-span-pilot(tune)
→ span-pilot-validate → review-span-pilot(all) → calibrate-gates
→ approve-span-pilot → span-discover-full → audit-full-spans
→ freeze-resolution-contract → calibrate-resolution-gates → run-resolution
→ merge-human → finalize-library → validate-library → publish-stage1-library
```

只有 `INCONCLUSIVE` 时才能在校准与批准之间插入连续的 `span-pilot-extend` / `review-span-extension`；任何阶段都不得自动跨过人工批准。

```bash
PYTHONPATH=src "$WP3_PYTHON" -m unittest \
  src.tests.test_stage1_terminology_span_pipeline -q

PYTHONPATH=src "$WP3_PYTHON" -m unittest \
  src.tests.test_stage1_train_only_lexicon -q

PYTHONPATH=src "$WP3_PYTHON" -m unittest \
  src.tests.test_stage1_retrieval_bundle -q
```

正式验收必须同时满足：5,165 条终态齐全、A5 通过、所有启用的自动分支满足冻结置信区间、人工未决项为零、所有依赖和配置哈希冻结、Stage 1 深度验证通过、正式推理不实时联网。
