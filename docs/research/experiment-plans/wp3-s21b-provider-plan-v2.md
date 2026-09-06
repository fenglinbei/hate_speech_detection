# WP3 S2.1b current-run provider plan v2

> **状态：ACCEPTED IMMUTABLE PREDECESSOR / SUCCESSOR IMPLEMENTED / CALLS GATED**
>
> 生效日期：2026-08-30（Asia/Shanghai）
>
> plan：`wp3-s21b-provider-plan/v2`
>
> ID：`wp3runplan-7c4b4c8bda5fd85a3ce9f26cae6069aa4bc8bb52497a2d5a6fb97194371ea90f`

> **Successor amendment（2026-08-30）：** 本 planning-only plan 保持不变，执行实现已进入新的
> content-addressed successor lifecycle；当前仍被公开源、人工 G3 reference 和官方价格证据门阻断。
> 详见 [`wp3-s21b-g3-runner-implementation-handoff.md`](wp3-s21b-g3-runner-implementation-handoff.md)。

## 1. 决定

本 revision 仅替换旧 S2.1b v1 协议中的 `/current_generator_pilot`，旧 raw-gold、frame、schema、
validator 和 legacy generator 文件全部保持不变：

- 所有 `Qwen3.8-27B` development slot 改为智谱官方 API 的 `glm-5.3-flash`；
- `deepseek-v4-flash` 保持不变；
- G2 继续采用 GLM + DeepSeek 两路 exact-occurrence union/no vote；
- G1 在 424-case development pilot 中采用 GLM + DeepSeek 的完整双 provider × 双 pass factorial；
- G1 是否把 DeepSeek 保留到 S2.2/full-fit，必须由 development provider ablation 决定，当前不永久承诺。

智谱官方文档给出的精确 Model Code 是 `glm-5.3-flash`，默认 API base 为
`https://open.bigmodel.cn/api/paas/v4`。本仓库没有该模型的本地权重或 manifest，因此本计划明确使用
官方外部 API，不将展示名、D14 的 `glm-5.3` 或任意本地 served alias 当作同一模型。

参考：

- [GLM-5.3-Flash 官方模型页](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-5.3-flash)
- [DeepSeek thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)
- [DeepSeek V4 Flash API](https://api-docs.deepseek.com/api/create-chat-completion/)

## 2. 版本边界

禁止修改或重发：

- `wp3_candidate_generators_v1.json` 与 `terminology_candidate_generators.py`；
- `wp3_candidate_generators_s21b_v1.json`、locked raw schema 与 v2 raw-gold validator；
- active frame `wp3devframe-0a28…`；
- active raw gold `wp3rawgold-41f16…`，payload `35b655…`。

v1 中的 Qwen grid 状态为 `superseded-before-execution`。新的 run artifact 将只依赖 content-only frame
投影和本 provider plan。执行 loader/CLI 已删除 `raw_gold_ref` 输入，只读取已锁定 frame 的公开
`case_id/blind_alias/content`，再将 `case_id` 映射为 `task_id`；只有 development evaluation 才能首次
合并 run 与 raw gold，避免 gold、route、人工 notes 或历史提案进入模型请求。

frame 绑定不仅检查 424 计数，还锁定 artifact/payload、cases canonical SHA、case-ID set SHA 和完整 task
projection SHA。任意替换一条内容、alias、ID，或传入另一组同样为 424 条的记录，均不能生成合法 grid。

## 3. 两模型共同参数

GLM 与 DeepSeek 的所有 G1/G2 请求引用同一个不可覆盖 profile：

| 参数 | 冻结值 |
|---|---|
| API shape | `openai-compatible-chat-completions/v1` |
| `temperature` | `1` |
| `top_p` | `0.95` |
| `max_tokens` | `4096` |
| `stream` | `false` |
| `response_format` | `{"type":"json_object"}` |
| `thinking.type` | `enabled` |
| `reasoning_effort` | `low` |

选择共同 thinking profile 的原因是 GLM-5.3-Flash 官方接口不支持关闭 thinking；对两端均发送其官方
接口定义的 `low` 档，但不声称同名档位代表可比的推理强度或 token 预算。provider 节点不得覆盖上述
profile。request body 的严格等价投影只排除 `model`；transport、精确 endpoint 和 credential 作为
provider 身份单独验证，不属于 body 等价性。

这保证请求字段一致，但不声称底层采样等价：DeepSeek 官方说明 thinking 模式下
`temperature/top_p` 不生效，而 GLM 的实现行为由智谱服务决定。报告必须记录这一差异，并分别保存
requested/returned model、origin、usage、finish reason、request ID 与完整 request hash。

## 4. G1 双 provider 评估

推荐在 development pilot 对 G1 同时运行两模型，原因是：

1. GLM-5.3-Flash 尚无 WP3 rewrite/mention 实测，只跑单模无法测量 provider unique TP；
2. G2 direct mention 不能替代 G1 rewrite，两者错误模式和结构约束不同；
3. 两 pass × 两 provider 可以判断 `surface_decode` 是否已被 G3 覆盖，以及第二 provider 是否只对
   `lexical_pragmatic` 有增益；
4. 两模型本来都会接收相同的 424 条 content-only G2 输入，G1 双模不新增数据接收方或记录范围。

聚合仍为 exact-occurrence union/no vote。相同 occurrence 合并 candidate，但保留所有 provider、pass、
mechanism、replacement 和 context provenance；不同边界、嵌套与重叠继续保留。一致不能自动 accept，
单路 failure 不能折为空结果或删除另一模型的 observation。

development 报告必须分别给出：GLM/DeepSeek × 两 pass、每 provider 的 G1 union、双 provider G1 union、
G2 provider/union 和最终 G1/G2/G3 union 的 recall、precision、exact F1、unique TP、边界错误、失败率、
proposal volume 与 provider ablation。

DeepSeek-G1 的保留门在看到结果前量化为：以“GLM-G1 + 双路 G2 + G3”为 baseline，加入
DeepSeek-G1 后，full union exact recall 的未舍入绝对增益至少 `0.01`，或带来至少 2 个、分布在至少
2 个 case 的 unique exact TP；同时 full-union exact F1 必须不低于 `0.85`，相对 baseline 的未舍入
绝对下降不超过 `0.005`。指标按 exact-occurrence micro union 去重计算。

decision 是三态：全部 2,544 个 model slots terminal、G3 完成、terminal failure 为 0 且 offset replay
为 100% 后，才进入 `retain/drop`；任一完整性门失败均为 `inconclusive`，不得借此删除 DeepSeek。
完整后，增量门与全部 F1 质量门同时成立则 `retain`，否则 `drop` 并让 S2.2/full-fit 的 G1 回落为
GLM-only。该公式已作为 plan 字段和离线 evaluator 固定，比较使用未舍入有理数。该结论不能仅凭
development 指标声称为 sealed 证据。

## 5. 调用规模

| 生成器 | 计算 | logical slots |
|---|---:|---:|
| G1 | 424 × 2 passes × 2 providers | 1,696 |
| G2 | 424 × 1 pass × 2 providers | 848 |
| G3 | offline deterministic | 0 |
| 合计 | 每条 6 个模型 slot | 2,544 |

GLM 与 DeepSeek 各 1,272 个 logical slots。相较 v1 的 1,696 增加 848（+50%）。physical attempts、
timeout、concurrency 和费用 cap 尚未冻结；attempt 不能计作额外 logical result。

## 6. 当前实现与剩余门

已实现：

- content-addressed provider plan、JSON Schema 与深度 validator；
- GLM/DeepSeek 精确 model、origin、credential env 与 returned-model policy；
- 单一共享 request profile，provider override fail-closed；
- 离线 request renderer 和两 provider request-body 等价投影；
- 绑定 exact 424-case frame 的确定性 2,544-slot grid；
- execution loader/CLI 无 raw-gold 输入，schema 为强制 hard-pin；
- 精确 API base 与 chat-completions endpoint 合约；
- Qwen 残留、model/origin/profile/count/authorization/schema/plan ID 篡改测试；
- active v1 raw-gold current-source/current-validator 回归。

本 v2 明确是 planning-only、永久不可直接翻转为可执行状态。尚未实现或授权：provider HTTP runner、
response envelope schema、checkpoint/failure ledger、raw response vault、usage/billing ledger、
retry/physical-attempt cap、完整 run artifact、development evaluation、
Phase-B-v2、S2.2、full-fit 和任何模型调用。正式调用前还必须发布 G1 prompt v2，明确最多 8 项并加入
prompt-injection 防护；完成两端单-slot endpoint/returned-model alias preflight；把 `finish_reason=length`、
空 content、非 JSON、schema/normalizer failure 都记为失败而不是零候选。上述内容完成并单独批准前，
必须以新 plan ID 发布 successor（prompt-v2 + runner contract），不能让 run artifact 引用本 v2 ID 后
替换 prompt。execution flags 继续全部为 `false`。

验证命令：

```bash
/usr/bin/python3 scripts/stage1/wp3_candidate_run_plan.py
/usr/bin/python3 -m unittest src.tests.test_wp3_candidate_run_plan
```
