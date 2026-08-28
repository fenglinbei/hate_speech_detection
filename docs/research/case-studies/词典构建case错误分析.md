# 本文目的

本文为一次出错词典构建过程的case分析，构建中的初次可观察问题表现为，474次候选构建中，只有39 included、435 rejected，纳入率 8.23%。
同时，第475条数据构建由于出现敏感信息被API拦下出现错误，整个运行意外终止。

可审计信息为：

Formal scope ID：fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50
Checkpoint ID：fchk-c6f627082268f76c0df5df31250b30516b1c77033c38708ed96ccdb4ff4bf419
已提交前缀：rank 1–474
出错候选：rank 475
出错 attempt sequence：2852

## 主要路径

[Checkpoint 根目录](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50)
[Checkpoint manifest](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/manifest.json)
[Attempt 账本 HEAD](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/attempts/HEAD.json)
[最后一个完整提交 rank 474](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000474.json)
[rank 475 的失败 attempt](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/attempts/000002852.json)
[Formal scope anchor](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoint-scopes/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50.json)

# Case分析得到的主要错误点

## 候选挖掘产生大量碎片——这是高拒绝率的主要来源

这些拒绝本身基本正确：

- 女拳不
- 女拳都
- 女拳自
- 拳在
- 些女拳
- 都让黑蛆喝了吧

它们多是分词/ngram 截断结果或完整句子，不适合作为词典项。例如 女拳不 被正确判断为“女拳 + 否定词不”的句子片段。
所以 435/474 的拒绝并不全是 DeepSeek 太严格，而是上游候选池混入了大量明显不可能入选的碎片，浪费了 Tavily 和 DeepSeek 请求。

解决方案：优化上游分词结果，分词要更加智能化，需要囊括一些潜在的仇恨语句，分词层未通过的话不进行下一次构建

## exact-term 规则执行不一致

Case A：easy 被纳入
实际语料：
easy girl可不是国男喊的
在easy girl眼里，这也比国男好得多

模型承认贬义来自完整短语 easy girl，却把候选 easy 单独纳入 Sexism/Racism。
问题：单独的 easy 绝大多数时候只是“容易”，不应继承完整短语的含义。这是典型的 substring attribution 错误。[rank 303](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000303.json)
相反，普信 被拒绝，理由是它只有和“男/女”组合时才有贬义。这个理由与 easy 的处理正好矛盾：
普信女写的关于男性的条件
这关普信男什么事呢？

因此这里存在明显的裁决不一致：easy girl → easy 可以投射，普信男/女 → 普信 却不可以。

解决方案：统一处理规则，这部分错误一定程度也来自上游分词问题，需要协同解决

## 类别被周围语境污染

Case B：变态 → LGBTQ
实际语料：
变态同性恋，用词精准
男人变态有什么错

context judge 原本正确认为：变态 是通用侮辱词，并非专门针对 LGBTQ；但 Web/final judge 又把它纳入 LGBTQ。
问题：类别来自旁边的“同性恋”，不是候选 变态 本身。[rank 74](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000074.json)
Case C：炼铜 → LGBTQ
实际语料：
基佬炼铜是人尽皆知的
北欧是一群炼铜海盗哦

模型定义“炼铜”为“恋童”的谐音，却将其归入 LGBTQ。
问题：恋童不是 LGBTQ 类别。这个类别显然是由第一个样本中的“基佬”投射而来，属于比较严重的 taxonomy leakage。[rank 163](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000163.json)
Case D：哥布林 → Racism
Web 证据实际说明它用于嘲讽“外貌不佳、社交尴尬或单身男性”，但最终类别是 Racism。
问题：即使决定纳入，也更接近 Sexism 或 others，Racism 与模型自己的定义不一致。[rank 55](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000055.json)

解决方案：继续细化整体处理流程，这类错误看起来是逻辑不够闭环

## 通用辱骂被当成仇恨词

例如：
德行 → others
辣鸡 → others
低能 → others
德行 的实际语料是：
国内女权的德行
上海人不是超过一半就这个德行吗？

它在句子里可能带贬义，但也是非常普通的中性词。Web 证据只是说明“瞧你那副德行”可能是骂人，最终模型便将其纳入。
如果目标是“所有攻击性词汇词典”，这可以接受；如果目标是辅助群体仇恨检测，这类词会产生大量无关命中。[rank 308](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000308.json)

解决方案：还是继续细化整体处理流程，一个可行方案是按版本管理整理处理方式，每一版完成后，要经过人审，确保候选词通过率到一定程度才启动构建

## Web evidence 使用标准前后矛盾

Case E：黑蛆 被纳入
语料本身非常明确：
对待黑蛆直接清空弹夹完事
黑蛆也吃蛆

它被纳入 Racism 在语义上合理。但实际 Web evidence 数量是 0，web_evidence_judge 却返回：
supported=true
evidence_ids=[]
这是结构性矛盾：没有 Web evidence，却声称 Web 已支持。[rank 4](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000004.json)

Case F：母畜 被拒绝
实际语料：
也就母畜能干得出来这种事
立场偏向母畜方的小作文

context judge 认为它确实是侮辱性称呼，但 final judge 因缺少独立 Web 证据而拒绝。[rank 252](/data/liaozijie/hate_speech_detection/exps/causal_context/stage1_p0/lexicons/.formal-checkpoints/fscope-02c208164043152753235da13840e12b23114df160a906c4e08022f4f2226c50/candidates/000252.json)
这和 黑蛆 的处理不一致：同样没有 Web evidence，一个纳入，一个拒绝。说明 final judge 实际上没有稳定执行“数据证据和 Web 证据如何组合”的规则。

解决方案：继续优化final judge

## 导致退出的 case

rank 475 候选是：
亡我之心不死
失败发生在 web_evidence_judge。该请求包含 9 条 Tavily 搜索结果，user prompt 长约 5518 字符，DeepSeek 返回 Content Exists Risk。
不能断言是候选本身触发，更可能是某条搜索摘要或完整组合内容触发了服务端风控。真正的工程问题是：一次 provider abstention 被记录成永久 terminal failure，导致整个 1000-case 构建停止。

解决方案：本地部署Qwen3.8-27B用来当作第二模型，两者同步执行，并记录冲突结果，当主模型再次因为风控不可用时，可以信任第二模型

# 修复后的复核结论（2026-08-26）

本次先修复了能够由代码确定、无需改变研究口径的问题：

- 混合中文语料现在会抽取连续英文短语，不再只产生英文 unigram；`easy girl` 会进入候选，`easy` 在其语料支持几乎都来自该长短语时会被标记为 `substring_fragment`。
- 中文候选增加了边界碎片和句子片段过滤；`女拳不`、`女拳都`、`女拳自`、`拳在`、`些女拳` 会在请求供应商前被抑制，`都让黑蛆喝了吧` 不再作为词条候选。
- 候选语言由候选字符串本身决定，而不是由数据集决定，避免英文候选按中文检索和裁决。
- `web_evidence_judge.supported=true` 现在必须引用非空且确实存在的 `evidence_ids`。
- `final.include=true` 现在必须至少得到 context 或 Web 一方支持；如果只有 context 支持，最终类别不得与 context 类别完全冲突。
- `Content Exists Risk`、`content_filter` 等响应现在被明确记录为 `provider_abstention`，不再与普通语义失败混为一类。

在完整 5165 条 fit 记录上的离线重挖结果为：旧候选 120566 个，新候选 120281 个；新旧 top-1000 重合 954 个，46 个旧候选被替换。被移出的 46 个候选中，31 个被判定为 substring fragment，15 个不再被抽取；其中 29 个曾出现在已提交的前 474 项里，28 个本来就被模型拒绝，1 个被纳入。说明修复命中了真实浪费项，但也说明“固定填满 1000 项”仍会从大量低质量候选中继续补位，不能只靠增加若干字符串规则解决。

以上修改会改变候选清单、代码哈希和正式构建身份。因此，原 `fchk-c6f6...bf419` 不应在新代码下继续发布；其 474 条结果可保留为错误分析和试验样本，但不能直接迁移成新正式词典。

# 新方案：以词条资格为中心的双模型构建协议

## 1. 先固定词典目标，而不是让模型自行解释目标

正式主词典只收录“词条自身稳定指向某类群体，且在脱离当前句子的情况下仍具有攻击、贬损或排斥含义”的表达。通用辱骂、行为描述和仅靠邻近词获得类别的表达不进入主词典，可进入独立的辅助攻击性词表。

每个候选先确定 `term_scope`：

- `standalone_group_term`：词条本身可独立成立且稳定指向群体；
- `productive_stem`：可在多个组合中稳定保留同一含义；
- `phrase_only`：只有完整短语成立；
- `generic_abuse`：通用辱骂，但不稳定指向群体；
- `context_fragment`：截断、句子片段或语法残片；
- `behavior_or_phenomenon`：描述行为、犯罪或现象，不是群体称谓。

只有前两类可以自动进入主词典候选。`easy` 应因含义只来自 `easy girl` 而归为 `phrase_only`；`普信` 是否是 `productive_stem` 必须用独立语料和组合一致性判断，不能因为出现过“普信男/女”就自动纳入或自动拒绝。

## 2. 候选阶段采用质量阈值，不再强制填满 1000 项

候选生成分成三层：显式标注/已知变体、稳定短语、统计回退 ngram。每个候选必须通过词边界、完整性、支持来源覆盖率和对照语料惩罚后才能进入供应商阶段。`max_candidates=1000` 只作为上限，不再作为必须填满的配额。

建议新增以下硬门：

- 句子、语法残片和被更长表达解释的子串直接离线拒绝；
- 单次出现的回退 ngram 原则上不外发，除非来自显式 target/rationale；
- 子串只有在存在独立支持记录、且意义不依赖长短语时才能保留；
- 对 `generic_abuse` 和 `behavior_or_phenomenon` 建立单独输出，不与群体仇恨主词典混合；
- 低于质量分界线时停止选取，不用后续低分项补齐名额。

## 3. 将类别判断与“是否为词条”解耦

模型必须先输出与类别无关的定义和目标类型，再允许选择 taxonomy：

1. 定义候选本身，不得复述周围群体词；
2. 判断它是否独立成词、是否稳定攻击群体；
3. 给出 `target_basis` 和可引用的样本/网页证据；
4. 最后根据 `target_basis` 映射类别。

确定性校验器拒绝以下结果：定义是通用辱骂但选择具体群体类别；定义是犯罪/行为却选择身份类别；类别只来自同句邻近词；唯一支持 judge 的类别与 final 完全冲突。这样 `变态→LGBTQ`、`炼铜→LGBTQ`、`哥布林→Racism` 会被挡在自动纳入之外，而不是寄希望于 final judge 自行纠错。

## 4. Web 只提供证据，不拥有最终决定权

Tavily 只对通过本地词条资格门的候选执行，最多三条查询；结果去重后只把直接出现候选、来源可追溯的少量摘要交给模型。Web 协议采用固定真值规则：

- 没有结果或没有有效 `evidence_id`：`supported=false`；
- `supported=true`：必须引用实际提供的 ID，且摘要必须直接支持候选本身；
- Web 只能补强或反驳候选语义，不能把相邻词的类别投射给候选；
- context 有强证据而 Web 无结果时，不因“网上搜不到”自动拒绝，而是进入双模型一致性判断或人审。

因此，`黑蛆` 的空证据不能再伪装成 Web 支持；`母畜` 也不会仅因无 Web 结果而被 final 任意拒绝。

## 5. DeepSeek 与本地 Qwen 独立裁决，不采用无条件替补

本地 Qwen 27B 的准确型号、量化、推理参数和 prompt 必须在 freeze 时记录。两模型读取同一个已经限长、去重的 evidence pack，互相不可见对方答案，并输出同一份结构化 schema。建议把原来的三个串行 LLM stage 合并为一次结构化裁决，但保留 `context_assessment`、`web_assessment` 和 `final_decision` 三个独立字段，由本地代码检查字段间一致性。

决策矩阵如下：

| DeepSeek | Qwen | 处理 |
|---|---|---|
| 均纳入，scope 与类别一致 | 一致 | 暂定纳入，进入人审抽检 |
| 均拒绝 | 一致 | 自动拒绝 |
| 纳入/拒绝冲突或类别冲突 | 不一致 | `review_required` |
| 一方 `provider_abstention` | 另一方有结果 | 保存已有结果并标记 `review_required`，继续后续候选 |
| 两方均不可用 | 无结果 | 保留为未决，不纳入、不终止整个构建 |

不能在 DeepSeek 被风控时“直接信任第二模型并自动纳入”，因为这会把双模型协议降级成未经验证的单模型协议。第二模型的正确作用是保证任务继续、保全证据，并把该候选送入人审。

## 6. Checkpoint 改为候选状态机

每个候选和每个供应商调用都在发送前预留 slot，返回后原子落盘。候选终态至少包括 `included`、`rejected`、`review_required` 和 `pending_retry`。`provider_abstention` 是可提交的候选状态，不应成为整个 build 的 terminal failure。

并发执行时由多个 worker 负责检索/推理，单独 writer 串行提交状态；崩溃恢复只重做没有成功响应的 slot，不重复已经完成的 Tavily/DeepSeek/Qwen 调用。正式身份还必须绑定候选清单、代码、prompt、两个模型 ID、本地模型权重哈希、量化和推理配置。任何一项变化都生成新 checkpoint，不覆盖旧账本。

## 7. 分阶段门禁与建议的下一次执行顺序

不把“最终纳入率”当成质量门槛：真正的仇恨词很少时，低纳入率可能完全合理。应分别衡量候选质量和纳入精度。

1. **离线候选门（80 case，分层抽样）**：由一位人工标注者逐条判断 exact candidate；“可成立”指形式完整且属于 `standalone_group_term` 或 `productive_stem`，不是最终纳入率。该比例至少 70%，明显碎片不超过 5%，句子片段为 0；不通过则继续改候选生成，不调用付费 API。
2. **本地模型门（同一 80 case）**：验证 schema 合规、exact-term 和 taxonomy；结构错误必须为 0，scope/类别人审一致率至少 90%。
3. **双模型 live smoke（50 case）**：按高分候选、边界候选、预计拒绝项分层抽样，而不是只取前 50；人工复核暂定纳入项，目标 precision 至少 90%、类别正确率至少 90%、off-scope 纳入不超过 5%，并实测单个拒答不会中止任务。
4. **新正式 build**：只有前三道门全部通过后才冻结候选和协议；候选数量由质量阈值决定。Tavily 仅用于过门候选，DeepSeek 原则上每候选一次结构化裁决，本地 Qwen 同步独立裁决。
5. **正式发布前人审**：100% 审核暂定纳入和 `review_required`，再对自动拒绝做分层抽检。人审修改作为独立覆盖层保存，不改写原始模型输出。

原 474 条记录建议冻结为 `legacy_failed_build` 审计集：39 条 included 全量复核，再从 435 条 rejected 中按碎片、exact-term、taxonomy、Web 缺失和通用辱骂分层抽样。它们适合用于新协议的回归测试和 pilot 对照，但不计入新正式 build 的已完成前缀。

当前 80-case 门已冻结为 `lcgate-2bb8d8fa91a79500a5e2cce5cf5e968b47436842dc24eafb5717f858011142d6`：从修复后 1000 个 provider-bound fit-only 候选中按 track 分层，并在各 track 内覆盖不同排序区间；配额为 annotation anchor 13、variant/slang 39、group reference 3、ngram backoff 24、contrastive phrase 1。标注页面不展示候选分数、语料类别、旧模型票据或 Web 结果；生成和审计阶段外部调用数为 0。

该门的单人完整标注结果为：7/80 可进入供应商阶段、28/80 为碎片、21/80 为句子，其余 24/80 虽形式完整但不属于主词典 scope；旧候选生成方案因此未通过。基于其中 49 个碎片/句子 case，已进一步冻结 exact-span 修订包 `lcsr-5ab78e12d5dd89403af86cb1f9dca60b2de179063e978da0e024ad2f26aadbc3`。每条只能标记“无有效词条”或提交一个/多个在所选 fit-only 上下文中逐字连续匹配、scope 为 `standalone_group_term`/`productive_stem` 的正确 span；外部调用数仍为 0。
