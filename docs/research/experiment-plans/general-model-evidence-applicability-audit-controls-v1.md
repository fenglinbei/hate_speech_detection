# 证据适用性与上下文干扰：联合审核、输入分解及局部机制验证规划 v1

**文档日期：2026-09-08（Europe/Berlin）**  
**实现基准：`fenglinbei/hate_speech_detection` 远端 `main`，提交 `10580e71a51ebad71bb83ece37f325d78e10103f`**  
**承接结果：`nolabel-01` → `paired-cases-02`**  
**建议入库路径：`docs/research/experiment-plans/general-model-evidence-applicability-audit-controls-v1.md`**  
**状态：规划，尚未执行本规划中的新增审核裁决、输入实验或激活干预。**

> 本规划将后续工作收束为三步：① 查询—示例—词典命中的联合审核；② 示例记录与其独有词典贡献的分离干预；③ 具体义项、立场与误命中的定向对照，随后有条件地进入内部恢复／破坏实验。研究重点从“加入资源是否提高对原 Gold 的符合程度”，推进到“模型是否、以及如何判断外部证据适用于当前查询”。原标签、冻结资源、原实验结果及 discovery/reserve 划分不覆盖。

---

## 0. 执行摘要与阶段门槛

### 0.1 核心研究问题

**给定查询、任务规则和一组外部资料，模型能否区分“可靠且适用的证据”“可靠但不适用的证据”与“本身存在问题的证据”，并据此选择性地使用词义解释、示例及其标签？这种选择经过什么内部计算路径？**

随后才回答原目标：在证据质量与适用性得到明确描述的条件下，哪些查询需要词典、需要示例、需要二者，或无需补充；哪些情况是资源可替代、无效或相互干扰。

“需要”始终相对于固定模型、任务、提示方式和当前可用资源，不解释为对所有可能资源的必要性。“不相关”“定义不适配”“示例误标”是不同变量，不能合并成一个噪声标记。

### 0.2 三步的交付与前置条件

| 步骤 | 要回答的问题 | 核心交付 | 进入下一阶段的条件 |
|---|---|---|---|
| **第一步：联合审核与双参考评估** | 原 Gold、示例答案与词典义项各有哪些可核实的问题？ | 三类审核记录、规则版本、按任务的裁决覆盖层、原／审核参考下的配对表、案例用途清单 | 拟实验案例具备明确审核状态；争议与未知不被强行裁成硬标签 |
| **第二步：示例—词典来源分解** | 变化来自示例包、由它引入的词典，还是两者共同出现？ | 冻结的四条件小样本包、原输入重放回执、完整候选分数、输入效应表、位置控制状态 | 干预对象与保持项清楚；新增输入通过数值与来源检查 |
| **第三步：定向对照与局部机制验证** | 是词义误迁移、立场处理、匹配错误，还是其他因素？ | 定向输入结果；对合格对照的双向 patching；负对照与适用边界 | 仅在输入效应稳定、位置可解释、采集工具无副作用时定位内部路径 |

验收以可追溯、可复算和解释边界为准，不以“改后准确率提高”“找到正交互”或“发现某个头”为成功条件。第一步可以独立完成；第二、三步有独立的 GPU 执行门槛，任何 CPU 命令不得自动扩展为模型运行。

---

## 1. 最新远端基准、已知能力与不可覆盖边界

### 1.1 基准身份

本次核对的远端最新提交为 **2026-09-08 的 `10580e7`**，完整 SHA 如页首。该提交记录了正式人审服务迁移；前一提交 `ec4932f` 更新了审核指导和部署准备。本规划依据固定 SHA 下的源码与公开结果，不以随时变化的 `main` 文件作为执行身份。[R01]

实现提交、历史实验提交和运行时源码身份必须分别记录：

```text
implementation_base_commit = 10580e71a51ebad71bb83ece37f325d78e10103f
scientific_source_commit   = f80fe355263c267606c9601ca5d12aabee26da10
source_scoring_run         = nolabel-01
source_paired_run          = paired-cases-02
paired_source_identity    = a8c1fc6a97f83b339f02e36c286da5954d233a568b47ec235e4a55b217842201
paired_manifest_sha256    = 4f9b48b76e2102aa49e65deafb55ac6566c798fee346fe1be6e5cd2c5ba771d2
frozen_lexicon_sha256     = 31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385
```

这些身份值来自公开 manifest、export manifest 和配置；实际执行仍须对文件逐字节计算并核对，不能以本文抄录代替校验。历史 manifest 的 `execution_commit` 记录为 `f80fe35`，另有 `code_sha256` 与源码快照作为运行身份；不要把后续发布提交误写成历史执行时的源码版本。[R04][R05][R09]

新实现记录自己的 `execution_commit`、工作树状态和实际源码哈希。若编码期间远端又有更新，保留本基准，先产出差异及兼容性检查，再明确修订版本；不自动变更已冻结输入。

### 1.2 当前公开材料已经够做什么

`paired-cases-02` 已发布逐查询结果、32 条 discovery 的结构化案例与完整提示，以及原始 dev 查询和压缩冻结上下文；上游完整原始候选评分及大体积分析文件仍未随该结果公开。因此，联合审核和基于既有预测的标签敏感性分析可从公开产物起步；某些新参考标签的连续分数重算仍需本地原始候选分数。[R03][R05]

首选机器输入是 `cases/card_data/*.json`、`tables/query_profiles.jsonl` 与来源 manifest，不从 Markdown 展示文本反向解析程序数据。以 3169 的结构化案例为例，可用字段包括：

| 内容 | 现有字段 |
|---|---|
| 查询、既有投影 | `query.content`、`query.projection` |
| 逐条件预测 | `profile.conditions[condition][task].prediction.labels` |
| 连续读数与口径标记 | 同层 `readouts`、`score_mode_sensitive`、`score_mode_predictions` |
| 来源图 | `profile.resources.lq_ids / ld_ids / ld_only_ids / union_ids / demo_match_ids` |
| 冻结词条与义项 | `lexicon_entries[]` 中的 `lexicon_id`、`senses[].sense_id` |
| 示例正文与答案 | `demonstrations[].content / id / projection[task]` |
| 完整条件提示 | `contexts[]` 的任务、条件、提示文本和相关身份信息 |

首次导入须验证其他案例的 schema 和必需字段；缺失不填造。展示卡的 `focus_task` 只表示原选样重点，不代表另一任务已获人审认可。[R12]

### 1.3 现有审核记录与新审核的关系

现有人审工作台采用三栏、分阶段揭示、自动保存及 revision 冲突保护。其 v1 store 仅接受严格固定的 `resources` 与 `assessment` 字段；`gold_verdict` 为 `agree / dispute / uncertain`，并没有逐任务的正式替代标签，也没有逐示例、逐命中记录。不能把“有争议”自动解释为已更正 Gold，更不能直接向旧 schema 塞入新字段。[R06][R07]

**推荐实现：沿用同一工作台与持久化组件，新增独立的“证据适用性审核”模式和附属记录层；原 v1 人审会话只读引用，不原地改写。** 新记录明确关联原会话导出快照、来源 manifest 和 reviewer。旧人工意见是带身份与暴露状态的输入，不自动变成新裁决；AI 初审仅作为单独意见。

### 1.4 正式服务与安全边界

截至基准提交的部署记录，唯一正式旧审核会话位于远端：

```text
/var/lib/hsd-general-model-paired-review/session.json
```

开发机旧 `reviews/paired-cases-02/session.json` 只是受 `session.json.remote-authority.json` 保护的备份。部署记录中的 **3/12** 是迁移时保留的进度，不是本文读取到的当前实时审核数量；本次没有访问私人正式会话内容。[R02][R06]

本规划不授权部署变更。将来部署新增审核模式时，必须保留唯一写入方、原记录、revision 和事件历史；不可删除迁移标记、启动旧 writer、用旧备份覆盖新裁决，或让自动测试填写正式记录。新增附属会话与原会话各自保持单一写入权威，不能成为同一批裁决的竞争副本。生产数据、凭据及运行日志不入 Git。部署只能影响 hsd 服务，不变更或重启同机 PDF 服务。[R02]

---

## 2. 科学问题分解与共同约定

### 2.1 三个评价对象，三条独立通道

1. **查询参考判断**：描述模型应该如何解释当前查询；修订它改变评估，不改变模型输入。
2. **示例正文及答案**：作为模型输入，可能传递词义、判断规则、格式或标签先验；修改答案必须作为新输入实验。
3. **词典义项与命中**：即使词条本身有合理释义，也可能被匹配到错误位置或错误语境；修改匹配边或释义同样是新输入实验。

任何一次实验只能在预先列明的通道上改动。不得同时更正查询 Gold、改示例答案、过滤词条，再把总体改善归因于单个因素。

### 2.2 区分四个证据属性

| 属性 | 审核单位 | 与其他属性的区别 |
|---|---|---|
| 内容／标签可靠性 | 示例及其任务标签；词典具体义项 | 材料自身是否有依据，不以模型是否受益决定 |
| 词义／话题适配性 | 查询—示例、文本片段—义项 | 词面接近不等于义项一致；异话题不等于无用 |
| 判断规则适配性 | 查询—示例—任务 | 例如不同主题都在反对群体泛化，可能具有规则关联 |
| 实际因果效用 | 固定查询下的受控输入对比 | 只有干预后才可描述；不能由人工“相关”评分替代 |

现有示例检索带类别配额，而不是纯相似度前十；词典采用查询命中与示例命中的并集。因此，召回选择会同时影响示例本身与词典来源。新实验暂不改检索器，不重排示例，不按审核 Gold 重新分桶召回。[R08][R09]

### 2.3 延续原任务与分数口径

继续使用两个独立任务：`hate` 为二分类；`group` 为五类的集合预测。不得把 `group` 自动设为 `[]`，只因为 `hate` 判为 non-hate；不得把身份被提及等同于被评价／攻击。原提示明确要求区分立场和实际指向，本轮审核规则必须说明其与数据集原投影的关系。[R10]

主决策保持不含 EOS 的 `answer_sum` argmax，精确并列取最小 canonical ordinal；group 仍评估完整 32 个候选，不改成逐标签阈值。既有其他计分模式仅作敏感性诊断。数值稳定不等于标注正确，计分口径敏感也不等于随机重复不稳定。[R11][R12]

S 仍表示无显式类别字段的词条及释义；G 表示显式类别字段；D 表示示例正文及其答案。S 不等于已剥离类别语义的“纯知识”。第二步默认 G 关闭，第三步的 G 实验单独命名，不与 S 的处理混合。

---

## 3. 第一步：查询—示例—词典命中的联合审核

### 3.1 审核范围与选样边界

以冻结的 **32 条 discovery** 为完整审核队列，不重新选择“更干净”的 32 条。对它们实际引用的示例做去重审核，同时保留每个查询—示例关系；审核所有实际呈现的词典义项及其来源命中。原 16 条 reserve 不进入定位工作台、不用于修订规则或挑选解释。[R03][R13]

去重键不能只有文本：至少使用 `source_dataset_id + demo_id + content_sha256`，各任务标签单独记录。同 ID 的正文或答案不一致时停止合并并报身份冲突；不同 ID 但相同文本可记录 family 关系，不静默合并原标签。记录去重后的示例数、命中数和待审数，不预填估计数量。

这批材料是按模型行为富集的，不用于估计整个 dev 的误标率。若后续需要总体质量估计，另行冻结与模型对错无关的抽样方案；不从本案例集外推百分比。

### 3.2 先冻结判定规则，再裁具体标签

新增 `annotation-policy-v1.md`，由项目负责人确认。最低需明确：

- 当前任务如何区分攻击、普通冒犯、个体批评、群体泛化、引用、反驳、反讽和自述；不得临时套入与数据集不同的“只有某些受保护群体才算 hate”等新口径。
- group 判断的是何种被评价／攻击对象；机构、地区、个人及 `others` 的适用边界；提及对象与实际目标如何区分。
- 原四元组、现有 projection 与审核后句级标签的对应；审核不默认重写完整四元组。
- 缺上下文、多个合理解释、目标遗漏如何记录；不得把召回示例当作查询真实缺失的上文。

如果争议实质上是两种任务定义的冲突，保留 `policy_ambiguous`，或分别形成明确命名的规则分支；不能把更符合某个模型结果的口径选为唯一真值。规则变更必须版本化，并列出需重新审核的记录。

### 3.3 审核顺序：先查询，再资源，再轨迹与 AI

新增模式采用以下服务器端门控，不改变原 v1 已锁定快照：

**A. 查询独立判断。** 只展示查询、任务规则和任务名。隐藏原查询 Gold、原候选桶、所有预测、召回资源及 AI 意见。按任务记录文本能支持的判断、攻击对象／说话者立场、证据片段和缺失语境。保存不可覆盖的初读快照。

**B. 资源审核。** 显示冻结词典与示例，核查词义适配及规则对应。示例先基于自身文本作判断，再显示其原答案，记录是否认可；对已在旧页面看过材料或答案的人如实标记暴露。不要让示例中的叙事替查询补造背景。

**C. 对照原标注与模型轨迹。** 独立判断保存后再揭示原 Gold 和预测，记录保留／修订理由及候选解释；AI 初审在人工初判保存后可选展开。

本轮已有大量结果暴露，统一标为非盲或部分遮蔽审核；新页面的隐藏功能不能消除历史暴露。旧工作台初读时已同时展示资源与示例答案，不能追溯性宣称既有记录完成了上述 A/B 顺序。[R06][R07]

### 3.4 新记录契约

以下是**拟新增契约**，不是现有 store 已支持的字段。采用 `schema_version = general-model-evidence-review/v1`，并使用 JSON/JSONL 保存完整结构，CSV 仅为扁平审阅导出。

**共同身份字段**：`record_id`、`source_manifest_sha256`、`source_payload_sha256`、`policy_version`、`reviewer_id`、`review_kind`、`created_at`、`updated_at`、`revision`、`exposure`、`supersedes`、`amendment_reason`。`review_kind` 区分 human、human_with_ai、ai_note；AI 不得自动获得 human 的确认状态。

| 记录表 | 主键与必需业务字段 |
|---|---|
| `query_reviews.jsonl` | `query_id + task + reviewer + policy_version`；原标签、独立判断、原标注审核状态、证据片段、立场／目标说明、信息缺口、阶段揭示时间 |
| `demo_label_reviews.jsonl` | `demo_key + task + reviewer + policy_version`；原答案、独立判断、标签可靠性、原四元组与投影问题、理由 |
| `query_demo_reviews.jsonl` | `query_id + demo_key + task`；词义／话题关联、判断规则关联、可解释对应片段、是否会引入不适配词典、不能判断的原因 |
| `lexicon_hit_reviews.jsonl` | `source_kind + source_id + text_sha256 + span + lexicon_id + sense_id`；原始与规范化位置、匹配方式、源语境义项适配、与当前查询的适配、实际进入提示的来源边 |
| `adjudications.jsonl` | `entity_key + task + policy_version`；所依据的审核记录 ID、裁决标签或 null、裁决依据、裁决人、争议与单人审核标记 |
| `eligibility.jsonl` | `query_id + task`；是否可做参考标签依赖分析、可做何种输入对照、待核验事项、不可用原因；保留原 bucket 与新审核后分型 |

原标注审核状态固定为 `accepted / suspected_error / policy_ambiguous / context_insufficient`。最终可用性另用 `resolved / unresolved`，审核进度另用 `unreviewed / draft / confirmed`。三个维度不能混用：确认提交了“信息不足”不意味着得到一个可靠硬标签。

示例标签可靠性按任务记录为 `accepted / disputed / unknown`。关联性分别记录 `direct / partial / none / unclear`，另附依据，不汇总成单一总分。命中问题支持 `valid_sense / substring_mismatch / wrong_sense / overly_narrow_definition / uncertain / not_rendered`，允许一条记录具备多个问题标记。

`[]` 是有效的 group 空集合；`null` 表示没有裁决。hate 与 group 分别可裁决，不能因一项通过而默认另一项通过。

### 3.5 命中边与位置的核验

现有 matcher 在规范化字符串中生成候选，并保留原始位置及规则匹配信息；词面命中不等于义项已被语义消歧。[R14]

优先使用封存来源图与 trace。若公开卡只给 entry ID、没有所有 span，可在**相同冻结词典和 matcher 源码**上进行 CPU 重建，并校验聚合命中集合是否与封存集合一致；不一致则标为 `provenance_mismatch`，不能拿新版 matcher 的结果替换历史输入。重建只服务审查，不重新生成旧实验结果。

所有 span 采用原文 Unicode code point 的半开区间 `[start,end)`，同时记录 normalized span 与映射。必须验证 `raw_text[start:end] == raw_surface`；不能用 UTF-8 字节索引或 JavaScript UTF-16 下标冒充 Python 字符位置。多个出现位置、多个义项和共享来源分别保留。

### 3.6 裁决与原／审核双参考评估

优先让两位人类审核者独立判断主要机制案例及其疑似问题示例，再处理分歧。若只有一位，允许推进，明确 `single_review`，不宣称独立一致性。尚无共识的样本保留；可以研究其不依赖 Gold 的行为变化，但不以“纠正”“有害”作为主要结论。

裁决形成不可覆盖的 `gold_overlay`，绑定文本哈希、任务和规则版本。采用如下分离：

```text
原始输入与原预测：不改
原始 Gold：不改
审核参考：单独存储，未裁决为 null
示例答案修订：只作为以后实验中的显式输入条件
```

CPU 上对同一已裁决子集、相同预测分别计算原参考与审核参考下的正确性、转换表、16 种核心位型；同时报告已审／未审／未决分母、任务构成和个别案例的分型改变。四条件次序继续为 `(C0, CLnewNoCat, CD, CLDnewNoCat)`。原总体 643 条的表保留原样，不把局部审核子集的结果冒充全量清洗结果。

重新计算连续分数须使用新参考：hate 可依据二候选差重定方向；group 更换 Gold 后，旧 `gold_margin` 或逐标签聚合 log-odds 通常不足以恢复新集合的分数。必须读取绑定哈希的完整 32 候选分数；没有则输出 `revised_margin = null` 与 `candidate_scores_missing`，先交付离散评估，不能为补齐 CPU 审核而自动重跑 GPU。

### 3.7 第一步验收

完整交付应让 32 条 discovery 的两个任务均有审核状态，对全部去重示例、查询—示例关系及实际呈现命中给出已审或明确未决记录。每个拟进入后续实验的查询，必须具备其重点任务的可解释参考状态、所有被操作资源的审核记录和冻结的假设／替代解释。

允许按案例提前通过门槛进入第二步，但报告应标为部分队列完成；不能把少量合格案例说成完成了全批联合审核。缺少原始候选分数仅阻塞相关连续读数，不阻塞可执行的离散评估。

---

## 4. 第二步：把示例包与其独有词典贡献分开

### 4.1 目标与首批范围

本步检验：一组待检查示例及其引入词典，分别或共同如何改变当前查询的判断。**这里的“示例包”包含正文和冻结答案，不是已分离答案作用的纯正文。** 四条件实验识别示例包与词典来源的效应，不能单独证明示例正文语义或输入—标签映射的作用。

首批建议不超过 **8 个原始 discovery query ID**，仅作为资源预算，不是功效计算；3169、5086、541优先考虑，6037、61、1128可作为定向候选或对照。每条是否进入、重点任务和所用资源由第一步的状态决定，不以新实验效果挑选。若仍需补充，限定在原32条内，先按审核属性匹配，再按固定哈希排序；不从reserve补位。

每个案例预先冻结一个主要待检查示例集合 U。U 优先满足：其任务答案已有可信审核，而词义／话题关联及判断规则关联均缺乏明确支持。标签存在疑问的示例进入“标签问题”分支，不混入“标签可靠但不适用”的主要结论。异话题但规则相关的示例不得仅因主题不同被归入 U。

### 4.2 来源集合定义：以封存的 entry ID 为单位

对查询 q，原有10条示例为 D，保留示例为 `D_r = D \ U`。L 表示原冻结匹配关系产生的词条集合，而不是本次审核后重新检索的集合：

$$
L_R=L_q\cup L(D_r)
$$

$$
L_U=L(U)\setminus L_R
$$

于是 `L_R ∩ L_U = ∅`，`L_R ∪ L_U = Lnew`。保留组和待检查组都保持原示例相对顺序；词典 entry 按原规则顺序呈现，entry 内的义项不改变。多来源共享词条保留在 L_R，不能因某个示例被移除而连同查询仍需要的词条一起删掉。

本步只区分**词条来源贡献**。`L_U` 不等于“语义上无关词条”，`L_R` 也不保证没有义项错误；审核属性另存。暂不在本步同时清理所有误命中。

如果 `L_U = ∅`，该案例的词典来源因素没有实际操纵，不把重复提示计为一个有效2×2实验；可以只测示例包效应，或进入第三步共享义项的定向干预。例如，3169的“嘿嘿”同时由查询和示例3660命中，只移除3660并不会让该词条从查询词典中消失。[R15]

### 4.3 四条件矩阵

基础上下文 C 包含不变任务指令、当前查询、保留示例 D_r，以及 L_R 的去类别词典。以下 G 均关闭：

| 新条件 ID | 示例输入 | 词典输入 | 解释 |
|---|---|---|---|
| `EA_U0_L0` | D_r | L_R | 不含待检查示例，也不含其独有词典 |
| `EA_U1_L0` | 原 D | L_R | 只加回待检查示例包 |
| `EA_U0_L1` | D_r | L_R ∪ L_U | 只加回其独有词典 |
| `EA_U1_L1` | 原 D | 原 Lnew | 两者都加回，应恢复原 SD 输入 |

新实验条件使用新命名空间，不能冒充或覆盖原 `CLDnewNoCat` 结果。对 `EA_U1_L1`，必须核验其完整实际输入与源 SD 的文本、token ID及模板身份一致；否则记录为新模板，不能声称已经重建原条件。

词典集合写入每个条件的显式清单，不根据“当前显示了哪些示例”自动重算。不重检索、不调整类别配额、不删除未被指定的词条、不修改示例答案。评分时四条件都在新运行中重放，旧分数只用于回归核验，不直接混入新条件做差。

### 4.4 自然增删与位置控制分两层报告

**层 N：自然增删。** 先用原组件的实际加入／删除建立条件效应。示例数、标签分布、上下文长度和后续位置也会变化，因此结论仅为整段输入操作的总效应。

**层 M：固定槽位控制。** 对准备进入机制定位的主要对比，保留原示例槽位、词典槽位及边界，分别使用至少两套预先冻结的替换材料。验证完整提示中的 token 边界、查询位置与回答起始位置；不以末尾补 padding、相同字符数或块内孤立 token 数声称完成位置对齐。

不能给中性正文随意配一个原 hate 标签后，把它当作“没有信息”的示例。若使用完整示例替换，须审核其正文—答案一致性，明确控制的是数量、标签构成、长度还是内容；若只删除正文保留答案，则单列为“标签骨架”干预。两类都不是天然中性的零输入。

现有 renderer 的普通 placebo 允许 token 数容差，不保证逐槽位等长，不能直接作为精细 patching 的位置控制证明。[R10] 若无法构造合格等长替换，保留自然增删结果，限制因果解释；不得放宽规则直到出现期望行为。

### 4.5 主读数与对比

令 `s_t(c,a)` 为任务 t、条件 c 下候选答案 a 的 `answer_sum`。对hate，固定无Gold方向读数：

$$
b(c)=s_h(c,\text{hate})-s_h(c,\text{non-hate})
$$

对有明确审核参考的任务，报告审核Gold相对最佳非Gold的margin，并同时保存原Gold读数。group另预先固定一个候选对比 `s_g(c,y)-s_g(c,f)`：y为裁决集合，f为在新条件执行前冻结的具体foil；新条件中不随意更换foil。完整32候选的最终预测仍是主决策。

对任一固定读数 z，逐查询计算：

$$
\Delta_U^{L=0}=z_{10}-z_{00},\quad
\Delta_U^{L=1}=z_{11}-z_{01}
$$

$$
\Delta_L^{U=0}=z_{01}-z_{00},\quad
\Delta_L^{U=1}=z_{11}-z_{10}
$$

$$
J_{U,L}=z_{11}-z_{10}-z_{01}+z_{00}
$$

每个结果同时给出四个原始读数、预测、top gap、口径敏感标记、token长度及审核状态。J仅表示该分数尺度上的行为交互；只有组合预测错误或正确，不足以证明内部协同。对于未决Gold，只报告不依赖Gold的固定对比，不把负向偏移直接称为有害。

首批小样本以逐案例效应和方向一致性为主，不增加以“显著”为目标的批量检验。重复同一提示用于数值核验，不是独立样本；不同中性替换或同源改写也不能冒充独立查询扩大样本量。若后续需要汇总区间，另冻结按原始query/family聚类的方案，并报告选样富集及多重比较边界。

### 4.6 标签问题分支：与相关性干预隔离

对第一步裁决为示例答案有问题的记录，可新增 `demo_answer_original` 与 `demo_answer_adjudicated` 对比：固定正文、顺序、全部词典内容，只改变指定任务的答案字段。未裁决的答案不自动修正。

答案修改可能同时改变标签比例和token长度；这一对比只称为“答案修订的总效应”。若进一步研究输入—标签映射，需单独冻结保标签分布的对应关系干预或合适的标签先验控制，不把答案修订直接等同于纯映射机制。不得因答案修订重新检索、重新匹配或重新生成词典并集。

### 4.7 第二步验收与分流

必须交付：U的审核依据、L_R/L_U集合与共享边核验、四条件完整提示、原SD重建结果、数值预检、全部候选分数、上述效应表和无效果／反向案例。

- 若效果主要来自 U，下一步优先检查示例内容、答案与判断规则对应。
- 若主要来自 L_U，优先定位词典适配、来源绑定或信息量变化。
- 若只在 U 与 L_U 同时存在时出现，进入交互候选；先排除位置和共同标签偏好，再研究串行路径。
- 若预期问题输入的操作没有相应效应，降低该解释优先级；保留负结果，不追着改U直到“证明有干扰”。

---

## 5. 第三步：具体问题的定向对照与局部内部验证

### 5.1 案例用途表：当前线索不是最终人审裁决

| 案例 | 初审线索 | 本规划中的用途 | 必须先处理 |
|---|---|---|---|
| **3169** | 普通笑声可能被套入“嘿嘿”的单一群体贬损义；存在同词示例3660 | 义项适配与同词示例迁移的首要候选 | 查询两任务分别审核；示例3660的任务答案与指代关系核验 |
| **5086** | 有直接相关示例127、7464，同时包含多种示例来源词条 | 第二步来源分解的首要候选 | 不把全部Ld-only当噪声；U及其答案先审核 |
| **541** | 反对由个体推广到群体；示例826在判断规则上相关 | 立场与适用示例对照，防止只研究真hate漏报 | 查询参考、词典义项、示例答案分别核验 |
| **6037** | 示例5230的“法国女性”命中“国女” | 字符串误命中与组合成功的分离 | 冻结来源图；先测该错误边，而非整体清洗 |
| **4026、1160** | 立场／目标归因或缺失语境争议 | 标签／规则边界审查，必要时仅做无Gold方向观察 | 不作为主要“模型失败恢复”证据，除非裁决明确 |
| **2297** | hate与group可能面临不同的目标归因问题 | 分任务裁决；词内误命中或义项过窄诊断 | 不因hate有依据而默认group完备 |
| **61、1128** | 原删除有害候选与稳定对照 | 保留反例和一般扰动对照 | 仍需审核，不能因原bucket而默认有效 |

相关词条和示例线索来自冻结案例；表内用途是本规划建议，不是已有机制结论。[R13][R15][R16][R17][R18][R19][R20][R21]

### 5.2 3169：共享义项不能用“删除示例独有词典”替代

操作对象为 `lex-0419:sense:01` 与示例3660，其他输入保持不变。准备两种定义版本：

- 原冻结定义；
- 经审核的多义／适用条件定义，说明普通笑声用法与特定群体指称用法的区别，不直接写“本句应为non-hate”等答案提示。

与“保留3660／受控替换3660”交叉。**删除示例时仍保留由查询命中的lex-0419**，这样才能把共享词义说明与示例作用分开。优先采用NoCat环境；另行改变G属于其他命名条件。

定义补充会改变长度和信息量，需加入语义保持改写或布局匹配对照，不能把所有恢复都归因于消歧。若原定义与经审核定义在受控条件下没有不同效应，应降低“单一义项解释主导错误”的假设权重。

进一步可构造两个经人工审核的同表面词最小语境：普通笑声使用与明确群体指称使用。它们是新增功能性诊断查询，具有新ID、独立Gold及`parent_query_id`，不计入原643条或原reserve。选择性利用的证据应表现为：模型在适用语境仍能受相关信息影响，在不适用语境不再机械套用，而不是把该词的所有作用都消除。

### 5.3 5086：先分解额外内容，再研究整合失败

保留查询直接相关词条及示例127、7464作为候选参考，但“保留”仍以审核通过为前提。由第一步选择一个固定U，执行第二步矩阵；不因某条示例主题跨类就自动判定它无用。

若去掉特定词典贡献能稳定改变行为，再细分其定义、类别字段或与示例的绑定，不同时重写所有词条。若只在缩短提示时改善，优先研究位置与长度；若等长内容替换也重现，再提高内容干扰路径的优先级。

若清理后原组合失败消失，结论首先是“当前输入构造对行为有贡献”，不是“内部机制问题已解决”。它仍可进一步说明模型为何缺乏适用性选择，但不能再把这一现象描述成干净证据之间的整合困难。

### 5.4 541：检查立场，而不是将词条定义直接等同于hate证据

以查询的反泛化立场与示例826的规则关联为重点，分开运行两类对比：

**字段对比**：固定释义、示例及其答案，比较原G、删除G、至少两套原位置替换G。  
**规则示例对比**：固定词典和其他示例，改变是否保留经审核的相关规则示例；明确替换操作的标签构成和位置影响。

两类不在同一个未分解条件中同时改变。若某内部干预只是让所有查询更倾向non-hate，不能称为恢复了反泛化判断；需要搭配已审核的主动泛化或明确攻击表达，检验是否仍保持合理区分。

### 5.5 6037及其他误命中：操作来源边，不全局删除词条

对“法国女性→国女”的问题，先确认 `demo_id=5230` 的确切命中span与其原始match trace。只屏蔽这条不适配的来源边；若该entry没有其他保留来源，则在实验副本中不呈现该entry；若仍有合法共享来源，不能全局删除。[R18]

固定示例正文与答案、其他词典和查询，比较原始匹配输入、错误边屏蔽输入及合适的位置对照。这检验具体误命中，不是一次新retriever实验。把“查询与示例一致的词义”与“示例中偶然出现的字符片段”区分开后，再讨论组合成功。

“毒鸡汤→鸡”“贞操→操”以及“骗婚”的过窄义项，可用同样审查单元记录；只有完成实际来源核验的边才进入输入实验。不以简单增加分词器或全局最长匹配作为既定答案，因为有效暗语也可能出现在重叠表达中。[R21]

### 5.6 功能性最小对照

可借鉴HateCheck的功能测试思路，另建小规模中文诊断集，区分普通义／暗语义、引用／赞同、反对泛化／主动泛化、提及对象／实际评价对象。借用的是设计方法，不直接移植其英文标签口径。[R25]

第一轮建议不超过8条新增诊断查询，全部在模型运行前冻结正文、任务参考和关系假设。每个family分组记录，标明human-authored或AI-assisted-and-human-reviewed。人工合成材料只支持局部功能检验，不称为新的自然分布泛化结果。

### 5.7 进入activation patching的门槛

只有同时满足以下条件的对比才能进入主要内部定位：

1. 对照的文本变化、来源与目的已经明确；参考依赖结论所需的人审状态可用。
2. 原冻结输入在新工具中能回放，单纯采集激活和同运行激活原样写回不会改变目标评分。
3. 输入效应不只是当前数值误差内的波动；其位置与长度混杂已经控制，或被明确限制在可解释的共同位置。
4. donor／recipient及效应方向在扫描内部位置前冻结；定义变体或替换失败的结果也保留，不只选择最有效版本。

历史epsilon不能直接当成新hooks或新运行环境的稳定性证明。若只能通过更换模型精度、attention实现或评分模式获得效果，须先建立新的桥接与回归证据。

### 5.8 粗定位、双向干预与负对照

首轮最多选择4组已经通过门槛的同查询对照，不做全模型、全位置的无边界扫描。先检查层×少数语义位置的残差流或模块输出，再对有候选作用的位置细分attention与MLP。

位置优先包括：查询关键表达的预先定义token位置、查询末端、相关示例答案位置、回答开始前的位置。保存完整token映射和每个位置的角色；不能按删除前后相同下标盲目对齐。QK/V等细分干预仅在对应模块已定位后开展，并核验模型实际头结构、RoPE位置及允许的信息流；词典若位于某示例之后，不能假设它改写了更早示例token的因果状态。

评分仍可teacher force候选答案，但用于研究资源读取的donor激活取自**查询答案出现前**，并对所有候选应用同一提示位置干预。不得使用含已输入查询Gold的状态来证明模型事先知道答案。

令A、B为冻结的两个条件，m为预先确定的固定分数对比，定义：

$$
R=m(B\leftarrow h_A)-m(B)
$$

$$
K=m(A)-m(A\leftarrow h_B)
$$

分别报告恢复与反向破坏，连同未经干预的A/B原分数、argmax及top gap。不要只报告归一化恢复率；当原分数差很小时，该比值不稳定，原始差值必须保留。

最低负对照包括：同条件原样写回、冻结的随机位置或无关位置、合适的无关donor、稳定案例中的同类干预。还应在适用／不适用的匹配语境中检验选择性；如果干预只是总体推动hate或non-hate，不足以解释证据适用性。

整个末层决策向量的替换至多是工程上界检查，不单独承担“找到词典利用路径”的结论。注意力热图、表示距离或可解码探针只能辅助筛选；主要证据来自针对目标差异的恢复、破坏和负对照。方法上保留多种合理输入替换与固定读数，是因为patching的腐化方式和指标选择可能改变定位结果。[R26]

### 5.9 机制留出与失败结果

在定位规则、干预位置、donor构造、readout和判据全部冻结后，才进入原reserve的后续机制评估；reserve不回流用于规则调整。其来自已暴露dev，不称为全新测试集，也不默认与discovery词条或示例完全不重叠。

若留出中出现新的标签争议，按同一规则标记参考不可用或仅作无Gold方向分析，不在看过patch结果后修订标签以扩大成功数量。若某假设未复现，报告其对词条、立场或输入结构的局部边界；不继续挑头直到复现。

---

## 6. 新实现如何接到当前仓库

### 6.1 复用与新增的边界

| 当前入口／模块 | 本轮复用方式 | 不应直接沿用的假设 |
|---|---|---|
| `tools/general_model_paired_review_ui/` | 三栏布局、搜索、草稿／确认／更正、分阶段展示与导出 | 旧schema没有逐任务裁决及逐命中审核，不以自由文本自动产生新Gold |
| `store.py` 与 `server.py` | 来源绑定、文件锁、原子写入、revision、受限材料接口 | 不把未知字段塞进旧notes；不默认以本地旧session启动正式服务 |
| `general_model_coverage_package.py` | 词典来源图、entry集合与候选catalog校验思路 | 不按当前是否显示D自动重新求词典；不重新运行retriever |
| `general_model_nolabel_package.py` | NoCat渲染的兼容性参考、原释义和示例答案保持规则 | 原包是固定六条件，不把新条件冒充旧条件 |
| `general_model_numeric_v2.py` 的 `score_pass` | 复用已有候选评分、原始分数记录和身份校验；需新plan适配与测试 | 低层函数存在不代表小样本新plan已经受现有完整执行器支持 |
| `general_model_nolabel_execution.py` | 数值桥接、预检、身份记录和生命周期管理的设计参考 | 原验证器固定643条、7716块、六条件和四GPU，不能直接用于新小样本 |
| 原paired-case分析与测试 | 预测规则、集合比较、位型和转换的纯CPU逻辑 | 新参考标签下重新计算correct，不沿用旧correct字段；不覆盖原报告 |

上述复用边界经当前源码核对。旧执行器的全量矩阵约束应保留，新实验另建动态矩阵验证器，而不是为让新实验通过而删除旧审计门槛。[R07][R22][R23][R24][R27][R28]

特别注意：`general_model_ld_run_v2.json`可用于理解历史检索配置，但其生成runtime不是本轮NoCat数值运行的最终FP32配置；不能直接复制其bfloat16运行参数或旧评测指标覆盖已封存数值口径。[R08][R22]

### 6.2 新附属审核层的兼容策略

建议新增 `evidence_store.py` 和独立API命名空间，复用原UI组件。旧 `session.json`、初读锁定快照及AI揭示快照均不改写；新模式绑定一个经过revision检查的旧会话JSON导出快照，另存新增判断。

**必须明确的兼容行为：**

- 旧记录完整保留，未填的新结构化字段为未审核，不以空值或默认标签冒充结论。
- 导入旧“dispute”只能产生争议线索，不能自动生成替代标签或resolved裁决。
- 新工作台按query-only → resources → trajectories/AI推进，API也执行同样访问门槛；不能只在前端隐藏实际已发出的数据。
- 审核导出优先用JSON以保留快照、事件和原文；CSV仅为工作表并保留公式注入防护。
- 新附属层如需上线，由独立部署变更实施；先验证隔离会话、旧会话只读兼容和重启续审，再处理生产部署。本文不包含自动部署操作。

建议的新远端附属目录是 `/var/lib/hsd-general-model-paired-review/evidence-applicability-v1/`，**此路径为拟新增位置，不表示当前已存在**。离线开发与测试使用独立临时／被忽略目录，不能连接或写入正式人工会话。[R02][R07][R28]

### 6.3 拟新增文件

以下路径均为本规划建议，除明确列为复用的文件外，不能当作当前可运行实现：

```text
docs/research/experiment-plans/
  general-model-evidence-applicability-audit-controls-v1.md

docs/research/annotation-guidelines/
  evidence-applicability-annotation-policy-v1.md

config/stage1/
  general_model_evidence_applicability_v1.json

src/diagnostics/
  general_model_evidence_audit.py          # 来源导入、审核覆盖层、双参考评估
  general_model_evidence_controls.py       # 冻结U/词典集合、输入构造、位置证明
  general_model_evidence_execution.py      # 动态小样本评分及数值预检
  general_model_evidence_analysis.py       # 固定读数、条件效应、完整报告
  general_model_evidence_patching.py       # 第三步门槛通过后才实现/启用

scripts/stage1/
  general_model_evidence_applicability.py

tools/general_model_paired_review_ui/
  evidence_store.py                      # 独立附属schema，不覆盖旧store数据
  # 对现有app/core/server增加兼容的新模式，并增加对应测试

src/tests/
  test_general_model_evidence_audit.py
  test_general_model_evidence_controls.py
  test_general_model_evidence_analysis.py
  test_general_model_evidence_execution.py

exps/causal_context/general_model_evidence_applicability_v1/
  README.md
  reviews/                              # 默认忽略；非生产竞争副本
  plans/                                # 已冻结案例、条件、模板和预算
  runs/                                 # 执行产物，不覆盖历史实验
  results/                              # 审批后导出的结果与核验回执
```

若选择更少模块，可以合并文件，但不得合并三类数据权限、旧新运行身份或阶段门槛。

### 6.4 输入身份与依赖降级

定义：

```text
PAIRED = exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02
NEW    = exps/causal_context/general_model_evidence_applicability_v1
```

每次执行冻结：本规划及annotation policy的哈希、实现commit与源码、PAIRED的manifest及export manifest、实际使用案例文件、旧人审导出快照与revision、新审核裁决快照、模型/tokenizer身份、候选catalog、每个条件prompt与token IDs。

manifest与export manifest覆盖范围不同：新增公开输入应按export manifest校验，不能因不在旧运行manifest内就错误拒绝，也不能跳过校验。完整压缩上下文需要分别验证压缩文件和解压后源字节哈希；优先只使用授权discovery的结构化卡，不为方便向审核者展开其他query。[R05]

绝对路径变化仅允许显式path mapping，并验证内容身份；不得自动查找同名文件替代。缺私人审核导出不妨碍生成空审核队列，但状态必须是 `awaiting_human_review`；不能用公开AI意见替代正式人工记录。缺原始候选分数不触发GPU补跑，只将需要该分数的读数标记缺失。

### 6.5 动态计划与安全接口

建议统一入口提供以下子命令；这是**待实现接口契约，不是当前已存在的可运行命令**：

| 子命令 | 允许动作 | 禁止动作 |
|---|---|---|
| `audit-prepare` | 只读提取discovery和去重资源、校验来源、建立审核清单 | 模型forward、改原Gold、写正式人审 |
| `audit-import` | 导入明确指定的人工导出快照、检查schema/revision | 从旧备份推断实时进度、AI自动确认 |
| `audit-evaluate` | 对裁决子集做CPU双参考评估 | 改输入、补跑模型、写回旧报告 |
| `controls-build` | CPU构建新prompt、来源清单与token映射；可载入本地tokenizer | 载入模型权重、启动CUDA或自动评分 |
| `validate` | 校验完整动态矩阵、预算、只读输入及阶段门槛 | 自动放宽门槛、自动改plan |
| `run-controls` | 仅对已冻结plan进行小样本GPU预检与评分 | 沿用旧全量授权自动扩展643条 |
| `run-patching` | 仅对已通过第三步门槛的冻结对照和位置集合运行 | 无边界扫描、读取reserve后调参 |
| `report` | 汇总实际完成、失败和未决结果 | 把跳过／失败写成complete |

不提供跨审核、评分、patching的自动`all`。模型运行命令必须显式指定冻结plan、输出目录和forward许可；CPU阶段的配置不能继承旧配置中的`auto_expand_on_pass=true`。

建议配置的核心约束如下，实际实施还必须补全路径、审核快照、案例条件清单及模型身份；`status: planned`不能直接用于执行：

```json
{
  "schema_version": "general-model-evidence-applicability/v1",
  "status": "planned",
  "implementation_base_commit": "10580e71a51ebad71bb83ece37f325d78e10103f",
  "source_paired_run": "paired-cases-02",
  "analysis_kind": "posthoc-exploratory",
  "allow_test_content": false,
  "overwrite_sources": false,
  "auto_expand": false,
  "query_gold_in_scoring": false,
  "allow_model_forward_default": false,
  "review_scope": "frozen-discovery-only",
  "reference_policy": "original-plus-versioned-adjudicated-overlay",
  "prediction_score": "answer_sum",
  "tie_rule": "smallest-canonical-ordinal",
  "primary_controls": "NoCat-demo-package-by-exclusive-lexicon-2x2",
  "pilot_limits": {
    "original_query_ids": 8,
    "new_functional_queries": 8,
    "initial_patching_pairs": 4
  }
}
```

每个case plan另含：`task`、`query_id`、`U_demo_ids`、`retained_demo_ids`、`shared_lexicon_ids`、`exclusive_lexicon_ids`、来源边、定义版本、原位替换版本、模板与prompt哈希、审核reference与foil引用、预期保持项、可反驳条件、允许的层／位置，以及明确的执行预算。

GPU输入文件只含提示、候选catalog和运行身份，不携带查询Gold、审核解释或“应答某标签”的注释。示例任务答案是明确允许的输入，查询参考只在评分完成后的分析层连接。

---

## 7. 数值桥接、产物与验收测试

### 7.1 运行桥接

原则上沿用已封存数值评分的Qwen3-8B、FP32、eager attention、batch-1、`use_cache=false`、固定候选编码及不截断输入策略。实际模型/tokenizer/模板、精度、计算库、硬件和设备列表均写入新运行身份；不要求为了小样本机械复制四GPU调度，但改变设备组织后必须核验可比性。[R22][R23]

新预检至少覆盖原始条件重放、重复运行、参考评分一致性、padding／候选顺序检查；涉及多副本时增加副本一致性。内部采集另测不挂hook、只读取hook、同条件原样patch三种状态。预检样本从已审核discovery中的长度与结构边界选取，不通过读取reserve寻找边界样本。

历史参考／重复容差为`1e-4`，旧epsilon为`0.0013427734375`；新计划应在运行前声明沿用或建立新数值策略，不能看见失败后放宽。若运行环境改变导致历史桥接失败，先修复或单独标记新profile，不混用不兼容分数。[R09][R22]

动态矩阵大小依据实际case plan求和：每个hate条件2个候选，每个group条件32个候选。运行前输出精确query数、prompt-task块数、候选数、最大序列长度及干预调用预算；超过预算或8192 token上限则停止，不静默裁剪或扩充计划。

采用现有生命周期管理原则，停止和清理仅影响本次run拥有的进程、worker与临时目录；不能结束其他实验或人审服务。

### 7.2 最低交付物

| 产物组 | 必交内容 |
|---|---|
| **来源与规则** | `manifest.json`、`config.frozen.json`、annotation policy及哈希、输入审计、源码与运行环境、旧审核快照引用 |
| **第一步** | 三类审核及关系表、`adjudications.jsonl`、`gold_overlay.jsonl`、`eligibility.jsonl`、双参考转换／位型表、未决清单、人工／AI暴露记录 |
| **第二步计划** | case plans、U及L_R/L_U来源图、`contexts.jsonl`、token／span映射、原SD重建证明、预算、替换材料版本 |
| **第二步结果** | 预检回执、全部候选原始分数、固定读数、四条件效应、自然与位置控制分表、失败和无效果案例 |
| **第三步** | 定向对照及新功能query清单、采集无副作用核验、donor/recipient映射、patch干预清单、恢复／破坏／负对照、保留集访问记录 |
| **总结** | `REPORT.md`：已支持与未支持解释、标签敏感性、可用范围、下一步停止或推进依据 |

公开结果至少包含来源哈希、规则、派生统计和必要的匿名化依据。过去公开案例全文的授权，不自动扩展为公开新增私人审核会话、reviewer原始记录或凭据；新增人工数据默认私有，只导出经确认的研究覆盖层和必要片段。

### 7.3 最低测试要求

| 范围 | 必测情形 |
|---|---|
| 身份与只读 | manifest/hash错误、ID重复、同ID异文、输入目录重叠、原文件运行前后哈希不变、压缩／解压双哈希 |
| 审核兼容 | 原v1会话可继续读取且字节不变；旧dispute不能自动resolved；新增字段不写入旧notes；导入快照revision不符拒绝 |
| 阶段隔离 | query-only接口不返回Gold、资源或bucket；轨迹与AI揭示有服务器门控；reserve查询被拒绝；旧暴露状态保留 |
| 人工记录 | 独立测试session、draft与confirmed区分、更正保历史、并发revision冲突不覆盖、断线／重启恢复、CSV公式防护 |
| 标签契约 | `[]`与null区分；hate/group独立；未决不默认non-hate；新group参考缺完整候选分数时连续margin明确缺失 |
| 词典来源 | 共享词条不误删、entry多义项不遗漏、U为空或L_U为空标记无操纵、规范化与Unicode跨度往返一致 |
| 输入构造 | 四条件原始11精确回到源SD；仅指定内容变化；不重新检索；不向候选评分输入泄漏查询裁决或审核说明 |
| 数值与统计 | 四读数相同则各差与J为零；固定foil不漂移；精确并列按ordinal；无forward的阶段不能初始化模型；不把重复运行当新样本 |
| Patching | 读取hook无副作用、self-patch一致、干预只落在授权位置、候选共享同一prompt干预、数值容差失败立即停止 |
| 部署边界 | 自动测试不触达正式记录；旧迁移标记有效；新模式与旧记录不形成双写；没有PDF服务变更 |

### 7.4 明确的状态与停止条件

建议至少使用：`planned`、`awaiting_human_review`、`review_partial`、`ready_for_controls`、`controls_complete`、`ready_for_patching`、`complete`，以及以下阻塞／失败理由：

```text
source_identity_mismatch
review_snapshot_missing
policy_unresolved
reference_unresolved
provenance_mismatch
candidate_scores_missing
intervention_noop
position_alignment_failed
baseline_replay_failed
numeric_preflight_failed
budget_exceeded
reserve_access_denied
```

“有争议”“无效果”“没有适合patching的案例”可以是研究结果，不是必须被修复成成功的程序故障。只有所有声明完成的阶段确实交付，运行才可标为complete；没有进入第三步的报告应明确截止于输入因果证据。

---

## 8. 最终问题如何推进：从材料审计到适用性选择

### 8.1 四层证据，不跨层声称结论

| 层次 | 本轮能够建立的证据 | 尚不能自动推出 |
|---|---|---|
| **E1：材料与参考可靠性** | 某个标签有争议、某条命中义项不适配、某示例规则关联不足 | 该问题已经造成模型错误 |
| **E2：输入因果使用** | 在固定查询下定向改变资料，预测或固定分数随之改变 | 模型以预期语义读取了资料，或资料具有普遍必要性 |
| **E3：选择性内部路径** | 合理对照间的局部激活恢复／破坏，并在适用与不适用语境中呈现差异 | 已找到跨任务、跨词条不变的“词典头”或通用路由器 |
| **E4：条件化资源需求** | 在审核后的新四条件框架与留出评估中，对缺口、收益及干扰形成可预测规律 | 只凭已看过LD的激活就证明能够事前决定是否检索 |

本规划优先完成E1、E2，并为少量案例尝试E3。E4需要进一步的外推检验，不作为本轮必须达成的指标。

### 8.2 待检验假设及其反证

| 假设 | 支持它需要看到什么 | 什么结果会削弱它 |
|---|---|---|
| **H1：参考标签问题反转了部分“收益／伤害”解释** | 同一固定预测在原／裁决参考下发生明确分型改变 | 裁决与原标注一致，或关键行为结论不随参考变化 |
| **H2：误匹配或过窄义项引发不适用的语义迁移** | 定向义项／边操作在控制后有选择性作用 | 只有缩短输入有效，或修改目标义项并不改变行为 |
| **H3：示例与其词典贡献产生双通道影响** | U与L_U的分离实验给出可重复的条件效应 | 词典因素没有实际操纵，或独立改变它无效 |
| **H4：模型缺少情境化证据选择，而非只缺知识** | 同资料在不同适用语境中的利用应不同；相应内部干预能选择性恢复这种区别 | 干预只造成全局标签偏置、一般性能力下降或末层答案覆盖 |

这些是假设，不是本文判定已经成立的机制。首批案例定向性强，只能支持局部解释；全部对照和反例应共同进入报告。

### 8.3 返回“什么时候需要词典／示例”

材料审核后，为合格案例定义新的、版本化的资源S*与D*，冻结其构造规则，再比较 `(0, S*, D*, S*D*)`。这是一组新条件，不与旧S/D混称，不把人工挑出的适用资源当作自动选择器已经实现。

保留全部16种正确性位型，以及连续分数。重点区分：词典独特收益、示例独特收益、只有组合成功、两种资源可替代、均无效、稳定正确和组合干扰。每个模式仍只是行为状态；是否对应词义缺口、判断边界缺口或适用性选择失败，需要前述干预证据。

尤其要区分两种失败：**缺少足够且适用的证据**，以及**已有足够信息却错误利用不适用的证据**。减少后一类干扰不一定意味着模型获得了新知识；给出更多材料也不一定意味着补足了前一类缺口。

若未来研究自动资源选择，必须先明确决策时可见的信息。读取词典／示例之前的选择器不能使用读完SD后的隐藏状态、原Gold或人工已知答案；读取之后的证据过滤器则应如实计入读取和审核成本。两种系统目标不得混为“模型提前知道自己需要什么”。

### 8.4 本轮最小可接受结论

一项足够扎实的阶段成果可以是：

> 在明确的审核规则与冻结模型下，确认一类不适配证据通过可分离的输入通道改变查询判断；针对该证据建立可重复的局部恢复与破坏，并界定它在词义、立场或示例对应关系上的适用范围。

若只完成输入效应而没有内部恢复证据，仍交付E2结论；若审核发现核心案例参考不可靠而无法支持“错误恢复”，则交付参考敏感性与新的可证伪案例设计。不要为了保持原研究叙述而把未决标签强制改成有利答案。

---

## 9. 实施顺序与提交建议

**提交1：审核基础。** 加入本规划、annotation policy草案、来源只读适配器、附属审核schema与测试；复用工作台增加新模式，但不改生产部署。先验证旧v1记录不受影响。

**提交2：联合审核结果。** 导入真实人工导出快照，完成按任务裁决与双参考评估，冻结首批case plans。提交必要的匿名化派生结果与核验回执，私人会话保留在忽略目录。

**提交3：小样本输入分解。** 实现动态条件包和独立执行器；通过原输入重放与新数值预检后，运行第二步与第三步的定向输入条件，完整报告预期、反向和无效结果。

**提交4：局部机制验证。** 仅对通过门槛的对照冻结层／位置方案并执行恢复、破坏与负对照；最后才使用机制留出，记录不复现的范围。

首次真正开始实施时，优先完成**提交1及现有人工记录的只读绑定**，而不是先清洗原数据、重跑643条或启动大规模patching。每次提交在README中说明已完成的证据层级、未完成门槛和下一步允许执行的阶段。

---

## 附录 A. 来源与核对范围

本规划核对了固定提交下的公开协议、代码、manifest和discovery案例，没有读取正式私人审核会话、没有复跑历史GPU实验，也没有独立重算全部原始候选分数。页首“最新远端”指文档编制时远端提交页所示版本；本规划产生之后的远端变化不自动并入。

R01–R24、R27–R28均固定到实现基准SHA。R25、R26为方法参考；本文的工作包、schema扩展、条件矩阵、预算和验收要求是拟定方案，不是这些来源已经实现的功能。

- **R01**：[远端基准提交与提交说明][R01]。
- **R02**：[AGENTS：工作台复用、正式审核权威与部署边界][R02]。
- **R03**：[配对案例实验README：当前发布范围与本地产物边界][R03]。
- **R04**：[paired-cases-02运行manifest][R04]。
- **R05**：[公开导出manifest：源manifest及输入文件哈希][R05]。
- **R06**：[现有人审工作台README][R06]。
- **R07**：[现有人审store及严格字段、快照、revision逻辑][R07]。
- **R08**：[历史检索配置与类别配额][R08]。
- **R09**：[NoCat配置：词典身份、干预定义与数值策略][R09]。
- **R10**：[任务提示与上下文renderer][R10]。
- **R11**：[配对分析配置：主分数、位型与选样边界][R11]。
- **R12**：[3169结构化案例：机器输入字段与预测结构][R12]。
- **R13**：[32条discovery索引][R13]。
- **R14**：[controlled matcher：规范化、跨度与规则判断][R14]。
- **R15**：[3169冻结资源页][R15]。
- **R16**：[5086冻结资源页][R16]。
- **R17**：[541冻结资源页][R17]。
- **R18**：[6037冻结资源页][R18]。
- **R19**：[4026冻结资源页][R19]。
- **R20**：[1160冻结资源页][R20]。
- **R21**：[2297冻结资源页][R21]。
- **R22**：[NoCat执行器：固定矩阵、runtime和预检边界][R22]。
- **R23**：[数值评分入口score_pass][R23]。
- **R24**：[coverage包：来源集合与catalog校验][R24]。
- **R25**：[Röttger等，2021，HateCheck: Functional Tests for Hate Speech Detection Models][R25]。
- **R26**：[Zhang与Nanda，ICLR 2024，Towards Best Practices of Activation Patching in Language Models: Metrics and Methods][R26]。
- **R27**：[NoCat包：字段删除与上下文构造][R27]。
- **R28**：[现有人审HTTP接口：分阶段访问、导出与会话保护][R28]。

[R01]: https://github.com/fenglinbei/hate_speech_detection/commit/10580e71a51ebad71bb83ece37f325d78e10103f
[R02]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/AGENTS.md
[R03]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/README.md
[R04]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/manifest.json
[R05]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/export_manifest.json
[R06]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/tools/general_model_paired_review_ui/README.md
[R07]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/tools/general_model_paired_review_ui/store.py
[R08]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/config/stage1/general_model_ld_run_v2.json
[R09]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/config/stage1/general_model_ld_nolabel_v1.json
[R10]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/src/diagnostics/general_model_contexts.py
[R11]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/config/stage1/general_model_ld_nolabel_paired_cases_v1.json
[R12]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/card_data/07-3169.json
[R13]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards_index.json
[R14]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/src/rag/controlled_lexicon_matcher.py
[R15]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/07-3169-1-resources.md
[R16]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/06-5086-1-resources.md
[R17]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/01-541-1-resources.md
[R18]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/30-6037-1-resources.md
[R19]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/42-4026-1-resources.md
[R20]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/08-1160-1-resources.md
[R21]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/cards/24-2297-1-resources.md
[R22]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/src/diagnostics/general_model_nolabel_execution.py
[R23]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/src/diagnostics/general_model_numeric_v2.py
[R24]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/src/diagnostics/general_model_coverage_package.py
[R25]: https://aclanthology.org/2021.acl-long.4/
[R26]: https://arxiv.org/abs/2309.16042v2
[R27]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/src/diagnostics/general_model_nolabel_package.py
[R28]: https://github.com/fenglinbei/hate_speech_detection/blob/10580e71a51ebad71bb83ece37f325d78e10103f/tools/general_model_paired_review_ui/server.py
