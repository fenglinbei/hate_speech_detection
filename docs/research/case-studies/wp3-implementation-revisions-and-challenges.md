# WP3 实施修订与困难复盘（讨论稿）

> 快照日期：2026-08-27（Asia/Shanghai）
>
> 讨论范围：Stage 1 P0 的 WP3，从既有词典资源梳理、旧版 fit-only 群体损贬词典构建，
> 到当前无类别术语理解库试运行。
>
> 当前结论：WP3 尚未完成。当前正式路线已推进到 **A1 的 200 条模型输出和人工复审包均已
> 生成，等待 A1 人工穷尽复审**；A2 锁定验证、全量扫描、解释门控、人工清零和
> `lexicon_ref.json` 发布均未完成。

## 1. 一页摘要

WP3 的目标经历了两次实质变化：

1. 从“直接沿用既有带类别词典”，转为“从冻结训练数据构建 fit-only 词典”；
2. 又从“筛选稳定损贬群体的词条”，转为“构建不提供任务答案、只帮助理解词义和语用的
   category-free 术语理解库”。

第二次变化不是简单删除 `category` 字段。旧版候选发现、排序、纳入条件和定义生成都使用过
`target/argument/targeted_group/hateful` 或其派生统计；即使最终不渲染类别，词条成员资格
本身仍会携带任务先验。当前协议因此要求从 5,165 条 fit 记录的原始 `content` 重新开始，
并以 A0–A5、解释校准、人工补集和不可变发布链证明资源边界。

截至本快照：

- 正式 data 与 train partition 已冻结：5,781 条 train 中 5,165 fit / 616 calibration；
- 新 WP3 A0 已覆盖全部 5,165 条 fit content，dev/test 与任务标签贡献均为 0；
- A0 产生 555,546 条规则 term 记录，并冻结 200 条 A1、100 条 A2 和 60 个隐藏复测关系；
- A1 的 200 条 Qwen 任务已有 200 个成功结果；另有 1 次可重试失败，随后成功；
- A1 的 200-case 离线匿名复审包已生成；仓库内尚无完成后的 A1 annotations；
- `lexicon_ref.json`、正式 train/dev context、正式 `M_LD/M_drop` 训练均未开始。

## 2. 先区分四类容易混淆的资源

| 资源 | 实际定位 | 已知边界或问题 | 当前角色 |
|---|---|---|---|
| `data/lexicon/annotated_lexicon.json` | 原始主实验默认使用的人工词典；文件说明称 830 条，实际 `.terms` 为 829 条 | 仓库内未携带可验证的构建 provenance；词表成员资格和定义以“隐式仇恨俚语”为目标，包含显式群体/贬义先验 | legacy 资源；是否可作为独立 external-resource arm，需要先做来源审计 |
| `data/lexicon/generated/full/lexicon.json` | 后来的 `full_llm_lexicon` 专项实验资源，656 条 | 明确使用 `train.json` 与 `test.json`，并保存 `hate_count/category` 等标签派生信息 | 不得进入当前正式 WP3；仅作历史实验资源 |
| 旧 WP3 fit-only 群体损贬词典 | 从 5,165 条 fit 记录筛选“稳定指向并损贬群体”的词条 | 虽然 fit-only，但候选、排序和 LLM payload 仍读取任务标签；候选质量、provider 稳定性和 checkpoint 恢复均出现问题 | 已退役；失败记录用于审计、回归和成本对账 |
| 当前 `terminology-understanding-library/v1` | 只解释词义、语域、歧义和变体的无类别术语理解库 | 必须通过试运行、人工门控、全量抽检、解释校准和人工清零 | 唯一正式主实验路线 |

讨论中“旧词典”如无额外限定，应优先指
[`annotated_lexicon.json`](../../../data/lexicon/annotated_lexicon.json)，而不是后来生成的
`generated/full/lexicon.json`。旧 WP3 的失败 build 则应称为“旧 fit-only 群体损贬协议”，
避免继续混用名称。

## 3. WP3 的科学任务与冻结边界

WP3 为后续 `L`（terminology evidence）条件提供正式资源。该资源需要支持
`C0/CL/CD/CLD` 因果比较，因此不能把当前查询的 `targeted_group` 或 `hateful` 答案提前编码
进上下文。

当前冻结边界是：

- 唯一语料来源为正式 train partition 中的 5,165 条 fit `content`；
- 616 条 calibration 只用于 checkpoint selection，对术语候选、support、LLM 输入贡献为 0；
- dev/test 对候选发现、阈值、人工选择和定义生成贡献为 0；
- candidate discovery 不得读取 target、argument、group、hate 或其派生统计；
- 模型可见和 semantic retrieval 使用同一 category-free block；
- 公开条目只能包含 `term`、`definition`、`usage_notes`、`ambiguity_notes`、`variants`；
- 收录表示“解释该表达有助于理解句意或语用”，不表示当前句子必然 hate，也不表示任务类别；
- 正式推理不联网，联网证据只允许在冻结前的解释阶段使用。

### 3.1 WP3 与后续 `M_LD / M_drop` 的关系

WP3 本身不负责训练正式 hate-speech extraction 模型。它的交付物是冻结的
`lexicon_ref.json`；WP4/WP5 再据此生成可检索、可渲染的 `L` 证据与统一 context，之后 WP6
才能训练两个主要 checkpoint：

| 模型 | 训练期上下文 | 实验用途 |
|---|---|---|
| `M_LD` | 每条训练样本始终同时看到术语证据 `L` 与示例证据 `D` | 表示“始终按完整方法训练”的模型，测量一个专门适应完整上下文的模型在 C0/CL/CD/CLD 下如何响应证据删除 |
| `M_drop` | 与 `M_LD` 使用相同的 per-epoch demo 顺序随机化，但以 `p=0.5` 独立丢弃 `L` 和 `D` | 让训练覆盖四种上下文组合，降低推理时删去某类证据造成的 train–test mismatch；用于检验词典和示例效应是否依赖训练期始终存在的上下文 |

两者都会在相同 query 上评估 C0、CL、CD、CLD；主计划把 `M_drop` 作为 confirmatory，
`M_LD` 作为 secondary。它们不是用来生成或审核 WP3 词条的模型，也不会帮助 WP3 过门；
相反，它们是 WP3 发布后的下游消费者。只要正式 `lexicon_ref` 和 context 尚不存在，二者的
正式训练就必须继续阻断。

完整规范见
[`stage1-p0-implementation.md`](../experiment-plans/stage1-p0-implementation.md) 第 6.5 节和
[`wp3-terminology-understanding-library.md`](../experiment-plans/wp3-terminology-understanding-library.md)。

## 4. 从启动到当前的修订时间线

### 4.1 R0：梳理既有词典，发现不能直接充当正式 WP3

原项目主实验默认加载 `data/lexicon/annotated_lexicon.json`。该文件本身是有价值的人工资源，
但当前仓库内没有生成者、数据来源、冻结时间、去重规则、人审协议或与本实验 split 独立性的
完整证明。其标题和定义还明确以“仇恨俚语”为收录目标。

本次复核确认，它与 `data/full/raw/annotated lexicon.json` 字节完全一致，SHA-256 均为
`a4a2d1e7826419a365962ded70806a610345d21a998454adeda8b1f999738565`；仓库根 README 和多份
legacy runner/data builder 也都直接指向 `data/lexicon/annotated_lexicon.json`。因此，讨论中的
“原有/旧词典”确实应指这个人工文件，而不是后生成的 656 条 `generated/full/lexicon.json`。

因此，当研究问题从“词典注入能否提高性能”升级为“术语证据是否产生可归因的推理时因果
效应”后，直接把旧词典去掉 `category` 并不能满足正式识别条件：词条是否被收录和定义如何
书写仍可能是任务标签的代理变量。

这一阶段形成的原则是：

- 原人工词典不删除，保留为 legacy/external 候选资源；
- 未完成来源审计前，不把它接入正式 confirmatory pool；
- 后续如果设置 external-resource sensitivity arm，必须与 fit-only 主资源分开冻结和报告。

### 4.2 R1：建立 fit-only 群体损贬词典协议

正式 data 与 train partition 冻结后，第一版 WP3 改为只从 5,165 条 fit 记录构建词典，试图
解决 dev/test 泄漏和不可重放问题。旧协议从 120,566 个原始候选中排序选择 1,000 个候选，
每个候选执行：

- 3 个 Tavily 搜索 logical slots；
- DeepSeek 的 context judge、web evidence judge、final judge 三阶段裁决；
- 保存 raw/normalized evidence、provider model、usage、配置和输入 lineage；
- 通过内容寻址 target/ref 发布。

这一版解决了“数据来源必须 fit-only”的一部分问题，但没有解决标签路径：候选 frame 仍包含
`hate_count`、`category_purity`、`primary_category`、target/argument 来源和带标签的 sample
contexts。资源目标仍是“群体损贬词典”，不是当前的中性术语理解资源。

### 4.3 R2：付费执行前的工程加固

在正式联网前，WP3 增加了：

- 只读 preflight、环境凭据存在性检查和一次性 capability；
- config/data/partition/code 精确绑定；
- Web evidence ID、direct-term match 和响应字段校验；
- provider model/usage capture 与安全 billing ledger；
- Tavily/DeepSeek 请求上限和费用估算；
- 失败即阻断的 artifact publication；
- 两轮 bounded smoke。

旧准备包记录的名义规模是 3,000 个 Tavily logical requests 与 3,000 个 DeepSeek logical
calls。工程上随后又加入 provider-slot checkpoint、发送前 reservation、pending → ambiguous、
单 writer、`0700/0600` 权限、atomic replace 和 `fsync`，目的是避免中断后重复计费或把未知
送达状态当作未发送。

这些措施后来大部分被保留为新协议的基础设施，但旧成本估算、授权、preflight 和 receipt
不能继承到当前资源角色。

### 4.4 R3：live build 暴露 provider 与恢复问题

仓库现有材料记录了至少两条已退役的 live 执行 lineage：

1. **billing-ledger lineage**：attempt 1 在 0 HTTP / 0 计费时 fail-closed；修复入口后，
   attempt 2 完成 321/1,000 candidate，在第 322 个候选首个 Tavily 请求处遇到 TLS EOF。
   该轮记录 964 次 Tavily attempts（963 成功、1 失败）和 963 次成功 DeepSeek calls。
   旧协议没有可恢复的 provider-slot checkpoint，因此 321 个完成项不能导入新 build。
2. **content-addressed checkpoint lineage**：`fchk-c6f6…bf419` 提交了 rank 1–474，
   第 475 个候选在 DeepSeek `web_evidence_judge` 收到 `Content Exists Risk` 后终止。
   前 474 条只有 39 included、435 rejected。

顶层执行状态把“attempt 1/2”主要用于描述第一条 lineage，而 case study 单独记录第二条
lineage；当前文档尚未给出二者统一的 run-name/authorization 映射。为避免错误合并，本复盘
把它们视为两个独立历史账本。后续应补一张 authoritative run registry，明确每次执行的
scope ID、config hash、key/budget scope、授权、开始/终止原因及可恢复性。

### 4.5 R4：从 provider 失败转向候选质量诊断

对 474 条 checkpoint 结果的 case 分析表明，高拒绝率主要不是模型“过于严格”，而是大量
低质量候选已经进入了付费阶段。典型问题包括：

- n-gram/分词碎片：`女拳不`、`女拳都`、`拳在`；
- 完整句子或语法片段：`都让黑蛆喝了吧`、`属实培养奴性`；
- substring attribution：`easy girl` 的意义错误投射给 `easy`；
- 邻近类别污染：`变态 → LGBTQ`、`炼铜 → LGBTQ`；
- taxonomy 与定义矛盾：`哥布林 → Racism`；
- 通用辱骂或普通词误收：`德行`、`辣鸡`、`低能`；
- Web evidence 为空却返回 `supported=true`；
- context 与 Web 使用标准不一致；
- 单个 provider abstention 终止整批任务。

随后完成的确定性修订包括：

- 混合中文语料抽取连续英文短语；
- 增加中文边界碎片、句子片段和 substring suppression；
- 候选语言由候选字符串而非数据集决定；
- `supported=true` 必须引用实际存在的 evidence IDs；
- final include 必须至少得到 context 或 Web 一方支持；
- `Content Exists Risk/content_filter` 归类为 `provider_abstention`，不再等同于语义失败。

离线重挖后，原 120,566 个候选变为 120,281 个；新旧 top-1,000 重合 954 个。修复命中了
部分真实错误，但“固定填满 1,000 项”仍会让低分候选自动补位，说明局部规则修补不足以
解决候选目标定义问题。

### 4.6 R5：加入候选人工门，旧候选方案未通过

为避免继续付费试错，WP3 新增 80-case 离线人工候选门。人工只判断 exact candidate，页面
不展示类别、排序分数、旧模型票据或 Web 结果。

结果为：

- 7/80 可进入供应商阶段；
- 28/80 是碎片；
- 21/80 是句子；
- 其余 24/80 形式可能完整，但不符合旧“独立群体词/可生产词干”scope。

旧候选方案因此未通过。随后针对 49 个碎片/句子 case 生成 exact-span 修订包，要求人工
提交原文逐字连续 span 或“无有效词条”。该包已经冻结，但仓库内未看到完成后的 49-case
修订结果，因此不能把它写成已闭环。

### 4.7 R6：尝试双模型直接抽 span，发现一致性不等于正确性

下一步探索让 Qwen3.8-27B 与 DeepSeek v4 flash 只读取原始 `content`，直接抽取 exact span。
冻结 frame 包含 240 条唯一 fit 记录，每个模型另有 48 条隐藏复测，共 288 个任务/模型。

主要结果：

- 两模型 exact span 集完全一致：181/240 = 75.42%；
- 其中 141/240 是双方都输出空集；
- 非空 exact span 完全一致仅 40/240；
- 59/240 存在 span 集分歧；
- has-span Cohen's κ = 0.5833；
- Qwen 判有 span 66/240，DeepSeek 判有 span 88/240；
- 两模型各自隐藏复测 exact-set 一致率均为 44/48 = 91.67%；
- JSON 与 exact-substring 结构错误为 0。

结论是：协议约束可靠、模型内稳定性尚可，但跨模型一致的大部分收益来自“共同为空”；模型
一致不能替代人工正确性检查。288-case 盲化人工检查包已经生成，但仓库内没有完成后的人工
annotations，因而该探索没有产生可直接升级为 gold 的结论。

### 4.8 R7：科学目标改为 category-free 术语理解库

候选门和双模型 audit 暴露出更根本的问题：旧协议问的是“候选是否属于群体损贬词”，而
Stage 1 真正需要的是“这段解释是否帮助模型理解词义/语用，同时不提前给出 group/hate
答案”。于是资源角色改为 `terminology-understanding-library/v1`。

核心变化如下：

| 维度 | 旧群体损贬协议 | 当前术语理解协议 |
|---|---|---|
| 准入问题 | 是否稳定指向并损贬某群体 | 解释该表达是否实质帮助理解句意或语用 |
| 候选来源 | fit，但使用 target/argument/labels 及派生统计 | 仅 fit 原始 content |
| 中性身份词 | 通常排除 | 可以作为有理解价值的候选 |
| 歧义/回收用法 | 主要作为排除或单一判断 | 在 `ambiguity_notes` 中显式保留 |
| 公开字段 | term、category、definition 等 | term、definition、usage_notes、ambiguity_notes、variants |
| 条目含义 | 收录本身接近任务先验 | 收录不代表当前句子的 group/hate 结论 |
| 正式推理联网 | 不应联网 | 明确禁止联网 |

这次 schema、候选目标、prompt、entry ID、renderer 和 retrieval block 全部变化，使旧 build、
checkpoint、cache、receipt、成本估算和 live 授权全部失效。旧结果只能作为 audit/regression
材料，不得迁移为新正式库的已完成前缀。

### 4.9 R8：实现 A0–A5 与解释阶段的门控生命周期

当前正式流程固定为：

```text
A0 全量规则普查
→ A1 200 条可调试试运行 + 人工穷尽复审
→ 冻结 prompt/model/rule/offset/ranker
→ A2 100 条锁定验证 + 60 个隐藏复测
→ 统计 gate + 人工 PASS
→ A4 覆盖全部 5,165 条
→ A5 200 occurrence 全量抽检
→ 冻结解释执行契约
→ 解释 gate
→ Qwen/DeepSeek/BGE/Web 分支
→ 人工队列与 10% 自动路线 QC
→ 未决项清零
→ finalize / validate / publish-stage1-library
```

主要新增机制：

- A1 只用于调试，不报告锁定性能；任何 prompt、模型、规则或 offset/ranker 改动都会重置
  A0/A1 identity；
- A2 才是锁定验证，只有 `PASS` 才能启动 A4；`INCONCLUSIVE` 只能按连续 100 条 block
  扩充，不能放宽阈值；
- 人工必须能删除、修边界并主动补充模型未提出的 span；
- A4 每条最终只能是 `success`、`empty` 或三次失败后人工形成的
  `exception_resolved`；
- A5 固定审核 200 个 occurrence，并以 Clopper–Pearson 95% 上界控制边界错误率；
- 解释阶段按校准结果决定自动路线是否开启，BGE 只作为特征而非真值；
- 自动路线稳定抽取 10% 做 QC；发现实质错误时关闭对应路线并补齐该路线全量人工决定；
- 发布器重新验证 fit 原文、全部 gate、人工作业、公开字段和 `lex:v2` 身份。

### 4.10 R9：当前真实进度已到 A1 人工复审入口

2026-08-26 晚间实际物化产物比当天较早的状态文档更靠后：

- A0 census：
  `spancensus-91b6d57e…e014`；
- `records.jsonl`：5,165 行；
- `rule_terms.jsonl`：555,546 行；
- `task_label_contribution_count=0`；
- `dev_test_contribution_count=0`；
- A1 checkpoint：200 个成功结果，1 次 retryable failure 后成功；
- A1 review package：`termspan-review-4ed4230bd227ab2744be`，200 cases，offline。

当前仓库没有 A1 完成人工 annotations，也没有 A2 checkpoint、span gate、人工 PASS、A4 span
frame、A5 decision、resolution gate 或 `lexicon_ref.json`。因此准确状态是：

> A0 已完成；A1 模型侧与复审包生成已完成；正在等待 A1 人工穷尽复审；正式 WP3 尚未通过。

## 5. 迄今遇到的主要困难

### 5.1 资源目标反复变化

“仇恨词典”“群体损贬词典”和“术语理解库”不是同一个对象。前两者天然以标签相关性或
群体攻击为准入条件，后者以理解价值为准入条件。早期没有先冻结资源角色，导致大量工程
加固建立在后来被废弃的科学目标上。

### 5.2 去掉类别字段不等于消除标签路径

标签可能进入：候选是否出现、排名、support sample、定义、歧义处理和最终纳入。只隐藏
`category` 无法消除 membership prior。当前方案要求 candidate discovery 从 content-only
重做，代价是旧付费结果不能复用。

### 5.3 中文术语边界和语境依赖难以自动化

中文缺少天然词边界，谐音、缩写、字符替换、生产词干、完整短语、普通身份称谓和句子碎片
混在一起。规则召回很容易产生海量 n-gram；单模型或双模型又会在“是否需要解释”和最小
span 上系统分歧。A0 的 555,546 条规则 term 也说明全量召回与精确筛选之间存在较大规模差。

### 5.4 模型稳定性、模型一致性与正确性是三件事

隐藏复测可以测稳定性，双模型一致可以测互相印证，但两者都不能证明语义正确。双方共同
输出空集、共同把中性身份词当术语、共同继承上下文恶意，都会产生表面一致。因此必须保留
人工穷尽标注、漏召回检查和自动路线 QC。

### 5.5 Web evidence 质量与 provider 行为不可控

Web 搜索可能无直接命中、返回相邻概念、包含过长或敏感摘要，也可能触发 TLS 中断或内容
风控。旧协议还出现空 evidence 却 `supported=true` 的结构矛盾。新协议需要同时处理：

- 直接词面支持与来源去重；
- evidence ID 约束；
- eTLD+1 独立来源计数；
- provider abstention；
- 超长字段与 public-suffix 数据漂移；
- 是否采用 trusted-source registry。

### 5.6 中断恢复与计费安全比预期复杂

外部请求在“已发送、未收到响应、尚未落盘”时无法确定 provider 是否计费。简单 resume 会
重复调用，简单 fail-fast 又会丢失完成前缀。因此引入 reservation、ambiguous、per-slot 与
global cap、单 writer、原子写和 fsync。它提高了可靠性，但也显著扩大了代码、测试和
artifact identity 的复杂度。

### 5.7 每次科学修订都会使工程产物整体失效

候选目标、prompt、模型参数、offset 算法、schema、代码 hash 或 renderer 任一变化，都可能
改变候选 frame 和公开语义。内容寻址生命周期要求产生新 ID，不能“局部修补后继续旧
checkpoint”。这保护了可重复性，也使前期 321/474 条付费结果成为沉没成本。

### 5.8 人工工作量不可省略，只能通过门控精确分配

当前至少包含：A1 200 条穷尽复审、A2 100 条及隐藏复测、必要的 extension、A5 200 条抽检、
解释 Gold、自动路线 10% QC 和最终人工队列。真正困难的不是制作页面，而是冻结判定口径、
保持盲化、避免同一 reviewer 同时调参与做锁定结论，并保证未决项精确清零。

### 5.9 历史状态和命名发生漂移

目前同时存在 321-candidate billing ledger 与 474-candidate content-addressed checkpoint，
但顶层“attempt 1/2”命名没有覆盖所有 lineage；较早状态文档还写着“只完成 A0”，而文件系统
随后已经完成 A1 模型运行和 review package。讨论和汇报时应明确快照时间，并规定：

1. immutable artifact/manifest；
2. 当前实施手册；
3. 执行状态快照；
4. 已标记 retired 的历史准备包；

作为从高到低的状态证据优先级。

## 6. 当前完成项、未完成项与阻塞关系

| 状态 | 内容 |
|---|---|
| 已完成 | 正式 data 与 5,165/616 train partition；category-free schema/config/code；A0 全量 census；A1 200 条模型结果；A1 200-case 复审包 |
| 正在等待 | A1 人工穷尽复审及 annotations 回收 |
| 尚未开始 | 冻结后 A2、统计 gate、人工 PASS、A4 全量扫描、A5、解释 gate、正式 resolution、人工清零、finalize/validate/publish |
| 不存在 | `exps/causal_context/stage1_p0/refs/lexicon_ref.json` |
| 下游影响 | WP4 正式 retrieval bundle、WP5 正式 train/dev context、`M_LD/M_drop` 正式训练均继续等待 WP3 |
| 已退役 | 旧群体损贬协议的 cache/checkpoint/preflight/readiness/receipt/live 授权；旧结果只作 audit/regression/billing |

WP3 当前不是“代码未实现”，而是“代码和 A1 模型侧已实现，但质量门和人工闭环尚未通过”。
不能用工程 smoke、模型一致率或 A1 调试输出替代正式 PASS。

## 7. 建议的近期推进顺序

1. 完成 A1 200 条穷尽复审，保留 missed span、边界错误、普通短语、句子碎片和 occurrence
   错误分布；
2. 根据 A1 只做允许的调试；如果 prompt/rule/offset/ranker/model contract 改变，重新物化
   A0 与完整 A1；
3. 明确签署“已冻结”记录，再启动 A2 的 100 条锁定验证；
4. 生成包含 300 原始任务与 60 隐藏复测的 blind review package，完成 annotations；
5. 运行 gate：`PASS` 才能人工批准；`INCONCLUSIVE` 只按连续 100 条 block 扩展；`FAIL`
   返回协议修订；
6. 获得人工 PASS 后运行 A4 全量 5,165 条；对三次失败项执行精确人工 exception resolution；
7. 完成 A5 固定 200 occurrence 抽检；
8. 冻结 resolution contract，制作解释 Gold 并校准自动路线；
9. 执行 Qwen/DeepSeek/BGE/Web resolution、10% QC 和人工队列清零；
10. finalize、validate、发布 `lexicon_ref.json`，随后再生成新的 verification receipt 和正式
    train/dev context。

## 8. 建议讨论并冻结的决策

### 8.1 原始人工词典的角色

建议在以下选项中明确选择并预注册：

- **主张 A（当前默认）**：仅作为 legacy baseline/case source，不进入 confirmatory WP3；
- **主张 B**：完成来源、时间、数据独立性和人审 provenance 审计后，建立独立 external-resource
  sensitivity arm；
- 不建议把它简单去掉 `category` 后与新 fit-only 库合并，因为这样无法区分 external prior
  与 fit-derived evidence 的效应。

### 8.2 “需要解释的术语”scope

需要统一中性身份简称、普通群体称谓、通用辱骂、行为描述、网络梗、生产词干、完整短语和
回收用法的标注口径。尤其要确认“解释有帮助”是否要求：

- 对原句理解有非平凡增益；
- 脱离当前句子仍存在稳定词义；
- 普通字面义与网络义如何并存；
- 一个 surface 是否允许多义、多语域和相反语用。

### 8.3 Reviewer 分工

建议明确 A1 调参 reviewer、A2 锁定 reviewer、A5 auditor、resolution Gold reviewer 和最终
负责人是否必须分离；若人力不足，至少冻结查看顺序和禁止信息，避免调参知识污染锁定结论。

### 8.4 Web 路线

当前 trusted-source registry 为空，自动 Web 路线要求两个独立 eTLD+1。需要决定：

- 是否继续 search-first fallback；
- 是否建立可信来源表；
- no-web 是否作为独立版本化方案；
- provider abstention 是进入人工还是允许另一模型自动接管。

### 8.5 历史执行账本

需要补充一个统一表，将 321-candidate ledger 与 474-candidate checkpoint 分别绑定到准确
run name、config hash、scope/authorization、请求预算和终止原因。该表只用于审计，不能改变
它们的 retired 状态。

### 8.6 上位六阶段计划同步

当前六阶段计划仍保留 `L-Category`、`L-CategorySwap` 等旧词典干预。category-free WP3
冻结后，应把这些条件改成或拆分为：正确释义、释义改写、错误释义、歧义说明、无关术语、
变体映射和人工 semantic oracle；类别注入只能作为显式的人工 shortcut control，不能再描述
为正式术语库的自然字段。

## 9. 关键证据与产物索引

- 当前 WP3 实施手册：
  [`wp3-terminology-understanding-library.md`](../experiment-plans/wp3-terminology-understanding-library.md)
- 原始主实验人工词典：
  [`annotated_lexicon.json`](../../../data/lexicon/annotated_lexicon.json)
- 后生成 656 条 LLM 词典的配置：
  [`full_llm.json`](../../../config/lexicon/full_llm.json)
- Stage 1 P0 执行状态：
  [`stage1-p0-execution-status.md`](../experiment-plans/stage1-p0-execution-status.md)
- Stage 1 P0 实施规格：
  [`stage1-p0-implementation.md`](../experiment-plans/stage1-p0-implementation.md)
- 旧付费准备包（retired）：
  [`stage1-p0-wp3-paid-build-readiness.md`](../experiment-plans/stage1-p0-wp3-paid-build-readiness.md)
- 旧 web-first preflight（retired）：
  [`stage1-p0-wp3-preflight.md`](../experiment-plans/stage1-p0-wp3-preflight.md)
- 474-case 错误分析：
  [`词典构建case错误分析.md`](词典构建case错误分析.md)
- 双模型 span audit：
  [`双模型span抽样一致性分析.md`](双模型span抽样一致性分析.md)
- A0 census manifest：
  [`manifest.json`](../../../exps/causal_context/stage1_p0/terminology_spans/census/spancensus-91b6d57ea07adedbe8d01d191b02ed59886fffb062f62a3b5cf85d9f2e05e014/manifest.json)
- A1 review package manifest：
  [`manifest.json`](../../../exps/causal_context/stage1_p0/review_packages/termspan-review-4ed4230bd227ab2744be/manifest.json)

## 10. 本文的状态维护规则

后续每次推进只更新以下三项，避免再次把“实现完成”写成“实验通过”：

1. 在开头更新快照日期和一句话状态；
2. 在第 6 节移动已完成/未完成项；
3. 在本节追加实际 artifact ID、人工 decision 和 gate 结果，不预写未来 ID。

任何 prompt、模型、规则、offset/ranker、采样或 schema 改动，必须同时记录：变更原因、受影响
的旧 artifact、是否要求重跑 A0/A1、是否使 A2/A4/A5 或 resolution decision 失效。
