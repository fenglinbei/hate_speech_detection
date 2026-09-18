# 适用关系字段规范 v1

适用协议：`task-applicability-scoring/v1`。本规范落实此前已认可的关系划分；不把新字段追认为旧审核已经填写的字段。字段结构见 [relation-record.schema.json](relation-record.schema.json)。

## 记录单位与身份

一行对应一条有方向的资料—文本关系，不给整包证据一个笼统的“适用”标签。`relation_kind` 仅取：

| 值 | 来源 source | 目标 target | 必填关系维度 |
|---|---|---|---|
| `lexicon_to_query` | 一个具体词典义项 | 当前查询 | `sense_fit` |
| `lexicon_to_demo` | 一个具体词典义项 | 一条示例 | `sense_fit` |
| `demo_to_query` | 一条示例 | 当前查询 | `semantic_reference_fit` 与 `rule_fit` 分别填写 |

词典与示例用 L、D 简写仅为文档阅读方便，不进入模型提示。每个义项使用独立 `sense_id`；同一词形的不同义项不得合并。同一材料有多个出现位置时分别建边；多处共同构成一项判断时可以同边记录多个片段，并写明范围。换一种措辞不自动产生独立家族。

`relation_id` 唯一；`family_id` 绑定材料构造家族。同一源材料在不同查询中复用时保留材料身份和正文哈希，但重新审核目标关系。`supersedes` 指向被修订关系 ID，初版为 null，旧记录不覆盖。

source / target 包含 `material_id`、`kind`、`raw_text`、`text_sha256`、`sense_id`。词典 raw_text 是模型实际会看到的一个义项的完整文本（含词条及释义），不是只用于检索的摘要；非词典的 sense_id 为 null。同 ID 不同正文或义项发生冲突时，须版本化，不能去重合并。

所有位置是原文 Unicode code point 半开区间 `[start,end)`；必须满足 `raw_text[start:end] == text`，不能使用 UTF-8 字节偏移或 token 偏移。此版本的关系审核与词面比较直接使用原文，不做 Unicode 归一化、大小写折叠或标点删除；若保留旧 matcher 的归一化位置，应在其原来源中连同映射保存，不能替换本字段。

## 资料自身正确性单列

`source_quality` 记录当前资料自身的正确性：词典使用 `definition_quality`；示例使用 `demo_answer_reliability`，按本任务规则核对该示例自身答案。值为 `accepted / disputed / unknown / null`：分别表示认可、存在具体问题、审核后仍未知、尚未审核。它不由对查询的适用性、查询标签或模型结果生成。

示例质量另存 `reviewed_answer`，明确核对的是哪一个“有／无”答案；词典的该字段为 null。本字段同时保存 `rationale`，并在 provenance 中保存 `review_kind`、`authorship`、`adoption` 和 `decision_ref`；AI 的 accepted 建议仍是 AI 草稿。采用旧审核必须验证材料、义项、任务口径及版本相同，保留来源，不把规则级确认变成新材料的逐项裁决。查询自己的任务参考另存，不能混入 source_quality 或关系值。

主要“正确但不适用”比较要求来源质量已被明确采纳为 accepted；disputed / unknown / 未审资料保留在审核或单独可靠性诊断中，不混进主适用性对照。语义关系可先于质量裁决起草，二者不互相覆盖。

## 三个关系维度

每个有效维度包含 `value`、`information`、`source_spans`、`target_spans`、`rationale`、`limitation`、`decision_relevance`。

| 维度 | 回答的问题 | 不自动意味着 |
|---|---|---|
| `sense_fit` | 这个具体义项能否解释目标指定位置的实际用法？ | 作者实施攻击；词典对另一个位置也适用 |
| `semantic_reference_fit` | 示例中哪项词义／所指信息可用于解释查询的哪个成分？ | 同话题足够；示例答案能够照搬 |
| `rule_fit` | 示例中的哪条对象绑定、作者立场、引用／反驳、泛化或攻击判断规则适用于查询？ | 相同规则必然给出相同标签 |

词典边的后两维整体为 null；示例边的 sense_fit 整体为 null。这表示该维度在本记录结构中不适用，不表示审核者已经判定 none。

| value | 操作含义 |
|---|---|
| `direct` | 指定信息／规则在指定位置直接适用，无需补造未提供的身份、语境或前提 |
| `partial` | 只有明确可指出的一部分适用，或迁移存在实质限制；limitation 必须说明可迁移与不可迁移的范围 |
| `none` | 有依据地确认所指定信息／规则不适用于所指定位置；不声称整份资料对模型绝无任何作用 |
| `unclear` | 已审核，但合理解释之间仍无法确定关系；limitation 必须说明缺失信息或冲突 |
| null | 尚未审核；不能代替 none 或 unclear，原因、证据与判题相关性保持空值 |

direct / partial / none 均需非空来源与目标片段、具体 information 和理由。unclear 保留已有片段并注明缺失信息。对示例两维分别评级；不能用 OR、AND 或平均分合成“总适用分”。旧“词义／话题关联”若只证明同话题而没有可指出的信息迁移，不能自动映射为本版 semantic_reference_fit=direct。

## 判题相关性与词面重合

`decision_relevance` 随每个关系维度分别记录，而非整条边共用一个结论：

- value 为 `task_relevant / background / unclear / null`。
- `steps` 可多选 `sense_interpretation`、`referent_binding`、`author_stance`、`quotation_scope`、`generalization_scope`、`attack_presence`、`severity_boundary`。
- task_relevant 必须写出判题环节；background 的 steps 为空；unclear 说明不确定处；未审为 null。
- 描述这项信息与判题环节的关系，不断言模型不知道它、不知道就一定答错，或已实际使用它。sense_fit=direct 可以同时只是 background。

`lexical_overlap` 独立记录预先指定的 `focal_forms`：主比较使用原文中完整字面串的精确共现，value 为 `present / absent / unclear / null`。`shared_forms` 必须等于 focal_forms 中同时出现在 source.raw_text 和 target.raw_text 的项，去重且保留 focal_forms 次序；额外保存两侧全部字面命中的片段。present 表示至少一项共现，absent 表示没有；focal_forms 尚未确定时为 unclear 或未审 null，而不是 absent。

这里的字面命中是机械字符串事实，即使是子串误匹配也只证明字面出现，不证明词义。无大小写折叠、同义词扩展或语义相似度阈值。focal_forms 在看目标结果前写定，不能事后选择产生目标格子的词。不要把“低重合”解释为全文没有任何共有字；本规范使用“指定词形出现／未出现”的确定定义。

## 呈现、审核来源与暴露

`presentation_refs` 列出实际出现该边相关资料的条件与提示哈希；未构造输入时为空数组。每项另记 `source_present` 和 `target_present`，可区分候选资料、没有进入提示的来源以及实际呈现。它不改变语义关系本身。后续输入冻结须核对 renderer 的实际字节，而不是仅相信来源列表。

示例边的 `introduced_lexicon_relation_ids` 引用其关联的词典来源边；词典未实际呈现不得当作模型已接收。该列表不自动说明示例有害，具体适配以对应边为准。

`provenance` 保存 `authorship`（ai/human/mixed）、`review_kind`（ai_note/human_with_ai/human_authored）、`adoption`（none/individual/bulk）、`accepted_fields`、`decision_ref` 及四项暴露：当前目标模型结果、原 Gold、AI 意见、相关历史结果。accepted_fields 用 JSON Pointer 精确列出采纳字段；整批采纳另有具体对象清单来源，不冒充独立逐项裁决。AI 起草与人工核对沿用既有方式，不新增“人必须先于 AI 作答”的门槛。

规则级冻结不填写任何材料的 accepted_fields。历史暴露不可清零；当前目标结果已见的记录仍可作探索，但不能称为对该结果的预先审核。新家族确认还需在输入计划中排除参与开发的家族；本 schema 本身不宣布任何材料为保留集。

## 主比较准入与保留项

后续每个对比明确指定被操纵的关系维度、来源质量、词面状态、标签对应，以及应保持的其他关系。主要二元比较只使用该维度已明确采纳的 direct / none，且任务参考、必要资料质量和词面字段已解决；partial、unclear、未审和质量争议另列，保留数量与原因，不根据模型表现清除。

一个维度的 none 不要求另一个维度也为 none；例如词义信息不适用而反泛化规则适用的示例应原样保留。任务标签与适用关系交叉，适用可以对应“有”或“无”。无数据支持时不能强行补齐四格，特别是词典的“无字面重合但义项适用”不能仅因设计需要而造出。

新增字段不回填旧人审记录。复用旧案例须建立新版本关系并指向旧记录，保留暴露；本包的 schema 示例只有 AI 作者身份，没有人工采纳、Gold 或模型预测。
