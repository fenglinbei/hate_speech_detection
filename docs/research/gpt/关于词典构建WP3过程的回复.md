## 结论先行

当前不应继续扩大 n-gram 候选或增加更多 LLM 投票。真正的瓶颈不是抽取模型能力不足，而是 **WP3 的标注对象和决策边界尚未冻结**。

旧流程实际上让模型同时回答了三个不同问题：

[
\underbrace{\text{哪里存在特殊表达}}*{\text{mention detection}}
\quad\rightarrow\quad
\underbrace{\text{该表达在这里是什么意思}}*{\text{sense interpretation}}
\quad\rightarrow\quad
\underbrace{\text{它是否构成群体损贬、属于哪个类别}}_{\text{downstream task label}}
]

第三个问题正是后续 hate-speech extraction 要预测的变量。一旦三者混在一起，就会出现你观察到的全部问题：普通群体名被收录、通用辱骂被收录、短 substring 继承整段含义、异常编码漏检，以及两个模型各自稳定但彼此口径不同。

LexSRAG 原论文中的词典条目本身是 ((\text{keyword},\text{group},\text{definition}))，检索文本还直接序列化 Group；同时词典消融是整套方法中影响最大的部分，Avg-F1 从 0.3664 降到 0.3445。因此，重新构建不直接编码任务标签的 WP3 资源，不是一般的数据清洗，而是后续来源因果效应能否成立的前置条件。

建议把 WP3 正式重定义为：

> **构建一个 fit-attested、sense-level、ontology-free、verdict-free 的术语理解资源。它解释输入中非透明表达的形式、核心含义和适用语境，但不输出下游类别、hate 判断或当前样本的结构化答案。**

实施上采用：

* **候选发现高召回**：多生成器并集；
* **正式发布高精度**：所有发布条目经过人工审核；
* **A-core / B-contextual / C-quarantine 分层**；
* 不再以“必须产出 1,000 条”为目标，而以覆盖饱和与质量门槛为停止条件。

---

# 一、先冻结“category-free”的准确含义

## 1. Category-free 不等于 referent-free

这里最关键的区分是：

> **category-free 应当表示 ontology-free + verdict-free，而不是把词义中的指称对象也删掉。**

例如：

```text
txl：常见为“同性恋”的拼音首字母缩写。
```

这是正常的词义解释，应当允许。它没有输出：

```text
Group = LGBTQ
Hateful = true
当前句子的 target 是……
```

如果连“同性恋”这一语义都不能出现，那么词典实际上无法帮助模型理解 `txl`，CL 条件也就失去了科学意义。

应严格禁止的是：

1. 数据集类别词：`Racism / Region / LGBTQ / Sexism / Others / Non-Hate`；
2. hate/non-hate 判断；
3. “该词属于针对某群体的仇恨词”等任务结论；
4. 当前句子的 target、argument 或四字段答案；
5. 从原标注复制出的目标类别或结构；
6. “看到该词就应判为某类”的决策提示。

允许的是：

1. 标准写法或解码结果；
2. 字音、字形、缩写、混写等形成机制；
3. 常规词义或语境中的特定词义；
4. 语体和歧义提示；
5. “语气、立场需结合上下文”等非决策性说明。

语义解释自然可能提高下游类别的可预测性。这是 **L 证据应有的因果效应**，不是泄漏。真正需要用 placebo 和反事实实验控制的是“词条被收录”本身产生的先验，以及解释中的显式任务答案。

---

## 2. 把资源对象拆成 mention、entry 和 sense 三层

不要继续把“字符串”直接等同于“词条”。建议使用三层对象：

### Mention：语料中的一次具体出现

[
m=(\text{record_id}, start, end, surface, context)
]

例如同一个 `txl` 在不同句子中是不同 mention。

### Entry：形式族

一个 entry 可以包含：

```text
headword: 国女
surface_forms: [国女, 郭女]
```

形式族只处理写法关系，不先验地假设所有上下文含义相同。

### Sense：上下文中的一种释义

同一形式可以具有多个 sense：

```text
entry: 基佬
sense_1: 非正式称呼，带负面语气
sense_2: 群体内自称或回收使用
sense_3: 引述、讨论该词本身
```

下游检索的实际对象应是：

[
(\text{entry},\text{sense})
]

而不是无条件的字符串匹配。这样才能处理多义、回收用法、引用用法和语境依赖。

---

# 二、建议采用的详细术语纳入规范

可以把正式纳入条件写成：

[
\operatorname{Admit}(m,s)=
A_{\text{fit}}
\land U_{\text{unit}}
\land G_{\text{gap}}
\land E_{\text{support}}
\land P_{\text{portable}}
\land N_{\text{neutralizable}}
]

其中：

* (A_{\text{fit}})：存在于冻结 fit partition；
* (U_{\text{unit}})：是最小且语义自足的语言单位；
* (G_{\text{gap}})：表面直接理解不足以恢复当前词义或语用；
* (E_{\text{support}})：释义有上下文或外部材料支持；
* (P_{\text{portable}})：不是只对一个句子成立的任意解释；
* (N_{\text{neutralizable}})：能在不输出类别和 hate 判断的情况下给出解释。

## 1. “解释缺口”是核心纳入标准

一个 span 值得收录，当且仅当至少满足下面一类情况。

| 类型         | 判定依据                   | 示例                     |
| ---------- | ---------------------- | ---------------------- |
| 形式解码缺口     | 表面写法与标准形式不同            | `ai紫 → 艾滋`、`J生虫 → 寄生虫` |
| 缩写或混写缺口    | 需要拼音、字母、数字或跨语言知识       | `txl`                  |
| 词汇语义缺口     | 不能按字面组合得到常规网络含义        | `舔狗`                   |
| 固定搭配缺口     | 含义属于整个短语，不能投射到单个子词     | `easy girl`            |
| 隐喻/转喻/典故缺口 | 需要社区或文化背景才能理解          | 特定的地域刻板印象代称            |
| 语用缺口       | 指称基本清楚，但立场、反讽、回收用法依赖语境 | `基佬` 的不同使用方式           |
| 语境特定多义     | 同一表面词在当前语境使用非常规义       | 某些普通词的隐晦代称用法           |

CodedLang 将“编码语言”限定为表面形式经过明确的语音、字形、符号或跨语言编码，特别强调普通方言、行话或共享知识造成的不透明，并不自动属于编码语言。该工作采用七类机制，并通过精确 span 标注和人机循环扩展词典。([arXiv][1])

你的 WP3 范围比 CodedLang 更广，因为还要包含俚语、固定表达和语用特殊词。因此建议使用两个正交字段，而不是强行将所有现象放进一个 taxonomy：

```text
form_mechanism:
  none
  homophone
  phonetic_abbreviation
  orthographic_variant
  split_or_insertion
  emoji_or_symbol
  mixed_script
  cross_lingual
  cipher

interpretation_mechanism:
  conventional_slang
  non_compositional_mwe
  metaphor_or_metonymy
  stereotype_or_allusion
  context_polysemy
  irony
  reclaimed_or_quoted
```

---

## 2. 明确排除规则

以下类型原则上不进入正式资源。

### 普通透明表达

如 `黑人` 在普通指称意义下不需要额外解释，应排除。它可能与下游类别相关，但不存在术语理解缺口。

### 通用辱骂或普通负面词

如 `低能`、`德行`、`辣鸡`，若只是常规词义或一般辱骂，不应因为出现在 hate 数据集中就纳入。只有出现特殊变体、非常规义或固定编码时才处理对应 sense。

### 非自足 n-gram

如：

* `女拳不`
* `女拳都`
* `拳在`

应先判断其中是否存在独立词条 `女拳`；语法附着成分不进入 span。

### 完整句子或临时修辞

如 `都让黑蛆喝了吧`、`属实培养奴性`，除非有证据表明它们已成为跨语境复现的固定表达，否则不作为词典条目。

### Substring 意义投射

`easy girl` 若具有短语级含义，必须保留整个短语。不能将解释附给 `easy`。判定原则是：

> 将候选 span 单独替换为释义后，原句是否仍保持目标解释？

如果不能，则边界过短。

### 只在当前句子中成立的推测

单一罕见组合、无独立证据、两个解释都合理时，不应被迫进入 included/rejected 二分，而应进入 C-quarantine。

---

## 3. 对你列举案例的建议判定

| 表达        | 建议                    | 原因                              |
| --------- | --------------------- | ------------------------------- |
| `ai紫`     | A-core                | 明确形式解码，解释可写为“艾滋的谐音/变体写法”        |
| `J生虫`     | A-core                | 混合文字与音形替代，可稳定映射                 |
| `txl`     | A-core                | 拼音首字母缩写；定义不需要 hate 判断           |
| `幕刃`      | B-contextual，证据充分后升 A | 需要确认是否稳定对应“母人”，不能只依赖单个模型推断      |
| `国女 / 郭女` | A 或 B                 | 可拆为标准形式、变体关系和语境说明               |
| `舔狗`      | A-core                | 非透明、常规化网络词；即使不是群体仇恨词也应收录        |
| `龟男`      | 视证据进入 A/B             | 不是因其是否仇恨，而是看是否具有稳定、非透明释义        |
| `基佬`      | B-contextual          | 指称较稳定，但语体、贬义、回收使用有明显语境依赖        |
| `黑人`      | Reject                | 普通透明词，除非具体 occurrence 使用了非常规编码义 |
| `女拳不`     | Reject span；重提取 `女拳`  | 前者是语法碎片                         |
| `easy`    | Reject；考虑 `easy girl` | 不能把短语意义投射给子串                    |
| `低能 / 德行` | 通常 Reject             | 普通词或通用辱骂，除非存在特殊 sense           |

---

## 4. 发布层级

### A-core：用于阶段一 confirmatory CL

要求：

* 边界清楚；
* 核心义稳定；
* 不存在未解决的主要反例；
* 可以生成简洁、无类别的解释；
* 人工审核通过。

### B-contextual：用于探索性分析或经过 sense gate 后使用

包括：

* 多义词；
* 立场依赖表达；
* 回收用法；
* 仅部分语境成立的代称；
* 证据相对有限但并非错误的条目。

建议初始阶段不将 B 全部放入 confirmatory CL。可以在阶段二作为独立因素研究。

### C-quarantine：不进入正式检索

包括：

* 可能存在特殊含义，但证据不足；
* 模型分歧大；
* 边界或标准形式不确定；
* 网络材料相互矛盾。

### R-reject：保留拒绝原因

不要删除 rejected 记录。至少保留：

```text
fragment
transparent
generic_insult
sentence_level
substring_projection
unsupported_sense
one_off_creation
label_derived
duplicate
wrong_boundary
```

这会使 WP3 可审计，也可以定量报告失败模式。

---

# 三、自动化 span 提取：不要再用全量 n-gram，改用多生成器弱监督

全量 n-gram 的问题不是阈值没调好，而是对象错了。它枚举的是字符串子序列，而不是语言单位。对于一条短句，绝大多数 n-gram 都没有词汇性，因此后续 LLM judge 被迫在极低先验的候选空间中工作。

建议采用：

[
C(x)=
G_{\text{rewrite}}
\cup G_{\text{lexical}}
\cup G_{\text{form}}
\cup G_{\text{phrase}}
\cup G_{\text{bootstrap}}
\cup G_{\text{uncertainty}}
]

每个生成器是一个 labeling function，只负责召回一部分现象。多弱标注源输出 span，再进行聚合，是弱监督 NER 中已有的成熟思路；相关方法允许不同 labeling function 只覆盖部分现象，并用 HMM 或其他聚合器处理不完整、冲突的 span 标签。([ACL Anthology][2])

## 1. 先建立真正的 span gold set

不应等自动流程完成后再抽查。建议：

* 将已有 240 条改造成 **guideline development set**；
* 保留 48 条重复样本检查 annotator/model 稳定性；
* 再从 fit 中冻结至少 240 条作为 **sealed span audit set**；
* 资源允许时扩展到 400–600 条；
* 采样只依据文本属性，不看 `targeted_group` 和 `hateful`。

分层维度可以包括：

* 字符串长度；
* 中英/数字/符号混写；
* 低频字或异常字形；
* 模型是否提出 span；
* 两模型是否分歧；
* 是否包含已知网络语；
* 是否含多个潜在表达；
* 是否为 Non-Hate 样本，但标注者不可见该标签。

标注字段至少应为：

```text
has_interpretation_relevant_span
start
end
surface
minimality
form_mechanism
interpretation_mechanism
canonical_or_paraphrase
context_dependence
review_status
```

你现有结果中，两模型各自复测一致率达到 91.67%，但跨模型 has-span κ 只有 0.5833。这更像是 **各自稳定执行了不同的概念边界**，而不是随机解码噪声。因此增加第三个模型投票不会从根本上解决问题；应先用人工 adjudication 冻结 guideline。

---

## 2. 原始文本与规范化视图必须分开

始终保留 immutable raw text，并生成若干平行视图：

```text
raw
unicode_normalized
fullwidth_halfwidth_normalized
simplified_traditional_view
lowercased_latin_view
punctuation_compact_view
pinyin_view
```

但所有规范化视图必须保留：

```text
normalized_index -> raw_start/raw_end
```

最终 span 一律回写到 raw offsets。否则容易在全半角、繁简转换、空格和 emoji 上产生不可重放的边界。

---

## 3. 六类候选生成器

### G1：最小改写—差异对齐生成器

这是当前最值得优先实现的生成器。

不要直接问“有哪些群体损贬词”，而是要求模型：

> 将句子改写为语义等价、标准、直白的中文，只修改必须依赖网络文化、字音字形、缩写、隐喻或特殊语用知识才能理解的局部表达。不要判断仇恨、冒犯、群体类别或情感标签。

输出：

```json
{
  "source_span": "幕刃",
  "start": 8,
  "end": 10,
  "replacement": "母人",
  "mechanism": "orthographic_or_phonetic_variant",
  "requires_context": true
}
```

随后通过 source–rewrite alignment 产生候选。这比“从头找 span”更容易捕获 `幕刃`、`ai紫`、`J生虫` 一类局部变换。

建议拆成两个独立 pass：

1. **surface decoding pass**：只找音、形、缩写、混写和符号编码；
2. **lexical/pragmatic paraphrase pass**：只找俚语、固定搭配、隐喻和语境特殊义。

两个任务分开可以减少模型把普通负面表达也全部标成“特殊词”。

### G2：直接术语 mention 提取器

使用已经冻结的定义和大量正反例，要求输出：

> 若不了解网络语、缩写、编码形式、固定搭配或特定语用，就可能无法正确理解当前句子的最小连续表达。

明确负例：

* 普通人名、地名和群体名；
* 一般情绪词；
* 普通辱骂；
* 语法碎片；
* 整句；
* 仅因其与 hate 标签相关而被选中的词。

Qwen 与 DeepSeek 在此生成器上取 **并集**，而不是交集。交集适合高精度，但会进一步放大你已经观察到的漏检。

### G3：形式规则生成器

规则只负责产生候选，不直接决定纳入。至少覆盖：

* 同音、近音和声调忽略的拼音距离；
* 拼音首字母；
* 拉丁字母、数字和汉字混写；
* 全角、拆字、偏旁或视觉近似；
* 中间插入符号或字符；
* emoji 替代；
* 字符分离与合并；
* 跨语言近音；
* 繁简或非标准异体。

中文有毒文本在同音和 emoji 变换下会显著影响模型判断，这支持将这两类机制作为独立候选生成通路，而不是依赖一般语义模型自行发现。([ACL Anthology][3])

### G4：受约束的词组挖掘

不再枚举所有 n-gram。只有满足至少一个触发条件的片段才进入候选池：

* 在两个以上独立上下文重复；
* PMI、左右熵或其他搭配强度较高；
* 分词器间边界分歧明显；
* LLM 改写对齐命中；
* 包含混合脚本或异常字符；
* 与已接受 entry 的语境 embedding 接近；
* 出现在一般网络语资源中，并且确实在 fit 中出现。

统计指标只用于候选排序，不能独立证明其具有特殊含义。

### G5：已接受条目的迭代传播

一旦人工接受某个 entry：

1. 在全部 fit 记录中检索其表面形式；
2. 聚类不同上下文；
3. 找出新 sense；
4. 对相邻形式做音形扩展；
5. 人工验证后更新 entry；
6. 继续迭代直到新条目趋近于零。

CodedLang 采用的就是小规模种子、字符串和拼音异常召回、人工验证、词典扩展、再召回的迭代方案，并以没有新验证 span 为收敛条件。该工作还发现主要分歧来自“普通方言”与“有意编码”的边界，说明精确概念规范比一次性大模型抽取更重要。([arXiv][1])

### G6：分歧与不确定性挖掘

将以下样本放入人工高优先级队列：

* Qwen 有、DeepSeek 无；
* 两模型 span 边界不同；
* 两模型 replacement 不同；
* 同一表面形式在上下文聚类中分成多个簇；
* LLM 无法产生稳定直白改写；
* 规则检测到形式异常，但 LLM 未标；
* 改写前后语义相似度显著下降。

分歧不是应当删除的噪声，而是最有可能暴露 guideline 边界和新类型的样本。

---

## 4. 固定 span 边界规则

建议把下面规则写入标注手册，并以最小对照样例解释。

### 最小语义充分原则

选择能独立承载特殊含义的最短 span，但不是机械地选择最短字符串。

```text
女拳不行 → 女拳
easy girl → easy girl，而不是 easy
```

### 替换可成立原则

将 span 替换为解释后，句子应基本保持预期含义和语法结构。

### 功能词排除原则

否定词、程度副词、语气词、体标记等通常不进入 span，除非整个组合已词汇化。

### 嵌套原则

若短 span 和长 span 对应同一 sense，保留最小自足 span；若长 span 有独立的非组合义，则保留长 span。

### 多次出现原则

同一表达在一句中多次出现，应标为多个 mention，共享 entry/sense。

### 不连续表达原则

若确有插入或拆分编码，可以记录：

```json
{
  "segments": [[3, 4], [6, 8]],
  "envelope": [3, 8]
}
```

但主资源优先连续 span。不连续条目单独报告，不应被强制压缩成包含大量无关字符的连续短语。

### Sense 发生原则

对多义词，只有当前 occurrence 使用特殊 sense 时才标。不能因为某个字符串在其他句子中是网络语，就标记其所有出现。

---

## 5. 多模型结果如何聚合

建议把每个生成器输出转成候选级特征：

```text
qwen_direct
deepseek_direct
qwen_rewrite
deepseek_rewrite
phonetic_rule
orthographic_rule
phrase_score
fit_context_count
cross_context_consistency
accepted_entry_match
boundary_valid
```

初期可以训练一个简单的 logistic regression 或 gradient-boosted ranker，仅用于 **人工审核排序**。不要让聚合器直接发布词条。

待积累足够人工 mention 后，再训练一个字符级 BILOU span tagger：

* 输入：raw content；
* 主任务：interpretation-relevant span；
* 辅助任务：form mechanism；
* 不输入任何 hate/group 标签；
* 弱标注用于预训练，人工 gold 用于校准和最终评估。

这样自动化系统的职责是减少人工扫描量，而不是取代词典编纂。

---

# 四、条目解释流程：LLM 知识只作为假设，联网材料作为证据

## 1. 不要让一个 LLM 同时“猜意思、搜证据、裁决并写定义”

旧流程的 context judge、web evidence judge、final judge 虽然分了阶段，但如果三个阶段仍沿用同一个初始假设，很容易形成确认偏误。

建议将每个 sense 拆成若干可验证 claim：

```text
C1: surface 与 canonical_form 的映射
C2: 核心词义
C3: 形成机制
C4: 适用语境
C5: 可能的其他义
C6: 语用或语体特征
```

每条外部 evidence 对 claim 标注：

```text
support
refute
qualify
background
```

以及：

```text
direct
partial
context_only
irrelevant
```

这可以借鉴 EviTrace 的 atom–evidence mapping 思路。但 EviTrace 自身的人审结果也表明，LLM 生成的 relation 一致性只有中等水平，而自报 confidence 校准较弱，因此不能把 LLM confidence 当作发布依据。

---

## 2. 推荐的七阶段解释流水线

### 阶段 A：构造 fit-only context pack

每个候选 entry 收集：

* 全部 fit occurrences；
* occurrence 前后文；
* 表面形式；
* 改写生成器提出的 canonical form；
* 机制候选；
* 语料频次；
* 上下文聚类。

绝不能放入：

* 原始 group；
* hateful；
* target/argument；
* 原词典类别；
* dev/test 内容。

外部搜索只用于解释 **fit 中已经出现的候选**，不能通过 Web 搜索新增未在 fit 中出现的正式 headword。

### 阶段 B：参数知识生成多个假设

让 LLM 给出 1–3 个竞争解释，而不是一个最终答案：

```json
{
  "hypotheses": [
    {
      "canonical": "...",
      "gloss": "...",
      "mechanism": "...",
      "supporting_context_ids": ["..."],
      "would_need_evidence": ["..."]
    }
  ]
}
```

模型参数知识不计入独立证据，只用于生成检索假设。

### 阶段 C：上下文 sense 聚类

用 occurrence 周围文本的 contextual embedding 聚类。对同一形式：

* 若多个簇共享同一解释，合并；
* 若不同簇对应不同含义，创建多个 sense；
* 若只有一个孤立 occurrence，标记为低证据；
* 不因字符串一致就默认语义一致。

中文网络词定义研究表明，UGC 数量和质量都会影响定义准确性，而且对未见新词，模型常依赖先验记忆而非真正从上下文推断。因此，不能仅把一个候选出现句交给模型后直接接受定义。

### 阶段 D：自适应联网检索

不再对每个候选固定执行三次搜索。按风险分层：

#### Tier 0：确定性形式转换

如非常清楚的拼音缩写、全半角、明显拆字。可不搜索，或只做一次确认。

#### Tier 1：常见稳定网络语

执行：

```text
"词语" 是什么意思
"词语" 网络用语
"词语" 用法
```

#### Tier 2：罕见、敏感或多义表达

增加：

```text
"词语" 出处
"词语" 另一种含义
"词语" 原句片段
"词语" "候选标准形式"
```

搜索应同时包含开放式查询与假设验证查询，不能全部写成“X 是否表示 Y”，否则会强化首个模型猜测。

#### Tier 3：证据冲突

进入人工调查，不自动发布。

来源建议分层：

1. 学术论文、正式辞书、官方或高质量语言资源；
2. 可信媒体对网络用语的解释；
3. 专门的网络语、方言或亚文化资源；
4. 论坛、社交媒体和评论，仅作为真实用例；
5. 搜索摘要不能独立作为证据，必须保存底层页面。

对于不稳定俚语，建议至少：

* 两个相互独立的来源；或
* 一个较权威来源，加两个一致的 fit 上下文。

镜像转载和相互抄录不算独立来源。

### 阶段 E：基于证据的结构化合成

定义生成模型只能看到：

* fit context pack；
* 已筛选 evidence；
* claim–evidence map；
* 输出规范。

建议定义拆成四个字段：

```text
canonical_form
gloss_core
formation
ambiguity_note
```

另存但默认不在阶段一渲染：

```text
pragmatic_note
register
reclaimed_or_quoted_usage
```

### 阶段 F：独立反证检查

第二个模型不负责“打一个总分”，而逐项检查：

* span 是否过短或过长；
* 是否存在 substring attribution；
* 释义是否与至少一个 fit 上下文冲突；
* 是否遗漏主要替代义；
* 是否把语气推断写成核心词义；
* 是否出现下游类别或 hate 判断；
* 是否加入证据中不存在的文化背景；
* 是否把关联含义误写成字典义。

LLM 自动生成辞书内容存在明显的扩写和文化失真风险。2026 年一项人工评估在 LLM 改写的辞书条目中发现约 19% 含有幻觉，典型问题包括无依据扩写、文化填充和语用极性反转。([ACL Anthology][4])

### 阶段 G：人工发布审核

建议所有进入 A-core 或 B-contextual 的正式条目至少经过一次人工审核；高风险条目双人审核。

高风险条件：

* 仅一个 fit occurrence；
* 两模型解释不一致；
* sense 多义；
* 可能存在回收或引述用法；
* Web 来源只有社区帖子；
* 涉及罕见编码；
* 释义包含较强语用或立场判断；
* 自动 verifier 发现冲突。

CHEER 的中文网络词资源同样采用 LLM 合成后人工核查和修改，审核重点包括语义准确、简洁和语言流畅，并明确删除来源未支持的额外内容。

---

## 3. 不使用 LLM 自报 confidence

最终 confidence 应由可观测证据计算，而不是让模型输出 0.87：

```text
deterministic_transform
fit_context_count
independent_source_count
source_tier
cross_context_consistency
cross_model_agreement
alternative_sense_count
negative_evidence_count
human_review_count
```

可在已人工审核条目上训练或校准：

[
P(\text{entry valid}\mid \phi_{\text{evidence}})
]

但正式状态仍由规则和人工决策控制。

---

# 五、建议的 `lexicon_ref.json` 结构

```json
{
  "resource_version": "wp3-v1.0.0",
  "construction_partition": "fit-only",
  "entries": [
    {
      "entry_id": "lex_000123",
      "headword": "txl",
      "surface_forms": ["txl", "TXL"],
      "senses": [
        {
          "sense_id": "lex_000123_s1",
          "canonical_form": "同性恋",
          "gloss_core": "“同性恋”的拼音首字母缩写。",
          "formation": {
            "form_mechanism": "phonetic_abbreviation",
            "source_form": "txl",
            "decoded_form": "同性恋"
          },
          "ambiguity_note": "具体语气和立场需结合上下文判断。",
          "pragmatic_note": null,
          "interpretation_mechanism": "abbreviation",
          "sense_conditions": [],
          "fit_mentions": [
            {
              "record_id": "fit_xxx",
              "start": 4,
              "end": 7,
              "surface": "txl",
              "context_hash": "..."
            }
          ],
          "evidence": [
            {
              "source_id": "src_xxx",
              "source_tier": 2,
              "supports": ["canonical_form", "formation"],
              "relation": "support",
              "directness": "direct",
              "content_hash": "...",
              "accessed_at": "..."
            }
          ],
          "confidence_features": {
            "fit_context_count": 3,
            "independent_source_count": 2,
            "cross_model_agreement": true
          },
          "release_tier": "A",
          "review": {
            "reviewer_count": 2,
            "decision": "accept"
          }
        }
      ],
      "provenance": {
        "candidate_generators": [
          "qwen_rewrite",
          "deepseek_direct",
          "phonetic_rule"
        ]
      }
    }
  ]
}
```

正式资源中不应存在：

```text
group
targeted_group
hateful
hate_label
task_prediction
```

`fit_mentions`、外部证据原文和审核信息用于 provenance，默认不渲染进 L 上下文。

数据来源、构建时期、文本特征、审核者背景、版本和来源继承关系应形成独立的 data statement/provenance appendix；已有数据说明规范也强调衍生数据应记录源数据及其适用限制。

---

# 六、阶段一的 L 条件应该渲染什么

建议 WP3 保存丰富字段，但 WP5 为不同实验提供多个 renderer。

## `L_form`

只提供形式解码：

```text
表达：郭女
标准形式：国女
形成方式：谐音或变体写法
```

用于检验形式归一化效应。

## `L_sem`

提供核心词义：

```text
表达：舔狗
常见含义：在关系中长期单方面过度讨好、缺乏对等回应的人。
```

建议将它作为阶段一主 CL 的候选版本。

## `L_prag`

增加语用和语体：

```text
语用说明：通常为非正式网络表达，具体褒贬和说话者立场需结合上下文。
```

建议放到阶段二路径分解，不与核心语义混成一个不可拆分的 L。

## `L_full`

全部字段，仅作为 exploratory 或后续实验。

这样阶段二可以直接比较：

[
L_{\text{form}},
L_{\text{sem}},
L_{\text{prag}},
L_{\text{full}}
]

而不必重新构建词典。

---

# 七、placebo 必须控制“被词典命中”这个先验

你担心的 label prior 不仅来自定义内容，也来自：

> 模型发现某个 query span 被系统专门列入“背景信息”。

因此 PL 不能只放随机无关词条。最有效的是 **同 headword、同位置、同格式、近似 token 长度** 的 matched placebo。

## PL-membership

```text
表达：txl
说明：这是原句中的一个表达，具体含义需结合上下文。
```

控制“这个 span 被词典系统标出”产生的先验。

## PL-shuffled

保留 `txl`，但从另一个长度和机制匹配的条目中抽取无关 gloss。

控制字段数量、解释长度和语言流畅度。

## CL-counterfactual

保留形式和合理性，但给出错误解码：

```text
表达：txl
常见含义：一个语义上合理但错误的解释。
```

它不是 placebo，应放入阶段三来源冲突或 gold-vs-counterfactual margin 实验。

如果 PL 不包含同一个 headword，那么 CL–PL 的差异中仍混有“该词是否被资源命中”的 membership effect，无法干净解释为语义利用。

---

# 八、WP3 自身应有独立验收指标

不要用后续 `M_LD/M_drop` 的 hate extraction F1 来筛选 WP3 条目，否则会产生循环选择。WP3 应先根据词汇学和证据质量冻结。

以下阈值可以作为初始 go/no-go 标准，但应在第一轮 pilot 后校准，并非通用文献标准。

| 维度     | 建议指标                       | 建议初始门槛       |
| ------ | -------------------------- | ------------ |
| 候选召回   | sealed mention recall      | ≥ 0.90       |
| 边界质量   | exact boundary F1          | ≥ 0.85       |
| 分类型召回  | form/slang/MWE/polysemy 分层 | 每类单独报告       |
| 核心释义   | 人工 semantic correctness    | A-core ≥ 95% |
| 证据覆盖   | 发布条目是否有可追溯支持               | 100%         |
| 直接泄漏   | ontology/verdict/答案字段      | 0            |
| offset | raw exact-substring 有效性    | 100%         |
| 来源合规   | headword 是否 fit-attested   | 100%         |
| 审核     | A/B 是否人工通过                 | 100%         |
| 重放性    | 配置、模型、prompt、hash 是否齐全     | 100%         |

还应增加两个专门审计。

## 1. 术语理解效用测试

给评估者或冻结模型两个版本：

```text
原句
原句 + 词条解释
```

要求其仅完成：

* 将候选 span 改写为标准直白表达；
* 选择正确释义；
* 判断两个上下文是否使用同一 sense。

这里不提供 hate/group 标签。若解释不能提高词义恢复，就没有必要进入 L。

## 2. Membership prior 审计

资源冻结后，在 calibration 数据上测试：

[
\operatorname{AUC}(Y;\text{lexicon-hit})
]

以及：

[
I(Y;\text{lexicon-hit})
]

比较：

* 仅知道是否被词典命中；
* 同 headword placebo；
* 真正 semantic gloss；
* 随机未匹配词条。

标签只用于此后验审计，不应用于决定哪些条目被收录。

---

# 九、建议重写后的 WP3 工作包

## WP3.0：冻结任务定义和标注手册

交付：

* category-free 的定义；
* inclusion/exclusion 规则；
* span 边界规则；
* 两轴机制 taxonomy；
* A/B/C/R 状态；
* 至少 50–100 个正反最小对照样例。

已有 240 条数据用于 guideline development，不再作为最终无偏评测。

## WP3.1：建立 sealed mention benchmark

* 从 fit 新抽取 240–600 条；
* 双人盲标；
* 不显示任务标签；
* 保留原始分歧与 adjudication；
* 冻结后不用于修改生成器 prompt。

## WP3.2：实现首批三个高收益生成器

先只实现：

1. surface rewrite + alignment；
2. direct terminology mention extraction；
3. phonetic/orthographic/mixed-script rules。

不要先实现复杂 phrase mining。先在 sealed benchmark 上确认互补召回。

## WP3.3：多生成器并集与人工审核

* 对全部 5,165 条 fit 记录运行；
* 候选按弱监督分数和模型分歧排序；
* 人工执行 accept / trim / expand / split / reject；
* 不要求固定候选数或固定词条数。

## WP3.4：entry 聚合与 sense 聚类

* mention 合并为 form family；
* occurrence embedding 聚类；
* 分离多义；
* 接受条目回扫全部 fit；
* 迭代到新 sense 收益趋近于零。

建议收敛标准同时使用：

* 连续两轮每 100 个新审核候选中新增 A/B sense 少于预设比例；
* sealed mention recall 不再显著提高。

## WP3.5：证据获取与定义生成

* LLM 参数知识生成竞争假设；
* 自适应 Web 检索；
* claim–evidence mapping；
* grounded synthesis；
* independent falsification；
* 人工发布审核。

## WP3.6：构建 controls 和 renderers

正式交付：

```text
lexicon_ref.json
lexicon_mentions.jsonl
lexicon_evidence.jsonl
lexicon_rejects.jsonl
renderer_L_form
renderer_L_sem
renderer_L_prag
placebo_membership
placebo_shuffled
counterfactual_glosses
wp3_audit_report
```

## WP3.7：内容寻址、冻结和封存

冻结：

* 资源 hash；
* fit record lineage；
* 搜索结果快照；
* 模型版本；
* prompt；
* reviewer decision；
* renderer 版本；
* tokenizer token cost。

随后才进入 WP4/WP5。`M_LD` 和 `M_drop` 仍然只是冻结资源的下游消费者，不参与 WP3 过门。

---

# 十、对整体研究路线的影响

采用这一方案后，WP3 不再只是“换一个质量更好的词典”，而会形成一个更强的研究对象：

> **外部知识在不显式提供任务答案时，能否通过形式解码、核心语义和语用信息，因果性地改变结构化 hate-speech extraction？**

它还会直接为后续阶段提供接口：

* 阶段一：`L_sem` 是否产生独立于 membership placebo 的效应；
* 阶段二：形式、语义、语用分别贡献多少；
* 阶段三：正确解释与反事实解释如何和 demonstrations 仲裁；
* 阶段四：效应可归因到哪个 entry/sense；
* 阶段五：可分别 patch `canonical_form`、`gloss_core`、`pragmatic_note` 对应的 token 段；
* 阶段六：CUCS 可以在 entry/sense/field 粒度估计 causal utility 和 token cost。

最重要的战略取舍是：**阶段一只使用保守的 A-core，不追求词条数量。** B-contextual 可以保留并用于后续探索，但不应让高度歧义、弱证据条目进入首轮 confirmatory CL。这样即使最终 A-core 规模较小，因果结论也会比一个自动生成的千条词典可信得多。

[1]: https://arxiv.org/pdf/2601.19932 "\"Newspaper Eat\" Means \"Not Tasty\": A Taxonomy and Benchmark for Coded Language in Real-World Chinese Online Reviews"
[2]: https://aclanthology.org/2020.acl-main.139.pdf "Named Entity Recognition without Labelled Data: A Weak Supervision Approach"
[3]: https://aclanthology.org/2025.findings-acl.742.pdf?utm_source=chatgpt.com "Exploring Multimodal Challenges in Toxic Chinese Detection"
[4]: https://aclanthology.org/2026.c3nlp-1.5/ "LLM-Adapted Colombian Spanish Lexicography: Proficiency Control, Hallucination, and Cultural Distortion - ACL Anthology"
