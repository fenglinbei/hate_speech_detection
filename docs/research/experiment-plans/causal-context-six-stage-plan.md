# 声明式知识与检索示例的因果错误诊断：六阶段实验计划

## 1. 研究目标

本文不再把词典注入和 SRAG 仅视为两个性能增强模块，而是把它们视为模型在推理时获得的两类外部证据：

- 词典提供声明式知识（declarative knowledge）：词义、群体指向和类别信息；
- 检索示例提供案例知识（episodic/procedural evidence）：相似语境、target–argument 关系、输出结构和标签映射。

核心研究问题是：

> 模型何时真正利用外部证据完成语义理解与结构类比，何时退化为词典类别、示例标签、输出模板或位置捷径；能否定位导致每个字段错误的具体证据与内部通路，并据此选择有益上下文、抑制有害上下文？

实验将形成一条连续证据链：

1. 证明词典与示例对同一冻结模型具有推理时因果效应；
2. 将总效应分解为语义、结构、标签、格式和位置路径；
3. 研究两类证据冲突时模型如何仲裁；
4. 把错误归因到具体词典条目、示例和输出字段；
5. 在模型内部定位语义利用与捷径复制通路；
6. 用字段级因果效用训练选择器，完成针对性修复。

## 2. 研究对象与统一记号

设：

- $x$：待判断文本；
- $L=\{l_1,\ldots,l_m\}$：检索到的词典条目；
- $D=\{d_1,\ldots,d_k\}$：检索到的示例集合；
- $C=(L,D)$：完整外部上下文；
- $y=(y_t,y_a,y_g,y_h)$：结构化输出，对应 target、argument、targeted group 和 hateful；
- $f\in\{t,a,g,h\}$：一个输出字段。

行为层的主要因果量采用字段级 gold-vs-counterfactual margin：

\[
m_f(C)=
\overline{\log p(y_f^{\mathrm{gold}}\mid x,C)}
-
\overline{\log p(y_f^{\mathrm{cf}}\mid x,C)}.
\]

其中 target 和 argument 使用按 token 数归一化的序列 log-probability，group 和 hate 使用相应标签片段的 log-probability。反事实输出只修改被研究字段，其余字段和解码前缀保持不变。

词典和示例的推理时总效应分别定义为：

\[
TE_L^{(f)}=m_f(L,\varnothing)-m_f(\varnothing,\varnothing),
\]

\[
TE_D^{(f)}=m_f(\varnothing,D)-m_f(\varnothing,\varnothing).
\]

两类证据的交互效应定义为：

\[
I_{L,D}^{(f)}=
m_f(L,D)-m_f(L,\varnothing)-m_f(\varnothing,D)+m_f(\varnothing,\varnothing).
\]

最终自由生成指标用于判断任务效果，teacher-forcing margin 用于识别预测改变发生在哪个字段、哪一个生成位置和哪条因果路径。

## 3. 实验材料

### 3.1 完整测试集

完整测试集用于：

- 主任务性能；
- 主要上下文干预；
- 词典条目与示例的 leave-one-out 归因；
- causal-utility selection 的最终比较。

### 3.2 因果困难子集

从测试集错误和不稳定样本中构造约 300–500 条人工复核子集，分层覆盖：

- 隐晦词、多义词和词典未登录变体；
- 引用、转述、否定、反驳和 counterspeech；
- 群体提及但不构成攻击；
- 个人攻击与群体攻击的边界；
- 同一关键词在 hateful/non-hate 语境中的最小对照；
- target–argument 边界、绑定和多三元组结构；
- 词典与示例给出冲突证据的样本；
- 长文本、示例截断和位置敏感样本。

该子集用于完整反事实矩阵、证据冲突实验和人工案例分析，不替代完整测试集结果。

### 3.3 机制子集

从行为结果中选择四类具有清晰预测差异的样本：

1. semantic success：正确语义证据使错误预测恢复；
2. lexicon shortcut：保持定义不变、仅交换类别即可翻转预测；
3. demonstration shortcut：保持示例输入不变、仅交换输出标签即可翻转预测；
4. source conflict：词典和示例单独作用不同，组合后产生明确仲裁或冲突。

每类先选择约 30–50 个样本，用于 activation patching；根据效应一致性决定是否扩展。

## 4. 实验前置

前置工作只服务于反事实可比性，不作为论文主贡献：

- 统一四字段数据、prompt、解析和评测，保证 group 与 hate 可独立干预；
- 在冻结 train split 后先建立独立、不可变的 `train-partition`：仅做 `CRLF→LF` 后按内容
  SHA-256 聚类，同内容簇以最小 canonical 数字 query ID 为代表；再对代表 ID 使用
  `salt=stage1-train-calibration-v1` 的 `sha256-query-id-v1`，按
  `bucket = hash mod 10000`、`bucket < 1000` 分入名义 10% calibration，其余分入 fit；
- 采用 full-information-isolated 协议：训练 loss、词典构建和 demo 候选池都只消费 fit，
  calibration 只用于 checkpoint selection，且它的 demo 顺序与 L/D dropout presentation
  固定为第 1 epoch 的确定性呈现，不随训练 epoch 改变；当前 5781 条 ID frame 的确定性
  分配为 5165 fit / 616 calibration；
- 词典和示例候选池因此是严格 fit-only；冻结每个查询的候选与选择 manifest，并让所有
  下游 artifact 精确绑定同一个 `train-partition` dependency；

当前上述 partition **只冻结了协议**，尚未发布正式 `train_partition_ref.json`：必须先
完成 Stage 1 P0 的 20 条 group–hate 二轮人审与 4 条 field-type 人审，签署并验证
`data_ref.json` 后，才能从该 data artifact 构建、重放并发布正式 partition。5165/616
不能被当作已经存在的 artifact ID、locator 或训练执行记录。
- 为每条词典和示例保存 ID、相似度、来源类别、位置、token 数和截断状态；
- 修正 uniform/global random 等对照条件，使实验名称与实际采样过程一致；
- 对跨类别重复示例进行全局去重，将集合选择与最终排序分离。

除专门研究长度、位置和截断的实验外，反事实条件固定模型、查询、候选 ID、上下文数量、顺序和近似 token budget。

---

## 阶段一：确定两类外部证据的推理时因果效应

### 研究问题

1. 词典与 SRAG 的收益是否来自模型对当前上下文的在线利用？
2. 还是主要来自训练期 prompt 形式、长度分布或上下文依赖？
3. 两类证据是互补、冗余，还是在部分字段上互相干扰？

### 核心假设

- H1.1：在同一冻结模型上，正确词典和正确示例都能提高至少一部分输出字段的 gold margin；
- H1.2：词典与示例的效应具有字段差异，不能被一个总体 F1 充分描述；
- H1.3：$I_{L,D}^{(f)}$ 在不同错误类型上同时存在正交互和负交互，负交互是后续证据冲突与上下文选择的研究入口。

### 实验一：同模型四上下文评估

冻结当前完整模型 $M_{LD}$，在相同测试样本上运行：

| 条件 | 词典 | 示例 | 目的 |
|---|---:|---:|---|
| C0 | 无 | 无 | 无外部上下文基线 |
| CL | 有 | 无 | 词典总效应 |
| CD | 无 | 有 | 示例总效应 |
| CLD | 有 | 有 | 完整方法与交互效应 |

主要比较为 $CL-C0$、$CD-C0$ 和 $CLD-CL-CD+C0$。同时报告自由生成任务指标和字段级 margin。

为区分真实内容效应与 prompt 形态效应，增加一个等长中性上下文条件：保留段落和输出结构，但用与任务无关的自然文本替换内容。该条件只作为长度/格式参照。

### 实验二：训练上下文依赖

主计划使用两个 checkpoint：

- $M_{LD}$：始终使用词典与示例训练；
- $M_{drop}$：训练时独立随机丢弃词典和示例，并随机化示例顺序。

两个模型都在 C0、CL、CD、CLD 下评估。若资源允许，再补充 $M_0$、$M_L$、$M_D$，形成完整的 train-context × test-context 矩阵。

### 主要指标

- Hard/Soft/Avg triple F1；
- target、argument、group、hate 的字段指标；
- 四字段 gold-vs-counterfactual margin；
- 加入上下文后由错变对、由对变错的双向 flip rate；
- 格式有效率和 tuple 数正确率；
- 词典、示例主效应与交互效应。

### 阶段结论判据

- 若同一冻结模型在正确上下文下产生稳定字段增益，则支持推理时利用；
- 若独立证据有益但组合后出现负交互，则支持来源冲突或注意力竞争；
- 若只有对应训练上下文的模型受益，则主要效应包含训练分布依赖，后续机制结论应限定在该模型行为上。

### 预期论文产出

- 一张 train/test context 矩阵表；
- 一张词典、示例对四字段的主效应与交互效应图；
- 对“外部上下文在线利用”的第一个因果结论。

---

## 阶段二：分解语义利用、结构类比与上下文捷径

### 研究问题

1. 词典的收益来自定义语义，还是类别标签？
2. 示例的收益来自输入语义、target–argument 结构、标签映射，还是输出格式？
3. 哪些组件导致正确预测，哪些组件导致 shortcut error？

### 核心假设

- H2.1：若模型利用词义，definition-only 应保留主要收益，语义等价改写应保持效果，而定义反转应显著改变预测；
- H2.2：若模型进行示例类比，正确 input-output mapping 应优于仅标签分布或仅输出模板；
- H2.3：标签交换主要影响 group/hate，跨度与结构交换主要影响 target/argument；若跨字段传播超出该模式，则说明模型形成了耦合捷径。

### 实验一：词典内部信息干预

固定词典条目 ID 和顺序，构造：

| 条件 | 操作 | 主要诊断 |
|---|---|---|
| L-Full | 关键词、定义、类别均正确 | 原始词典效果 |
| L-Definition | 保留定义，隐藏类别 | 纯语义利用 |
| L-Category | 保留关键词和类别，用中性占位替换定义 | 类别先验/标签捷径 |
| L-Paraphrase | 定义语义不变、词面改写 | 语义不变性 |
| L-CategorySwap | 定义不变、类别换为反事实类别 | 类别复制敏感性 |
| L-DefinitionSwap | 类别不变、定义替换为另一含义 | 是否实际读取定义 |
| L-Irrelevant | 等长无关词典内容 | 注意力与长度干扰 |

类别交换优先选择语义上可区分且格式相同的类别；定义交换保持语言流畅，避免用明显乱码制造额外分布偏移。

### 实验二：示例内部信息干预

固定示例 ID、数量和顺序，构造：

| 条件 | 操作 | 主要诊断 |
|---|---|---|
| D-Full | 输入和完整四字段输出均正确 | 原始 SRAG 效果 |
| D-Input | 保留示例输入，遮蔽具体输出 | 输入语义提示 |
| D-Schema | 输入与字段值替换为占位符，仅保留结构 | 格式和 tuple 模板 |
| D-Span | 保留输入、target 和 argument，遮蔽 group/hate | 关系与跨度类比 |
| D-Label | 使用等长无关输入，保留 group/hate | 标签先验 |
| D-WithinLabelShuffle | 同标签示例间交换输出 | 保持标签分布，破坏实例对应 |
| D-CrossLabelShuffle | 跨标签交换输出 | input-label mapping |
| D-Paraphrase | 输入语义不变、词面改写 | 语义类比与词面匹配 |
| D-OppositeStance | 保留关键词，反转引用、否定或立场 | 语用理解与 lexical lure |
| D-Order | 对同一示例集合做循环换位 | 位置和近因效应 |

完整测试集运行主条件 D-Full、D-Input、D-Schema、D-Span、D-Label、D-CrossLabelShuffle；其余条件在因果困难子集上运行。

### 字段传播矩阵

分别腐化上下文中的 definition、category、demo target、demo argument、demo group 和 demo hate，计算：

\[
C_{r\rightarrow f}
=m_f(C_{\mathrm{clean}})-m_f(C_{r\text{-corrupt}}),
\]

得到“上下文组件 $r$ → 预测字段 $f$”的传播矩阵。

### 核心诊断指标

- Definition Semantic Effect：正确定义与错误定义之间的字段 margin 差；
- Definition Paraphrase Stability：定义改写前后的预测一致率；
- Lexicon Label-Follow Rate：交换类别后输出跟随反事实类别的比例；
- Demo Label-Follow Rate：交换示例标签后查询输出跟随示例标签的比例；
- Mapping Sensitivity：正确 mapping 与 shuffled mapping 的差值；
- Structural Transfer Effect：D-Span 相对 D-Schema 对 target/argument 的增益；
- Order Variance：同一示例集合换序后的字段 margin 方差和预测 flip rate。

### 阶段结论判据

- “语义利用”需要同时满足：对语义腐化敏感、对语义等价改写稳定、且效果不能由 category/label-only 条件解释；
- “标签捷径”由保持输入语义不变的标签交换导致系统性字段翻转来识别；
- “结构类比”由 span mapping 对 target/argument 的特异性贡献识别；
- “格式模仿”由 schema-only 主要改善格式、tuple 数而不改善语义字段来识别。

### 预期论文产出

- 一张上下文组件到输出字段的因果传播热图；
- 词典语义利用与 category shortcut 的定量分解；
- 示例语义类比、结构迁移、label copying 和 schema copying 的定量分解。

---

## 阶段三：研究声明式知识与案例证据的冲突仲裁

### 研究问题

1. 词典和示例冲突时，模型在不同输出字段上相信哪一种来源？
2. 仲裁依据是查询语义、证据可靠性，还是标签频率与位置？
3. 两个单独有益的上下文是否会因组合而导致错误？

### 核心假设

- H3.1：证据支配关系具有字段特异性，target/argument 与 group/hate 不一定跟随同一来源；
- H3.2：模型在语义相反但标签一致、语义一致但标签相反时表现不同，可据此区分语义仲裁和标签仲裁；
- H3.3：部分冲突错误表现为来源交互，而非任何单条证据的独立错误。

### 主实验：3×3 来源状态矩阵

构造：

\[
L\in\{\text{Absent},\text{Correct},\text{Counterfactual}\},
\]

\[
D\in\{\text{Absent},\text{Correct},\text{Counterfactual}\}.
\]

形成九个条件：

| 词典 \ 示例 | Absent | Correct | Counterfactual |
|---|---|---|---|
| Absent | 无上下文 | 仅正确示例 | 仅错误示例 |
| Correct | 仅正确词典 | 两类证据一致正确 | 正确词典 vs 错误示例 |
| Counterfactual | 仅错误词典 | 错误词典 vs 正确示例 | 两类错误证据 |

Counterfactual 分两轮构造：

1. label conflict：保持定义/示例输入语义，交换 category 或输出标签；
2. semantic conflict：保持标签表面不变，改变定义含义、关系或 stance。

对同一冲突集合做循环换位，使两类证据和不同示例轮流处于近端位置。

### 仲裁指标

在 $L\neq D$ 的条件下，按字段统计：

- Lexicon Dominance：输出跟随词典证据的比例；
- Demo Dominance：输出跟随示例证据的比例；
- Query Recovery：同时拒绝错误证据并根据查询恢复 gold 的比例；
- Unsupported Output：不跟随任一证据且错误的比例；
- Conflict Degradation：冲突条件相对 C0 和正确单来源条件的下降；
- Position-Conditioned Dominance：来源位置改变前后的 dominance 差；
- $I_{L,D}^{(f)}$：两类证据在每个字段上的交互效应。

### 重点分析切片

- 词典 exact match 与 semantic match；
- demo 高相似度与低相似度；
- hate 与 counterspeech；
- 单三元组与多三元组；
- target 正确但 argument 错误；
- group 正确但 hate 错误；
- 模型初始置信度高与低的样本。

### 阶段结论判据

- 若冲突输出主要由证据标签决定且对语义反转不敏感，则属于标签仲裁捷径；
- 若模型能根据查询拒绝与语境不一致的词典或示例，则支持可靠性敏感的语义仲裁；
- 若单来源均正确而组合后错误，则属于 source-interaction error，并成为阶段六路由器的直接训练目标。

### 预期论文产出

- 一张 Lexicon × Demo 冲突相图；
- 一张四字段来源支配图；
- 对模型如何仲裁声明式知识与案例证据的核心科学结论。

---

## 阶段四：实例级因果归因与错误分解

### 研究问题

1. 能否定位导致某个字段正确或错误的具体词典条目和示例？
2. embedding similarity 是否等价于真实 causal utility？
3. 错误来自检索、选择、利用、证据冲突还是生成解码？

### 上下文单元的字段级因果效用

对上下文单元 $c_i\in L\cup D$，定义 leave-one-out 效用：

\[
U_i^{(f)}=m_f(C)-m_f(C\setminus c_i).
\]

- $U_i^{(f)}>0$：该上下文支持字段 $f$；
- $U_i^{(f)}<0$：该上下文伤害字段 $f$；
- 不同字段符号不同：该上下文存在跨字段权衡。

同时计算 add-back 效用：

\[
A_i^{(f)}=m_f(C_{base}\cup\{c_i\})-m_f(C_{base}),
\]

用于区分上下文本身无效与其作用被其他上下文覆盖。对于最有益、最有害和冲突最大的若干单元，在困难子集上进一步估计二阶交互：

\[
U_{ij}^{(f)}=
m_f(C)-m_f(C\setminus c_i)-m_f(C\setminus c_j)
+m_f(C\setminus\{c_i,c_j\}).
\]

### 归因忠实性验证

按预测效用排序，进行：

- 删除 top-harmful 上下文，观察错误修复率；
- 删除 top-helpful 上下文，观察正确预测破坏率；
- 向无上下文或中性上下文中 add-back top-helpful，观察恢复率；
- 与随机删除、最低相似度删除和最高相似度删除比较；
- 计算预测效用排序与真实删除效应之间的相关性。

### Oracle 实验与错误分解

在冻结候选池内枚举或近似搜索最有益的词典/示例集合，构造 candidate-pool oracle；在困难子集上人工提供最匹配定义和示例，构造 semantic oracle。

据此定义：

- Retrieval miss：候选池未包含可修复该错误的上下文；
- Selection error：候选池包含有益上下文，但当前策略未选择；
- Harmful-context error：删除某条已选上下文后错误恢复；
- Utilisation failure：正确 oracle 已提供，但模型仍未利用；
- Label-copy shortcut：仅修改上下文标签即可控制预测字段；
- Structural imitation：预测跨度或 tuple 结构跟随示例而偏离查询；
- Source-conflict error：上下文单独有益，组合产生负二阶效应；
- Position/truncation error：集合不变，仅换序或恢复被截断上下文即可修复；
- Decoding error：teacher-forcing margin 指向 gold，但自由生成结果错误。

### 关键科学分析

1. BGE similarity 与 $U_i^{(f)}$ 的 Spearman 相关；
2. 高相似负效用示例的比例及主要错误类型；
3. 同一上下文在不同字段上的 utility sign disagreement；
4. demo 数增加时，新增信息量、冗余和负效用比例的变化；
5. 词典效用与示例效用之间的互补、覆盖和冲突关系。

### 阶段结论判据

- 删除/加回实验显著优于随机操作，才认为实例归因具有因果忠实性；
- 相似度与 causal utility 的系统性偏离，构成改进现有检索目标的直接证据；
- retrieval、selection 和 utilisation 三类错误必须通过 oracle 条件分开报告。

### 预期论文产出

- similarity–utility 散点图；
- helpful/harmful utility 分布及字段差异；
- 错误来源分解图；
- 若干“高相似但有害”“单独有益但组合有害”的代表案例。

---

## 阶段五：定位语义利用与捷径复制的内部因果通路

### 研究问题

1. 词典定义语义与词典类别是否通过不同内部表征影响输出？
2. 示例输入/跨度与示例标签是否通过不同层或注意力头影响查询？
3. 行为层发现的 semantic path、label-copy path 和 source-conflict 是否能在模型内部得到因果验证？

### Clean–corrupt 配对

从阶段二至四选择效应明确的 prompt 对：

- 正确定义 vs 定义语义反转；
- 正确类别 vs category swap；
- 正确 demo mapping vs output shuffle；
- 正确 span vs boundary corruption；
- 正确 stance vs same-term opposite stance；
- 正确词典/示例组合 vs 来源冲突组合。

每一对只改变一个被研究变量，并以对应字段的 gold-vs-counterfactual margin 作为 patching 输出。

### 第一步：层级与 token 区域定位

先对 residual stream 做粗粒度 activation patching，token 区域包括：

- lexicon keyword；
- lexicon definition；
- lexicon category；
- demo input；
- demo target/argument；
- demo group/hate；
- schema/separator；
- query target、predicate 和 stance tokens；
- query answer boundary。

对每一层和 token 区域计算 patch 后的字段 margin 恢复比例：

\[
R_{r,l}^{(f)}=
\frac{m_f(\mathrm{patched}_{r,l})-m_f(\mathrm{corrupt})}
{m_f(\mathrm{clean})-m_f(\mathrm{corrupt})}.
\]

### 第二步：组件定位

在恢复效应集中的层进一步定位 attention heads 和必要时的 MLP：

- patch 单个或小组 attention-head output；
- knockout 候选组件，检验 clean 行为是否消失；
- 比较 label corruption 与 semantic corruption 的关键组件重叠度；
- 检验关键组件对非目标字段和普通语言任务的影响，判断其作用是否具有字段特异性。

### 候选因果通路

重点检验：

1. definition semantic path：定义 token → 查询关系/stance 表征 → argument/group；
2. lexicon label path：category token → 标签复制组件 → group/hate；
3. demonstration analogy path：demo input/span → 查询 binding 表征 → target/argument；
4. demonstration label path：demo output label → 标签复制组件 → group/hate；
5. schema path：separator/tuple pattern → 格式和 tuple 数；
6. conflict path：词典与示例表征在查询输出前汇合并发生来源竞争。

若 residual/head patching 已形成稳定、可重复的字段特异性证据，不把完整 circuit reconstruction 设为必要条件。DAS 或 function-vector 实验只用于进一步检验“语义、标签、schema”是否可分离编码。

### 机制结论判据

- patch 定义表征能够恢复语义字段，而 patch 类别表征主要恢复标签字段；
- patch demo span 能恢复 target/argument，patch demo label 主要控制 group/hate；
- 对关键组件 knockout 后相应行为效应消失，并且其他路径相对保留；
- 单独的 attention visualization 或 probe accuracy 不作为因果机制结论。

### 预期论文产出

- token region × layer × output field 的 patching 热图；
- 少量关键 attention heads/MLP 的必要性与充分性实验；
- 一张声明式语义、案例类比和标签复制通路的机制示意图。

---

## 阶段六：Causal-Utility Context Selection 与针对性修复

### 研究问题

1. 能否在未知测试样本上预测每条词典和示例对各字段的因果效用？
2. 能否选择更小但更有益的上下文集合，并主动拒绝有害上下文？
3. 修复能否同时提高任务表现、降低 shortcut 敏感性并减少上下文开销？

### 方法概述

方法暂称 **Causal-Utility Context Selection（CUCS）**，包括候选召回、效用预测、集合选择和来源路由四步。

#### 1. 候选召回

- BGE 分别召回 $N_L$ 个词典候选和 $N_D$ 个示例候选；
- 召回阶段追求覆盖，不直接把 embedding similarity 当作最终价值；
- 候选只来自训练数据与训练词典资源。

#### 2. 字段级效用预测

使用阶段四在训练/验证样本上产生的 LOO/add-back 标签，学习：

\[
\hat{U}(x,c_i)=
[\hat U_i^{(t)},\hat U_i^{(a)},\hat U_i^{(g)},\hat U_i^{(h)}].
\]

同时预测：

- harmful probability；
- label-copy risk；
- order instability；
- 与已选上下文的冗余或冲突。

效用模型可先采用 query–context cross-encoder；若字段回归不稳定，改为 helpful/neutral/harmful 分类与 pairwise ranking 的多任务目标。

#### 3. 集合选择

在 token budget 下联合选择 $L'\subseteq L$、$D'\subseteq D$：

\[
\max_{L',D'}
\sum_{c_i\in L'\cup D'}\sum_f w_f\hat U_i^{(f)}
+\lambda_{cov}\operatorname{Coverage}
+\lambda_{con}\operatorname{Contrast}
-\lambda_{red}\operatorname{Redundancy}
-\lambda_{risk}\operatorname{ShortcutRisk}
-\lambda_{conf}\operatorname{Conflict},
\]

满足：

\[
\operatorname{Tokens}(L',D')\le B.
\]

选择器允许动态 $k$，包括不选择任何 demo 或词典。对否定、引用、反驳和 counterspeech 等高风险查询，contrast 项鼓励同时选择：

- 一个语义/结构近邻；
- 一个同词但 stance 相反的 counter-example。

#### 4. 来源路由

根据预测效用与冲突风险，在四种上下文模式中选择：

| 路由 | 使用条件 |
|---|---|
| No context | 两类证据的预测效用均不为正 |
| Lexicon only | 声明式知识有益，示例风险较高 |
| Demo only | 结构类比有益，词典不相关或冲突 |
| Lexicon + Demo | 两类证据互补且冲突风险低 |

### 训练与推理策略

- selector 的监督只使用 train/validation 上的因果效用；
- 主因果比较先固定生成模型，只替换选择策略；
- 端到端版本再使用 CUCS 构建训练上下文，并加入独立 lexicon/demo dropout 与示例顺序随机化；
- 推理时按 utility-per-token 选择，截断不再简单删除列表尾部类别。

### 比较方法

- No context；
- Lexicon only / SRAG only；
- BGE global top-k；
- corrected stratified BGE；
- random 与 stratified random；
- MMR/DPP；
- similarity + coverage；
- CUCS；
- candidate-pool oracle。

所有选择方法首先在同一冻结生成模型上比较，以确保性能差来自上下文选择，而非不同 checkpoint。

### 方法消融

- 去掉 causal utility，只使用 similarity；
- 去掉字段分解，使用单一全局 utility；
- 去掉 harmful/shortcut-risk penalty；
- 去掉 contrastive selection；
- 去掉 Lexicon–Demo conflict penalty；
- 固定 k vs 动态 k；
- SRAG-only utility selector vs 词典与示例联合 selector；
- 不允许 no-context 回退；
- 不使用 context dropout/order randomization。

### 成功判据

方法结论同时要求：

1. 提高完整测试集 Hard/Avg F1 和至少两个关键字段指标；
2. 在因果困难子集上降低 Label-Follow Rate、Order Variance 和 harmful-context rate；
3. 删除预测 harmful 上下文后错误修复率提高，证明 selector 使用了与因果归因一致的信号；
4. 平均上下文 token 数不高于原 k=10 SRAG；
5. 在另一模型规模或另一数据切片上保持主要方向；
6. 性能提升不能仅由更多上下文或更长 prompt 解释。

若 F1 提高但标签交换敏感性、位置敏感性或有害上下文比例同时上升，则只视为性能改进，不视为本文主张的 shortcut 修复。

### 预期论文产出

- 主方法结果表与字段级结果；
- 任务性能—shortcut risk—上下文成本三维比较；
- utility predictor 的排序忠实性；
- CUCS 对 retrieval、selection、conflict 三类错误的分项修复结果。

---

## 5. 跨阶段评测与统计方案

### 5.1 主任务指标

- Triple Hard/Soft/Avg F1；
- target、argument 的跨度/语义指标；
- targeted group 与 hateful 的 Precision、Recall、F1；
- 格式有效率、tuple 数和多三元组完整率。

### 5.2 因果诊断指标

- 字段 gold-vs-counterfactual margin；
- correct-to-wrong 与 wrong-to-correct flip rate；
- Lexicon/Demo Label-Follow Rate；
- Definition/Paraphrase Semantic Effect；
- Mapping Sensitivity；
- Order Variance；
- Lexicon/Demo Dominance 与 Query Recovery；
- helpful/harmful context rate；
- attribution deletion/add-back fidelity；
- retrieval、selection、utilisation 和 decoding error 占比。

### 5.3 统计原则

- 因果干预使用同一 checkpoint、同一样本的配对比较；
- teacher-forcing 与主行为干预使用确定性推理；
- 主性能差异报告 paired bootstrap 95% confidence interval；
- 二元预测翻转使用配对检验并报告效应量；
- 顺序实验对同一集合做循环换位或 Latin-square；
- 最终方法至少报告三个独立训练种子，探索性机制实验优先报告样本级重复一致性；
- 预先指定主比较，其他大量切片作为机制解释而非独立性能结论。

## 6. 六阶段之间的依赖与执行顺序

| 顺序 | 阶段 | 进入下一阶段所需结果 |
|---:|---|---|
| 1 | 推理时来源效应 | 至少一种外部证据对冻结模型存在稳定字段效应 |
| 2 | 路径分解 | 找到可重复的 semantic、structure 或 shortcut 行为对 |
| 3 | 来源冲突 | 建立字段级 dominance 与交互效应 |
| 4 | 实例归因 | LOO/add-back 归因通过删除与恢复验证 |
| 5 | 内部机制 | 至少一条语义路径和一条捷径路径获得 patching 支持 |
| 6 | 因果效用修复 | 同时改善任务指标、shortcut 指标和上下文成本 |

阶段二和阶段四是整个计划的行为科学核心；阶段三把词典与 SRAG 统一为同一研究问题；阶段五提供机制深度；阶段六将解释结果转化为可验证的方法贡献。

## 7. 论文主张与实验对应关系

| 论文主张 | 主要证据 |
|---|---|
| 词典和示例具有推理时因果效应 | 阶段一同 checkpoint 四上下文比较 |
| 词典可通过定义语义或类别捷径影响预测 | 阶段二 definition/category 干预与字段传播矩阵 |
| SRAG 同时包含语义类比、结构迁移和标签/格式复制 | 阶段二 demo component 干预 |
| 两类证据冲突时存在字段特异性仲裁 | 阶段三 3×3 冲突矩阵与 dominance 指标 |
| 可定位导致具体错误的知识条目和示例 | 阶段四 LOO/add-back、oracle 与错误分解 |
| 语义利用与捷径复制具有可分离的内部因果通路 | 阶段五 activation patching 与 knockout |
| 因果效用能够指导选择并修复错误 | 阶段六 CUCS、消融与 shortcut robustness |

## 8. 最小完整版本与增强版本

### 最小完整版本

- 完成实验前置；
- 使用 $M_{LD}$ 和 $M_{drop}$；
- 在完整测试集完成阶段一、阶段二主条件和阶段四 LOO；
- 在人工困难子集完成阶段三；
- 在四类机制样本上完成 residual-stream patching 和少量 head ablation；
- 实现 demo-level causal-utility selector，并加入 lexicon/demo 路由；
- 在当前主模型和另一模型规模上报告结果。

### 增强版本

- 完整 train-context × test-context 矩阵；
- 词典与示例的统一字段级 selector；
- 二阶上下文交互与近似 Shapley；
- attention-head/MLP 级机制定位或 DAS；
- 第二数据集或跨域测试；
- 完整的 context-dropout、order-robust 和 contrastive training。

## 9. 预期论文叙事

整篇论文以“证据如何导致判断”为主线，而不是按两个工程模块分章：

1. 外部上下文总体有效，但收益并非随信息量单调增加；
2. 词典定义和示例语境可支持语义理解与结构绑定；
3. 同样的上下文也会通过类别、标签、模板和位置产生捷径；
4. 词典与示例冲突时，模型表现出字段特异、位置敏感的证据仲裁；
5. 实例级因果效用和内部 patching 能定位错误来源；
6. 依据这些因果信号选择、组合或拒绝上下文，可以在提高准确率的同时减少 shortcut error。

对应的总论点为：

> Knowledge-augmented structured hate-speech detection succeeds when external context causally supports semantic and relational representations, but fails when category, demonstration-label, schema, or positional signals dominate. Field-wise causal diagnosis makes these failures attributable and enables utility-aware context selection to repair them.
