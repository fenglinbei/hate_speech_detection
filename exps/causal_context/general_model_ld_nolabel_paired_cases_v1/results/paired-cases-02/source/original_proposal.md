# Qwen3-8B 去词典类别字段后的逐查询配对分析与机制候选集规划 v1

**阶段：第一步 / CPU 事后分析与案例准备**  
**文档日期：2026-09-07（Europe/Berlin）**  
**基准提交：`f80fe355263c267606c9601ca5d12aabee26da10`**  
**基准运行：`nolabel-01`**  
**状态：供实现与执行的规划；尚未运行本规划中的新增分析。**

> 本阶段的交付是可追溯的行为证据和内部干预候选集，不是内部机制结论。仅复用已封存的本轮六条件分数；不新增模型 forward、不读取 test、不修改词典、示例、Gold 或原报告。所有新分析均为已查看总体结果后的探索性扩展，不能追溯性称为原实验预注册端点。

## 1. 目标、起点与研究边界

### 1.1 本阶段要完成什么

先恢复每条查询在不同资源条件下的完整预测轨迹，回答三个问题：删除类别字段后究竟哪些查询被纠正、哪些变差；去类别词典与示例分别何时有效、组合何时有效或有害；哪些案例足够清晰，适合下一阶段做受控输入干预与 activation patching。

本阶段依次完成输入核验、全量配对统计、补充探索性区间、候选集筛选与审阅。总体统计始终使用完整 643 条 dev；为机制定位挑选的小样本集不得用于估计总体收益。

### 1.2 已知结果仅作为选题依据

基准运行使用 Qwen3-8B、643 条 dev、六条件与 hate/group 两个独立任务，共 7,716 个评分块、131,172 个候选。公开回执记录分类与原四个配对区间已经复算。大型 raw 和逐查询 analysis 留在本地，公开汇总不能还原查询之间的配对关系。[R1][R2]

| 条件 | Hate Macro-F1 | Group Micro-F1 | 两任务同时正确数 |
|---|---:|---:|---:|
| C0 | 36.26% | 61.01% | 210 |
| CLnew | 41.03% | 65.44% | 239 |
| CD | 71.32% | 70.83% | 345 |
| CLDnew | 62.36% | 72.24% | 281 |
| CLnewNoCat | 42.10% | 58.00% | 244 |
| CLDnewNoCat | 66.84% | 73.47% | 323 |

上表来自本轮 `classification.csv`，只用于核对与说明研究动机，不是本规划新计算的结果。[R3]

原四个主要差值显示：无示例时删除类别字段的 Group Micro-F1 变化为 −7.44 个百分点；有示例时 Hate Macro-F1 变化为 +4.47 个百分点。二者的逐点描述性区间不跨零。另两个差值的区间跨零。后续必须分析双向转换，不能把净变化直接当成“被救回的样本数量”。[R4]

### 1.3 本阶段不做什么

不运行新提示、模型重评分、隐藏状态提取、探针训练、QK/V 交换或 patching；不扩大到 14B/27B；不访问 test；不依据分析结果调整任务标签、决策阈值、示例排序或冻结词典。若输入缺失，记录阻塞原因，不通过重跑 GPU 补齐。

不把“预测正确”视为“模型理解了词义”，不把“只有组合正确”视为内部协同，不把“C0 正确”视为外部信息从未被使用。本阶段只生成这些解释的候选证据。

## 2. 资源与条件定义

记 **S** 为词条、义项及定义，**G** 为词典的显式类别字段，**D** 为固定示例及其答案。S 仍可能包含类别语义；D 也仍含答案标签。NoCat 只表示删除词典显式类别字段。[R5]

| 解释记号 | 实际 condition ID | 用途 |
|---|---|---|
| 0 | C0 | 无词典、无示例基线 |
| S+G | CLnew | 完整合并词典，无示例 |
| D | CD | 仅固定示例 |
| S+G+D | CLDnew | 完整合并词典与示例 |
| S | CLnewNoCat | 去类别合并词典，无示例 |
| S+D | CLDnewNoCat | 去类别合并词典与示例 |

解释记号只用于报告，数据中保留原始 condition ID 和原始顺序，不重命名旧产物。研究 S 与 D 的四条件轨迹固定为 **(C0, CLnewNoCat, CD, CLDnewNoCat)**。

两项关键限制必须随表披露。第一，Lnew 来自查询命中词条与固定示例命中词条的并集，D 关闭时仍保留示例贡献的词条，所以 S 不是独立于示例检索构造的资源。第二，删除字段也缩短输入；原输入核验记录平均减少约 36.286 tokens。因此本轮对比不能分离类别语义、长度与位置效应。[R5][R6]

## 3. 输入、版本绑定与只读核验

### 3.1 基准身份

```text
repository_commit = f80fe355263c267606c9601ca5d12aabee26da10
source_run = nolabel-01
plan_id = gmlnolabel-3f09715c5e57531565c88e61da5d6bf4087b1cbd9629a5c8579d0f1f8a0c046f
plan_sha256 = f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f
raw_scores_sha256 = 505f8da36596f1493ad4bf6b55985ab527dcfd5157a512a9722c5e1b1dac7eeb
analysis_sha256 = 5e5630736b3da89a05457fc13917c7203c87d7fe0692c439f5994c6cd55b2f77
lexicon_sha256 = 31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385
```

上述身份取自本轮 README、配置和公开报告回执。执行时核验实际文件，不能以文档抄录的哈希代替校验。[R1][R2][R7]

### 3.2 必需文件与解析方式

令 `BASE = exps/causal_context/general_model_ld_nolabel_v1`，`RUN = BASE/runs/nolabel-01`。

| 输入 | 用途与定位方式 |
|---|---|
| BASE/plan_ref.json | 解析 target_path、plan_id、plan_sha256 |
| target_path/plan.json | 固定 frame 顺序、候选目录、上下文描述符、源包路径 |
| RUN/run_manifest.json | 确认 complete、raw 与 analysis manifest 绑定 |
| RUN/dev-b1/manifest.json 和 scores.jsonl | 验证完整原始候选分数并重建预测 |
| RUN/analysis/manifest.json 和 analysis.json | 读取 per_query，并与 raw 重建结果核对 |
| target_path/contexts.dev.jsonl | 查询/任务/条件的冻结提示、trace 与长度 |
| target_path/resource_inventory.json | 本轮 union_ids、词条数与条件长度 |
| plan.package_path 下的 manifest.json、queries.dev.jsonl | 核验查询文本及既有 projection，不重新标注 |
| BASE/results/nolabel-01/ 下公开报告文件 | 核对点值、区间与来源回执 |

这些路径与连接关系由当前构建、执行和报告代码确认。原服务器绝对路径失效时，只允许显式配置路径映射并核对文件内容哈希；不得按相似文件名自动寻找替代输入。[R8][R9]

词条来源的细分信息优先读取父级 coverage plan 已封存的 `resource_inventory.json`：其中有 `lq_ids`、`ld_ids`、`intersection_ids`、`ld_only_ids`、`demo_ids` 和逐示例命中映射。本轮 inventory 只保留较简的并集元数据，不能假定上述字段直接存在于 nolabel inventory。[R10][R11]

### 3.3 核验门槛

按以下顺序核验：来源哈希与 complete 状态；643 个唯一字符串 query ID 与 plan.frame 顺序；六条件、两任务的完整笛卡尔积；hate 每块 2 候选、group 每块 32 候选；候选身份、canonical ordinal、分数有限性与 answer_sum 算术；raw 与 analysis 的预测、Gold、readout 对齐；最后重算公开分类点值和原四个区间。

整数计数要求完全一致；分类点值按绝对误差不超过 1e−12 核对。原区间按原 frame 顺序、PCG64 seed 42、10,000 次抽样重建，并使用 `atol=1e−12, rtol=1e−12` 核对。保留数组顺序和库版本，不能通过改变排序使结果“接近”。

输入缺失、哈希不符、重复/缺失 block 或预测不一致时停止该分析链。缺上下文元数据但核心配对输入完整时，可交付明确标注的行为统计；候选集标记 `context_incomplete`，不能验收为可直接用于内部干预。

**只读实现要求：** 不直接运行现有 `report_and_classification_audit.py`，因为它会写回原 results 目录；可借鉴其计算方法，在新命名空间做独立重算。也不调用旧 run/build-plan 入口。CPU 检查不应要求复现 GPU 环境或加载模型。[R12]

## 4. 数据契约与派生变量

### 4.1 现有字段如何读取

`analysis.json` 的 `per_query` 已含 query_id、lex_hit、gold，以及每个 condition/task 的 prediction 与 readouts。prediction 是含 labels、ordinal、top_score_gap、tied_top_count、within_two_epsilon 的对象，不是一个直接可比较的标签字符串。[R13]

| 派生字段 | 现有来源或计算规则 |
|---|---|
| pred_labels | conditions[c][t].prediction.labels |
| correct_hate | 单一预测标签是否等于 gold.hate |
| correct_group | 预测集合是否完整等于 gold.group，忽略列表顺序 |
| gold_margin | readouts["answer_sum/gold/best_nongold_margin"] |
| hate_logodds | readouts["answer_sum/margin/hate"] |
| group_label_logodds | readouts["answer_sum/margin/" + label] |
| gold_mass / gold_nll | readouts 中 answer_sum/gold/mass 与 nll |
| pred_group_size | 预测 group 集合基数 |
| group_error_count | 预测与 Gold 的对称差集合大小，范围 0–5 |
| core_mask | 依固定四条件顺序串接正确位，长度 4 |
| six_condition_mask | 依 plan 条件顺序串接正确位，长度 6 |

readout 的具体键和含义来自现有数值分析实现。gold_margin 是 Gold 候选相对最高分非 Gold 候选的差；group_label_logodds 是 32 个候选集合中“包含该类/不包含该类”的聚合分数差。后者不能代替整体集合 argmax 规则，gold_mass 也不是经校准的真实世界正确概率。[R14]

导出两类基本表：`query_profiles.jsonl` 每个查询一行，共 643 行；`condition_task_rows.csv` 每个查询/条件/任务一行，共 7,716 行。全部新增字段带 schema version，缺失值用 null 与 reason 表示，不用 0 冒充。

### 4.2 决策、并列与数值筛选

主预测继续使用不含 EOS 的 `answer_sum` argmax；精确并列取最小 canonical ordinal。group 不做逐标签阈值、不根据 hate 清空。正确性直接由预测决定，不能用 gold_margin > 0 代替，以免误处理并列。[R5]

继承 epsilon = 0.0013427734375，并记录 gap ≤ 2×epsilon 的近并列标记。完整 dev 统计不剔除这些查询；主要机制候选要求定义该候选类型所涉及的条件均 `tied_top_count == 1` 且 gap > 2×epsilon。边界查询另存，不因本次筛选而改写原预测。[R7][R13]

这是沿用工程容差的筛选，不是跨提示格式、跨运行或语义稳定性的证明。四种既存 score mode 可在 CPU 上重取 argmax，标记 `score_mode_sensitive`；不因此更换主口径，也不隐去敏感案例。它们在未来干预之前仍需要重复性核验。

## 5. 全量逐查询配对分析

### 5.1 转换表与逐标签变化

对两个删除对比、S+D 相对 D，以及 S 相对 0、D 相对 0、S+D 相对 S，分别统计正确→正确、正确→错误、错误→正确、错误→错误。每张表的四格之和应等于该层查询数。

hate 进一步按 Gold 为 hate/non-hate 分开，报告 FN→TP、TP→FN、FP→TN、TN→FP。group 除集合完全匹配外，对五个标签逐一输出相同转换，并报告预测基数与 group_error_count 变化。

这样可以保留“集合仍错，但减少了一个误报”之类 0→0 的改进。不能将 group exact-match 的恢复数量等同于 Micro-F1 的贡献，也不能将所有逐样本 F1 平均后当成 Micro-F1。

### 5.2 核心四条件的 16 种行为模式

两个任务分别枚举全部 16 个 core_mask，包括零计数模式，不将所有查询强行划为四类。重点解释如下，其他模式保留原始位型并描述各条件对错。

| core_mask（0/S/D/SD） | 行为描述 | 解释边界 |
|---|---|---|
| 0001 | 只有 S+D 正确 | 是组合成功候选，不证明计算协同 |
| 0101 | S 足够，D 单独不足 | 相对于当前固定 S、D |
| 0011 | D 足够，S 单独不足 | 不等于任何词典均无用 |
| 0111 | S 或 D 都能解决 | 资源可替代，不是必须两者 |
| 0000 | 四条件均未解决 | 可能缺信息，也可能未正确使用 |
| 1111 | 四条件均正确 | 可能未用，也可能冗余使用资源 |
| 0110 | S、D 单独正确，组合错误 | 组合干扰候选 |
| 其余九种 | 包括单资源被组合破坏、C0 正确后退化 | 不合并为“都不需要” |

C0 正确仅表明当前任务在该固定输入下不依赖补充资源取得正确预测，不代表机制上不读取外部资料。

### 5.3 机制候选标签：允许重叠，保留反例

以下 q 的正确性均针对表中指定任务。候选标签是行为筛选规则，不能改写成原因标签。

| 候选标签 | 判定条件 | 下一阶段要解释的问题 |
|---|---|---|
| H_rescue | hate：D 对、SGD 错、SD 对 | 删除操作消除了什么干扰？ |
| H_residual | hate：D 对、SGD 错、SD 错 | 删除后为何仍劣于仅示例？ |
| H_removal_harm | hate：SGD 对、SD 错 | 哪些情况下显式字段反而有用？ |
| G_category_support | group：SG 对、S 错 | 无示例时字段提供了什么帮助？ |
| G_category_harm | group：SG 错、S 对 | 类别字段是否也会带来群体误判？ |
| H_joint_only / G_joint_only | 对应任务 core_mask = 0001 | 是证据相加还是信息间依赖？ |
| Stable_correct / Stable_wrong | 对应任务六条件全部对 / 全部错 | 稳定对照与未解决对照 |

额外给 H_rescue 标注 Gold hate、Lq_hit 等属性，但不预设它一定是漏报恢复。全部候选保留所有标签；用于小样本配额的 primary_bucket 只是调度字段，不改变总体计数。正文必须同时给出纠正和退化数量。

### 5.4 连续读数与分层

逐查询计算：

```text
E_remove_with_D(q,t) = margin_SD(q,t) - margin_SGD(q,t)
E_remove_without_D(q,t) = margin_S(q,t) - margin_SG(q,t)
E_S_given_D(q,t) = margin_SD(q,t) - margin_D(q,t)
I_S_D(q,t) = margin_SD(q,t) - margin_S(q,t) - margin_D(q,t) + margin_0(q,t)
```

此处 margin 固定指 gold_margin；同时另报不依赖 Gold 方向的 hate_logodds 变化，用于区分“更倾向 hate”和“更接近正确标签”。报告均值、中位数、四分位数及正负比例，连续读数本阶段不新增区间。I_S_D 是指定分数尺度上的行为交互，不是内部回路证明。

固定分层为 all、Lq_hit、Lq_no_hit、Gold hate/non-hate、Gold group 大小 0/1/≥2，以及 Lq_hit × Gold hate 的交叉层。每层同时报告 n、标签构成、转换表和点指标；空层为 NA。不得从原本标签构成不同的切片差异直接推导资源需求规则。

元数据分析记录 Lq/Ld-only 词条数、词典 tokens、完整 prompt tokens、删除 tokens 及示例标签构成。词条来源取封存 inventory，长度取任务/条件对应值，不用一个查询级长度代替两个任务。长度与错误转换的关系仅作描述；不称为已控制长度的因果效应。

## 6. 新增统计分析：与原端点分离

### 6.1 固定新增四项探索性区间

令 F_H(c) 为完整样本上的 Hate Macro-F1，F_G(c) 为五类 Group Micro-F1。只对下列两个对比、两个任务新增区间：

```text
E_S_given_D_F1(t) = F_t(SD) - F_t(D)
J_remove_by_D_F1(t) = [F_t(SD) - F_t(SGD)] - [F_t(S) - F_t(SG)]
```

第一个回答删除类别后，相对仅示例还剩什么差距；第二个直接量化删除效应在有/无示例时的差，避免用“一个区间跨零、另一个不跨零”代替差值检验。J 是聚合 F1 尺度的差中差，不等同于平均逐查询 I_S_D，也不识别模型内部机制。

### 6.2 重抽样与报告规则

沿用配对查询 bootstrap：完整 643 条、10,000 次、NumPy PCG64、seed 42，所有条件、两个任务和全部端点共享同一批 query 索引。每次抽样重新汇总 TP/FP/FN，再计算 F1 及差值；不得相减已有置信区间端点，不得平均逐查询 F1。[R13]

hate 始终固定两个类别，即使某次抽样缺一类；group 始终固定五类，零分母记 0。使用 2.5% 与 97.5% percentile，`method="linear"`。输出原四端点的核对结果，以及新增四端点的探索性结果，分文件保存。

全部新增区间为逐点、描述性、未作多重比较校正；不输出确认性 p 值，不以是否跨零设置“实验成功”。分层、逐标签、去除 others 的敏感性表只给点值；不临时增加有利区间。若报告去除 others 后的四类 Micro-F1，必须与原五类指标并列，明确它不是替代主指标。

## 7. 机制候选集：目标 48 条，不强凑样本

### 7.1 总体池、定位集与留出集分开

保留完整候选池。初始详细案例集目标为 48 个不重复 query ID：32 个定位案例、16 个机制留出案例；这是资源预算而非功效计算。实际数量由满足规则的案例决定，低于目标必须如实报告，不放宽数值条件或修改类别定义来补足。

先按查询文本的 NFC 规范化、统一换行后的 SHA256 建立完全重复内容 family；仅用于划分，不修改真实 prompt。固定 `family_key` 后计算：

```text
h = SHA256("paired-cases-v1|20260907|split|" + family_key)
reserve if int(h[:16], 16) % 3 == 0 else discovery
```

同一查询所有条件、任务和重复内容 family 必须处于同一侧。先固定划分与规则，再详细阅读案例；粗粒度行为标签可以用于分层，但留出内容不用于路径定位和调参。

此处 reserve 仅指“后续未参与路径定位”，仍来自已暴露 dev，且其行为结果已用于候选资格判断。它不是新 test、未见数据或确认性泛化证据。共享词条和示例的程度需报告；本阶段不承诺词条或示例完全不重叠。

### 7.2 配额、优先级与排序

| 主配额桶，按处理顺序 | discovery 上限 | reserve 上限 |
|---|---:|---:|
| H_rescue | 4 | 2 |
| H_residual | 4 | 2 |
| H_removal_harm | 4 | 2 |
| G_category_support | 4 | 2 |
| H_joint_only | 4 | 2 |
| G_joint_only | 4 | 2 |
| Stable_correct（任一任务；记录 focus_task） | 4 | 2 |
| Stable_wrong（任一任务；记录 focus_task） | 4 | 2 |

G_category_harm 保留完整候选清单和结果表，本版不另设强制名额；后续增加配额须版本化记录。稳定对照优先轮换 hate/group，任务内再分层；若另一任务不足则报告实际构成。

每桶先应用第 4.2 节的数值筛选，再在 hate 的 Gold label × Lq_hit，或 group 的 Gold size × Lq_hit 子层中轮询取样。子层按 `SHA256(seed|bucket|subcell_key)` 排序，其中 subcell_key 是任务、Gold 层和命中状态的规范 JSON；层内查询按 `SHA256(seed|bucket|query_id)` 排序，哈希并列再按字符串 ID 排序。每轮从每个非空子层取一条，直至配额用满或全部耗尽；不按效应绝对值或“故事好讲程度”排序。

同一 query/family 已被更早桶选中，后续桶跳过；完整成员资格仍保留。每桶发现集和留出集独立取样，不跨侧补位、不跨桶补位。输出每次跳过、缺额和排除的原因，确保第三方仅凭输入与配置可得到相同名单。

### 7.3 案例卡与人工审阅

每张案例卡包含 query_id、focus_task、来源哈希、全部条件预测与 Gold、top gap、gold_margin、四/六位轨迹、所有候选标签、词条来源、定义与类别字段、示例 ID/顺序和相关片段、长度变化、数值/计分敏感标记及未知项。

审阅分两步：先只看查询与固定资源，记录词义歧义、引用/否定/立场、定义适配性、类别字段关系和示例对应性；再显示预测轨迹，记录候选解释、替代解释与下一阶段可证伪的干预。审阅时不显示未执行干预的假设性结果。

不在本阶段修改 Gold。疑似标注争议单列；原总体统计保留，候选是否暂缓进入 patching 需有理由。若只有一位审阅者，明确 single-review；多人审阅记录各自判断和分歧，不把共同脚本复算表述为独立研究者审计。含仇恨表达的完整案例默认保留本地，公开报告仅放必要、去标识化片段。

## 8. 实现工作包与安全执行接口

### 8.1 建议新增文件

以下均为拟新增路径，不是当前提交已经存在的实现。

```text
docs/research/experiment-plans/
  general-model-ld-nolabel-paired-cases-v1.md
config/stage1/
  general_model_ld_nolabel_paired_cases_v1.json
src/diagnostics/
  general_model_nolabel_paired_cases.py
scripts/stage1/
  general_model_nolabel_paired_cases.py
src/tests/
  test_general_model_nolabel_paired_cases.py
exps/causal_context/general_model_ld_nolabel_paired_cases_v1/
  runs/paired-cases-01/
```

新增模块使用标准库、NumPy 和项目既有的纯 CPU 评分逻辑；尽量避免导入模型、执行调度器和原 plan loader。原 loader 绑定冻结运行环境，不宜直接作为新 CPU 事后分析的通用加载器。[R8]

### 8.2 工作包与完成条件

| 工作包 | 输入与动作 | 完成条件 |
|---|---|---|
| A0 绑定与核验 | 只读解析 manifests、raw、analysis、plan | 哈希、完整性、原点值与区间通过 |
| A1 全量轨迹 | 生成长表、16 模式、转换与逐标签表 | 每表计数闭合，raw/analysis 预测一致 |
| A2 探索性统计 | 新四端点、既定分层与连续读数 | 共用抽样、方法记录、无端点漂移 |
| A3 候选冻结 | 标签、数值筛选、划分与确定性配额 | 名单可复现、无跨侧重复、缺额有记录 |
| A4 案例审阅与交接 | 案例卡、替代解释、未来干预对象 | 交付定位集；留出集访问状态可追踪 |

CPU 实现先冻结配置、源码 commit、环境、输入哈希和分析规则，再运行新统计。若修改筛选规则，建立 v2 并同时保留 v1 产物；不得覆盖已有结果。

下面是**待实现接口规范**，不是已经可运行的仓库命令：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python \
  scripts/stage1/general_model_nolabel_paired_cases.py \
  --config config/stage1/general_model_ld_nolabel_paired_cases_v1.json \
  --phase all
```

拟支持 `validate / analyze / select / review-export / all`。只读输入与输出目录显式分离；输出目录已存在且身份不同则拒绝写入。`CUDA_VISIBLE_DEVICES` 只是附加防护，测试还应断言没有模型加载或任何 forward 调用。

## 9. 产物清单、测试与验收

### 9.1 必交产物

| 目录/文件 | 内容 |
|---|---|
| manifest.json、config.frozen.json | 基准与执行 commit、来源哈希、版本、随机规则、运行状态 |
| audit/input_audit.json、baseline_reproduction.json | 完整性、点值/原区间核对、异常与缺失 |
| tables/query_profiles.jsonl、condition_task_rows.csv | 643 条全轨迹与 7,716 条长表 |
| tables/transitions.csv、core_masks.csv | 双向转换、全部 16 模式及分层 |
| tables/per_label_transitions.csv、strata.csv | 五类标签变化、构成与点值 |
| tables/paired_readouts.csv、exploratory_ci.csv | 连续效应与新增四项区间 |
| cases/candidate_pool.jsonl、selection_manifest.json | 全候选、标签、优先桶、确定性取样记录 |
| cases/discovery.jsonl、reserve.jsonl、boundary_cases.jsonl | 定位/留出/数值边界集合及访问状态 |
| cases/review_template.csv、cards/ | 结构化审阅字段与定位案例卡 |
| REPORT.md | 结果、反例、限制、缺额和下一阶段建议 |

逐样本与 raw 引用默认本地保存；公开版只导出汇总、配置、核验回执及必要案例。reserve 案例卡不默认向定位审阅者展开。

### 9.2 最低测试集

测试须覆盖：重复/缺失/额外 query 与 block 拒绝；哈希错误拒绝；精确并列按 ordinal、gap 等于 2×epsilon 时标记边界；16 种合成位型及重叠候选标签；group 0→0 但逐标签误差改变；空 Gold/空预测及单类 bootstrap 的零分母策略；差中差在每次抽样内重新计算；所有条件同预测时差值与区间为零；frame 顺序固定后批处理与直接索引重抽样一致；取样重复运行同名单、无 query/family 跨侧、稀有桶不强凑；原目录内容哈希运行前后不变；缺失上下文只能降级交付；无 test 读取与模型执行。

原仓库已包含配对 F1、完整矩阵和并列规则的相关测试，可复用其思想，新增测试仍需覆盖本规划特有的候选分桶、差中差和选样行为。[R15]

### 9.3 验收与停止规则

验收依赖正确性与可追溯性，不依赖效应为正、区间不跨零或候选达到目标数量。完整统计、分层分母和转换计数应相互闭合；主预测与原结果一致；原四端点不被改写；新四端点带 exploratory 标记；全部候选有规则与来源；发现/留出划分无重复；原输入不变；没有新增模型执行或 test 内容访问。

失败状态必须明确到 `input_missing`、`identity_mismatch`、`baseline_mismatch`、`context_incomplete` 或 `analysis_failed`。保留日志与已生成核验信息，但不能把失败运行标为 complete。只缺少某类候选不是运行失败，应标记该问题当前缺少行为案例支持。

## 10. 如何据此进入内部机制阶段

若找到不处于数值边界的 H_rescue，优先为这些案例设计类别字段原位长度/布局对照，再定位删除可缓解的负作用路径。若 H_residual 较多，则为其准备“释义内容、示例来源词条、资源排列”的后续对照，不能继续把全部损失归因于显式类别字段。

若 G_category_support 与存在 D 时的表现形成清晰对比，可优先检验“类别字段与示例是否提供可替代的任务映射”。若存在 H/G_joint_only，下一阶段同时保留“两个独立贡献跨阈值”和“串行依赖”两种解释。若候选主要是近并列或计分敏感案例，先做受控重复与答案编码检查，而非直接宣称稳定回路。

本阶段最终报告应给出哪些假设值得进入内部干预、哪些仍缺案例、哪些受混杂限制；不宣布发现“词典头”“示例头”或事前资源选择能力。下一阶段的输入对照、激活采集、donor/recipient 配对与恢复/破坏端点应另写协议，不由本规划自动授权 GPU 执行。

## 附录 A. 配置冻结要点

下列为待实现配置的核心约束，不替代程序的完整 schema：

```json
{
  "schema_version": "general-model-nolabel-paired-cases-config/v1",
  "base_commit": "f80fe355263c267606c9601ca5d12aabee26da10",
  "source_run": "nolabel-01",
  "analysis_kind": "posthoc-exploratory",
  "allow_model_forward": false,
  "allow_test_content": false,
  "overwrite_sources": false,
  "query_count": 643,
  "prediction_score": "answer_sum",
  "tie_rule": "smallest-canonical-ordinal",
  "epsilon": 0.0013427734375,
  "bootstrap": {
    "repetitions": 10000,
    "seed": 42,
    "rng": "PCG64",
    "shared_query_draws": true,
    "interval": "pointwise-descriptive-percentile-95"
  },
  "selection": {
    "seed": 20260907,
    "target_total": 48,
    "discovery_target": 32,
    "reserve_target": 16,
    "backfill_across_buckets": false,
    "backfill_across_splits": false,
    "rank_by_effect_size": false
  }
}
```

正式配置还须含第 3 节完整文件路径与哈希、第 5 节分层及候选规则、第 6 节新增端点、第 7 节桶顺序与算法版本。缺少这些字段时不能仅凭本示例运行默认分析。

## 附录 B. 核对来源

全部来源固定到基准提交，而非可移动的 main。本文只核对了公开代码、协议及汇总；尚未取得本地逐查询运行产物，未生成真实候选 ID 或新增区间。文中的数量目标、工作包和选样规则均为本次拟定方案。

[R1] [实验 README：运行规模、plan 身份与本地产物边界。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/exps/causal_context/general_model_ld_nolabel_v1/README.md)

[R2] [report_manifest.json：raw/analysis 哈希与原报告核验回执。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/exps/causal_context/general_model_ld_nolabel_v1/results/nolabel-01/report_manifest.json)

[R3] [classification.csv：六条件分类点值。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/exps/causal_context/general_model_ld_nolabel_v1/results/nolabel-01/classification.csv)

[R4] [primary_differences.csv：原四个配对区间。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/exps/causal_context/general_model_ld_nolabel_v1/results/nolabel-01/primary_differences.csv)

[R5] [general-model-ld-nolabel-v1.md：冻结协议、决策与解释边界。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/docs/research/experiment-plans/general-model-ld-nolabel-v1.md)

[R6] [input-audit.json：字段删除与长度变化核验。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/exps/causal_context/general_model_ld_nolabel_v1/audits/input-audit.json)

[R7] [general_model_ld_nolabel_v1.json：配置、词典身份与 epsilon。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/config/stage1/general_model_ld_nolabel_v1.json)

[R8] [general_model_nolabel.py：plan、上下文文件及加载约束。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/diagnostics/general_model_nolabel.py)

[R9] [general_model_nolabel_execution.py：run/analysis 来源绑定。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/diagnostics/general_model_nolabel_execution.py)

[R10] [general_model_coverage_package.py：父级词条与示例来源 inventory。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/diagnostics/general_model_coverage_package.py)

[R11] [general_model_nolabel_package.py：本轮精简 inventory 与字段删除实现。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/diagnostics/general_model_nolabel_package.py)

[R12] [report_and_classification_audit.py：只读复算设计参考；原脚本会写回报告。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/exps/causal_context/general_model_ld_nolabel_v1/audits/report_and_classification_audit.py)

[R13] [general_model_nolabel_analysis.py：per_query schema、并列标记、配对 F1。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/diagnostics/general_model_nolabel_analysis.py)

[R14] [general_model_numeric_analysis.py：margin、Gold 与四种分数口径。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/diagnostics/general_model_numeric_analysis.py)

[R15] [test_general_model_nolabel_analysis.py：当前 CPU 统计与并列测试。](https://github.com/fenglinbei/hate_speech_detection/blob/f80fe355263c267606c9601ca5d12aabee26da10/src/tests/test_general_model_nolabel_analysis.py)
