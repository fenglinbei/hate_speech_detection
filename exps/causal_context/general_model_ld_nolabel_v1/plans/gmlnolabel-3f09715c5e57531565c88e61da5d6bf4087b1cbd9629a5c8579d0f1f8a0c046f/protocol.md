# Qwen3-8B 显式词典类别字段删除实验 v1

日期：2026-09-07（Asia/Shanghai）。状态：用户已批准，所有科学输入与实现须在第一条新 GPU 分数之前封存。

## 授权与研究问题

用户先要求运行去词典标签变体、冻结项先商量；本任务提出 Lnew 六条件、分类指标为主、沿用原 8B/643 dev/FP32/真实 batch 1 并预检后自动完整运行。用户随后明确答复“可以，批准运行”。本轮据此使用四张 L20（0–3），运行至完成；不另设预约截止时间。任何数值门禁失败停止并保留失败证据，不调宽阈值、自动换 profile 或重试失败运行。

研究问题：在合并词典 Lnew 中，删除显式类别字段是否改善分类；这一差异在有、无示例 D 时分别如何。删除会缩短输入，且定义本身可能包含群体语义，因此本轮估计的是显式字段删除的总体输入效应，不是纯标签语义或内部机制识别。

## 输入冻结

- 模型、tokenizer、chat template：沿用已完成 `coverage-01` 的冻结 Qwen3-8B 身份；不微调。
- 同一 643 条 dev，query 等权；原 Lq 命中 223、未命中 420。dev 已参与开发，不称为新未见数据。
- 使用同一 833 条机制词典，SHA256 `31240193eeba29f712560e2e80e89ee9dd5c3c969aa180d10b95bf2451882385`。
- Lnew 为原 query 命中词条与同一组 10 个固定示例命中词条的并集，按冻结 ID 排序去重。直接复用封存输入的成员、顺序与全义项，不重检索、不改词典。
- 四个原条件的上下文、prompt 与 token 身份保持原样，但本轮重新计分，不导入旧分数。
- 新变体只删词典渲染中的显式 `类别：[...]` 字段。保留词条、义项、完整定义、顺序、系统指令、query、示例文本及示例答案。按完整词典前缀重建替换，不全局按文本行删除；定义或示例中恰好出现“类别：”的文字保持。
- D 关闭仅隐藏示例，Lnew 仍保留由这些示例贡献的词条。词典为空时无占位块；对应去标签条件退化为完全相同的输入。
- 不运行 extraction，不读 test，不修 Gold。query Gold 不用于构建、候选、预检或 GPU 计分；只在完整 raw 与预检证据核验封存后由 CPU 分析连接。

## 六条件与对比

| ID | 词典 | D |
|---|---|---|
| C0 | 无 | 无 |
| CLnew | 完整 Lnew | 无 |
| CD | 无 | 原固定 D |
| CLDnew | 完整 Lnew | 原固定 D |
| CLnewNoCat | Lnew 去显式类别字段 | 无 |
| CLDnewNoCat | Lnew 去显式类别字段 | 原固定 D |

正式 dev 共 643 × 2 × 6 = 7,716 blocks，643 × 6 × 34 = 131,172 candidates。

主对比：`NoCat_without_D = CLnewNoCat − CLnew`；`NoCat_with_D = CLDnewNoCat − CLDnew`。
辅助对比：CD−C0、CLnew−C0、CLnewNoCat−C0、CLDnew−CD、CLDnewNoCat−CD。无中性替代词典、标签打乱或额外 Lq 变体。

## 分类与统计冻结

主分数沿用答案 token logprob 总和，不含 EOS；保存逐 token、EOS、答案均分、含 EOS 总分和均分。hate 两个规范 JSON 候选；group 五标签固定顺序的 32 个规范集合，不增加排列。

分类取 `answer_sum` 最高分候选；精确并列按最小 canonical ordinal。记录最高分并列数，以及最高/次高差不大于 2×epsilon 的条数，不改变或豁免预测规则。group 从候选集合整体 argmax，不逐标签另选阈值，也不按 hate 强制清空。

- 主指标：Hate Macro-F1（hate/non-hate 两类等权）、Group Micro-F1（五类汇总 TP/FP/FN）。两指标 × 两主对比，共四个主要差值。
- 所有条件报告 Accuracy、正类 hate Precision/Recall/F1、Group Macro-F1、Sample-F1、集合完全匹配、逐类别 TP/FP/FN 与 F1、两任务同时正确率。零分母记 0；Group Gold/预测皆空时 Sample-F1 记 1。
- 四个主要差值仅在全部 643 条上做配对 query bootstrap：10,000 次，seed 42，NumPy PCG64；每次对所有条件和两个任务共享同一抽样。每个抽样重新汇总 TP/FP/FN 后计算 F1，不平均逐 query F1。
- 95% percentile 逐点描述性区间，无同时覆盖、多重比较校正或确认性 p 值；不按结果改端点或选样。
- all、Lq_hit、Lq_no_hit、Gold group 大小 0/1/>=2 六层保留分类与辅助点汇总。空层为 NA。辅助读数不新增 CI。
- 保留原四种计分口径下全部 Gold mass/NLL/rank、hate/group margin、集合大小和 EOS 辅助描述。它们不替代主分类指标，也不参与数值通过条件。

## 执行与门禁

沿用四个独立同模型 FP32 副本，GPU 0–3，每卡真实 forward batch 1。eager attention、TF32 关闭、无 KV cache、动态右 padding、最大 8,192 tokens、seed 42、每副本 CPU threads 4。沿用全序列 transformer 与仅答案/EOS 位置词表投影的已验证 scorer。query/task/condition 按冻结顺序确定分片，保持单 writer、完整原子 block 与 plan/runtime/input/assignment 绑定。

每卡能放下一个 8B FP32 副本，四卡以不同 query 并行；不使用张量并行将同一可单卡容纳模型拆开。原 forward batch 4 曾未通过等价门禁，因此不为吞吐恢复 batch 4，也不自动尝试缓存、降精度或新 profile。预计吞吐以新预检及正式进度的实测为准，不预先承诺完成时刻。

沿用原 regression 8 条、validation 24 条；按六条件输入元数据再选择最多 4 条边界代表（空 Lnew、最多词条、最长词典块、最长完整输入），字符串 query ID 解并列，先选代表再去重、不补位。冻结所有候选边界，额外 64 padding 也须在长度上限内。

各非空 cohort 六遍：baseline+同 logits CPU FP64 算术参考、repeat、额外 64 padding、完整无缓存逐前缀参考、候选顺序挑战、物理卡循环迁移一位。继承 E8=0.00067138671875、epsilon=0.0013427734375；repeat/CPU 参考 <=1e-4，其余挑战 <=epsilon。每项覆盖、有限性、几何、物理身份与来源要求 100%。batch4/尾批/批内位置对真实 batch1 为 NA。

每个 pass 和分片封存并用 CPU 重放检查证据；所有预检通过后自动完整 dev。任一失败阻止下游，完整 raw 验证前不接 Gold。独立生命周期 guard 在 CUDA 初始化前确认接管，仅管理本次 PID/start-ticks 身份及后代；用户 stop 文件、主进程退出遗留后代时执行有界 SIGINT→SIGTERM→SIGKILL 清理。无运行截止时间；guard 意外失联则主进程自停。

## 产物与可追溯性

新命名空间 `exps/causal_context/general_model_ld_nolabel_v1`。配置、协议、源码及继承源码、输入、边界、候选、模型身份均在首个 GPU 分数前哈希封存。保留旧实验及其失败状态。对中断只允许同一身份下完整已提交 block 的恢复；失败不自动重试。

最终交付完整运行状态、数值预检证据、分类与四个配对区间、Gold/margin 辅助表、数据及实现核验说明。分析复算与来源验证不表述为另一个未实际执行的独立研究者审计。
