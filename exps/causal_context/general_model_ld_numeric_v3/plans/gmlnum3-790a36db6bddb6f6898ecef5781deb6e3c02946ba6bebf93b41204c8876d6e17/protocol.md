# General-model L/D numerical measurement v3

日期：2026-09-05（Asia/Shanghai）。本版落实父注册已预先列明的真实 batch 1 后备路径，不重新选择数值精度、不改变研究对象或容差。用户后续已授权逐步优化直至得到完整 dev 科学结果，只有重大研究修订才重新讨论。

## 依据与失败保留

父科学注册 [v1](general-model-ld-numerical-measurement-v1.md) 第 7、8 节明确：batch 4 失败时可另建完整 batch 1 运行，必须新 run、新回执、适用的数值基准且禁止混合两种 batch 分数。旧流程的 automatic=false 要求原失败流程停止，不将失败自动改判；本版是在其已停止后按后续整体执行授权落实该既定备选。

[v2](general-model-ld-numerical-measurement-v2.md) 的 numeric-03 已封存为 preflight_failed。原 8 条六挑战全部通过，E8=0.00067138671875，epsilon=0.0013427734375。24 条重复及 CPU 算术参考通过，但 batch 4 最大误差 0.001621246337890625 超过冻结 epsilon。失败不能因 margin 未超阈值或其他读数通过而豁免。

三个条件记录的 group-24 答案总分及含 EOS 总分超过门槛。独立检查未发现候选、索引、边界或求和错误；证据与 FP32 前向形状依赖相容，尚未定位到某个具体层。不存在已经证实且足以保证修复的单行计算错误，因此不在已暴露 24 条上搜索新 precision、padding 或参照定义。

原计划、源文件快照、失败回执和全部原始分数保留。v3 计划以哈希绑定父 v2 plan、numeric-03 终态及预检报告，确认后备事先登记、原 8 条完整通过且 epsilon 由原公式得到。旧分数只作失败与校准来源，不拼接进新的预检或 dev 主产物。

## 唯一生产变化

真实 batch 4 改为真实 batch 1。四张 GPU 仍运行相同 FP32 副本，按原子组分配；四张卡并行不等于单个 forward 的 batch 为 4。每次生产 forward 仅有一个候选，不补 dummy 行、不伪报 batch，不为加速改为 prefix 或 KV-cache 生产。

以下全部继承父 v2 冻结身份：同一 Qwen3-8B 参数、FP32 Transformer 和输出层、禁用 TF32、eager attention、动态右 padding、无 KV-cache、全序列 teacher forcing、仅答案/EOS 预测位置的全词表投影、tokenizer、seed 42 及 CPU threads 4。kernel、pool 和父版 scorer 源码不改。

科学内容完全不变：643 条 dev、hate 两候选、group 32 个固定顺序规范集合、两任务六条件、7,716 block、131,172 候选。答案总 logprob 不含 EOS 为主；四种分数、六类 margin、八项配对比较、主总体/命中分层/gold 大小分层、集合大小与 EOS 检查，以及 10,000 次 seed 42 联合 query bootstrap 均继承原注册。

## 后备验收

新 run 从头对原 8 条及原 24 条依次计分，不复用 numeric-03 的候选分数。所有注册读数仍包括逐 token、EOS、四种候选分数及全部条件级 margin；身份、边界、有限性和完整度仍须 100%。

1. 真实 batch 1 运行两遍，重复误差不超过 1e-4；首遍同次 FP32 logits 的 CPU float64 算术参考误差不超过 1e-4。
2. 四类生产相关比较均对首遍 batch 1 的全部读数验证：额外 64 个 masked padding；现有完整、无 cache、无 padding 的唯一已发生前缀参考；任务特定候选执行顺序变化；真实物理卡循环迁移一位且 batch 1 几何不变。
3. 前缀算法仅作独立参考，仍覆盖每个候选所有答案和 EOS 位置。候选顺序检查在 batch 1 下检验执行顺序/状态依赖，不声称改变不存在的批内行位置。
4. 两个 cohort 的全部四类比较均使用已冻结的 epsilon=0.0013427734375。新观测只记 observed_fallback_max，不重新计算 E8、不调整 epsilon，也不以 24 条校准。
5. batch 4 等价、size-2 尾批及批内位置轴不再适用于真实 batch 1 生产，明确记为不适用；它们的原失败及证据保留，不能写成本版通过原六挑战。全 dev 的最后一条仍必须覆盖，batch 1 不存在另一个 size-2 生产尾批。
6. 任一失败即封存并停止，不扩展 dev，不自动搜索其他 profile。只有两组后备适用门槛全部通过，才执行完整 dev-b1。

这 24 条已经暴露于 numeric-03 的 batch 4 验收。本次是预注册备选的重验，不称为首次未见的独立验证集；不更换样本，不据其误差挑选其他数值实现。若预定后备仍失败，则停止自动执行并就新的测量或验收设计与用户讨论。

## 封存、分析与解释

新 plan、config、协议及新增编排代码共同封存；旧 v1/v2 源码保持其冻结哈希。每个候选记录真实设备、batch 1、运行和 prompt 身份，原子 block 断点只复用同一新身份下已验证的完整 block。禁止不同 plan、runtime 或 batch 混合。

完整 dev raw 的顶层/四 shard 身份、哈希、规范排序和 7,716/131,172 覆盖全部核验并封存后，才允许读取冻结 query gold 及生成原分析；不得读取 test。仅有预检通过或部分 raw 时不能发布科学结果。

独立审计复算所有候选分数、margin、配对读数、主/辅助分层汇总和全部 192 个端点在六个主分层的 CI，并输出 EOS 与集合大小附表。普通比较数值带仍为 2*epsilon，LxD 为 4*epsilon。经验数值带不是数学误差保证，未分辨不等于科学零效应。

结果属于 dev 的类别偏好变化分析，不将输入非加性称为已识别内部因果机制。固定序列化顺序、答案长度、受限候选空间、模型/数据历史暴露、标注质量及金标大小混杂等解释限制不变。
