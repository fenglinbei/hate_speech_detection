# General-model L/D numerical measurement v2

日期：2026-09-05（Asia/Shanghai）。这是 v1 的数值实现修订，不是研究问题或数据口径修订。用户已授权逐步优化、预检通过后执行完整 dev 并得到科学结果；只有改变重大研究约定才重新讨论。

## 保持不变的科学边界

- 父注册为 `general-model-ld-numerical-measurement/v1`，父数值 plan、输入 package、生成预检和源码均保持原身份。v1 失败回执不覆盖、不改判、不混入新分数。
- 固定本地 Qwen3-8B、完整 643 条 dev、hate/group 两任务、C0/CL/CD/CLD/PL/PD 六条件。共有 7,716 block 和 131,172 候选。
- hate 两种规范答案、group 五标签固定顺序的 32 个规范 JSON 集合、答案总 logprob（不含 EOS）主线、五桶边际 log-odds 和全部八个配对比较不变。
- query 等权、主总体与 lexicon hit/no-hit 分层、gold 大小分层、token 均分/EOS/集合大小敏感性、10,000 次 seed 42 联合 query bootstrap 和原有限解释边界不变。
- query gold 只在全部 raw 封存并通过验收后显式加入；不读 test，不运行微调模型、14B/27B、提取或额外自由生成。不能因效应方向、准确率或显著性选择数值实现。

## 数值修订依据

v1 E8 为 3.9062767，且 batch 1 同前缀跨完整候选长度也漂移，不能直接回退运行。新增诊断仅使用原 8 条内四个失败 context：1322:group:CD、3531:group:PD、7648:group:CLD、7648:hate:CL；没有新增开发样本或读取 gold。

实际 CUDA 对照显示 BF16 差异在早期 Transformer 层已经出现。关闭低精度归约、单独 FP32 输出层及固定 padding 都不能在全部四个 context 消除漂移。全 FP32 对照的答案总分最大跨形状差约 0.000542、所探查前缀 token 最大差约 0.000523，支持继续检验全 FP32，而不是放宽 0.005 上限。这些开发诊断不是正式预检，也不是科学效应。

## 冻结实现

- 参数从父运行已核验的 BF16 checkpoint 加载，精确扩展到 FP32 计算；不训练、不改变参数数值或 tokenizer。Transformer 与 lm_head 都是 FP32，TF32 禁用，BF16 reduced-precision reduction 禁用，eager attention、eval、seed 42、CPU threads 4、确定性算法不变。
- 仍是全序列 teacher forcing、右 padding、`use_cache=false`。仅在答案和 EOS 的预测位置执行全词表输出投影，避免为 prompt 和 padding 生成不使用的词表 logits。仍使用全词表 logsumexp，不裁剪归一化词表。
- 生产采用动态右 padding、最大 batch 4 和父注册的候选/任务/条件分组；其形状依赖必须通过下面所有挑战，不以固定 padding 的重复性代替可信性。
- 实际 dtype、后端开关、输出投影和 padding 策略构成新的 runtime 身份；父 BF16 生成 runtime 只作为模型来源证据，不宣称两者逐值等价。科学结果明确为同一冻结参数的 FP32 数值测量。
- 单个 query/task/condition 的全部候选仍为原子提交单位；断点续跑只复用相同 plan、runtime、pass 和完整验证 block。不同精度、不同版本分数不拼接。
- 数据并行固定使用物理 GPU 0/1/2/3 的四个相同 FP32 模型副本。按原 batch group 整组轮转分配，不拆候选原子 block、不改变 batch 成员或样本权重。统一 pool 身份包含每个物理设备及其实际数值 runtime，候选另记实际生产卡；顶层完整 pass 只有在所有 shard 哈希、身份和完整度核验通过后封存。

## 新预检门槛

原 8 条为工程选择与误差校准集，额外 24 条保持独立验收。每个 cohort 覆盖两任务、六条件、全部候选和全部答案/EOS 位置。

1. batch 1 两遍，全部注册读数重复差不超过 1e-4；首遍同次 FP32 logits 的 CPU float64 算术参考差不超过 1e-4。
2. 以下六种挑战分别与 batch 1 首遍比较：原 batch 4；hate size-2 尾批；在 batch 4 动态长度上增加 64 个 masked padding；不含未来答案的逐前缀参考；任务特定的 batch 4 成员/位置挑战；保持 batch 组不变并将物理执行卡循环移动一位的跨卡挑战。成员挑战对 group 使用循环移位一位后反转，对 hate 直接交换两个候选，保证两任务实际输入顺序都发生变化；封存前恢复规范候选顺序。
3. 逐前缀参考对每个不同已发生前缀独立运行无 padding、无 cache 的 batch 1 前向。同一前缀多个下一 token 共享一次全词表归一化；这只是相同条件概率的复用，不是 KV-cache。每个完整候选的答案与 EOS 分数都由这些 token 重建。
   跨卡挑战必须由实际设备 UUID 证明相对标准 batch 4 确实迁移，且 batch 成员、顺序和 padding 保持不变。数值同时与 batch 1 及标准 batch 4 比较，该家族误差取两者最大值。
4. 比较范围是每个 token、EOS、四种候选分数和所有注册条件级 margin，不能只检查均值、单 token 或个别候选。
5. E8 取原 8 条所有六种挑战、所有注册读数的最大绝对差，必须不超过 0.005。epsilon = min(0.005, max(0.0001, 2*E8))，只在 E8 通过时生效。额外 24 条的全部六种挑战均不得超过冻结 epsilon。
6. 候选身份、边界、有限性、来源和完整度要求 100%。任一门槛失败即封存失败并停止该运行；不得借此次失败回执启动 dev，不放宽阈值。

额外 24 条不用于反复选择数值实现。若其验收失败，保留失败并先判断能否仅通过确定性工程修复解决；涉及重新定义误差接受标准或科学测量目标时属于重大修订，需与用户讨论。

早期单卡 `numeric-02` 在原 8 条首遍阶段因发现 hate 置换公式退化为恒等而主动中断，未执行额外 24 条或完整 dev。该次 plan 快照和中断回执保留，不复用其部分分数放行；上述任务特定置换与数据并行策略重新冻结为独立 plan，并从头执行正式预检。

## 结果发布与解释

全部预检通过后自动运行完整 dev；全部 7,716 block、131,172 候选完整封存后才加入 gold 并运行原分析。独立审计复算候选分数、margin、八个 contrast、分层汇总和 bootstrap CI，并补充 EOS 单项贡献和各集合大小 log-evidence 附表。

主结果报告方向、幅度、描述性置信区间与数值未分辨区，不要求正效应。继续使用普通比较 2*epsilon、交互 4*epsilon 的经验数值带；不把它解释为数学误差保证或科学零效应。结果属于 dev 开发分析，固定候选顺序、长度、候选空间限制以及内部因果识别不足等限制全部保留。

具体配置、源码、父 plan 身份和本协议快照在新 plan 中共同哈希封存。只有正式预检和全量 raw 实际完成，才可以标记科学结果已完成。
