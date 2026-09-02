# Stage 1 词典/示例机制探索试验 v0

## 结论与核心目标

可以暂停当前词典扩建，先运行这一版试验。它回答的不是“现有词典是否已经足够好”，而是更窄、更有行动价值的问题：在固定的、机制富集的开发帧上，未做本任务训练的 Qwen3-8B 是否会对词典信息 L、示例信息 D 及其冲突作出可重复的输出变化；变化主要来自定义、类别、示例输入、示例标签，还是仅来自额外上下文长度和格式。

该结果只用于决定 WP3 后续优先修什么、正式主线值得怎样设计。它永久标记为 development-only、non-sealed、non-scientific，不估计总体性能，不触碰封存 test，不声称揭示已训练 LexSRAG 的内部机制。

## 冻结范围

- 模型：本地 `models/base/Qwen3-8B`，Transformers、单 GPU、BF16、greedy、`enable_thinking=false`。
- 词典：只读 `data/lexicon/annotated_lexicon.json`；不读取 WP3 生成中间物。
- 示例：只从 5,165 条 formal fit partition 检索；calibration、dev、test 均不可成为 demo。
- 查询：formal dev 的固定 64 条机制富集帧；gold 可供采样覆盖与事后严格评估，审核页不可见。
- 长度：总长 8,192；prompt 不超过 7,680；completion 上限 512；冻结后不裁剪、不删 demo，超长硬失败。
- 预算：完整自由生成 888 次；预检输出属于同一完整网格并按 prompt hash 复用。

## 固定开发帧

五层配额为 16/16/16/8/8：非透明词条、可疑词义/边界/指代、引用/否定/反仇恨/自称等语用上下文、多命中/嵌套/冲突、无精确命中的 D-only 控制。抽样器在看模型输出前冻结主样本和同层 reserve；人工驳回后只允许用预冻结的同层 reserve 依次替换。

精确命中为区分大小写的 substring。完全相同的 `(term, category, definition)` 三元组只保留一次；同一 term 的不同类别或定义均保留。排序依次按首次出现位置、较长 term、稳定词典编号，最多 5 条。

### 审核（适配 WP3 2.1 / G3 操作逻辑）

审核页不从零开发，也不原样搬用形式关系字段。`tools/exploratory_qwen3_ld_review_ui` 复用 WP3 2.1 / G3 已验证的工作台操作逻辑：候选队列、检索与过滤、快捷键、自动保存、CAS 并发冲突处理、确认后锁定及带原因重新打开；页面内容和校验契约则适配为本试验的输入质量审核。

页面直接显示盲化查询、精确命中 span、词典类别、定义及预注册 DefinitionSwap donor。动作解释如下：

- 接受：命中相关、边界正确、定义基本可用、当前 sense 匹配；预注册 swap 自然但不相容。
- 驳回：任一核心条件失败；备注写 `fail=relevance|boundary|definition|sense|swap` 中的一个或多个。
- 暂缓：允许中途保存，但冻结时必须为 0。
- 确认后锁定；需要更正时必须填写原因重新打开，保留 amendment 记录。
- 语用标签在页面中直接多选：引用、否定、反仇恨、自称/重领、反讽、讨论。

审核页不显示 gold、demo、条件名、模型输出或效果统计。

### 输入审核结果

首轮审核覆盖全部 128 条预冻结候选：初始结果为 92 接受、35 驳回、1 暂缓。主要失败集中于精确 substring 的边界和当前 sense：`relevance`、`boundary`、`sense_fit` 各有 27 条失败；31 条驳回来自 lex-hit，4 条来自 no-hit 覆盖疑问。单字或跨词边界命中是主要来源，例如“基本→基”“批判→批”“其它→它”。这些问题记录为后续 matcher/lexicon 修复输入，不在 v0 中回写词典。

人工复议保留完整 amendment 轨迹：

- `妈宝女 / 妈宝` 是不同词条，按 v0 已冻结的 substring 与 multi-hit 规则同时保留，改为接受；
- “唔好咩都唔理咁”是普通组合方言表达，不作为稳定仇恨词条，改为接受 no-hit。

复议后为 94 接受、34 驳回、0 暂缓，五层均满足配额。最终 64 条 frame ID 为 `pilot-frozen-995e5977f29aeb97121c31b423b2b8ead57031dd6bc80f8d1c79db51c915f43d`；16 条替换全部来自预冻结的同层 reserve。

## 干预矩阵

56 条 lex-hit 查询运行 15 条件：C0、L-Full、D-Full、LD-Full、PL、PD、L-Definition、L-Category、L-CategorySwap、L-DefinitionSwap、D-Input、D-Schema、D-CrossLabelShuffle、LD-CategorySwap、LD-DefinitionSwap。

8 条 no-hit 查询运行 6 条件：C0、D-Full、PD、D-Input、D-Schema、D-CrossLabelShuffle。总数固定为 888。

CategorySwap 使用合法类别的确定性异类替换，并优先接近 tokenizer 长度。DefinitionSwap 的 donor 必须来自不同 term 和 category、不得出现在查询中，按长度差和稳定 ID 决定。CrossLabelShuffle 对 10 个单四元组 demo 的 `(targeted_group, hateful)` 联合标签做无固定点旋转，保留输入、target、argument 与整体标签分布。

PL、PD 是自然、无任务答案、无查询 token 的长度匹配安慰剂；与对应真实资源的差异必须在 `max(8 tokens, 2%)` 内，否则该上下文构造不合格。

## 技术预检与停止规则

预先从五层选 8 条（2/2/2/1/1），按各自完整矩阵生成。只看模型加载、runner 状态、token/hash、严格 JSON 和确定性，不看准确率或条件效果。

- 基础设施/模型加载失败必须为 0。
- C0、L-Full、D-Full、LD-Full 的严格 JSON 率至少 90%。
- 固定 4 个 prompt 重放时 raw output 与 token IDs 完全一致。
- 失败即停止整批；修改必须产生新 plan/run ID，不逐行重试。

当天 09:20 后不启动新 GPU batch；先 flush ledger 与 checkpoint，09:30 前释放 GPU。若自由生成网格不完整，不给总体效果结论；margin 可作为独立、同 hash 的可续跑覆盖层。

## 报告口径

自由生成报告严格格式率、四字段指标、相对 C0 的成对 flips、标签跟随和 L×D 交互。11 个诊断条件另做 targeted_group/hateful teacher-forcing 序列对数似然 margin；hate foil 为二元相反标签，group foil 为固定合法类别轮换，gold/foil 从不进入提示或抽样。

统计只报告固定富集帧上的成对差和 bootstrap 区间，不给总体 p 值，不设“必须显著”的科学成功门槛。零效应、反向效应和格式失败均是有效诊断结果。

## 2026-09-02 实际运行记录

上下文网格按冻结契约生成 888/888 条，grid ID 为 `pilot-contexts-d0f47e0ce2df8858df1db0b5a7eb3b1685589c79756dd0264bf5635fb1ea3b00`，最大 prompt 为 1,557 tokens，未触发 8K 长度上限。因 GPU 窗口改到晚间，命令行只覆盖工程停止时刻为 23:30；模型、数据、矩阵、prompt 和生成参数均未改变。

最终预检覆盖 8 条查询、111 个条件输出：基础设施失败 0，核心条件严格 JSON 率 0.933，4 个固定 prompt 的 raw output 与 completion token IDs 重放均完全一致。第一次重放检查把右侧 batch padding 误计入 completion token IDs，产生了假性不一致；原始失败账本和回执保存在 `failed_preflights/20260902T212326_padding_batch_geometry/`。修复仅把 completion 截到首个 EOS，并以相同有序 batch geometry 重放；冻结 frame/grid 未重建。这一处理偏离了“任意预检失败即新 plan/run ID”的字面规则，因此作为 instrumentation-only deviation 明示保留，而不能当作隐去失败。

自由生成于 21:42 前完成 888/888 条，runner failure 为 0；账本最终 SHA-256 为 `3fd089ef1fa506c1150f1a79c3089b1edd6145da8256c0193bd5ed6ad750e49c`。主评估显示，这个未训练模型的基线任务能力很低（C0 group F1 0.076、hate F1 0.242、hard tuple F1 0.030），因此下面的变化应理解为资源敏感性，而非可用系统性能。

在固定开发帧的组别/仇恨联合 exact 上，D-Full 相对 C0 为 +0.172，95% bootstrap CI `[0.062, 0.297]`；LD-Full 相对 C0 为 +0.143，CI `[0.036, 0.250]`；L-Full 相对 C0 为 +0.054，CI `[-0.036, 0.143]`。直接对照进一步显示 D-Full 优于只给 demo 输入的 D-Input（+0.172，CI `[0.062, 0.281]`）、只给 schema 的 D-Schema（+0.203，CI `[0.094, 0.312]`）和长度安慰剂 PD（+0.188，CI `[0.078, 0.297]`）。D-Full 与 CrossLabelShuffle 的差为 +0.047，CI `[-0.016, 0.125]`，不足以在这一小帧上区分完整 demo 标签与轮换标签。

这些直接对照由冻结条件和预先写明的解释顺序派生，但具体汇总表是在完整生成后的报告审查中加入；其中 `L-Full - L-Category` 是为补齐定义的条件增量而追加。它们没有改变样本、prompt、条件或模型输出，但按开发性派生分析而非独立确认性检验解释。

词典侧的证据更弱且不均匀：L-Full 相对长度安慰剂 PL 为 +0.071，CI `[0.000, 0.161]`；L-Category 相对 CategorySwap 为 +0.107，CI `[0.018, 0.214]`，说明正确类别有可检测贡献；L-Full 相对 L-Definition 为 +0.071，CI `[0.018, 0.143]`，但相对 DefinitionSwap 仅 +0.018，CI `[-0.036, 0.071]`。LD-Full 相对 L-Full 为 +0.089，CI `[-0.018, 0.196]`，相对 D-Full 为 0.000，CI `[-0.089, 0.107]`；L×D 的联合 exact 交互为 -0.054，CI `[-0.179, 0.071]`，没有协同证据。CrossLabelShuffle 的 rendered-label 集合覆盖过宽，原集合重合率不是可识别的“标签跟随率”，只保留作描述性诊断。

teacher-forcing margin 第一次用完整 FP32 `log_softmax` 时在写入 2 条后 OOM，失败账本已隔离到 `failed_margins/20260902T2142_full_fp32_softmax_oom/`；后续实现改为只 gather 目标 token、沿时间轴分块计算 FP32 `logsumexp`，并将 gold/group foil/hate foil 逐条前向，单元测试已与完整计算对齐。一个 20:44 启动的外部四卡训练随后占用每卡约 30–33GB；在重新核验 PID/工作目录且获得用户明确授权后向该任务发送 `SIGTERM`，进程自行退出，没有使用更强信号。

margin 覆盖层最终完成 616/616 条，未发生失败或截止中断，账本 SHA-256 为 `5539002e3de32f9d00d8f973864433e42278eb1d1a49e4359d69f82df5d65602`。完整 demo 相对 CrossLabelShuffle 的 group/hate margin 分别增加 +4.549（CI `[3.149, 6.003]`）和 +2.452（CI `[1.127, 3.730]`），表明 demo 的正确输出标签确实改变模型偏好，而不只是输入与格式起作用。

词典机制呈现清晰的分工。正确 Category 相对 CategorySwap 的 group margin 增加 +13.592（CI `[11.132, 15.941]`），但 hate margin 没有对应改善；在正确 Category 上加入 Definition，group/hate margin 进一步增加 +1.233（CI `[0.319, 2.153]`）和 +2.170（CI `[1.127, 3.221]`）。完整词典相对 DefinitionSwap 也增加 +2.090 group margin 和 +1.156 hate margin，两个区间均不跨 0。也就是说，类别主要帮助目标组判断，正确、带用法信息的定义提供较小但可检测的增量。该概率偏好并未稳定转化为 greedy exact：自由生成中的 `L-Full - L-Category` 联合 exact 为 -0.018（CI `[-0.089, 0.036]`），这是低基线任务能力下“内部偏好变化”和“最终结构化输出正确”之间的区别。

LD-Full 相对 L-Full 主要提高 hate margin（+5.011，CI `[2.652, 7.311]`），相对 D-Full 主要提高 group margin（+7.174，CI `[4.672, 9.853]`）；这支持词典和示例在字段上的互补性，但不推翻自由生成 L×D 交互区间跨 0 的结果，因此不能声称存在协同效应。

### 对 WP3 的当前决策含义

可以继续暂停无差别扩充词条数量。下一轮更值得做的是有边界的手工修复：优先修 matcher 的 span/词边界与当前 sense 过滤，其次核对类别，再把“仅释义”定义补成包含实际贬损/指称用途的定义；与此同时保住 demo 的标签质量与类别覆盖。现有结果支持“一轮针对审核失败模式的修复后复跑同一开发诊断”，不支持先进行大规模词典扩建，也不支持把这 64 条开发帧结果解释为总体性能结论。

## 可复现入口

```bash
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py build-frame
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py serve-audit --port 8766
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py freeze-frame
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py build-contexts
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py preflight
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py run
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py evaluate
.conda/stage1-p0/bin/python scripts/stage1/exploratory_qwen3_ld_v0.py margins
```

默认产物目录为 `exps/causal_context/stage1_exploratory_qwen3_ld_v0/`，由仓库现有 ignore 规则排除。每一步写入输入文件 hash、配置 hash、prompt hash 和原始失败；已存在结果只按完全相同 prompt hash 续跑。
