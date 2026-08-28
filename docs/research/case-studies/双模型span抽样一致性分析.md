# 双模型 span 抽样一致性分析

## 1. 目的与边界

本轮不是构造 gold set，而是检验“Qwen3.8-27B + DeepSeek v4 flash 直接从原始记录抽取 exact span”是否具有足够质量，值得进入后续人工裁决或双模型生产流程。

模型请求只包含原始 `content`。数据集类别、`target`、`argument`、旧候选、旧模型结果、原始记录 ID 和抽样层均未进入 provider payload。

## 2. 冻结抽样

- 来源：Stage-1 正式 train partition 的 fit-only 5,165 条记录；
- 唯一记录：240 条；
- 六层：Sexism、Racism、Region、LGBTQ、others、non-hate 各 40 条；
- 只使用单一 hate 类别记录或纯 non-hate 记录，多标签记录不进入本轮；
- 每层按字符长度排序切成短、中、长三等份，分别抽取 14、13、13 条；
- 每层另选 8 条隐藏复测，按短、中、长 3、3、2 分配；
- 总任务：每模型 240 + 48 = 288 条；
- 重复任务使用不同 blind alias，页面位置至少相隔 12 条，并分别真实调用模型；
- frame：`spanframe-66a04b4dda19d4988e6f401e1d5d1938717b07f34ac0cfb053159c6e4a9ade26`。

## 3. 模型与执行协议

| 项目 | Qwen | DeepSeek |
|---|---:|---:|
| 模型 | Qwen3.8-27B，本地冻结权重 | deepseek-v4-flash |
| 任务完成 | 288/288 | 288/288 |
| 物理 attempts | 288 | 288 |
| 重试/错误 | 0/0 | 0/0 |
| temperature / top_p | 0.2 / 0.8 | 0.2 / 0.8 |
| max tokens | 768 | 768 |
| thinking | disabled | disabled |

Qwen 权重清单 `/data/models/Qwen3.8-27B.sha256` 的全部文件在启动前通过 `sha256sum --check`。专用环境为 `/data/liaozijie/conda/qwen38_27b_span_audit`，核心版本为 vLLM 0.17.1、torch 2.10.0+cu128、transformers 4.57.6；纯文本 `--language-model-only` 模式在 2×L20 上运行。[vLLM 官方配方](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)对该模型注明 vLLM 0.17+，并说明 transformers 5.8 的要求主要关系到 Qwen-VL processor；本轮不加载视觉处理器。

所有调用写入事务型 checkpoint。attempt 在 HTTP 前预留，进程若异常退出会把未完成 attempt 保守计入预算；provider 全局协议 hash 锁定 prompt、解码参数、endpoint 与模型，续跑不得混用配置。

## 4. 结果

### 4.1 跨模型（240 条唯一记录）

| 指标 | 结果 |
|---|---:|
| exact span 集合完全一致 | 181/240 = 75.42% |
| 双方均无 span | 141/240 = 58.75% |
| 非空 exact span 完全一致 | 40/240 = 16.67% |
| exact span 集合不一致 | 59/240 = 24.58% |
| 至少共享一个 exact surface | 49/240 = 20.42% |
| span Jaccard 均值 | 0.7757 |
| has-span Cohen's κ | 0.5833 |
| Qwen 判有 span | 66/240 = 27.50% |
| DeepSeek 判有 span | 88/240 = 36.67% |

因此，`75.42%` 不能直接解读为高质量一致性：其中大部分来自双方同时输出空集。κ=0.5833 只达到中等程度的一致性，且 DeepSeek 明显更倾向于收词。

分层 exact span 集合一致率中，non-hate 最高（97.5%，双方均无 span 95%），Sexism 最低（60%）；这表明主要争议集中在有潜在词条、且词面边界或“中性身份词是否误收”更难判断的记录。

### 4.2 模型内隐藏复测（各 48 对）

| 指标 | Qwen | DeepSeek |
|---|---:|---:|
| exact span 集合一致 | 44/48 = 91.67% | 44/48 = 91.67% |
| span Jaccard 均值 | 0.9306 | 0.9427 |
| 共享 span 的类型一致 | 100% | 100% |
| 共享 span 描述的字符 bigram Dice | 0.7293 | 0.6771 |
| 整条描述的字符 bigram Dice | 0.5985 | 0.6587 |

复测说明两个模型的 span 决策多数稳定，但仍各有 4/48 对发生 span 集合变化。描述指标只是字面一致性诊断，不等同于语义一致；不能凭 91.67% 复测一致就认定输出正确。

### 4.3 协议质量

- 两模型 576 条结果均可解析为规定 JSON；
- annotation-level 结构错误为 0；
- span-level 结构错误为 0；
- 非原文逐字子串为 0；
- Qwen 总 token：143,588；
- DeepSeek 总 token：134,795，其中 prompt cache hit 74,112、miss 34,353、completion 26,330。

这说明“输出协议与 exact-substring 约束”可靠，但不代表词条的语义资格可靠。

## 5. 人工检查建议

暂不把任一模型输出或交集自动升级为 gold。建议先在盲化 Web 包中检查：

1. 全部 59 条跨模型 span 集合分歧；
2. 288 个页面 case 独立检查，不提前揭示其中 48 条隐藏复测；审阅后再用 `audit/source_map.json` 配对检查人工与模型稳定性；
3. 从 141 条双方均为空中抽查漏召回；
4. 从 40 条非空完全一致中抽查“共同犯错”，特别是普通身份词、职业/称谓和上下文恶意被错误转移到中性词面的情况。

页面包含全部 288 条（240 条唯一原文 + 48 条隐藏复测），跨模型分歧可筛选，但重复关系不在主界面显示。模型真实身份默认隐藏为 A/B，建议完成判断后再查看包内 `audit/model_mapping.json` 与 `audit/source_map.json`。

## 6. 产物

- 完整 Qwen 结果：`exps/causal_context/stage1_p0/dual_model_span_audit/runs/spanrun-66a04b4d/results/qwen.results.jsonl`
- 完整 DeepSeek 结果：`exps/causal_context/stage1_p0/dual_model_span_audit/runs/spanrun-66a04b4d/results/deepseek.results.jsonl`
- 一致性汇总：`exps/causal_context/stage1_p0/dual_model_span_audit/runs/spanrun-66a04b4d/results/analysis.json`
- 唯一记录逐条比较：`exps/causal_context/stage1_p0/dual_model_span_audit/runs/spanrun-66a04b4d/results/cross_model_detail.jsonl`
- 隐藏复测逐对比较：`exps/causal_context/stage1_p0/dual_model_span_audit/runs/spanrun-66a04b4d/results/repeat_detail.json`
- 离线 Web 包：`exps/causal_context/stage1_p0/review_packages/dmspan-review-30583732ec1adb2f.zip`（manifest 固定 288 cases / 240 unique / 48 hidden repeats）。
