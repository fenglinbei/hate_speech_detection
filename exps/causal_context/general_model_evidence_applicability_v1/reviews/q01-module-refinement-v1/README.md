# Q01 六模块局部细化：运行交付

本交付把首轮提名的 block34 / pre_answer 细化为33、34、35层各自的 attention 输出和 MLP 输出。CPU准备已冻结并通过验证。本窗口于2026-09-16 00:52:01（Asia/Shanghai）正常暂停，四个worker退出码均为0；00:53:38确认全部本任务进程消失、四卡显存/利用率为0。GPU完整工程验收仍未完成，科学阶段尚未开始。

已完成首个工程遍次的2,880条请求／5,760次候选评分。96条历史基线重放与2,784条工程预期均为零误差；同logits FP64最大差异3.3418916984828684e-06，小于不变的0.0001门槛。完整采集来源和封存回执通过核对。独立60位Decimal审计重建86,400项候选分数数值并核对全部工程预期，最大算术重建误差约2.96e-16。

重复性遍次已保存1,711／2,880条请求，与首遍重叠部分差异为0；部分检查点的来源、几何、卡号/UUID、runtime和采集回执均通过CPU复核。本窗口累计9,182次候选评分及768次prompt-only采集。重复性部分仍不等于完整遍次验收，也未证明padding、prefix、候选顺序或换卡门槛。

剩余工程25,378次候选评分，科学32,640次，总计58,018次。根据本窗口实测速度，仅把全部遍次按普通评分速度估算就约160分钟，prefix还会更慢，因此未满足“正式实验能在1:00前完成”的条件。下一窗口从同一run的工程重复性检查点续接；必须先完成全部六遍工程门槛，再进入科学阶段。既有freeze和已封存/保存的分数不重写。

冻结 manifest：`63683f4ebedefbf710f0cb3bfd8aab2ef2b865fd6a0abf779eb84c3a212c05c4`。

计划 ID：`q01-module-595e22271060b4e3a94f0cd5371c2cae7e201cb4adc6cd5eacb0ad7e1d1d05a5`。

## 交付文件

| 文件 | 用途 |
|---|---|
| `frozen-01/PROTOCOL.md` | 正式范围、六单元、控制、双向定义、评分、选择与停止规则 |
| `frozen-01/units.json` | 六个不同(layer,module,role)单元 |
| `frozen-01/contexts.jsonl` | 96条来源提示，逐字节继承首轮 |
| `frozen-01/positions.csv` | 672条逐提示角色记录，含每个背景自己的答案前位置 |
| `frozen-01/capture-specs.jsonl` | 每提示的模块/层/角色/位置采集合同 |
| `frozen-01/candidate-boundaries.jsonl` | 192个完整JSON标签边界，不把多token标签简化为首token |
| `frozen-01/compact-matrix.csv` | 按类别/模块/层/位置/编码/探针汇总的紧凑矩阵 |
| `frozen-01/intervention-matrix.csv` | 全部5,504条请求，逐项含供体、受体、方向和工程预期 |
| `frozen-01/pair-proofs.jsonl` | 96组完整提示长度与变动槽外一致证明 |
| `frozen-01/scoring-spec.json` | 原总分/均分、NCC两类估计对象、A/B、single/LOO和数值界 |
| `frozen-01/benchmark-sources.jsonl` | 320条完整block历史桥的来源行绑定 |
| `frozen-01/acceptance-schedule.json` | 先工程六遍、后科学六遍；逐遍预算和挑战参数 |
| `frozen-01/budget.json` | 候选、评分forward及采集成本分开计数 |
| `frozen-01/implementation-snapshot.json` | 本版与继承代码的逐文件hash及文本快照 |
| `audits/cpu-acceptance-01.json` | CPU验收回执，绑定测试、字节检查和独立输入审计 |
| `audits/independent-engineering-decimal-01.json` | GPU工程首遍的独立60位Decimal复算回执 |
| `audits/window-closeout-01.json` | 本窗口完成量、检查点、剩余预算与释放状态 |
| `audits/resource-release-01.json` | 本任务进程消失、四卡显存和利用率归零的核对 |

首轮的全部冻结、运行、分析和指针保持不变。所有新科学输入文字与原提示一致。Q01仍是一个已暴露查询，无原数据Gold；适用性与标签来源继续分开，mechanism_ready=false。后续查询转移未纳入本运行。

## 矩阵与成本

主矩阵为6模块×4组×2示例×2方向×8变体，即768条请求、每遍1,536次候选评分。H/A分别指示例中的“嘿嘿”/“哈哈”版本，与输出标签的A/B编码分开。N1/N2中性控制和示例9位置控制完整保留。block34主/中性复现与block35输出诊断共320条历史桥，不参加模块提名。

工程每遍2,880条请求，科学每遍2,720条请求，均含96条基线。12遍共67,200次候选评分：工程34,560、科学32,640；评分backbone forward共100,640次。无中断时供体采集每卡上界1,152次、四卡4,608次；恢复冷缓存另计。工程prefix包含多个完整标签前缀，不能按普通遍次时长估算。

CPU已通过16项测试：六模块真实替换/自身替换、prompt-only缓存共用、最终模块无后续传播、早层仍可传播、缓存污染拒绝、36层随机小Qwen3 CPU结构、实际worker检查点复用及采集来源、六模块提名分离、12遍合成数值门槛、历史block桥门槛、NCC代数与探针、无目标停止、精确分析重建和工程阶段边界。合成/小模型测试不等于8B GPU数值验收。

完整字节重建通过；独立审计验证353个来源文件、96条提示模板/token重建、192候选边界、96匹配对及768主矩阵请求。GPU阶段仍需重新获得全部资格。

## 当前窗口与续跑

当前窗口：2026-09-16（Asia/Shanghai），0:52不再接收新请求，0:56执行本任务进程组的兜底退出，确保1:00前释放。用户补充允许在完整正式实验能于1:00前完成时进入正式阶段；窗口默认仅工程验收，完整阶段需要明确的完成预算。服务器supervisor负责退出，不是应用自动唤醒。

读取 `window-20260916-01/state.json` 和 `run-01/run_manifest.json`，不要启动重复executor。创建窗口目录中的 `CANCEL` 可取消本任务控制器/owned run；不影响其他用户进程。只允许同卡列表、UUID、runtime、计划和源码下的 paused 运行续接。complete/failed终态或已完成的请求阶段拒绝新forward。

CPU命令（仓库根目录）：

```bash
.conda/stage1-p0/bin/python scripts/review/prepare_q01_module_refinement.py check
.conda/stage1-p0/bin/python scripts/review/audit_q01_module_refinement.py
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py validate
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py resume-check
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py engineering-check
```

`resume-check`核对已封存遍次和部分检查点，不要求工程已全部完成；`engineering-check`要求六遍工程完整通过。两者只在worker正常释放后使用。

未来窗口的GPU命令模板（必须把截止时间替换为新授权窗口内的带时区时间，并安排退出supervisor）：

```bash
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py run --phase engineering --gpus 0 1 2 3 --stop-at '<ISO时间含+08:00>'
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py run --phase full --gpus 0 1 2 3 --stop-at '<ISO时间含+08:00>'
```

默认工程阶段；`full`仍会先执行所有尚未完成的工程遍次并验收，之后才执行科学阶段。正式结果仅在全部12遍及raw seal通过后生成：

```bash
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py check
.conda/stage1-p0/bin/python scripts/review/run_q01_module_refinement.py analyze
```

## 结果阅读约束

C与I分别选至多一个模块，原总分四组双向的最差残差改善为唯一排序标准，平局按层、attention先于MLP。辅助口径不能换赢家；全部反向、过度移动、未决和分类例外保留。模块恢复比例不可相加；两个模块各自有效不证明串联路径。中性对照、绝对目标大小和第35层输出端限制必须一起解释。
