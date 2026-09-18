# 跨词项新评分器：运行绑定与结果入口

用户于本轮明确授权实现评分器、绑定并使用当前空闲的四张GPU。为保持此前单卡验收方案，本轮固定使用GPU 0（NVIDIA L20，UUID `GPU-09b29c25-c372-62f4-3098-9734013e93c0`）；另外三卡未分配给此任务。

新的运行冻结位于 `reviews/cross-term-next-token-v1/frozen-01`，manifest SHA256为 `3fcb4793f42a62f15d66bef4294907c61c3919c8d0c543606682377292e5cbfb`。该冻结、源代码和依赖已绑定，不得原地修改。它是新实验，不重启任何已完成Q01运行。

## 输入与评分

沿用已确认的120条完整输入及252项比较。单卡FP32 eager、每批一条、无KV缓存；原权重配置中的bfloat16被显式覆盖为FP32，并逐张量核验。使用原生Qwen3 forward的 `logits_to_keep` 选择每条提示最后一个有效位置，保存该位置完整词表的FP32 logits，再用FP64算术计算 `m=z无−z有`。两候选共享同一次forward。

查询参考、分级与关系字段不进入评分输入；原Gold仍为空。正式分析只在工程资格及全部原始分数封存、GPU工作进程退出后读取分析参考。模型权重5个分片已全文哈希，运行环境、关键模型实现、tokenizer与物理GPU绑定在运行计划中。

## 验收与恢复

依序进行基准、重复、左padding、右padding、请求逆序五个工程pass，每个覆盖全部120输入。重复和请求逆序的margin最大差必须为0；padding最大差不超过0.001。通过后封存 `b=max(0.000001,2×最大工程margin差)`，再完成120次正式评分，并检查其与工程基准的差不超过b。12条无资料基线的自由生成诊断另计，保留原token和未经strip的可见答案。

每条请求先保存完整logits，再原子提交带来源、位置、producer、哈希和读数的回执。恢复必须保持同一冻结、环境及GPU，只复用已完成且复算通过的记录。失败和终态完成不自动重跑。往当前 `run-01` 目录创建 `STOP` 会在完成一条记录的边界暂停本任务；不会终止其他任务。

## 检查与结果

入口为 `scripts/review/run_cross_term_next_token_v1.py`，支持 validate、run、check、analyze。独立审计入口为 `scripts/review/audit_cross_term_next_token_v1.py`。使用仓库 `.conda/stage1-p0/bin/python`。

8项CPU测试已覆盖小型随机Qwen3、左右padding、最后有效位置、无截断、工程失败门控、原始向量复算及损坏拒绝。它们不替代GPU数值资格。

当前进展由本目录 `current.json` 及运行目录 `run_manifest.json` 指定。最终结果与解释须合读全部252项表达式及反向/未决项；这些仍是三个已暴露词项的开发材料，不作独立确认、纯语义或抽象适用性因果结论。
