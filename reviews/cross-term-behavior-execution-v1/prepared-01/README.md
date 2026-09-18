# 第二轮跨词项行为实验：GPU 前准备

36 条新输入、84 条历史端点和 168 项比较沿用人工已采纳的科学冻结包；全部提示与答案保持原字节。这里冻结执行实现和数值验收方案，实际设备绑定、GPU 验收及新评分均未进行。没有启动或排队 GPU 任务。

## 执行顺序与预算

1. 在新获准的 GPU 窗口中，核验第一轮使用的 GPU 0 UUID、驱动、Python、权重及源码完全相同，并确认该卡空闲。只需要该张 L20，不需要四卡。
2. 重放全部 84 条历史端点。主分数 m=z无−z有 必须逐项精确重现，最大差为 0；不覆盖旧分数，不把桥接计为新增科学条件。失败即停止。
3. 对 36 条新输入各做 reference、repeat、左 padding、右 padding、逆请求顺序五遍，共 180 次 forward。重复和逆序误差必须为 0，padding 上限沿用 0.001。新界 b=max(0.000001,2×新输入最大工程差)，不继承旧界。
4. 封存桥接和资格凭据后，正式评分新 36 条，各一次 forward，并与 reference 比较。每次只取同一全词表 FP32 向量，两个合法答案共享一次 forward，FP64 归约。
5. 封存原始评分并释放 worker，才允许 CPU 分析读入人工参考，复用 84 个旧物理分数及其原界，计算全部 168 项比较。共享物理项先抵消，再累加误差界；修复、损伤、连续移动与未决分开报告。

合计 **300 次 prompt-only forward / 600 个候选值**，包括 84 桥接 + 180 工程 + 36 正式。全词表向量约 183 MB（十进制），另加凭据。旧 36 条 N 条件通过历史结果完整保留，不进入新核心。原 12 个空示例基线的自由生成诊断保留为历史证据，本轮无额外生成；不能据此声称新输入自由生成已验收。

旧端点的界仍为 0.00045013427734375，保留其原资格文件；新端点界等待新 GPU 工程验收。桥接只认可同设备同运行环境的精确主分数重现，不增加事后容差。任何失败保留证据、停止，不自动重试或放宽阈值。

## 入口

CPU 核查：
```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py validate
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .conda/stage1-p0/bin/python scripts/review/test_cross_term_behavior_v1.py
```

以下步骤必须等待用户重新给出可用窗口；当前未执行，也没有空闲轮询：
```bash
.conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py bind --authorization-note '填写实际新授权原话'
.conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py run
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py check
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/run_cross_term_behavior_v1.py analyze
CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/audit_cross_term_behavior_v1.py results
```

bind 新建独立 bound-01.json，prepared-01 保持不变；run 只读取这个绑定。暂停用新 run 目录中的 STOP 文件；只在显式移除 STOP 并重新授权后用 run --resume，已完成和失败的 run 拒绝启动。不会重启任何历史 run。

所有材料都是三个已暴露词项上的开发条件；A/B、同词/异词改变整包文字，不能解释为纯义项或规则适用性的因果效应。没有新增独立确认、内部干预或机制结论。CPU 通过不等于 GPU 数值验收通过。
