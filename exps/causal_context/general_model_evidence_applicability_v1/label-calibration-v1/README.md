# 原标签、NCC 与 A/B 正反映射

新冻结入口为 `current.json`，绑定原来两个案例的 36 个条件与 64 项比较。完整细则见冻结目录的 `analysis_protocol.json` 和 `FREEZE.md`。原材料、接受回执、分析参考及上一轮结果均保留。

先重放全部 36 个历史原标签条件，再完成本轮的原标签、逐条件五探针 NCC 和 A/B 正反映射。任务说明、示例答案和完整候选同步映射；其余内容保持。计算与数值边界在新模型分数出现前冻结。

CPU 校验通过 8 个测试、288 提示重建、576 完整标签边界、36 原输入重放与 36 正反映射位置核对，冻结文件可逐字节重建。NCC 主口径为完整标签平均分，不含 EOS；同时报告每个单探针及每个留一探针。

执行命令（仓库根目录）：

```bash
.conda/stage1-p0/bin/python scripts/review/freeze_evidence_label_calibration.py --check
.conda/stage1-p0/bin/python scripts/review/run_evidence_label_calibration.py validate
.conda/stage1-p0/bin/python scripts/review/run_evidence_label_calibration.py run
.conda/stage1-p0/bin/python scripts/review/run_evidence_label_calibration.py check
.conda/stage1-p0/bin/python scripts/review/run_evidence_label_calibration.py analyze
.conda/stage1-p0/bin/python scripts/review/analyze_evidence_label_calibration.py --check
```

评分已在本轮用户授权下完成。最终执行状态与结果由独立的 `../label-calibration-results-v1/current.json` 选择，见[结果解读](../label-calibration-results-v1/INTERPRETATION.md)；冻结指针保留准备时的含义。已完成的 run 不可重新执行 forward。

实际评分顺序由 plan.blocks、schedule 与各 pass 的物理分配回执完整绑定；继承 runtime 中的排序描述保留为历史配置字段。本轮明确先原标签全条件、再分探针、再 A/B 正反映射。无跨案例批量 forward。
