# #541 / #3169 注意力实验：GPU 前交付

完整输入、采集、工程检查、未来 GPU 执行器、CPU 分析和交互查看器已实现。本目录不包含真实模型注意力结果。实际设备 allocation=null，GPU 数值资格待取得；未创建绑定、GPU run、队列或空闲轮询。

先打开 [完整材料](MATERIALS.md)、[位置表](positions.tsv)、[查看器](viewer.html) 和 [协议](../../../docs/research/experiment-plans/case-attention-v1/PROTOCOL.md)。查看器在 GPU 未运行时显示真实输入和灰色待测区域，不显示伪造热图。

本次使用当前完整任务指令与单 token“有／无”。两个真实查询、各自完整的 10 个真实示例和 2／6 个原词条；原顺序；六条件各一条，共 12 个 prompt。用户已选择按既有已审核答案展示，4 条示例答案变化可见 MATERIALS.md 和 materials.json；原始标签未覆盖。

主位置：答案前。辅助：词典结束／示例前、示例结束／查询前、查询末尾、查询全句平均、词形位置平均。后两种平均保留全部头，使用 FP64 归约。读取阶段的对比只作描述；未读到的片段为 NA。

工程：72 个完整前向 + 14 个原生截断前缀 + 所有 12 条 greedy 单字/EOS 检查；正常输出时多 12 次前向。通过并正常退出后，显式 full 再做 12 条。常规合计 110 次，上限 182 次。预先固定数值门槛、结果复建、STOP 暂停／显式恢复、损坏检测、终态保护和只读 hook 检查均已实现。

时间估计：在一张空闲 L20 48GB 上，GPU 工程＋正式阶段及 CPU 复核/报告约 **10–20 分钟**；建议留 **30 分钟**窗口。估计包含两阶段加载／权重核验，不含等待空闲或故障排查，不能当成新环境实测吞吐。原始注意力预计约 4–5 GB，建议至少留 8 GB 磁盘；FP32 权重约 32.76 GB，另需运行缓冲。当前没有查询实时 GPU 占用。

## CPU 入口

在仓库根目录，以固定 `.conda/stage1-p0/bin/python` 执行。所有命令加 `PYTHONUTF8=1`。

```bash
PYTHONUTF8=1 .conda/stage1-p0/bin/python scripts/review/run_case_attention_v1.py validate
PYTHONUTF8=1 .conda/stage1-p0/bin/python scripts/review/audit_case_attention_v1.py
PYTHONUTF8=1 CUDA_VISIBLE_DEVICES='' .conda/stage1-p0/bin/python scripts/review/test_case_attention_v1.py
```

## 下一次明确 GPU 窗口中的入口（本次未执行）

先 bind：指定届时核实空闲的 `--gpu INDEX`、新的 `--output` 绑定文件与具体 `--authorization-note`。bind 会核验权重、显存及占用，不能从旧空闲回执推定现在空闲。

```text
run_case_attention_v1.py bind --gpu INDEX --output NEW_BOUND --authorization-note ACTUAL_FUTURE_AUTHORIZATION
run_case_attention_v1.py run --bound NEW_BOUND --run reviews/case-attention-v1/run-01
run_case_attention_v1.py check --run reviews/case-attention-v1/run-01
run_case_attention_v1.py run --bound NEW_BOUND --run reviews/case-attention-v1/run-01 --phase full
run_case_attention_v1.py check --run reviews/case-attention-v1/run-01
run_case_attention_v1.py analyze --run reviews/case-attention-v1/run-01 --output reviews/case-attention-v1/results-01
```

上例的 run 首次只能 engineering，只有通过所有资格、格式检查且正常释放的 qualified 状态才可 full。暂停时保留 STOP，明确决定继续后移走 STOP 并用相同 phase 与 `--resume`；failed/complete 不可重启。任何异常保留现场，不自动降低精度、改阈值或重试。

最终查看器可离线选择 `*.view.json`，或在结果目录用 `python -m http.server --bind 127.0.0.1 PORT` 启动本地读取。查询、条件、角色、层、头、片段层级、密度／总质量和阶段／条件差图均可切换，统一色标可手动指定；可导出 SVG。图形副本 FP32，精确数值使用原始 FP64 注意力数组与 aggregates.json。

研究范围仅为两个已暴露案例。最新任务下是否重现历史错误尚未知；注意力热图不能独立证明适用性机制、词典损害、示例修复或跨样本规律。参考答案只在 raw seal 和正常 worker release 后加入 CPU 分析。
