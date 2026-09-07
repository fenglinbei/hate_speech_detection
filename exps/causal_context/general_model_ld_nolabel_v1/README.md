# Qwen3-8B 显式词典类别字段删除实验

2026-09-07 用户批准的 `nolabel-01` 已完整完成：18 项数值预检全部通过，643 条 dev 的六条件评分已封存，独立分类与 bootstrap 复算通过。守护进程于 08:35:30 UTC 释放，本次运行无残留进程。

结果解释见 [RESULTS.md](RESULTS.md)，完整分类表与主要区间见 [报告](results/nolabel-01/REPORT.md)。

- [冻结协议](../../../docs/research/experiment-plans/general-model-ld-nolabel-v1.md)
- [运行配置](../../../config/stage1/general_model_ld_nolabel_v1.json)
- Plan：`gmlnolabel-3f09715c5e57531565c88e61da5d6bf4087b1cbd9629a5c8579d0f1f8a0c046f`
- Plan SHA256：`f95c7519c1e12729dc59e0700caf3d4d71ba8bc54fdf6c67600a875899c6658f`
- 完整 dev：643 queries、6 conditions、2 tasks、7,716 blocks、131,172 candidates。
- 条件：C0、CLnew、CD、CLDnew、CLnewNoCat、CLDnewNoCat。
- 主指标：Hate Macro-F1、Group Micro-F1；去类别−保留类别在有/无 D 时的四项配对差值及描述性区间。
- 37 项 CPU 输入/统计/执行测试通过；10 项独立生命周期守护测试通过。
- [另一路输入重建核验](audits/input-audit.json)：5,144 个基线上下文完全一致；2,572 个去类别上下文仅移除显式字段；12 个空词典对应输入一致。平均少 36.286 tokens，最多少 101 tokens。

## 执行

GPU 0–3 为四张 L20，每卡一个相同 FP32 Qwen3-8B 副本、真实 forward batch 1。所有预检与 dev 使用原冻结数值 scorer。独立 guard 使用系统 Python；模型仍使用冻结 `.conda/stage1-p0` 环境。系统 `pidfd_open` 实际返回 ENOSYS，guard 沿用已测试的 PID/start-ticks 重核对 fallback，并在状态中记录其检查至信号间存在 PID 重用竞态的限制。

授权为运行至完成，无固定截止时间。guard 接管主进程及其身份范围内的后代；用户停止或主进程异常退出时清理本次进程，guard 丢失时主进程自停。原 14B/27B 队列保持停止。

## 本地运行证据

- `plan_ref.json` 与 `plans/`：冻结输入与来源。
- `runs/nolabel-01/execution.log`、`run_manifest.json`：进度与终态。
- `runs/nolabel-01/preflight/`：18 个预检 pass 与重放核验。
- `runs/nolabel-01/guard/state.json`：独立生命周期守护状态。
- `runs/nolabel-01/dev-b1/`、`analysis/`：预检通过后生成。
- [分类复算与报告导出脚本](audits/report_and_classification_audit.py)：仅完整 run 封存后使用。

与前序约定一致，大型运行产物保留本地。结果校验回执见 `results/nolabel-01/report_manifest.json`；版本库收录协议、实现、测试、主要分类表及复算回执；原始评分、运行目录和两个大型 auxiliary.csv 留在本地。
