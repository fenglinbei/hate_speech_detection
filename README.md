# Hate Speech Detection Utilities

## Paired Bootstrap for LLM Runner Outputs

脚本位置：`scripts/paired_bootstrap_llm.py`

这个脚本对 LLM runner 输出做 test instance 级别的 bootstrap。它复用 `metrics/core.py` 和 `metrics/metric_llm.py` 的评估逻辑，适合给单个系统估计置信区间，也适合对两个系统做 paired bootstrap 显著性比较。

默认会一次输出所有支持的 bootstrap 指标：

- `f1_hard`
- `f1_soft`
- `f1_avg`
- `f1_target`
- `f1_hate`

其中 `f1_target` 对应 `metric_llm.py` 中 `field_metrics.targeted_group.f1` 的逻辑，`f1_hate` 对应 `field_metrics.hateful.f1` 的逻辑。

### 启动方式

在仓库根目录运行。推荐使用 Python 3.9+；如果本地有仓库虚拟环境，可以直接用 `.venv\Scripts\python.exe`。

单系统：输出所有指标的 bootstrap 置信区间。

```powershell
.\.venv\Scripts\python.exe scripts\paired_bootstrap_llm.py `
  --system-a exps\ablation\wo_semantic_match\exp_3992dbfb11\runner_output\exp_3992dbfb11_s42.json `
  --n-bootstrap 10000 `
  --seed 42 `
  --output exps\ablation\wo_semantic_match\exp_3992dbfb11\runner_output\bootstrap_s42_all.json
```

双系统：两个系统必须包含同一批 test instance `id`，脚本会在每轮 bootstrap 中对两个系统使用同一组重采样 index。

```powershell
.\.venv\Scripts\python.exe scripts\paired_bootstrap_llm.py `
  --system-a exps\ablation\wo_semantic_match\exp_3992dbfb11\runner_output\exp_3992dbfb11_s42.json `
  --system-b runner\output\k_ablation\k10_s42.json `
  --n-bootstrap 10000 `
  --seed 42 `
  --output output\paired_bootstrap\wo_semantic_vs_main.json
```

只分析部分指标时，用逗号传给 `--metric`：

```powershell
.\.venv\Scripts\python.exe scripts\paired_bootstrap_llm.py `
  --system-a exps\ablation\wo_semantic_match\exp_3992dbfb11\runner_output\exp_3992dbfb11_s42.json `
  --system-b runner\output\k_ablation\k10_s42.json `
  --metric f1_avg,f1_target,f1_hate `
  --n-bootstrap 10000 `
  --output output\paired_bootstrap\wo_semantic_vs_main_selected.json
```

项目中也提供了一个启动脚本：

```bash
bash scripts/run_paired_bootstrap_wo_sem_vs_main.sh
```

它默认输出所有指标。可通过环境变量覆盖参数，例如：

```bash
METRIC=f1_avg,f1_target,f1_hate N_BOOTSTRAP=20000 bash scripts/run_paired_bootstrap_wo_sem_vs_main.sh
```

### 输入格式

输入文件应是 runner 输出 JSON，至少包含顶层 `results` 列表。每条样本需要有：

- `id`：test instance id；双系统比较时按这个字段配对。
- `gt_quadruples`：标准四元组列表。
- `pred_quadruples`：预测四元组列表。
- `status`：可选；用于统计 `success` 和 `success_rate`，不影响 F1 计算。

脚本的 bootstrap 单位是 test instance。每轮会从 `results` 的样本索引中有放回抽样，然后把抽到样本的 tuple-level TP/FP/FN 相加，再计算 micro precision/recall/F1。

### 输出指标

脚本会向 stdout 打印 JSON；如果传了 `--output`，也会把同一份 JSON 写入文件。

通用字段：

- `mode`：`single` 或 `paired`。
- `metric`：原始指标选择。默认是 `all`。
- `metrics`：实际输出的规范化指标列表。
- `n_instances`：参与 bootstrap 的 test instance 数。
- `n_bootstrap`：bootstrap 重采样次数。
- `seed`：随机种子。
- `similarity_threshold`：soft match 的 target/argument 相似度阈值。
- `ci_method`：当前为 percentile bootstrap。

`system_a` / `system_b` 字段：

- `path`：输入文件路径。
- `success`、`success_rate`：runner 成功数与成功率。
- `observed`：原始完整 test set 上的指标，包括：
  - `hard_precision`、`hard_recall`、`f1_hard`
  - `soft_precision`、`soft_recall`、`f1_soft`
  - `f1_avg = (f1_hard + f1_soft) / 2`
  - `target_precision`、`target_recall`、`f1_target`
  - `hate_precision`、`hate_recall`、`f1_hate`
- `bootstrap.<metric>`：每个输出指标的 bootstrap 摘要：
  - `observed`：原始完整 test set 上的指标值。
  - `mean`、`std`：bootstrap 分布均值与标准差。
  - `ci_lower`、`ci_upper`：percentile bootstrap 置信区间。

双系统模式额外包含 `comparison`：

- `delta_label`：当前固定为 `system_a_minus_system_b`。
- `comparison.<metric>.observed_delta`：原始完整 test set 上的 A-B 指标差。
- `mean_delta`、`std_delta`：bootstrap 差值分布的均值与标准差。
- `ci_lower`、`ci_upper`：A-B 差值的置信区间；若区间跨 0，通常说明差异不稳定。
- `prob_a_better`：bootstrap 样本中 A-B > 0 的比例。
- `prob_b_better`：bootstrap 样本中 A-B < 0 的比例。
- `p_value_two_sided`：基于 bootstrap 差值符号的双侧 p 值近似，越小表示 A/B 差异越稳定。

### 参数解释

| 参数 | 是否必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--system-a` | 是 | 无 | 第一个 runner 输出 JSON。单系统模式只需要这个参数。 |
| `--system-b` | 否 | 无 | 第二个 runner 输出 JSON。传入后进入 paired 双系统模式。 |
| `--metric` | 否 | `all` | 要 bootstrap 的指标。支持 `all` 或逗号分隔列表；别名包括 `f1_hard`/`hard_f1`、`f1_soft`/`soft_f1`、`f1_avg`/`avg_f1`、`f1_target`/`target_f1`/`targeted_group_f1`、`f1_hate`/`hate_f1`/`hateful_f1`。 |
| `--n-bootstrap` | 否 | `10000` | bootstrap 重采样次数。论文或正式报告建议至少 10000。 |
| `--seed` | 否 | `42` | 随机种子，用于复现实验。 |
| `--similarity-threshold` | 否 | `0.5` | soft match 中 target 和 argument 的 `SequenceMatcher` 相似度阈值。 |
| `--ci` | 否 | `95.0` | percentile bootstrap 置信区间宽度。 |
| `--output` | 否 | 无 | 输出 JSON 文件路径；不传则只打印到 stdout。 |
| `--indent` | 否 | `2` | JSON 缩进。设为 `0` 时输出单行 JSON。 |

### 匹配定义

- Hard match：`target`、`argument`、`targeted_group`、`hateful` 四个字段完全匹配，逻辑对应 `metrics/core.py` 的 `convert_quad` 和 `is_hard_match`。
- Soft match：`targeted_group` 和 `hateful` 必须匹配，`target` 与 `argument` 的字符串相似度都必须大于 `--similarity-threshold`，逻辑对应 `metrics/core.py` 的 `preprocess_quad` 和 `is_soft_match`。
- Target-F1：先用 `target` 和 `argument` 的相似度对预测四元组与标准四元组做贪心对齐，再评估 `targeted_group` 字段是否一致；未对齐预测计为 FP，未对齐标准答案计为 FN。
- Hate-F1：使用同一套对齐结果，评估 `hateful` 字段是否一致；未对齐预测计为 FP，未对齐标准答案计为 FN。
- 所有 F1 都是 tuple-level micro-F1，不是逐样本 F1 的平均。
