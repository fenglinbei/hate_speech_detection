# 仇恨言论检测项目

[English](README.md) | 中文

本仓库包含结构化仇恨言论检测所需的数据构建、模型微调、推理和评测工具。主流水线会构建四元组风格 prompt，微调指令模型，用 vLLM 启动 checkpoint 服务，并通过 `LLMmetrics` 评测 runner 输出。

## 目录结构

- `data/build_data.py`：从标准四元组数据构建 `train.jsonl`、`val.jsonl` 和 runner 可直接读取的 `test.json`。
- `data/cold_adapter.py`：将 COLD 风格原始数据转换为本项目标准格式。
- `data/exp_data/*/config.json`：数据构建配置。
- `finetune/train.py`：微调入口。
- `finetune/config/*/*.json`：微调配置。
- `runner/run.py`：LLM 推理与指标计算入口。
- `runner/config/*/*.json`：runner 配置。
- `scripts/exps/run_all.sh`：k-ablation 风格实验的一键式循环脚本。
- `scripts/exps/run_one_exp.sh`：运行由 `expctl.py` 生成的单个 `exp_*` 实验目录。
- `scripts/exps/expctl.py`：从 JSON spec 生成自包含实验目录。

## 数据格式

标准化训练和测试数据格式如下：

```json
{
  "id": "sample_id",
  "content": "input text",
  "quadruples": [
    {
      "target": "NULL",
      "argument": "input text or evidence span",
      "targeted_group": "Racism",
      "hateful": "hate"
    }
  ]
}
```

`data/build_data.py` 会把这个格式转换成：

- `train.jsonl`
- `val.jsonl`
- 带有 `id`、`content`、`gt_quadruples`、`messages_list` 的 `test.json`

## 快速启动：主实验

主实验对应 k=10 的 class-quota 设置：

- 数据构建配置：`data/exp_data/k_ablation/k10/config.json`
- 训练配置：`finetune/config/k_ablation/k10.json`
- 推理配置：`runner/config/k_ablation/k10.json`
- 示例数：`srag_top_k = 10`
- 选择策略：`stratified = true`，`weights = null`，即使用 `rag/core.py` 中默认的 class quota

### 前置条件

请在仓库根目录运行命令。一键脚本是 Bash 脚本，因此请使用 Linux、WSL、Git Bash 或其他兼容 Bash 的环境。

需要的运行工具：

- 已安装项目依赖的 Python 环境，包括 `torch`、`transformers`、`sentence-transformers`、`scikit-learn`、`numpy`、`tqdm`、`loguru`、`vllm`
- `curl`
- `nvidia-smi`
- `jq`，供 `scripts/exps/run_all.sh` 使用

需要的本地资源：

- 基座大模型：`models/base/Qwen2.5-7B-Instruct`
- 检索模型：`models/base/bge-large-zh-v1.5`
- 标准数据集：`data/full/std/train.json` 和 `data/full/std/test.json`
- 词典数据：`data/lexicon/annotated_lexicon.json`

### 端到端运行 k=10

在 Bash 中运行：

```bash
K_START=10 K_END=10 MODE=full \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_CUDA_VISIBLE_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
bash scripts/exps/run_all.sh
```

如果在 PowerShell 中启动，需要先设置环境变量，再调用 Bash：

```powershell
$env:K_START="10"
$env:K_END="10"
$env:MODE="full"
$env:TRAIN_CUDA_VISIBLE_DEVICES="0,1,2,3"
$env:VLLM_CUDA_VISIBLE_DEVICES="0,1,2,3"
$env:TENSOR_PARALLEL_SIZE="4"
bash scripts/exps/run_all.sh
```

该命令会依次执行：

1. 使用 `data/build_data.py --config data/exp_data/k_ablation/k10/config.json` 构建数据
2. 使用 `finetune/train.py --config finetune/config/k_ablation/k10.json` 微调模型
3. 在端口 `35010` 启动 vLLM
4. 使用 `runner/run.py --config runner/config/k_ablation/k10.json` 进行推理和评测

### 分阶段运行

只构建数据：

```bash
K_START=10 K_END=10 MODE=data bash scripts/exps/run_all.sh
```

构建数据并训练，但不推理：

```bash
K_START=10 K_END=10 MODE=train bash scripts/exps/run_all.sh
```

完整流水线：

```bash
K_START=10 K_END=10 MODE=full bash scripts/exps/run_all.sh
```

`run_all.sh` 默认会跳过已存在的产物：

- `SKIP_BUILD_IF_EXISTS=1`
- `SKIP_TRAIN_IF_EXISTS=1`
- `SKIP_RUN_IF_EXISTS=1`

如果想强制重跑某个阶段，把对应变量设为 `0`。

### 主实验输出

k=10 的主要输出路径如下：

- 构建后的数据：`data/exp_data/k_ablation/k10/train.jsonl`、`val.jsonl`、`test.json`
- 模型 checkpoint：`models/exps/k_ablation/k10/checkpoint-*`
- 运行日志：`logs/k_ablation/k10_*.log`
- 单 seed runner 输出：`runner/output/k_ablation/k10_s<seed>.json`
- 多 seed 汇总：`runner/output/k_ablation/k10_multi_seed.json`

runner 配置默认使用这些 seed：

```text
42, 4242, 424242, 42424242, 4242424242
```

## 重新生成 k-ablation 配置

如果 k-ablation 配置缺失，可以从 k=1 模板重新生成：

```bash
K_START=2 K_END=20 BASE_K=1 bash scripts/exps/make_config.sh
```

只生成 k=10：

```bash
K_START=10 K_END=10 BASE_K=1 bash scripts/exps/make_config.sh
```

`make_config.sh` 会写出：

- `data/exp_data/k_ablation/k{k}/config.json`
- `finetune/config/k_ablation/k{k}.json`
- `runner/config/k_ablation/k{k}.json`
- `runner/start_comand/k_ablation/k{k}.sh`

## 自包含实验目录

`scripts/exps/expctl.py` 可以从 spec 生成独立的 `exp_*` 实验目录：

```bash
python scripts/exps/expctl.py gen --spec path/to/spec.json
```

每个生成的实验目录都包含：

- `build_config.json`
- `train_config.json`
- `runner_config.json`
- `manifest.json`
- 局部 `data/`、`model/`、`runner_output/`、`logs/`、`progress/`、`prompts/`、`cache/` 目录

运行单个生成的实验：

```bash
MODE=full bash scripts/exps/run_one_exp.sh exps/some_project/exp_xxxxxxxxxx
```

运行某个根目录下的全部生成实验：

```bash
MODE=full bash scripts/exps/run_all_exps.sh exps/some_project
```

当 checkpoint 已经存在时，`run_one_exp.sh` 还支持 `MODE=infer`。

## COLD 固定切分转换

如果原始 COLD 文件已经下载到 `data/cold/raw`，并且包含 `train.csv`、`dev.csv` 或 `val.csv`、`test.csv`，可以直接转换固定 split，不会重新切分：

```bash
python data/cold_adapter.py --raw-dir data/cold/raw --output-dir data/cold/std
```

该命令会写出：

- `data/cold/std/train.json`
- `data/cold/std/val.json`
- `data/cold/std/test.json`

单文件转换仍然可用：

```bash
python data/cold_adapter.py --input path/to/cold.csv --output-dir data/cold/std
```

单文件模式在没有 split 列时可能会重新切分输入；如果已有固定 train/dev/test 文件，优先使用 `--raw-dir`。

## 只运行 Runner 评测

如果已经有 vLLM 或 OpenAI-compatible 服务在运行，可以直接调用 runner：

```bash
python runner/run.py --config runner/config/k_ablation/k10.json
```

请确认 `runner/config/k_ablation/k10.json` 中这些路径/地址正确：

- `model.params.api_base`
- `tester.test_data_file`
- `tester.output_dir`

## Paired Bootstrap

Bootstrap 工具位于 `scripts/paired_bootstrap/`。

单系统置信区间：

```bash
python scripts/paired_bootstrap/paired_bootstrap_llm.py \
  --system-a runner/output/k_ablation/k10_s42.json \
  --n-bootstrap 10000 \
  --seed 42 \
  --output output/paired_bootstrap/k10_s42.json
```

两系统 paired comparison：

```bash
python scripts/paired_bootstrap/paired_bootstrap_llm.py \
  --system-a path/to/system_a.json \
  --system-b runner/output/k_ablation/k10_s42.json \
  --metric f1_avg,f1_target,f1_hate \
  --n-bootstrap 10000 \
  --output output/paired_bootstrap/system_a_vs_k10.json
```

输入应为 runner 输出 JSON，且顶层包含 `results` 列表。

## 常见问题

- 缺少 `tqdm`、`transformers` 或 `sentence_transformers`：在当前 Python 环境中安装项目依赖。
- `jq not found`：安装 `jq`；或者使用 `scripts/exps/run_one_exp.sh`，它通过 Python patch JSON。
- vLLM 长时间未 ready：查看 `logs/k_ablation/k10_vllm_port35010.log`。
- 找不到 checkpoint：先运行 `MODE=train` 或 `MODE=full`；使用 `run_one_exp.sh` 时也可以设置 `MODEL_CKPT_OVERRIDE` 指向已有 checkpoint。
- 端口不对：k=10 默认使用 `PORT_BASE + k`，即 `35010`。
