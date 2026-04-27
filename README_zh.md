# 仇恨言论检测项目

[English](README.md) | 中文

本仓库包含结构化仇恨言论检测所需的数据构建、模型微调、推理和评测工具。主流水线会构建四元组风格 prompt，微调指令模型，用 vLLM 启动 checkpoint 服务，并通过 `LLMmetrics` 评测 runner 输出。

## 目录结构

- `src/data/build_data.py`：从标准四元组数据构建 `train.jsonl`、`val.jsonl` 和 runner 可直接读取的 `test.json`。
- `src/data/cold_adapter.py`：将 COLD 风格原始数据转换为本项目标准格式。
- `data/exp_data/*/config.json`：数据构建配置。
- `src/finetune/train.py`：微调入口。
- `config/finetune/*/*.json`：微调配置。
- `src/runner/run.py`：LLM 推理与指标计算入口。
- `config/runner/*/*.json`：runner 配置。
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

`src/data/build_data.py` 会把这个格式转换成：

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

- 已安装项目依赖的 Python 环境，包括 `torch`、`transformers`、`accelerate`、`deepspeed`、`sentence-transformers`、`scikit-learn`、`numpy`、`tqdm`、`loguru`、`vllm`
- `curl`
- `nvidia-smi`
- `jq`，供 `scripts/exps/run_all.sh` 使用

需要的本地资源：

- 基座大模型：`models/base/Qwen2.5-7B-Instruct`
- 检索模型：`models/base/bge-large-zh-v1.5`
- 标准数据集：`data/full/std/train.json` 和 `data/full/std/test.json`
- 词典数据：`data/lexicon/annotated_lexicon.json`

### 推荐 Conda 环境

对于 4x RTX 4090、NVIDIA 580.x 驱动的 Linux 服务器，建议使用全新的 conda
环境，并通过 `uv pip` 安装 PyTorch/vLLM 栈。这个环境里不要用 conda 安装
PyTorch；vLLM 对 PyTorch、CUDA、NCCL 的二进制组合比较敏感。

即使 `nvidia-smi` 显示 CUDA 13.0，这里也推荐安装 CUDA 12.8 wheel 栈。580.x
驱动可以运行 CUDA 12.8 runtime，而且 CUDA 12.8 是 vLLM 预编译 wheel 的默认目标。

创建环境：

```bash
conda create -n hsd-cu128 python=3.12 -y
conda activate hsd-cu128

conda install -y -c conda-forge git git-lfs curl jq cmake ninja packaging
python -m pip install -U pip uv setuptools wheel
```

安装 vLLM 以及匹配的 PyTorch CUDA 12.8 依赖：

```bash
uv pip install "vllm==0.11.1" --torch-backend=cu128
```

安装项目运行依赖：

```bash
uv pip install \
  "transformers>=4.51,<5" \
  "datasets>=2.19" \
  "accelerate>=0.33" \
  "deepspeed>=0.14" \
  "sentence-transformers>=3" \
  "modelscope>=1.18" \
  swanlab \
  pandas scikit-learn "numpy<3" tqdm loguru requests \
  faiss-cpu \
  matplotlib matplotlib-venn \
  fastapi "pydantic>=2,<3" uvicorn
```

如果是在已有环境上更新，单独安装这次新增的分布式训练依赖即可：

```bash
uv pip install "deepspeed>=0.14"
```

`src/finetune/train.py` 会用 `attn_implementation="flash_attention_2"` 加载模型，因此还需要安装
FlashAttention。如果环境里还没有 `nvcc`，先把 CUDA 12.8 编译组件安装到 conda 环境：

```bash
conda install -y -c nvidia/label/cuda-12.8.1 \
  cuda-nvcc cuda-libraries-dev cuda-nvtx cuda-cupti

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8

uv pip install --no-build-isolation "flash-attn==2.8.3"
```

验证环境：

```bash
python - <<'PY'
import accelerate, deepspeed, torch, vllm, flash_attn, faiss
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("accelerate ok")
print("deepspeed ok")
print("vllm ok")
print("flash-attn ok")
print("faiss ok")
PY
```

这台 4x RTX 4090 服务器推荐使用如下运行参数：

```bash
MODE=full \
TRAIN_BACKEND=deepspeed \
TRAIN_PROFILE=ds_zero2_safe \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_CUDA_VISIBLE_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
MAX_MODEL_LEN=8192 \
bash scripts/exps/run_one_exp.sh exps/some_project/exp_xxxxxxxxxx
```

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

1. 使用 `src/data/build_data.py --config data/exp_data/k_ablation/k10/config.json` 构建数据
2. 使用 `src/finetune/train.py --config finetune/config/k_ablation/k10.json` 微调模型
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

通用实验流程是：

```text
spec.json -> scripts/exps/expctl.py -> output_root/exp_<id>/ -> run_one_exp.sh -> build/train/vLLM/runner
```

这是现在推荐用于新 sweep 和 ablation 的流程。较早的
`scripts/exps/make_config.sh` 和 `scripts/exps/run_all.sh` 是 k-ablation
专用脚本，会把配置分散写到 `data/exp_data/`、`finetune/config/` 和
`runner/config/` 下。

### 1. 编写或复用 spec

spec 示例在 `exps/specs/` 下。一个 spec 会描述模板配置、参数网格和运行时默认值：

- `project`：实验组名称。
- `output_root`：生成的 `exp_*` 目录所在根目录。
- `base.build`、`base.train`、`base.runner`：要复制并修改的三个模板配置。
- `grid`：用 dot-path 覆盖配置字段，并做笛卡尔积展开。支持前缀
  `build.`、`train.`、`runner.`、`reuse.`。
- `port_base`：每个生成实验的端口是 `port_base + index`。
- `train_cuda_visible_devices`：训练阶段默认 GPU。
- `vllm`：vLLM 默认运行参数，例如可见 GPU、tensor parallel size、最大模型长度和 served model name。
- `reuse.data_dir`、`reuse.model_checkpoint`：可选的复用数据目录或 checkpoint。

示例：

```bash
python scripts/exps/expctl.py gen --spec exps/specs/example.json
```

`expctl.py` 会读取三份 base 配置，对每个 grid 组合应用覆盖项，然后为每个组合写出一个确定性的
`exp_<hash>` 目录。hash 来自 `project` 和 `overrides`，所以同一个 spec 通常会生成稳定的目录名。

### 2. 查看生成目录

每个生成的实验目录都是自包含的：

- `build_config.json`
- `train_config.json`
- `runner_config.json`
- `manifest.json`
- 局部 `data/`、`model/`、`runner_output/`、`logs/`、`progress/`、`prompts/`、`cache/` 目录

`expctl.py` 还会自动 patch 路径，让每个实验写到自己的目录下：

- build 输出写到 `exp_*/data/train.jsonl`、`val.jsonl`、`test.json`。
- finetune checkpoint 写到 `exp_*/model/`。
- runner output、progress、prompts、cache 都写到同一个 `exp_*` 目录下。
- `runner_config.json` 里的 `api_base` 先保留占位端口，真正端口由 `run_one_exp.sh` 运行时 patch。
- `manifest.json` 记录 overrides、端口、复用路径、生成路径和默认运行参数。

### 3. 运行单个实验

```bash
MODE=full bash scripts/exps/run_one_exp.sh exps/some_project/exp_xxxxxxxxxx
```

支持的模式：

- `MODE=data`：只构建数据；如果复用数据或数据已经存在，会跳过。
- `MODE=train`：构建数据，然后微调。
- `MODE=full`：构建数据、微调、启动 vLLM，然后运行 runner 评测。
- `MODE=infer`：只启动 vLLM 并运行 runner，需要已有 checkpoint。

`run_one_exp.sh` 会读取 `manifest.json`，按顺序执行：

1. `python src/data/build_data.py --config build_config.json`
2. `python -m torch.distributed.run ... src/finetune/train.py --config <temporary_train_config>`
3. `python -m vllm.entrypoints.openai.api_server ...`
4. `python runner/run.py --config <temporary_runner_config>`

训练配置和 runner 配置都会先复制到临时文件。训练临时配置会按所选分布式
profile patch，runner 临时配置会把 `model.params.api_base` patch 成实际端口。
如果设置了 `reuse.data_dir`，训练和测试数据路径会指向复用目录。如果设置了
`reuse.model_checkpoint`，训练会被跳过，除非显式设置 `FORCE_TRAIN=1`。

full finetune 的分布式训练由这些变量控制：

- `TRAIN_BACKEND=deepspeed|fsdp|single`，默认 `deepspeed`。
- `TRAIN_PROFILE=ds_zero2_safe|ds_zero2_bs1|ds_zero3_safe|ds_zero3_bs1|ds_zero3_offload|fsdp_safe|single`。
- `TRAIN_NPROC_PER_NODE` 默认等于 `TRAIN_CUDA_VISIBLE_DEVICES` 里的 GPU 数量。
- `TRAIN_MASTER_PORT` 默认等于 `PORT + 1000`。
- `TRAIN_MAX_STEPS=2` 可用于短 smoke test；正式训练时不要设置。

各 profile 的含义：

- `ds_zero2_safe`：ZeRO-2，无 CPU offload，micro-batch 2，gradient accumulation 1。
- `ds_zero2_bs1`：ZeRO-2，无 CPU offload，micro-batch 1，gradient accumulation 2。
- `ds_zero3_safe`：ZeRO-3，无 CPU offload，micro-batch 2，gradient accumulation 1。
- `ds_zero3_bs1`：ZeRO-3，无 CPU offload，micro-batch 1，gradient accumulation 2。
- `ds_zero3_offload`：ZeRO-3 + CPU offload，作为能跑通但较慢的显存兜底方案。
- `fsdp_safe`：PyTorch FSDP full-shard，并按 Qwen2 decoder layer 自动 wrap。
- `single`：旧的单进程调试路径，会保留 `device_map`。

2-step 分布式 smoke test 示例：

```bash
MODE=train \
TRAIN_MAX_STEPS=2 \
TRAIN_BACKEND=deepspeed \
TRAIN_PROFILE=ds_zero2_safe \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/exps/run_one_exp.sh exps/some_project/exp_xxxxxxxxxx
```

训练阶段会在 `exp_*/logs/` 下写出复现实验所需文件：
`train_runtime_config.json`、DeepSpeed profile 对应的 `ds_config_<profile>.json`、
`train.<backend>.<profile>.log`，以及兼容旧路径的 `train.log`。

常用运行时覆盖：

```bash
MODE=infer \
MODEL_CKPT_OVERRIDE=models/some_run/checkpoint-100 \
DATA_DIR_OVERRIDE=exps/some_project/exp_xxxxxxxxxx/data \
PORT=35010 \
VLLM_CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/exps/run_one_exp.sh exps/some_project/exp_xxxxxxxxxx
```

常用变量包括 `TRAIN_CUDA_VISIBLE_DEVICES`、`VLLM_CUDA_VISIBLE_DEVICES`、
`TRAIN_BACKEND`、`TRAIN_PROFILE`、`TRAIN_NPROC_PER_NODE`、`TRAIN_MASTER_PORT`、
`TRAIN_MAX_STEPS`、`TENSOR_PARALLEL_SIZE`、`MAX_MODEL_LEN`、`SERVED_MODEL_NAME`、
`DYNAMIC_GPU_MEM_UTIL`、`DEFAULT_GPU_MEM_UTIL`。

### 4. 运行某个根目录下的全部实验

```bash
MODE=full bash scripts/exps/run_all_exps.sh exps/some_project
```

`run_all_exps.sh` 会找到给定根目录下第一层的所有 `exp_*` 子目录，并用相同
`MODE` 顺序调用 `run_one_exp.sh`。

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
