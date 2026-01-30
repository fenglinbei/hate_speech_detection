#!/usr/bin/env bash
set -euo pipefail

# Auto-generated for k=12
K=12
PORT=35012

# You can override these at runtime:
# CUDA_VISIBLE_DEVICES="2,3" bash runner/start_comand/k_ablation/k12.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_ROOT="./models/exps/k_ablation/k${K}"
CKPT_DIR="$(ls -d "${MODEL_ROOT}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"

if [[ -z "${CKPT_DIR}" ]]; then
  echo "[ERROR] No checkpoint found under: ${MODEL_ROOT}/checkpoint-*"
  exit 1
fi

echo "[INFO] Using checkpoint: ${CKPT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  PORT=${PORT}"

python -m vllm.entrypoints.openai.api_server \
  --served-model-name "qwen2.5" \
  --model="${CKPT_DIR}" \
  --trust-remote-code \
  --tensor-parallel-size="2" \
  --port="${PORT}" \
  --max_model_len "8192"
