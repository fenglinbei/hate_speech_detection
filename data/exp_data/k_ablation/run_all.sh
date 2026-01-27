#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-configurable options
# =========================
K_START="${K_START:-6}"
K_END="${K_END:-20}"

# 端口：port = PORT_BASE + k  (例如 k=6 -> 35006)
PORT_BASE="${PORT_BASE:-35000}"

# 训练阶段可见 GPU（需与 finetune config 里的 device_map 对齐）
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 推理(vLLM)阶段可见 GPU
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5}"

# 是否跳过已存在的产物
SKIP_BUILD_IF_EXISTS="${SKIP_BUILD_IF_EXISTS:-1}"   # 1=若 train/val/test 已存在则跳过 build_data
SKIP_TRAIN_IF_EXISTS="${SKIP_TRAIN_IF_EXISTS:-1}"   # 1=若 checkpoint-* 已存在则跳过训练
SKIP_RUN_IF_EXISTS="${SKIP_RUN_IF_EXISTS:-0}"       # 1=若 runner 输出文件已存在则跳过推理评测

# vLLM 启动后等待就绪的最长秒数
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-180}"

# 日志目录
LOG_DIR="${LOG_DIR:-logs/k_ablation}"
mkdir -p "$LOG_DIR"

# =========================
# Helpers
# =========================
require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }
}

latest_checkpoint_dir() {
  local model_root="$1"
  local ckpt
  ckpt="$(ls -d "${model_root}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "$ckpt" ]]; then
    return 1
  fi
  echo "$ckpt"
}

wait_vllm_ready() {
  local port="$1"
  local deadline="$2"
  local url="http://127.0.0.1:${port}/v1/models"
  local i=0
  while [[ $i -lt $deadline ]]; do
    if curl -sS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    i=$((i+1))
  done
  return 1
}

kill_process_tree() {
  local pid="$1"
  if [[ -z "$pid" ]]; then return 0; fi
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill -TERM "$pid" >/dev/null 2>&1 || true
    # 给一点时间优雅退出
    for _ in $(seq 1 10); do
      if kill -0 "$pid" >/dev/null 2>&1; then
        sleep 1
      else
        break
      fi
    done
    # 仍未退出则强杀
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi
}

# 全局清理：避免脚本中断后 vLLM 残留
VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]]; then
    echo "[CLEANUP] Stopping vLLM (pid=${VLLM_PID}) ..."
    kill_process_tree "${VLLM_PID}"
    VLLM_PID=""
  fi
}
trap cleanup EXIT INT TERM

# =========================
# Preconditions
# =========================
require_cmd jq
require_cmd curl
require_cmd python

# =========================
# Main loop
# =========================
for k in $(seq "$K_START" "$K_END"); do
  echo
  echo "=============================="
  echo "[RUN] k=${k}"
  echo "=============================="

  # ---- paths ----
  BUILD_CFG="data/exp_data/k_ablation/k${k}/config.json"
  TRAIN_CFG="finetune/config/k_ablation/k${k}.json"
  RUN_CFG="runner/config/k_ablation/k${k}.json"

  TRAIN_DATA_DIR="data/exp_data/k_ablation/k${k}"
  TRAIN_JSONL="${TRAIN_DATA_DIR}/train.jsonl"
  VAL_JSONL="${TRAIN_DATA_DIR}/val.jsonl"
  TEST_JSON="${TRAIN_DATA_DIR}/test.json"

  MODEL_DIR="models/exps/k_ablation/k${k}"
  RUNNER_OUT_DIR="runner/output/k_ablation"
  RUNNER_OUT_FILE="${RUNNER_OUT_DIR}/k${k}.json"   # 一般 run.py 会写到 output_dir/output_name

  PORT=$((PORT_BASE + k))
  VLLM_LOG="${LOG_DIR}/k${k}_vllm_port${PORT}.log"
  BUILD_LOG="${LOG_DIR}/k${k}_build.log"
  TRAIN_LOG="${LOG_DIR}/k${k}_train.log"
  RUN_LOG="${LOG_DIR}/k${k}_runner.log"

  # ---- sanity checks ----
  [[ -f "$BUILD_CFG" ]] || { echo "[ERROR] Missing build config: $BUILD_CFG" >&2; exit 1; }
  [[ -f "$TRAIN_CFG" ]] || { echo "[ERROR] Missing train config: $TRAIN_CFG" >&2; exit 1; }
  [[ -f "$RUN_CFG" ]] || { echo "[ERROR] Missing runner config: $RUN_CFG" >&2; exit 1; }

  # =========================
  # 1) Build data
  # =========================
  if [[ "$SKIP_BUILD_IF_EXISTS" == "1" && -f "$TRAIN_JSONL" && -f "$VAL_JSONL" && -f "$TEST_JSON" ]]; then
    echo "[SKIP] build_data (found ${TRAIN_JSONL}, ${VAL_JSONL}, ${TEST_JSON})"
  else
    echo "[STEP] build_data for k=${k}"
    mkdir -p "$TRAIN_DATA_DIR"
    python data/build_data.py --config "$BUILD_CFG" 2>&1 | tee "$BUILD_LOG"
  fi

  # =========================
  # 2) Train
  # =========================
  CKPT_DIR=""
  if [[ "$SKIP_TRAIN_IF_EXISTS" == "1" ]]; then
    CKPT_DIR="$(latest_checkpoint_dir "$MODEL_DIR" || true)"
  fi

  if [[ -n "$CKPT_DIR" ]]; then
    echo "[SKIP] train (found checkpoint: ${CKPT_DIR})"
  else
    echo "[STEP] finetune for k=${k}"
    mkdir -p "$MODEL_DIR"
    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
      python finetune/train.py --config "$TRAIN_CFG" 2>&1 | tee "$TRAIN_LOG"

    CKPT_DIR="$(latest_checkpoint_dir "$MODEL_DIR")" || {
      echo "[ERROR] No checkpoint found after training under: ${MODEL_DIR}/checkpoint-*" >&2
      exit 1
    }
  fi

  # =========================
  # 3) Start vLLM
  # =========================
  if [[ "$SKIP_RUN_IF_EXISTS" == "1" && -f "$RUNNER_OUT_FILE" ]]; then
    echo "[SKIP] runner (found output: ${RUNNER_OUT_FILE})"
    continue
  fi

  # 确保没有残留 vLLM
  cleanup

  echo "[STEP] start vLLM for k=${k} on port ${PORT}"
  echo "[INFO] checkpoint=${CKPT_DIR}"
  # 后台启动 vLLM
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
    python -m vllm.entrypoints.openai.api_server \
      --served-model-name "$SERVED_MODEL_NAME" \
      --model "$CKPT_DIR" \
      --trust-remote-code \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --port "$PORT" \
      --max_model_len "$MAX_MODEL_LEN" \
      > "$VLLM_LOG" 2>&1 &

  VLLM_PID="$!"
  echo "[INFO] vLLM pid=${VLLM_PID}, log=${VLLM_LOG}"

  # 等待服务就绪
  if ! wait_vllm_ready "$PORT" "$VLLM_WAIT_SECONDS"; then
    echo "[ERROR] vLLM not ready after ${VLLM_WAIT_SECONDS}s (k=${k}, port=${PORT}). Tail log:" >&2
    tail -n 80 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  echo "[OK] vLLM is ready: http://127.0.0.1:${PORT}/v1/"

  # =========================
  # 4) Runner eval (patch api_base on the fly)
  # =========================
  echo "[STEP] runner eval for k=${k}"

  TMP_RUN_CFG="$(mktemp)"
  jq \
    --arg api "http://127.0.0.1:${PORT}/v1/" \
    --arg outname "k${k}.json" \
    --arg testfile "data/exp_data/k_ablation/k${k}/test.json" \
    '
    .model.params.api_base = $api
    | .output_name = $outname
    | .tester.test_data_file = $testfile
    ' "$RUN_CFG" > "$TMP_RUN_CFG"

  python runner/run.py --config "$TMP_RUN_CFG" 2>&1 | tee "$RUN_LOG"
  rm -f "$TMP_RUN_CFG"

  # =========================
  # 5) Stop vLLM
  # =========================
  echo "[STEP] stop vLLM for k=${k}"
  cleanup
  # 给端口释放一点缓冲
  sleep 2

  echo "[DONE] k=${k} finished."
done

echo
echo "[ALL DONE] k=${K_START}..${K_END} finished."
echo "[LOGS] ${LOG_DIR}"
