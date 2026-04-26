#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# MODE:
#   data  -> build_data only
#   train -> build_data + finetune (no inference)
#   full  -> build_data + finetune + vLLM + runner
# =========================================================
MODE="${MODE:-full}"
case "$MODE" in
  data)  DO_BUILD=1; DO_TRAIN=0; DO_INFER=0 ;;
  train) DO_BUILD=1; DO_TRAIN=1; DO_INFER=0 ;;
  full)  DO_BUILD=1; DO_TRAIN=1; DO_INFER=1 ;;
  *)
    echo "[ERROR] Unknown MODE=$MODE (use MODE=data|train|full)" >&2
    exit 1
    ;;
esac

# =========================
# User-configurable options
# =========================
K_START="${K_START:-6}"
K_END="${K_END:-20}"

# ===== multi-seed inference =====
# 逗号分隔，例如：SEEDS="42424242,1,2,3"
SEEDS="${SEEDS:-42,4242,424242,42424242,4242424242}"

# 端口：port = PORT_BASE + k  (例如 k=6 -> 35006)
PORT_BASE="${PORT_BASE:-35000}"

# 训练阶段可见 GPU（需与 finetune config 里的 device_map 对齐）
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 推理(vLLM)阶段可见 GPU
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5}"

# 是否跳过已存在的产物
SKIP_BUILD_IF_EXISTS="${SKIP_BUILD_IF_EXISTS:-1}"   # 1=若 train/val/test 已存在则跳过 build_data
SKIP_TRAIN_IF_EXISTS="${SKIP_TRAIN_IF_EXISTS:-1}"   # 1=若 checkpoint-* 已存在则跳过训练
SKIP_RUN_IF_EXISTS="${SKIP_RUN_IF_EXISTS:-1}"       # 1=若 runner 输出文件已存在则跳过推理评测

# ===== vLLM GPU memory utilization (dynamic) =====
DYNAMIC_GPU_MEM_UTIL="${DYNAMIC_GPU_MEM_UTIL:-1}"     # 1=动态计算 0=固定值
DEFAULT_GPU_MEM_UTIL="${DEFAULT_GPU_MEM_UTIL:-0.90}"  # 动态关闭时使用

GPU_MEM_HEADROOM_MB="${GPU_MEM_HEADROOM_MB:-1200}"    # 预留给碎片/驱动/波动的安全余量
GPU_MEM_UTIL_MARGIN="${GPU_MEM_UTIL_MARGIN:-0.92}"    # 在“可用比例”基础上再乘一个保险系数
GPU_MEM_UTIL_MIN="${GPU_MEM_UTIL_MIN:-0.10}"          # 下限，太小 vLLM 可能不可用/吞吐太差
GPU_MEM_UTIL_MAX="${GPU_MEM_UTIL_MAX:-0.95}"          # 上限，别太激进


# vLLM 启动后等待就绪的最长秒数
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-180}"

# 日志目录
LOG_DIR="${LOG_DIR:-logs/k_ablation}"
mkdir -p "$LOG_DIR"

# =========================
# Helpers
# =========================
parse_csv_list() {
  local s="${1// /}"
  IFS=',' read -r -a arr <<< "$s"
  echo "${arr[@]}"
}


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
    for _ in $(seq 1 10); do
      if kill -0 "$pid" >/dev/null 2>&1; then
        sleep 1
      else
        break
      fi
    done
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

# 写 jq 输出到临时文件，成功后再覆盖目标文件（避免空文件残留）
safe_jq_to_file() {
  local target="$1"
  shift
  local tmp
  tmp="$(mktemp)"
  if jq "$@" > "$tmp"; then
    mv "$tmp" "$target"
  else
    echo "[ERROR] jq failed while writing: $target" >&2
    rm -f "$tmp"
    exit 1
  fi
}

# 解析 CUDA_VISIBLE_DEVICES="2,3" -> 数组 [2 3]
parse_cuda_devices() {
  local s="${1// /}"
  IFS=',' read -r -a arr <<< "$s"
  echo "${arr[@]}"
}

# 读取单张 GPU 的 total/free（单位 MB）
# 输出格式：total free
gpu_mem_total_free_mb() {
  local gpu_id="$1"
  # 例输出: "24576, 18320"
  local line
  line="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "0 0"
    return
  fi
  # shellcheck disable=SC2001
  line="$(echo "$line" | sed 's/ //g')"
  local total free
  total="$(echo "$line" | cut -d',' -f1)"
  free="$(echo "$line" | cut -d',' -f2)"
  echo "$total" "$free"
}

# 基于指定 GPU 列表动态计算 vLLM --gpu-memory-utilization
compute_vllm_gpu_mem_util() {
  local cuda_list="$1"
  local headroom_mb="$2"
  local margin="$3"
  local umin="$4"
  local umax="$5"

  local -a gpus
  read -r -a gpus <<< "$(parse_cuda_devices "$cuda_list")"

  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "$DEFAULT_GPU_MEM_UTIL"
    return
  fi

  # 收集 total/free
  local pairs=()
  for gid in "${gpus[@]}"; do
    read -r total free < <(gpu_mem_total_free_mb "$gid")
    pairs+=("${total}:${free}:${gid}")
  done

  # 用 python 做浮点计算更稳
  python - "$headroom_mb" "$margin" "$umin" "$umax" "${pairs[@]}" <<'PY'
import sys

headroom = float(sys.argv[1])
margin   = float(sys.argv[2])
umin     = float(sys.argv[3])
umax     = float(sys.argv[4])
pairs    = sys.argv[5:]

utils = []
detail = []
for p in pairs:
    total_s, free_s, gid_s = p.split(":")
    total = float(total_s)
    free  = float(free_s)
    gid   = gid_s
    if total <= 0:
        u = 0.0
    else:
        avail = max(0.0, free - headroom)
        u = (avail / total) * margin
    utils.append(u)
    detail.append((gid, total, free, u))

u = min(utils) if utils else umin
u = max(umin, min(umax, u))

# 打印到 stdout：最终利用率（给 bash 接收）
print(f"{u:.3f}")

# 也把明细写到 stderr，方便你看每张卡的状态（不会影响 bash 取值）
for gid, total, free, uu in detail:
    print(f"[GPU-MEM] gpu={gid} total={int(total)}MB free={int(free)}MB -> raw_util={uu:.3f}", file=sys.stderr)
print(f"[GPU-MEM] chosen --gpu-memory-utilization={u:.3f} (headroom={headroom}MB margin={margin})", file=sys.stderr)
PY
}


# =========================
# Preconditions
# =========================
require_cmd python
# 推理阶段才强依赖 jq/curl
if [[ "$DO_INFER" == "1" ]]; then
  require_cmd jq
  require_cmd curl
fi

echo "[INFO] MODE=$MODE (build=$DO_BUILD train=$DO_TRAIN infer=$DO_INFER)"
echo "[INFO] K range: ${K_START}..${K_END}"

# =========================
# Main loop
# =========================
for k in $(seq "$K_START" "$K_END"); do
  echo
  echo "=============================="
  echo "[RUN] k=${k}"
  echo "=============================="

  BUILD_CFG="data/exp_data/k_ablation/k${k}/config.json"
  TRAIN_CFG="finetune/config/k_ablation/k${k}.json"
  RUN_CFG="runner/config/k_ablation/k${k}.json"

  TRAIN_DATA_DIR="data/exp_data/k_ablation/k${k}"
  TRAIN_JSONL="${TRAIN_DATA_DIR}/train.jsonl"
  VAL_JSONL="${TRAIN_DATA_DIR}/val.jsonl"
  TEST_JSON="${TRAIN_DATA_DIR}/test.json"

  MODEL_DIR="models/exps/k_ablation/k${k}"
  RUNNER_OUT_DIR="runner/output/k_ablation"
  RUNNER_OUT_FILE="${RUNNER_OUT_DIR}/k${k}.json"

  PORT=$((PORT_BASE + k))
  VLLM_LOG="${LOG_DIR}/k${k}_vllm_port${PORT}.log"
  BUILD_LOG="${LOG_DIR}/k${k}_build.log"
  TRAIN_LOG="${LOG_DIR}/k${k}_train.log"
  RUN_LOG="${LOG_DIR}/k${k}_runner.log"

  # ---- required files check based on mode ----
  if [[ "$DO_BUILD" == "1" ]]; then
    [[ -f "$BUILD_CFG" ]] || { echo "[ERROR] Missing build config: $BUILD_CFG" >&2; exit 1; }
  fi
  if [[ "$DO_TRAIN" == "1" ]]; then
    [[ -f "$TRAIN_CFG" ]] || { echo "[ERROR] Missing train config: $TRAIN_CFG" >&2; exit 1; }
  fi
  if [[ "$DO_INFER" == "1" ]]; then
    [[ -f "$RUN_CFG" ]] || { echo "[ERROR] Missing runner config: $RUN_CFG" >&2; exit 1; }
  fi

  # =========================
  # 1) Build data
  # =========================
  if [[ "$DO_BUILD" == "1" ]]; then
    if [[ "$SKIP_BUILD_IF_EXISTS" == "1" && -f "$TRAIN_JSONL" && -f "$VAL_JSONL" && -f "$TEST_JSON" ]]; then
      echo "[SKIP] build_data (found ${TRAIN_JSONL}, ${VAL_JSONL}, ${TEST_JSON})"
    else
      echo "[STEP] build_data for k=${k}"
      mkdir -p "$TRAIN_DATA_DIR"
      python src/data/build_data.py --config "$BUILD_CFG" 2>&1 | tee "$BUILD_LOG"
    fi
  else
    echo "[SKIP] build_data (MODE=$MODE)"
  fi

  # =========================
  # 2) Train
  # =========================
  CKPT_DIR=""
  if [[ "$DO_TRAIN" == "1" ]]; then
    if [[ "$SKIP_TRAIN_IF_EXISTS" == "1" ]]; then
      CKPT_DIR="$(latest_checkpoint_dir "$MODEL_DIR" || true)"
    fi

    if [[ -n "$CKPT_DIR" ]]; then
      echo "[SKIP] train (found checkpoint: ${CKPT_DIR})"
    else
      echo "[STEP] finetune for k=${k}"
      mkdir -p "$MODEL_DIR"
      CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
        python src/finetune/train.py --config "$TRAIN_CFG" 2>&1 | tee "$TRAIN_LOG"

      CKPT_DIR="$(latest_checkpoint_dir "$MODEL_DIR")" || {
        echo "[ERROR] No checkpoint found after training under: ${MODEL_DIR}/checkpoint-*" >&2
        exit 1
      }
    fi
  else
    echo "[SKIP] train (MODE=$MODE)"
    # 如果不训练但需要推理，可以在这里选择自动寻找已有 checkpoint
    if [[ "$DO_INFER" == "1" ]]; then
      CKPT_DIR="$(latest_checkpoint_dir "$MODEL_DIR" || true)"
      [[ -n "$CKPT_DIR" ]] || { echo "[ERROR] MODE=$MODE requires inference but no checkpoint found in $MODEL_DIR" >&2; exit 1; }
    fi
  fi

  # =========================
  # 3) Inference (vLLM + runner)
  # =========================
  if [[ "$DO_INFER" == "1" ]]; then
    if [[ "$SKIP_RUN_IF_EXISTS" == "1" && -f "$RUNNER_OUT_FILE" ]]; then
      echo "[SKIP] runner (found output: ${RUNNER_OUT_FILE})"
      continue
    fi

    cleanup

    echo "[STEP] start vLLM for k=${k} on port ${PORT}"
    echo "[INFO] checkpoint=${CKPT_DIR}"

    # ====== 在启动 vLLM 前动态计算 gpu-memory-utilization ======
    GPU_MEM_UTIL="$DEFAULT_GPU_MEM_UTIL"
    if [[ "${DYNAMIC_GPU_MEM_UTIL}" == "1" ]]; then
      # 注意：这里用的是“物理 GPU id 列表”，即 VLLM_CUDA_VISIBLE_DEVICES=2,3
      GPU_MEM_UTIL="$(compute_vllm_gpu_mem_util \
        "$VLLM_CUDA_VISIBLE_DEVICES" \
        "$GPU_MEM_HEADROOM_MB" \
        "$GPU_MEM_UTIL_MARGIN" \
        "$GPU_MEM_UTIL_MIN" \
        "$GPU_MEM_UTIL_MAX")"
    fi

    echo "[INFO] vLLM gpu-memory-utilization=${GPU_MEM_UTIL} (dynamic=${DYNAMIC_GPU_MEM_UTIL})"

    CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
      python -m vllm.entrypoints.openai.api_server \
        --served-model-name "$SERVED_MODEL_NAME" \
        --model "$CKPT_DIR" \
        --trust-remote-code \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --port "$PORT" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        > "$VLLM_LOG" 2>&1 &


    VLLM_PID="$!"
    echo "[INFO] vLLM pid=${VLLM_PID}, log=${VLLM_LOG}"

    if ! wait_vllm_ready "$PORT" "$VLLM_WAIT_SECONDS"; then
      echo "[ERROR] vLLM not ready after ${VLLM_WAIT_SECONDS}s (k=${k}, port=${PORT}). Tail log:" >&2
      tail -n 120 "$VLLM_LOG" >&2 || true
      exit 1
    fi
    echo "[OK] vLLM is ready: http://127.0.0.1:${PORT}/v1/"

    echo "[STEP] runner eval for k=${k}"

    TMP_RUN_CFG="$(mktemp)"
    # 用安全写法避免 jq 失败生成空文件
    safe_jq_to_file "$TMP_RUN_CFG" \
      --arg api "http://127.0.0.1:${PORT}/v1/" \
      --arg outname "k${k}.json" \
      --arg testfile "data/exp_data/k_ablation/k${k}/test.json" \
      '
      .model.params.api_base = $api
      | .output_name = $outname
      | .tester.test_data_file = $testfile
      ' "$RUN_CFG"

    python runner/run.py --config "$TMP_RUN_CFG" 2>&1 | tee "$RUN_LOG"
    rm -f "$TMP_RUN_CFG"

    echo "[STEP] stop vLLM for k=${k}"
    cleanup
    sleep 2
  else
    echo "[SKIP] inference (MODE=$MODE)"
  fi

  echo "[DONE] k=${k} finished."
done

echo
echo "[ALL DONE] MODE=$MODE  k=${K_START}..${K_END}"
echo "[LOGS] ${LOG_DIR}"
