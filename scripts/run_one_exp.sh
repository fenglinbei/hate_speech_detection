#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODE=full bash scripts/run_one_exp.sh /abs/or/rel/path/to/exp_dir
#
# Modes:
#   data  : build_data only (unless reuse data)
#   train : build_data + finetune (build may be skipped)
#   full  : build_data + finetune + vLLM + runner
#   infer : vLLM + runner only (requires model checkpoint)
MODE="${MODE:-full}"

EXP_DIR="${1:-${EXP_DIR:-}}"
if [[ -z "$EXP_DIR" ]]; then
  echo "[ERROR] Missing EXP_DIR. Usage: bash scripts/run_one_exp.sh <exp_dir>" >&2
  exit 1
fi
EXP_DIR="$(python - <<PY
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
"$EXP_DIR")"

MANIFEST="${EXP_DIR}/manifest.json"
BUILD_CFG="${EXP_DIR}/build_config.json"
TRAIN_CFG="${EXP_DIR}/train_config.json"
RUNNER_CFG="${EXP_DIR}/runner_config.json"

[[ -f "$MANIFEST" ]] || { echo "[ERROR] missing $MANIFEST" >&2; exit 1; }
[[ -f "$BUILD_CFG" ]] || { echo "[ERROR] missing $BUILD_CFG" >&2; exit 1; }
[[ -f "$TRAIN_CFG" ]] || { echo "[ERROR] missing $TRAIN_CFG" >&2; exit 1; }
[[ -f "$RUNNER_CFG" ]] || { echo "[ERROR] missing $RUNNER_CFG" >&2; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing command: $1" >&2; exit 1; }; }

require_cmd python
require_cmd curl
require_cmd nvidia-smi

# ===== Dynamic vLLM memory util (optional) =====
DYNAMIC_GPU_MEM_UTIL="${DYNAMIC_GPU_MEM_UTIL:-1}"
DEFAULT_GPU_MEM_UTIL="${DEFAULT_GPU_MEM_UTIL:-0.90}"
GPU_MEM_HEADROOM_MB="${GPU_MEM_HEADROOM_MB:-1200}"
GPU_MEM_UTIL_MARGIN="${GPU_MEM_UTIL_MARGIN:-0.92}"
GPU_MEM_UTIL_MIN="${GPU_MEM_UTIL_MIN:-0.10}"
GPU_MEM_UTIL_MAX="${GPU_MEM_UTIL_MAX:-0.95}"

parse_csv_list() { local s="${1// /}"; IFS=',' read -r -a arr <<< "$s"; echo "${arr[@]}"; }

gpu_mem_total_free_mb() {
  local gpu_id="$1"
  local line
  line="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
  if [[ -z "$line" ]]; then echo "0 0"; return; fi
  line="$(echo "$line" | sed 's/ //g')"
  local total free
  total="$(echo "$line" | cut -d',' -f1)"
  free="$(echo "$line" | cut -d',' -f2)"
  echo "$total" "$free"
}

compute_vllm_gpu_mem_util() {
  local cuda_list="$1" headroom_mb="$2" margin="$3" umin="$4" umax="$5"
  local -a gpus; read -r -a gpus <<< "$(parse_csv_list "$cuda_list")"
  local pairs=()
  for gid in "${gpus[@]}"; do
    read -r total free < <(gpu_mem_total_free_mb "$gid")
    pairs+=("${total}:${free}:${gid}")
  done
  python - "$headroom_mb" "$margin" "$umin" "$umax" "${pairs[@]}" <<'PY'
import sys, statistics
headroom=float(sys.argv[1]); margin=float(sys.argv[2]); umin=float(sys.argv[3]); umax=float(sys.argv[4])
pairs=sys.argv[5:]
utils=[]
for p in pairs:
    total_s, free_s, gid = p.split(":")
    total=float(total_s); free=float(free_s)
    if total<=0: u=0.0
    else:
        avail=max(0.0, free-headroom)
        u=(avail/total)*margin
    utils.append(u)
u=min(utils) if utils else umin
u=max(umin, min(umax, u))
print(f"{u:.3f}")
PY
}

# Read manifest runtime defaults
read_manifest_field() {
  local key="$1"
  python - <<PY
import json,sys
m=json.load(open("${MANIFEST}","r",encoding="utf-8"))
cur=m
for p in sys.argv[1].split("."):
    cur=cur.get(p)
print(cur if cur is not None else "")
PY
"$key"
}

PORT="${PORT:-$(read_manifest_field port)}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-$(read_manifest_field runtime_defaults.train_cuda_visible_devices)}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-$(read_manifest_field runtime_defaults.vllm.cuda_visible_devices)}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$(read_manifest_field runtime_defaults.vllm.tensor_parallel_size)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(read_manifest_field runtime_defaults.vllm.max_model_len)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(read_manifest_field runtime_defaults.vllm.served_model_name)}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-180}"

LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"
mkdir -p "$LOG_DIR"
BUILD_LOG="${LOG_DIR}/build.log"
TRAIN_LOG="${LOG_DIR}/train.log"
VLLM_LOG="${LOG_DIR}/vllm_port${PORT}.log"
RUN_LOG="${LOG_DIR}/runner.log"

# Overrides for reusing existing data/model (runtime)
DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-}"
MODEL_CKPT_OVERRIDE="${MODEL_CKPT_OVERRIDE:-}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"  # 若 MODEL_CKPT_OVERRIDE 存在但你仍想训练，设 FORCE_TRAIN=1

# Manifest reuse
MANIFEST_REUSE_DATA="$(read_manifest_field reuse.data_dir)"
MANIFEST_REUSE_MODEL="$(read_manifest_field reuse.model_checkpoint)"

# Resolve reuse choices (runtime override > manifest)
REUSE_DATA_DIR="${DATA_DIR_OVERRIDE:-$MANIFEST_REUSE_DATA}"
REUSE_MODEL_CKPT="${MODEL_CKPT_OVERRIDE:-$MANIFEST_REUSE_MODEL}"

# Some convenient derived paths
EXP_DATA_DIR="$(read_manifest_field paths.data_dir)"
EXP_MODEL_DIR="$(read_manifest_field paths.model_dir)"
RUNNER_OUT_DIR="$(read_manifest_field paths.runner_output_dir)"

# helper: patch JSON via python (avoid jq)
json_patch() {
  local src="$1"
  local dst="$2"
  shift 2
  python - "$src" "$dst" "$@" <<'PY'
import json,sys,os
src=sys.argv[1]; dst=sys.argv[2]
pairs=sys.argv[3:]
obj=json.load(open(src,"r",encoding="utf-8"))

def set_path(o, path, val):
    parts=path.split(".")
    cur=o
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p]={}
        cur=cur[p]
    cur[parts[-1]]=val

it=iter(pairs)
for k,v in zip(it,it):
    # simple type inference
    if v.lower()=="true": vv=True
    elif v.lower()=="false": vv=False
    else:
        try:
            if "." in v: vv=float(v)
            else: vv=int(v)
        except:
            vv=v
    set_path(obj,k,vv)

os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(obj, open(dst,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
PY
}

latest_checkpoint_dir() {
  local model_root="$1"
  ls -d "${model_root}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

# vLLM lifecycle
VLLM_PID=""
cleanup() {
  if [[ -n "${VLLM_PID}" ]]; then
    echo "[CLEANUP] stop vLLM pid=${VLLM_PID}"
    kill -TERM "${VLLM_PID}" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL "${VLLM_PID}" >/dev/null 2>&1 || true
    VLLM_PID=""
  fi
}
trap cleanup EXIT INT TERM

wait_vllm_ready() {
  local port="$1" deadline="$2"
  local url="http://127.0.0.1:${port}/v1/models"
  local i=0
  while [[ $i -lt $deadline ]]; do
    if curl -sS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1; i=$((i+1))
  done
  return 1
}

echo "[INFO] EXP_DIR=$EXP_DIR"
echo "[INFO] MODE=$MODE  PORT=$PORT"
echo "[INFO] TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "[INFO] VLLM_CUDA_VISIBLE_DEVICES=$VLLM_CUDA_VISIBLE_DEVICES"
echo "[INFO] REUSE_DATA_DIR=${REUSE_DATA_DIR:-<none>}"
echo "[INFO] REUSE_MODEL_CKPT=${REUSE_MODEL_CKPT:-<none>}"

# ===== Stage selection =====
case "$MODE" in
  data)  DO_BUILD=1; DO_TRAIN=0; DO_INFER=0 ;;
  train) DO_BUILD=1; DO_TRAIN=1; DO_INFER=0 ;;
  full)  DO_BUILD=1; DO_TRAIN=1; DO_INFER=1 ;;
  infer) DO_BUILD=0; DO_TRAIN=0; DO_INFER=1 ;;
  *) echo "[ERROR] unknown MODE=$MODE (data/train/full/infer)"; exit 1 ;;
esac

# ===== 1) build_data =====
if [[ "$DO_BUILD" == "1" ]]; then
  if [[ -n "${REUSE_DATA_DIR}" ]]; then
    echo "[SKIP] build_data because REUSE_DATA_DIR is set: $REUSE_DATA_DIR"
  else
    # 如果已有产物则可跳过
    if [[ -f "${EXP_DATA_DIR}/train.jsonl" && -f "${EXP_DATA_DIR}/val.jsonl" && -f "${EXP_DATA_DIR}/test.json" ]]; then
      echo "[SKIP] build_data (found existing data in ${EXP_DATA_DIR})"
    else
      echo "[STEP] build_data"
      python data/build_data.py --config "$BUILD_CFG" 2>&1 | tee "$BUILD_LOG"
    fi
  fi
fi

# ===== 2) train =====
CKPT_DIR=""
if [[ "$DO_TRAIN" == "1" ]]; then
  if [[ -n "${REUSE_MODEL_CKPT}" && "$FORCE_TRAIN" != "1" ]]; then
    echo "[SKIP] train because REUSE_MODEL_CKPT is set (set FORCE_TRAIN=1 to override)"
  else
    echo "[STEP] train"
    TMP_TRAIN_CFG="$(mktemp)"
    if [[ -n "${REUSE_DATA_DIR}" ]]; then
      # patch train/val paths to reuse data
      json_patch "$TRAIN_CFG" "$TMP_TRAIN_CFG" \
        data.train_data_path "${REUSE_DATA_DIR}/train.jsonl" \
        data.val_data_path "${REUSE_DATA_DIR}/val.jsonl"
    else
      cp "$TRAIN_CFG" "$TMP_TRAIN_CFG"
    fi

    CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
      python finetune/train.py --config "$TMP_TRAIN_CFG" 2>&1 | tee "$TRAIN_LOG"
    rm -f "$TMP_TRAIN_CFG"
  fi
fi

# Decide checkpoint for inference
if [[ "$DO_INFER" == "1" ]]; then
  if [[ -n "${REUSE_MODEL_CKPT}" && "$FORCE_TRAIN" != "1" ]]; then
    CKPT_DIR="$REUSE_MODEL_CKPT"
  else
    CKPT_DIR="$(latest_checkpoint_dir "$EXP_MODEL_DIR")"
  fi

  if [[ -z "$CKPT_DIR" || ! -e "$CKPT_DIR" ]]; then
    echo "[ERROR] No valid checkpoint found for inference. EXP_MODEL_DIR=$EXP_MODEL_DIR  REUSE_MODEL_CKPT=$REUSE_MODEL_CKPT" >&2
    exit 1
  fi
fi

# ===== 3) vLLM + runner =====
if [[ "$DO_INFER" == "1" ]]; then
  echo "[STEP] start vLLM"
  GPU_MEM_UTIL="$DEFAULT_GPU_MEM_UTIL"
  if [[ "${DYNAMIC_GPU_MEM_UTIL}" == "1" ]]; then
    GPU_MEM_UTIL="$(compute_vllm_gpu_mem_util \
      "$VLLM_CUDA_VISIBLE_DEVICES" \
      "$GPU_MEM_HEADROOM_MB" \
      "$GPU_MEM_UTIL_MARGIN" \
      "$GPU_MEM_UTIL_MIN" \
      "$GPU_MEM_UTIL_MAX")"
  fi
  echo "[INFO] vLLM --gpu-memory-utilization=${GPU_MEM_UTIL}"

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
  echo "[INFO] vLLM pid=${VLLM_PID} log=${VLLM_LOG}"

  if ! wait_vllm_ready "$PORT" "$VLLM_WAIT_SECONDS"; then
    echo "[ERROR] vLLM not ready after ${VLLM_WAIT_SECONDS}s. Tail log:" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  echo "[OK] vLLM ready on port $PORT"

  echo "[STEP] runner"
  TMP_RUN_CFG="$(mktemp)"
  # patch api_base + (optional) reuse test_data_file
  if [[ -n "${REUSE_DATA_DIR}" ]]; then
    json_patch "$RUNNER_CFG" "$TMP_RUN_CFG" \
      model.params.api_base "http://127.0.0.1:${PORT}/v1/" \
      tester.test_data_file "${REUSE_DATA_DIR}/test.json"
  else
    json_patch "$RUNNER_CFG" "$TMP_RUN_CFG" \
      model.params.api_base "http://127.0.0.1:${PORT}/v1/"
  fi

  python runner/run.py --config "$TMP_RUN_CFG" 2>&1 | tee "$RUN_LOG"
  rm -f "$TMP_RUN_CFG"

  echo "[STEP] stop vLLM"
  cleanup
fi

echo "[DONE] $EXP_DIR (MODE=$MODE)"
