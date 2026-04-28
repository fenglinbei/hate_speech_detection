#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODE=full bash scripts/exps/run_one_exp.sh /abs/or/rel/path/to/exp_dir
#
# Modes:
#   data  : build_data only (unless reuse data)
#   train : build_data + finetune (build may be skipped)
#   full  : build_data + finetune + vLLM + runner
#   infer : vLLM + runner only (requires model checkpoint)
MODE="${MODE:-full}"

EXP_DIR="${1:-${EXP_DIR:-}}"
if [[ -z "$EXP_DIR" ]]; then
  echo "[ERROR] Missing EXP_DIR. Usage: bash scripts/exps/run_one_exp.sh <exp_dir>" >&2
  exit 1
fi
EXP_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$EXP_DIR")"

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

count_csv_items() {
  local s="${1// /}"
  if [[ -z "$s" ]]; then echo "0"; return; fi
  local -a items
  IFS=',' read -r -a items <<< "$s"
  local n=0 item
  for item in "${items[@]}"; do
    [[ -n "$item" ]] && n=$((n+1))
  done
  echo "$n"
}

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
import sys
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

read_manifest_field() {
  local key="$1"
  PYTHONPATH=src python -c '
import json,sys
manifest_path=sys.argv[1]
key=sys.argv[2]
m=json.load(open(manifest_path,"r",encoding="utf-8"))
cur=m
for p in key.split("."):
    if isinstance(cur, dict) and p in cur:
        cur=cur[p]
    else:
        cur=None
        break
print("" if cur is None else cur)
' "$MANIFEST" "$key"
}

read_json_field() {
  local file="$1"
  local key="$2"
  PYTHONPATH=src python -c '
import json,sys,os
path=sys.argv[1]
key=sys.argv[2]
if not os.path.exists(path):
    print("")
    raise SystemExit(0)
payload=json.load(open(path,"r",encoding="utf-8"))
cur=payload
for p in key.split("."):
    if isinstance(cur, dict) and p in cur:
        cur=cur[p]
    else:
        cur=None
        break
print("" if cur is None else cur)
' "$file" "$key"
}

PORT="${PORT:-$(read_manifest_field port)}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-$(read_manifest_field runtime_defaults.train_cuda_visible_devices)}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-$(read_manifest_field runtime_defaults.vllm.cuda_visible_devices)}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$(read_manifest_field runtime_defaults.vllm.tensor_parallel_size)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$(read_manifest_field runtime_defaults.vllm.max_model_len)}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(read_manifest_field runtime_defaults.vllm.served_model_name)}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-180}"

TRAIN_BACKEND="${TRAIN_BACKEND:-deepspeed}"  # deepspeed | fsdp | single
case "$TRAIN_BACKEND" in
  single) TRAIN_PROFILE="${TRAIN_PROFILE:-single}" ;;
  fsdp) TRAIN_PROFILE="${TRAIN_PROFILE:-fsdp_safe}" ;;
  *) TRAIN_PROFILE="${TRAIN_PROFILE:-ds_zero2_safe}" ;;
esac
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-$(count_csv_items "$TRAIN_CUDA_VISIBLE_DEVICES")}"
if [[ -n "$PORT" ]]; then
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-$((PORT + 1000))}"
else
  TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29500}"
fi
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
TRAIN_DS_AUTOTUNE="${TRAIN_DS_AUTOTUNE:-0}"
TRAIN_DS_AUTOTUNE_FAST="${TRAIN_DS_AUTOTUNE_FAST:-1}"
TRAIN_DS_AUTOTUNE_OVERWRITE="${TRAIN_DS_AUTOTUNE_OVERWRITE:-1}"
TRAIN_DS_AUTOTUNE_METRIC="${TRAIN_DS_AUTOTUNE_METRIC:-throughput}"
TRAIN_DS_AUTOTUNE_START_PROFILE_STEP="${TRAIN_DS_AUTOTUNE_START_PROFILE_STEP:-3}"
TRAIN_DS_AUTOTUNE_END_PROFILE_STEP="${TRAIN_DS_AUTOTUNE_END_PROFILE_STEP:-5}"
TRAIN_DS_AUTOTUNE_NUM_MBS="${TRAIN_DS_AUTOTUNE_NUM_MBS:-3}"
TRAIN_DS_AUTOTUNE_MAX_TRAIN_BATCH_SIZE="${TRAIN_DS_AUTOTUNE_MAX_TRAIN_BATCH_SIZE:-}"
TRAIN_LORA="${TRAIN_LORA:-}"
TRAIN_LORA_R="${TRAIN_LORA_R:-}"
TRAIN_LORA_ALPHA="${TRAIN_LORA_ALPHA:-}"
TRAIN_LORA_DROPOUT="${TRAIN_LORA_DROPOUT:-}"
TRAIN_LORA_TARGET_MODULES="${TRAIN_LORA_TARGET_MODULES:-}"
TRAIN_LORA_MERGE="${TRAIN_LORA_MERGE:-}"
TRAIN_LORA_MERGE_MAX_SHARD_SIZE="${TRAIN_LORA_MERGE_MAX_SHARD_SIZE:-5GB}"

LOG_DIR="${LOG_DIR:-${EXP_DIR}/logs}"
mkdir -p "$LOG_DIR"
BUILD_LOG="${LOG_DIR}/build.log"
TRAIN_LOG="${LOG_DIR}/train.${TRAIN_BACKEND}.${TRAIN_PROFILE}.log"
TRAIN_LATEST_LOG="${LOG_DIR}/train.log"
TRAIN_RUNTIME_CFG="${LOG_DIR}/train_runtime_config.json"
TRAIN_DS_CFG="${LOG_DIR}/ds_config_${TRAIN_PROFILE}.json"
TRAIN_DS_AUTOTUNE_RESULTS_DIR="${TRAIN_DS_AUTOTUNE_RESULTS_DIR:-${LOG_DIR}/autotuning_results}"
TRAIN_DS_AUTOTUNE_EXPS_DIR="${TRAIN_DS_AUTOTUNE_EXPS_DIR:-${LOG_DIR}/autotuning_exps}"
VLLM_LOG="${LOG_DIR}/vllm_port${PORT}.log"
RUN_LOG="${LOG_DIR}/runner.log"

DATA_DIR_OVERRIDE="${DATA_DIR_OVERRIDE:-}"
MODEL_CKPT_OVERRIDE="${MODEL_CKPT_OVERRIDE:-}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

MANIFEST_REUSE_DATA="$(read_manifest_field reuse.data_dir)"
MANIFEST_REUSE_MODEL="$(read_manifest_field reuse.model_checkpoint)"

REUSE_DATA_DIR="${DATA_DIR_OVERRIDE:-$MANIFEST_REUSE_DATA}"
REUSE_MODEL_CKPT="${MODEL_CKPT_OVERRIDE:-$MANIFEST_REUSE_MODEL}"

if [[ "$TRAIN_DS_AUTOTUNE" == "1" ]]; then
  if [[ "$TRAIN_BACKEND" != "deepspeed" ]]; then
    echo "[ERROR] TRAIN_DS_AUTOTUNE=1 requires TRAIN_BACKEND=deepspeed (got ${TRAIN_BACKEND})" >&2
    exit 1
  fi
  require_cmd deepspeed
fi

EXP_DATA_DIR="$(read_manifest_field paths.data_dir)"
EXP_MODEL_DIR="$(read_manifest_field paths.model_dir)"
BUILD_SIGNATURE_FILE="${EXP_DATA_DIR}/build_signature.json"
BASELINE_METHOD="$(read_json_field "$RUNNER_CFG" baseline.method)"
BASELINE_METHOD="${BASELINE_METHOD:-$(read_json_field "$BUILD_CFG" baseline.method)}"
BASELINE_METHOD="${BASELINE_METHOD:-standard}"
if [[ "$BASELINE_METHOD" == "ddp" ]]; then BASELINE_METHOD="dpp"; fi
BASELINE_TASK_TYPE="$(read_json_field "$RUNNER_CFG" baseline.task_type)"
BASELINE_TASK_TYPE="${BASELINE_TASK_TYPE:-$(read_json_field "$BUILD_CFG" baseline.task_type)}"
BASELINE_TASK_TYPE="${BASELINE_TASK_TYPE:-structured}"
DPP_DEMOS_PATH="$(read_json_field "$BUILD_CFG" baseline.dpp.demos_path)"

dpp_demos_missing() {
  [[ "$BASELINE_METHOD" == "dpp" && -n "$DPP_DEMOS_PATH" && ! -f "$DPP_DEMOS_PATH" ]]
}

run_dpp_select() {
  if [[ "$BASELINE_METHOD" == "dpp" ]]; then
    echo "[STEP] dpp demo selection"
    PYTHONPATH=src python src/baselines/dpp_select.py --build-config "$BUILD_CFG" 2>&1 | tee "${LOG_DIR}/dpp_select.log"
  fi
}

build_config_signature() {
  PYTHONPATH=src python - "$BUILD_CFG" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, "r", encoding="utf-8"))
text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
print(hashlib.sha1(text.encode("utf-8")).hexdigest())
PY
}

build_signature_matches() {
  local current_sig
  current_sig="$(build_config_signature)"
  PYTHONPATH=src python - "$BUILD_SIGNATURE_FILE" "$current_sig" <<'PY'
import json
import os
import sys

path, current = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    raise SystemExit(1)
payload = json.load(open(path, "r", encoding="utf-8"))
raise SystemExit(0 if payload.get("build_config_sha1") == current else 1)
PY
}

build_requires_signature() {
  PYTHONPATH=src python - "$BUILD_CFG" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
required = bool(payload.get("require_build_signature", False))
required = required or str(payload.get("task_type", "")).strip().lower() == "cold_binary"
print("1" if required else "0")
PY
}

write_build_signature() {
  local current_sig
  current_sig="$(build_config_signature)"
  PYTHONPATH=src python - "$BUILD_SIGNATURE_FILE" "$current_sig" <<'PY'
import json
import os
import sys
import time

path, current = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(
    {
        "build_config_sha1": current,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    open(path, "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
PY
}

json_patch() {
  local src="$1"
  local dst="$2"
  shift 2
  PYTHONPATH=src python - "$src" "$dst" "$@" <<'PY'
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
    if v.lower()=="true": vv=True
    elif v.lower()=="false": vv=False
    else:
        try:
            if "." in v: vv=float(v)
            else: vv=int(v)
        except Exception:
            vv=v
    set_path(obj,k,vv)

os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(obj, open(dst,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
PY
}

train_config_lora_field() {
  local cfg="$1" field="$2"
  PYTHONPATH=src python - "$cfg" "$field" <<'PY'
import json
import sys

cfg_path, field = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
raw = cfg.get("lora", False)

def to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

if isinstance(raw, dict):
    enabled = to_bool(raw.get("enabled", True))
    merge_on_save = to_bool(raw.get("merge_on_save", True))
else:
    enabled = to_bool(raw)
    merge_on_save = True

values = {
    "enabled": enabled,
    "merge_on_save": merge_on_save,
}
print("1" if values[field] else "0")
PY
}

write_train_runtime_config() {
  local src="$1" dst="$2" runtime_cfg="$3" ds_cfg="$4"
  PYTHONPATH=src python - "$src" "$dst" "$runtime_cfg" "$ds_cfg" \
    "$TRAIN_BACKEND" "$TRAIN_PROFILE" "$TRAIN_NPROC_PER_NODE" "$TRAIN_MASTER_PORT" \
    "$TRAIN_CUDA_VISIBLE_DEVICES" "$TRAIN_MAX_STEPS" \
    "$TRAIN_LORA" "$TRAIN_LORA_R" "$TRAIN_LORA_ALPHA" "$TRAIN_LORA_DROPOUT" \
    "$TRAIN_LORA_TARGET_MODULES" "$TRAIN_LORA_MERGE" \
    "$TRAIN_DS_AUTOTUNE" "$TRAIN_DS_AUTOTUNE_FAST" "$TRAIN_DS_AUTOTUNE_OVERWRITE" \
    "$TRAIN_DS_AUTOTUNE_METRIC" "$TRAIN_DS_AUTOTUNE_START_PROFILE_STEP" \
    "$TRAIN_DS_AUTOTUNE_END_PROFILE_STEP" "$TRAIN_DS_AUTOTUNE_NUM_MBS" \
    "$TRAIN_DS_AUTOTUNE_MAX_TRAIN_BATCH_SIZE" "$TRAIN_DS_AUTOTUNE_RESULTS_DIR" \
    "$TRAIN_DS_AUTOTUNE_EXPS_DIR" <<'PY'
import json
import sys
from pathlib import Path

src, dst, runtime_cfg, ds_cfg = sys.argv[1:5]
backend, profile = sys.argv[5], sys.argv[6]
nproc, master_port, cuda_devices, max_steps = sys.argv[7:11]
lora_env, lora_r, lora_alpha, lora_dropout, lora_targets, lora_merge = sys.argv[11:17]
(
    ds_autotune,
    ds_autotune_fast,
    ds_autotune_overwrite,
    ds_autotune_metric,
    ds_autotune_start_profile_step,
    ds_autotune_end_profile_step,
    ds_autotune_num_mbs,
    ds_autotune_max_train_batch_size,
    ds_autotune_results_dir,
    ds_autotune_exps_dir,
) = sys.argv[17:27]

cfg = json.load(open(src, "r", encoding="utf-8"))
training = cfg.setdefault("training", {})

DEFAULT_LORA_TARGET_MODULES = ["q_proj", "v_proj"]

def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

def normalize_lora(raw) -> dict:
    if isinstance(raw, dict):
        payload = dict(raw)
        payload["enabled"] = str_to_bool(payload.get("enabled", True))
    else:
        payload = {"enabled": str_to_bool(raw)}
    if payload["enabled"]:
        payload.setdefault("r", 8)
        payload.setdefault("alpha", payload.pop("lora_alpha", 32))
        payload.setdefault("dropout", payload.pop("lora_dropout", 0.1))
        payload.setdefault("target_modules", DEFAULT_LORA_TARGET_MODULES)
        payload.setdefault("bias", "none")
        payload.setdefault("merge_on_save", True)
    return payload

def apply_lora_overrides() -> dict:
    lora = normalize_lora(cfg.get("lora", False))
    if lora_env:
        lora["enabled"] = str_to_bool(lora_env)
    if lora["enabled"]:
        lora.setdefault("r", 8)
        lora.setdefault("alpha", 32)
        lora.setdefault("dropout", 0.1)
        lora.setdefault("target_modules", DEFAULT_LORA_TARGET_MODULES)
        lora.setdefault("bias", "none")
        lora.setdefault("merge_on_save", True)
    if lora_r:
        lora["r"] = int(lora_r)
    if lora_alpha:
        lora["alpha"] = int(lora_alpha)
    if lora_dropout:
        lora["dropout"] = float(lora_dropout)
    if lora_targets:
        lora["target_modules"] = csv_list(lora_targets)
    if lora_merge:
        lora["merge_on_save"] = str_to_bool(lora_merge)
    cfg["lora"] = lora if lora.get("enabled", False) else False
    return cfg["lora"] if isinstance(cfg["lora"], dict) else {"enabled": False}

lora_runtime = apply_lora_overrides()

for key in (
    "deepspeed",
    "fsdp",
    "fsdp_config",
    "fsdp_min_num_params",
    "fsdp_transformer_layer_cls_to_wrap",
):
    training.pop(key, None)

def apply_common(micro_batch: int, grad_accum: int) -> None:
    training["per_device_train_batch_size"] = micro_batch
    training["per_device_eval_batch_size"] = micro_batch
    training["gradient_accumulation_steps"] = grad_accum
    training["gradient_checkpointing"] = True
    training.setdefault("bf16", True)

def autotuning_config() -> dict:
    payload = {
        "enabled": True,
        "results_dir": ds_autotune_results_dir,
        "exps_dir": ds_autotune_exps_dir,
        "overwrite": str_to_bool(ds_autotune_overwrite),
        "metric": ds_autotune_metric,
        "start_profile_step": int(ds_autotune_start_profile_step),
        "end_profile_step": int(ds_autotune_end_profile_step),
        "fast": str_to_bool(ds_autotune_fast),
        "num_tuning_micro_batch_sizes": int(ds_autotune_num_mbs),
        "arg_mappings": {
            "train_micro_batch_size_per_gpu": "--per_device_train_batch_size",
            "gradient_accumulation_steps": "--gradient_accumulation_steps",
        },
    }
    if ds_autotune_max_train_batch_size:
        payload["max_train_batch_size"] = int(ds_autotune_max_train_batch_size)
    return payload

def deepspeed_config(stage: int, offload: bool = False) -> dict:
    # Keep bucket values concrete. Some accelerate/deepspeed versions cannot
    # fill these fields when they are set to "auto".
    zero = {
        "stage": stage,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 200_000_000,
    }
    if stage == 2:
        zero.update({
            "allgather_partitions": True,
            "allgather_bucket_size": 200_000_000,
            "reduce_scatter": True,
            "round_robin_gradients": True,
        })
    else:
        zero.update({
            "stage3_prefetch_bucket_size": 20_000_000,
            "stage3_param_persistence_threshold": 100_000,
            "stage3_max_live_parameters": 1_000_000_000,
            "stage3_max_reuse_distance": 1_000_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
            "sub_group_size": 1_000_000_000,
        })
    if offload:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}

    config = {
        "bf16": {"enabled": "auto"},
        "zero_optimization": zero,
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    if str_to_bool(ds_autotune):
        config["autotuning"] = autotuning_config()
    return config

profiles = {
    "ds_zero2_safe": {"backend": "deepspeed", "stage": 2, "micro": 2, "accum": 1, "offload": False},
    "ds_zero2_bs4": {"backend": "deepspeed", "stage": 2, "micro": 4, "accum": 1, "offload": False},
    "ds_zero2_bs1": {"backend": "deepspeed", "stage": 2, "micro": 1, "accum": 2, "offload": False},
    "ds_zero3_safe": {"backend": "deepspeed", "stage": 3, "micro": 2, "accum": 1, "offload": False},
    "ds_zero3_bs1": {"backend": "deepspeed", "stage": 3, "micro": 1, "accum": 2, "offload": False},
    "ds_zero3_offload": {"backend": "deepspeed", "stage": 3, "micro": 1, "accum": 2, "offload": True},
    "fsdp_safe": {"backend": "fsdp", "micro": 2, "accum": 1},
}

runtime = {
    "backend": backend,
    "profile": profile,
    "nproc_per_node": int(nproc),
    "master_port": int(master_port),
    "cuda_visible_devices": cuda_devices,
    "max_steps_override": int(max_steps) if max_steps else None,
}

if backend == "single":
    if profile != "single":
        raise SystemExit(f"TRAIN_BACKEND=single requires TRAIN_PROFILE=single, got {profile}")
elif profile not in profiles:
    raise SystemExit(f"Unknown TRAIN_PROFILE={profile}")
else:
    spec = profiles[profile]
    if spec["backend"] != backend:
        raise SystemExit(f"TRAIN_PROFILE={profile} belongs to backend={spec['backend']}, got TRAIN_BACKEND={backend}")
    apply_common(spec["micro"], spec["accum"])
    runtime["effective_micro_batch"] = spec["micro"]
    runtime["effective_gradient_accumulation"] = spec["accum"]

    if backend == "deepspeed":
        ds = deepspeed_config(spec["stage"], spec.get("offload", False))
        Path(ds_cfg).parent.mkdir(parents=True, exist_ok=True)
        json.dump(ds, open(ds_cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        training["deepspeed"] = ds_cfg
        runtime["deepspeed_config"] = ds_cfg
        runtime["zero_stage"] = spec["stage"]
        runtime["offload"] = spec.get("offload", False)
        runtime["autotuning"] = {
            "enabled": str_to_bool(ds_autotune),
            "initial_micro_batch": spec["micro"],
            "initial_gradient_accumulation": spec["accum"],
            "results_dir": ds_autotune_results_dir if str_to_bool(ds_autotune) else None,
            "exps_dir": ds_autotune_exps_dir if str_to_bool(ds_autotune) else None,
            "fast": str_to_bool(ds_autotune_fast),
            "overwrite": str_to_bool(ds_autotune_overwrite),
            "metric": ds_autotune_metric,
            "start_profile_step": int(ds_autotune_start_profile_step),
            "end_profile_step": int(ds_autotune_end_profile_step),
            "num_tuning_micro_batch_sizes": int(ds_autotune_num_mbs),
            "max_train_batch_size": int(ds_autotune_max_train_batch_size) if ds_autotune_max_train_batch_size else None,
        }
    elif backend == "fsdp":
        training["fsdp"] = "full_shard auto_wrap"
        training["fsdp_transformer_layer_cls_to_wrap"] = "Qwen2DecoderLayer"
        training["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"],
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "limit_all_gathers": True,
            "use_orig_params": False,
        }
        runtime["fsdp"] = training["fsdp"]
        runtime["fsdp_transformer_layer_cls_to_wrap"] = "Qwen2DecoderLayer"
        runtime["fsdp_state_dict_type"] = "FULL_STATE_DICT"

if max_steps:
    steps = int(max_steps)
    training["max_steps"] = steps
    training["save_strategy"] = "steps"
    training["save_steps"] = steps
    training["eval_strategy"] = "no"

runtime["training"] = training
runtime["lora"] = lora_runtime

Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(runtime_cfg).parent.mkdir(parents=True, exist_ok=True)
json.dump(cfg, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(runtime, open(runtime_cfg, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PY
}

latest_checkpoint_dir() {
  local model_root="$1"
  ls -d "${model_root}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

latest_merged_checkpoint_dir() {
  local model_root="$1"
  ls -d "${model_root}"/merged-checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true
}

checkpoint_has_hf_weights() {
  local ckpt="$1"
  [[ -f "${ckpt}/pytorch_model.bin" \
    || -f "${ckpt}/pytorch_model.bin.index.json" \
    || -f "${ckpt}/model.safetensors" \
    || -f "${ckpt}/model.safetensors.index.json" \
    || -n "$(find "$ckpt" -maxdepth 1 -type f -name 'pytorch_model-*.bin' -print -quit 2>/dev/null)" \
    || -n "$(find "$ckpt" -maxdepth 1 -type f -name 'model-*.safetensors' -print -quit 2>/dev/null)" ]]
}

checkpoint_has_lora_adapter() {
  local ckpt="$1"
  [[ -f "${ckpt}/adapter_config.json" \
    && ( -f "${ckpt}/adapter_model.bin" \
      || -f "${ckpt}/adapter_model.bin.index.json" \
      || -f "${ckpt}/adapter_model.safetensors" \
      || -f "${ckpt}/adapter_model.safetensors.index.json" \
      || -n "$(find "$ckpt" -maxdepth 1 -type f -name 'adapter_model-*.bin' -print -quit 2>/dev/null)" \
      || -n "$(find "$ckpt" -maxdepth 1 -type f -name 'adapter_model-*.safetensors' -print -quit 2>/dev/null)" ) ]]
}

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
  local url="http://127.0.0.1:${port}/health"
  local i=0
  while [[ $i -lt $deadline ]]; do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1; i=$((i+1))
  done
  return 1
}

echo "[INFO] EXP_DIR=$EXP_DIR"
echo "[INFO] MODE=$MODE  PORT=$PORT"
echo "[INFO] TRAIN_CUDA_VISIBLE_DEVICES=$TRAIN_CUDA_VISIBLE_DEVICES"
echo "[INFO] TRAIN_BACKEND=$TRAIN_BACKEND  TRAIN_PROFILE=$TRAIN_PROFILE"
echo "[INFO] TRAIN_NPROC_PER_NODE=$TRAIN_NPROC_PER_NODE  TRAIN_MASTER_PORT=$TRAIN_MASTER_PORT"
if [[ "$TRAIN_DS_AUTOTUNE" == "1" ]]; then
  echo "[INFO] TRAIN_DS_AUTOTUNE=1  FAST=$TRAIN_DS_AUTOTUNE_FAST  METRIC=$TRAIN_DS_AUTOTUNE_METRIC  PROFILE_STEPS=${TRAIN_DS_AUTOTUNE_START_PROFILE_STEP}-${TRAIN_DS_AUTOTUNE_END_PROFILE_STEP}  NUM_MBS=$TRAIN_DS_AUTOTUNE_NUM_MBS"
  echo "[INFO] DeepSpeed autotuning dirs: results=$TRAIN_DS_AUTOTUNE_RESULTS_DIR exps=$TRAIN_DS_AUTOTUNE_EXPS_DIR"
fi
if [[ -n "$TRAIN_LORA" ]]; then
  echo "[INFO] TRAIN_LORA=$TRAIN_LORA  R=${TRAIN_LORA_R:-<default>}  ALPHA=${TRAIN_LORA_ALPHA:-<default>}  DROPOUT=${TRAIN_LORA_DROPOUT:-<default>}  TARGETS=${TRAIN_LORA_TARGET_MODULES:-<default>}  MERGE=${TRAIN_LORA_MERGE:-<default>}"
fi
echo "[INFO] VLLM_CUDA_VISIBLE_DEVICES=$VLLM_CUDA_VISIBLE_DEVICES"
echo "[INFO] REUSE_DATA_DIR=${REUSE_DATA_DIR:-<none>}"
echo "[INFO] REUSE_MODEL_CKPT=${REUSE_MODEL_CKPT:-<none>}"
echo "[INFO] BASELINE_METHOD=$BASELINE_METHOD  BASELINE_TASK_TYPE=$BASELINE_TASK_TYPE"

case "$MODE" in
  data)  DO_BUILD=1; DO_TRAIN=0; DO_INFER=0 ;;
  train) DO_BUILD=1; DO_TRAIN=1; DO_INFER=0 ;;
  full)  DO_BUILD=1; DO_TRAIN=1; DO_INFER=1 ;;
  infer) DO_BUILD=0; DO_TRAIN=0; DO_INFER=1 ;;
  *) echo "[ERROR] unknown MODE=$MODE (data/train/full/infer)"; exit 1 ;;
esac

if [[ "$DO_BUILD" == "1" ]]; then
  if [[ -n "${REUSE_DATA_DIR}" ]]; then
    echo "[SKIP] build_data because REUSE_DATA_DIR is set: $REUSE_DATA_DIR"
  elif [[ -f "${EXP_DATA_DIR}/train.jsonl" && -f "${EXP_DATA_DIR}/val.jsonl" && -f "${EXP_DATA_DIR}/test.json" ]]; then
    if [[ "$(build_requires_signature)" != "1" ]] || build_signature_matches; then
      if dpp_demos_missing; then
        echo "[STEP] build_data (DPP demos missing)"
        run_dpp_select
        PYTHONPATH=src python src/data/build_data.py --config "$BUILD_CFG" 2>&1 | tee "$BUILD_LOG"
        write_build_signature
      else
        echo "[SKIP] build_data (found existing data in ${EXP_DATA_DIR})"
      fi
    else
      echo "[STEP] build_data (existing data signature missing or stale)"
      run_dpp_select
      PYTHONPATH=src python src/data/build_data.py --config "$BUILD_CFG" 2>&1 | tee "$BUILD_LOG"
      write_build_signature
    fi
  else
    run_dpp_select
    echo "[STEP] build_data"
    PYTHONPATH=src python src/data/build_data.py --config "$BUILD_CFG" 2>&1 | tee "$BUILD_LOG"
    write_build_signature
  fi
fi

CKPT_DIR=""
if [[ "$DO_TRAIN" == "1" ]]; then
  if [[ -n "${REUSE_MODEL_CKPT}" && "$FORCE_TRAIN" != "1" ]]; then
    echo "[SKIP] train because REUSE_MODEL_CKPT is set (set FORCE_TRAIN=1 to override)"
  else
    echo "[STEP] train"
    TMP_BASE_TRAIN_CFG="$(mktemp)"
    TMP_TRAIN_CFG="$(mktemp)"
    if [[ -n "${REUSE_DATA_DIR}" ]]; then
      json_patch "$TRAIN_CFG" "$TMP_BASE_TRAIN_CFG" \
        data.train_data_path "${REUSE_DATA_DIR}/train.jsonl" \
        data.val_data_path "${REUSE_DATA_DIR}/val.jsonl"
    else
      cp "$TRAIN_CFG" "$TMP_BASE_TRAIN_CFG"
    fi

    write_train_runtime_config "$TMP_BASE_TRAIN_CFG" "$TMP_TRAIN_CFG" "$TRAIN_RUNTIME_CFG" "$TRAIN_DS_CFG"
    echo "[INFO] train runtime config: $TRAIN_RUNTIME_CFG"
    [[ "$TRAIN_BACKEND" == "deepspeed" ]] && echo "[INFO] DeepSpeed config: $TRAIN_DS_CFG"
    echo "[INFO] train log: $TRAIN_LOG"

    PROFILE_MICRO_BATCH="$(read_json_field "$TRAIN_RUNTIME_CFG" effective_micro_batch)"
    PROFILE_GRAD_ACCUM="$(read_json_field "$TRAIN_RUNTIME_CFG" effective_gradient_accumulation)"
    if [[ "$TRAIN_DS_AUTOTUNE" == "1" && -n "$TRAIN_MAX_STEPS" ]]; then
      MIN_AUTOTUNE_STEPS=$((TRAIN_DS_AUTOTUNE_END_PROFILE_STEP + 1))
      if [[ "$TRAIN_MAX_STEPS" -lt "$MIN_AUTOTUNE_STEPS" ]]; then
        echo "[ERROR] TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS} is too small for DeepSpeed autotuning profile window; set it to at least ${MIN_AUTOTUNE_STEPS} or unset TRAIN_MAX_STEPS." >&2
        exit 1
      fi
    fi

    if [[ "$TRAIN_BACKEND" == "single" ]]; then
      CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
        TRAIN_BACKEND="$TRAIN_BACKEND" \
        TRAIN_PROFILE="$TRAIN_PROFILE" \
        PYTHONPATH=src python src/finetune/train.py --config "$TMP_TRAIN_CFG" 2>&1 | tee "$TRAIN_LOG" "$TRAIN_LATEST_LOG"
    elif [[ "$TRAIN_DS_AUTOTUNE" == "1" ]]; then
      if [[ -z "$TRAIN_CUDA_VISIBLE_DEVICES" ]]; then
        echo "[ERROR] TRAIN_DS_AUTOTUNE=1 requires TRAIN_CUDA_VISIBLE_DEVICES to select local GPUs" >&2
        exit 1
      fi
      DEEPSPEED_INCLUDE="localhost:${TRAIN_CUDA_VISIBLE_DEVICES// /}"
      TRAIN_BACKEND="$TRAIN_BACKEND" \
        TRAIN_PROFILE="$TRAIN_PROFILE" \
        PYTHONPATH=src deepspeed \
          --autotuning run \
          --include "$DEEPSPEED_INCLUDE" \
          --master_port "$TRAIN_MASTER_PORT" \
          src/finetune/train.py \
          --config "$TMP_TRAIN_CFG" \
          --deepspeed "$TRAIN_DS_CFG" \
          --per_device_train_batch_size "$PROFILE_MICRO_BATCH" \
          --gradient_accumulation_steps "$PROFILE_GRAD_ACCUM" 2>&1 | tee "$TRAIN_LOG" "$TRAIN_LATEST_LOG"
    else
      if [[ "${TRAIN_NPROC_PER_NODE}" -lt 1 ]]; then
        echo "[ERROR] TRAIN_NPROC_PER_NODE must be >= 1 (got ${TRAIN_NPROC_PER_NODE})" >&2
        exit 1
      fi
      CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
        TRAIN_BACKEND="$TRAIN_BACKEND" \
        TRAIN_PROFILE="$TRAIN_PROFILE" \
        PYTHONPATH=src python -m torch.distributed.run \
          --nproc_per_node "$TRAIN_NPROC_PER_NODE" \
          --master_port "$TRAIN_MASTER_PORT" \
          src/finetune/train.py --config "$TMP_TRAIN_CFG" 2>&1 | tee "$TRAIN_LOG" "$TRAIN_LATEST_LOG"
    fi

    if [[ "$(train_config_lora_field "$TMP_TRAIN_CFG" enabled)" == "1" && "$(train_config_lora_field "$TMP_TRAIN_CFG" merge_on_save)" == "1" ]]; then
      ADAPTER_CKPT="$(latest_checkpoint_dir "$EXP_MODEL_DIR")"
      if [[ -z "$ADAPTER_CKPT" || ! -e "$ADAPTER_CKPT" ]]; then
        echo "[ERROR] No LoRA adapter checkpoint found under $EXP_MODEL_DIR" >&2
        exit 1
      fi
      if ! checkpoint_has_lora_adapter "$ADAPTER_CKPT"; then
        echo "[ERROR] Latest checkpoint is not a valid LoRA adapter checkpoint: $ADAPTER_CKPT" >&2
        exit 1
      fi

      ADAPTER_NAME="$(basename "$ADAPTER_CKPT")"
      MERGED_CKPT="${EXP_MODEL_DIR}/merged-${ADAPTER_NAME}"
      echo "[STEP] merge LoRA adapter -> ${MERGED_CKPT}"
      CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
        PYTHONPATH=src python src/finetune/merge_lora.py \
          --config "$TMP_TRAIN_CFG" \
          --adapter "$ADAPTER_CKPT" \
          --output "$MERGED_CKPT" \
          --max-shard-size "$TRAIN_LORA_MERGE_MAX_SHARD_SIZE" \
          --overwrite 2>&1 | tee -a "$TRAIN_LOG" "$TRAIN_LATEST_LOG"
      if ! checkpoint_has_hf_weights "$MERGED_CKPT"; then
        echo "[ERROR] Merged checkpoint is not vLLM-loadable; missing HuggingFace weight files in $MERGED_CKPT" >&2
        exit 1
      fi
    fi
    rm -f "$TMP_BASE_TRAIN_CFG" "$TMP_TRAIN_CFG"
  fi
fi

if [[ "$DO_INFER" == "1" ]]; then
  if [[ -n "${REUSE_MODEL_CKPT}" && "$FORCE_TRAIN" != "1" ]]; then
    CKPT_DIR="$REUSE_MODEL_CKPT"
  else
    CKPT_DIR="$(latest_merged_checkpoint_dir "$EXP_MODEL_DIR")"
    if [[ -z "$CKPT_DIR" ]]; then
      CKPT_DIR="$(latest_checkpoint_dir "$EXP_MODEL_DIR")"
    fi
  fi

  if [[ -z "$CKPT_DIR" || ! -e "$CKPT_DIR" ]]; then
    echo "[ERROR] No valid checkpoint found for inference. EXP_MODEL_DIR=$EXP_MODEL_DIR  REUSE_MODEL_CKPT=$REUSE_MODEL_CKPT" >&2
    exit 1
  fi
  if ! checkpoint_has_hf_weights "$CKPT_DIR"; then
    echo "[ERROR] Latest checkpoint is not vLLM-loadable; missing HuggingFace weight files in $CKPT_DIR" >&2
    echo "[ERROR] Check $TRAIN_LOG and profile config. For ZeRO-3, stage3_gather_16bit_weights_on_model_save must be true." >&2
    exit 1
  fi
fi

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
  if [[ -n "${REUSE_DATA_DIR}" ]]; then
    json_patch "$RUNNER_CFG" "$TMP_RUN_CFG" \
      model.params.api_base "http://127.0.0.1:${PORT}/v1/" \
      tester.test_data_file "${REUSE_DATA_DIR}/test.json"
  else
    json_patch "$RUNNER_CFG" "$TMP_RUN_CFG" \
      model.params.api_base "http://127.0.0.1:${PORT}/v1/"
  fi

  if [[ "$BASELINE_METHOD" == "ids" ]]; then
    PYTHONPATH=src python src/baselines/ids_runner.py --config "$TMP_RUN_CFG" --build-config "$BUILD_CFG" 2>&1 | tee "$RUN_LOG"
  else
    PYTHONPATH=src python src/runner/run.py --config "$TMP_RUN_CFG" 2>&1 | tee "$RUN_LOG"
  fi
  rm -f "$TMP_RUN_CFG"

  echo "[STEP] stop vLLM"
  cleanup
fi

echo "[DONE] $EXP_DIR (MODE=$MODE)"
