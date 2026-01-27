#!/usr/bin/env bash
set -euo pipefail

# =============================
# Configurable parameters
# =============================
K_START="${K_START:-6}"
K_END="${K_END:-20}"

# 端口映射：port = PORT_BASE + k
PORT_BASE="${PORT_BASE:-35000}"

# vLLM 启动参数（按需改）
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5}"

# 模板（默认用 k6 作为模板）
BASE_K="${BASE_K:-6}"
BASE_BUILD_CFG="${BASE_BUILD_CFG:-data/exp_data/k_ablation/k${BASE_K}/config.json}"
BASE_FINETUNE_CFG="${BASE_FINETUNE_CFG:-finetune/config/k_ablation/k${BASE_K}.json}"
BASE_RUNNER_CFG="${BASE_RUNNER_CFG:-runner/config/k_ablation/k${BASE_K}.json}"

# 是否覆盖已存在文件：1 覆盖，0 跳过
OVERWRITE="${OVERWRITE:-1}"

# =============================
# Helper functions
# =============================
need_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] Missing template file: $f" >&2
    exit 1
  fi
}

write_file() {
  local path="$1"
  if [[ -f "$path" && "$OVERWRITE" != "1" ]]; then
    echo "[SKIP] Exists: $path"
    return 1
  fi
  return 0
}

# =============================
# Preconditions
# =============================
command -v jq >/dev/null 2>&1 || { echo "[ERROR] jq not found. Please install jq." >&2; exit 1; }

need_file "$BASE_BUILD_CFG"
need_file "$BASE_FINETUNE_CFG"
need_file "$BASE_RUNNER_CFG"

mkdir -p "data/exp_data/k_ablation"
mkdir -p "finetune/config/k_ablation"
mkdir -p "runner/config/k_ablation"
mkdir -p "runner/start_comand/k_ablation"

# =============================
# Main loop
# =============================
for k in $(seq "$K_START" "$K_END"); do
  out_data_dir="data/exp_data/k_ablation/k${k}"
  out_build_cfg="${out_data_dir}/config.json"
  out_finetune_cfg="finetune/config/k_ablation/k${k}.json"
  out_runner_cfg="runner/config/k_ablation/k${k}.json"
  out_vllm_sh="runner/start_comand/k_ablation/k${k}.sh"

  port=$((PORT_BASE + k))

  mkdir -p "$out_data_dir"

  # ---- 1) build_data config.json ----
  if write_file "$out_build_cfg"; then
    jq \
      --arg outdir "$out_data_dir" \
      --argjson topk "$k" \
      '
      .data_paths.train_output_path = ($outdir + "/train.jsonl")
      | .data_paths.val_output_path  = ($outdir + "/val.jsonl")
      | .data_paths.test_output_path = ($outdir + "/test.json")
      | .retrieval_settings.srag_top_k = $topk
      ' "$BASE_BUILD_CFG" > "$out_build_cfg"
    echo "[OK]  $out_build_cfg"
  fi

  # ---- 2) finetune config k{k}.json ----
  if write_file "$out_finetune_cfg"; then
    jq \
      --arg exp "k${k}" \
      --arg outdir "models/exps/k_ablation/k${k}" \
      --arg cfg "data/exp_data/k_ablation/k${k}/config.json" \
      --arg train "data/exp_data/k_ablation/k${k}/train.jsonl" \
      --arg val "data/exp_data/k_ablation/k${k}/val.jsonl" \
      '
      .exp_name = $exp
      | .training.output_dir = $outdir
      | .training.run_name = $exp
      | .data.config_path = $cfg
      | .data.train_data_path = $train
      | .data.val_data_path = $val
      ' "$BASE_FINETUNE_CFG" > "$out_finetune_cfg"
    echo "[OK]  $out_finetune_cfg"
  fi

  # ---- 3) runner config k{k}.json ----
  if write_file "$out_runner_cfg"; then
    jq \
      --arg outname "k${k}.json" \
      --arg api "http://127.0.0.1:${port}/v1/" \
      --arg test "data/exp_data/k_ablation/k${k}/test.json" \
      '
      .output_name = $outname
      | .model.params.api_base = $api
      | .tester.test_data_file = $test
      ' "$BASE_RUNNER_CFG" > "$out_runner_cfg"
    echo "[OK]  $out_runner_cfg"
  fi

  # ---- 4) vllm start script k{k}.sh ----
  if write_file "$out_vllm_sh"; then
    cat > "$out_vllm_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

# Auto-generated for k=${k}
K=${k}
PORT=${port}

# You can override these at runtime:
# CUDA_VISIBLE_DEVICES="2,3" bash $out_vllm_sh
CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-$CUDA_DEVICES}"

MODEL_ROOT="./models/exps/k_ablation/k\${K}"
CKPT_DIR="\$(ls -d "\${MODEL_ROOT}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"

if [[ -z "\${CKPT_DIR}" ]]; then
  echo "[ERROR] No checkpoint found under: \${MODEL_ROOT}/checkpoint-*"
  exit 1
fi

echo "[INFO] Using checkpoint: \${CKPT_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES}  PORT=\${PORT}"

python -m vllm.entrypoints.openai.api_server \\
  --served-model-name "$SERVED_MODEL_NAME" \\
  --model="\${CKPT_DIR}" \\
  --trust-remote-code \\
  --tensor-parallel-size="$TENSOR_PARALLEL_SIZE" \\
  --port="\${PORT}" \\
  --max_model_len "$MAX_MODEL_LEN"
EOF
    chmod +x "$out_vllm_sh"
    echo "[OK]  $out_vllm_sh"
  fi

done

echo
echo "[DONE] Generated configs and scripts for k=${K_START}..${K_END}"
echo "Tip: If you want to skip overwriting existing files, run: OVERWRITE=0 bash <this_script>.sh"

