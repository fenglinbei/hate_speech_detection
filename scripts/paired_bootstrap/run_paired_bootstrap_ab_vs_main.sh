#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

SYSTEM_B="${SYSTEM_B:-output/runner/k_ablation/k10_s42.json}"
SYSTEM_B_LABEL="${SYSTEM_B_LABEL:-main}"
METRIC="${METRIC:-all}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-42}"
CI="${CI:-95.0}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-output/paired_bootstrap}"
INDENT="${INDENT:-2}"

SYSTEM_A_NAMES=(
  "w/o lex"
  "w/o exact matching"
  "w/o semantic retrieval"
  "w/o class-quota"
  "w/o truncation"
)

SYSTEM_A_FILES=(
  "output/runner/ablation/wo_lex.json"
  "exps/ablation/wo_exact_match/exp_e77aff03c5/runner_output/exp_e77aff03c5_s4242.json"
  "exps/ablation/wo_semantic_match/exp_3992dbfb11/runner_output/exp_3992dbfb11_s42.json"
  "exps/emonstration_selection/global/exp_744c07ab8e/runner_output/exp_744c07ab8e_s424242.json"
  "output/runner/ablation/wo_autolength.json"
)

if [[ "${#SYSTEM_A_NAMES[@]}" -ne "${#SYSTEM_A_FILES[@]}" ]]; then
  echo "SYSTEM_A_NAMES and SYSTEM_A_FILES must have the same length." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for idx in "${!SYSTEM_A_FILES[@]}"; do
  system_a_name="${SYSTEM_A_NAMES[$idx]}"
  system_a_file="${SYSTEM_A_FILES[$idx]}"
  output_file="$OUTPUT_DIR/${system_a_name}_vs_${SYSTEM_B_LABEL}.json"

  echo "Running paired bootstrap: ${system_a_name} vs ${SYSTEM_B_LABEL}"
  "$PYTHON_BIN" scripts/paired_bootstrap/paired_bootstrap_llm.py \
    --system-a "$system_a_file" \
    --system-b "$SYSTEM_B" \
    --metric "$METRIC" \
    --n-bootstrap "$N_BOOTSTRAP" \
    --seed "$SEED" \
    --ci "$CI" \
    --similarity-threshold "$SIMILARITY_THRESHOLD" \
    --output "$output_file" \
    --indent "$INDENT"
done
