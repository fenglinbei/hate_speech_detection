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
  "Ours(uniform)"
  "Cluster"
  "Global"
  "MMR"
  "Random"
)

SYSTEM_A_FILES=(
  "exps/emonstration_selection/uniform/exp_c44a603058/runner_output/exp_c44a603058_s42.json"
  "exps/emonstration_selection/cluster/exp_bd8221b10b/runner_output/exp_bd8221b10b_s42.json"
  "exps/emonstration_selection/global/exp_744c07ab8e/runner_output/exp_744c07ab8e_s424242.json"
  "exps/emonstration_selection/mmr/exp_22101d2758/runner_output/exp_22101d2758_s42.json"
  "exps/emonstration_selection/random/exp_c3700e5fe8/runner_output/exp_c3700e5fe8_s42424242.json"
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
