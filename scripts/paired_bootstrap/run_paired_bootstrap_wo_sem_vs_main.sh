#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SYSTEM_A="${SYSTEM_A:-exps/ablation/wo_semantic_match/exp_3992dbfb11/runner_output/exp_3992dbfb11_s42.json}"
SYSTEM_B="${SYSTEM_B:-runner/output/k_ablation/k10_s42.json}"
METRIC="${METRIC:-all}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
SEED="${SEED:-42}"
CI="${CI:-95.0}"
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.5}"
OUTPUT="${OUTPUT:-output/paired_bootstrap/wo_semantic_vs_main.json}"
INDENT="${INDENT:-2}"

"$PYTHON_BIN" scripts/paired_bootstrap_llm.py \
  --system-a "$SYSTEM_A" \
  --system-b "$SYSTEM_B" \
  --metric "$METRIC" \
  --n-bootstrap "$N_BOOTSTRAP" \
  --seed "$SEED" \
  --ci "$CI" \
  --similarity-threshold "$SIMILARITY_THRESHOLD" \
  --output "$OUTPUT" \
  --indent "$INDENT"
