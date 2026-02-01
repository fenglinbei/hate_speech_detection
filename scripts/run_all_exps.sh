#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${ROOT:-}}"
if [[ -z "$ROOT" ]]; then
  echo "[ERROR] Missing ROOT. Usage: bash scripts/run_all_exps.sh <experiments_root>" >&2
  exit 1
fi

MODE="${MODE:-full}"
# 可透传给 run_one_exp.sh 的 runtime override：
# DATA_DIR_OVERRIDE=... MODEL_CKPT_OVERRIDE=... FORCE_TRAIN=... 等

ROOT="$(python - <<PY
import os,sys
print(f"sys.argv: {sys.argv}")
print(os.path.abspath(sys.argv[1]))
PY
"$ROOT")"

echo "[INFO] ROOT=$ROOT MODE=$MODE"

shopt -s nullglob
exps=("$ROOT"/exp_*)
if [[ ${#exps[@]} -eq 0 ]]; then
  echo "[ERROR] No exp_* dirs under $ROOT" >&2
  exit 1
fi

for exp in "${exps[@]}"; do
  echo
  echo "======================================="
  echo "[RUN] $exp"
  echo "======================================="
  MODE="$MODE" bash scripts/run_one_exp.sh "$exp"
done

echo
echo "[ALL DONE] MODE=$MODE ROOT=$ROOT"
