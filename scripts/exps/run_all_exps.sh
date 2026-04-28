#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  MODE=full bash scripts/exps/run_all_exps.sh <experiments_root>
  MODE=full bash scripts/exps/run_all_exps.sh <exp_dir> [<exp_dir> ...]
  bash scripts/exps/run_all_exps.sh <exp_dir>/run.sh [<exp_dir>/run.sh ...]

Inputs:
  - experiments_root: directory containing exp_* children
  - exp_dir         : directory containing manifest.json
  - run.sh         : local experiment launcher; executed from the repo root

Environment:
  MODE=full|train|data|infer    passed to exp_dir/root targets (default: full)
  DRY_RUN=1                    print commands without running them
  CONTINUE_ON_ERROR=1          run remaining targets after a failure
EOF
}

if [[ $# -eq 0 ]]; then
  if [[ -n "${ROOT:-}" ]]; then
    set -- "$ROOT"
  else
    usage
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ONE="${RUN_ONE:-${SCRIPT_DIR}/run_one_exp.sh}"
MODE="${MODE:-full}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

[[ -f "$RUN_ONE" ]] || { echo "[ERROR] missing run_one_exp.sh: $RUN_ONE" >&2; exit 1; }

case "$MODE" in
  data|train|full|infer) ;;
  *) echo "[ERROR] unknown MODE=$MODE (data/train/full/infer)" >&2; exit 1 ;;
esac

abspath() {
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

declare -a TARGET_TYPES=()
declare -a TARGET_PATHS=()
declare -A SEEN_TARGETS=()

add_target() {
  local type="$1"
  local path="$2"
  local key="${type}:${path}"
  if [[ -n "${SEEN_TARGETS[$key]:-}" ]]; then
    return 0
  fi
  SEEN_TARGETS[$key]=1
  TARGET_TYPES+=("$type")
  TARGET_PATHS+=("$path")
}

expand_input() {
  local raw="$1"
  local path
  path="$(abspath "$raw")"

  if [[ -f "$path" ]]; then
    if [[ "$(basename "$path")" != "run.sh" ]]; then
      echo "[ERROR] File inputs must be experiment run.sh files: $raw" >&2
      exit 1
    fi
    add_target "script" "$path"
    return 0
  fi

  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Not found: $raw" >&2
    exit 1
  fi

  if [[ -f "${path}/manifest.json" ]]; then
    add_target "exp" "$path"
    return 0
  fi

  local -a children=()
  shopt -s nullglob
  children=( "$path"/exp_* )
  shopt -u nullglob
  if [[ ${#children[@]} -eq 0 ]]; then
    echo "[ERROR] No exp_* dirs under $path" >&2
    echo "[HINT] Pass an experiments output_root, an exp_* directory, or one or more exp_*/run.sh files." >&2
    exit 1
  fi

  local child
  for child in "${children[@]}"; do
    if [[ -d "$child" && -f "${child}/manifest.json" ]]; then
      add_target "exp" "$child"
    elif [[ -d "$child" && -f "${child}/run.sh" ]]; then
      add_target "script" "${child}/run.sh"
    fi
  done
}

run_target() {
  local type="$1"
  local path="$2"

  echo
  echo "======================================="
  if [[ "$type" == "exp" ]]; then
    echo "[RUN] exp: $path"
    echo "======================================="
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[DRY-RUN] MODE=%q bash %q %q\n' "$MODE" "$RUN_ONE" "$path"
      return 0
    fi
    (cd "$REPO_ROOT" && MODE="$MODE" bash "$RUN_ONE" "$path")
  else
    echo "[RUN] script: $path"
    echo "======================================="
    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[DRY-RUN] MODE=%q bash %q\n' "$MODE" "$path"
      return 0
    fi
    (cd "$REPO_ROOT" && MODE="$MODE" bash "$path")
  fi
}

for input in "$@"; do
  expand_input "$input"
done

if [[ ${#TARGET_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] No runnable experiments found." >&2
  exit 1
fi

echo "[INFO] MODE=$MODE targets=${#TARGET_PATHS[@]} repo=$REPO_ROOT"

failures=0
for i in "${!TARGET_PATHS[@]}"; do
  if run_target "${TARGET_TYPES[$i]}" "${TARGET_PATHS[$i]}"; then
    echo "[OK] ${TARGET_PATHS[$i]}"
  else
    status=$?
    failures=$((failures + 1))
    echo "[FAILED] ${TARGET_PATHS[$i]} (exit=$status)" >&2
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$status"
    fi
  fi
done

echo
if [[ "$failures" -gt 0 ]]; then
  echo "[ALL DONE WITH FAILURES] MODE=$MODE targets=${#TARGET_PATHS[@]} failures=$failures"
  exit 1
fi

echo "[ALL DONE] MODE=$MODE targets=${#TARGET_PATHS[@]}"
