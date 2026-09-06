#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
environment_dir="${repository_root}/.venv/wp3-g3"
requirements_lock="${repository_root}/environment/wp3-g3-requirements.lock"
source_intake_requirements_lock="${repository_root}/environment/wp3-g3-source-intake-requirements.lock"
wheelhouse="${repository_root}/.venv/wp3-g3-wheelhouse"

python3.12 -m venv "${environment_dir}"
if [[ -d "${wheelhouse}" ]]; then
  "${environment_dir}/bin/python" -m pip install \
    --require-hashes --no-index --find-links "${wheelhouse}" \
    -r "${requirements_lock}"
else
  "${environment_dir}/bin/python" -m pip install \
    --require-hashes -r "${requirements_lock}"
fi

if [[ -d "${wheelhouse}" ]]; then
  "${environment_dir}/bin/python" -m pip install \
    --require-hashes --no-deps --no-index --find-links "${wheelhouse}" \
    -r "${source_intake_requirements_lock}"
else
  "${environment_dir}/bin/python" -m pip install \
    --require-hashes --no-deps -r "${source_intake_requirements_lock}"
fi

PYTHONPATH="${repository_root}/src" \
  "${environment_dir}/bin/python" \
  "${repository_root}/scripts/stage1/wp3_g3_environment.py" check
