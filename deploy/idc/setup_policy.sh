#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_DATA_ROOT=/pfs/user/data/host_piper_paper
HOST_ENV_DIR="${HOST_DATA_ROOT}/envs/policy"
test "$(hostname)" = dev-instance-shenrongtian
test -d "${REPO_DIR}/.git"
mkdir -p "${HOST_DATA_ROOT}/cache/pip" "${HOST_DATA_ROOT}/tmp" "${HOST_DATA_ROOT}/logs"
export PIP_CACHE_DIR="${HOST_DATA_ROOT}/cache/pip"
export TMPDIR="${HOST_DATA_ROOT}/tmp"
export PYTHONNOUSERSITE=1
export DS_BUILD_OPS=0
unset PYTHONPATH
if [ ! -f "${HOST_ENV_DIR}/pyvenv.cfg" ]; then
  python3 -m venv "${HOST_ENV_DIR}"
fi
"${HOST_ENV_DIR}/bin/python" -m pip install --retries 0 --timeout 60 \
  -r "${REPO_DIR}/deploy/idc/requirements-policy.txt"
# Ensure all declared pins even when resuming an installation started from an older list.
"${HOST_ENV_DIR}/bin/python" -m pip install --retries 0 --timeout 60 \
  packaging==25.0 regex==2025.11.3 tqdm==4.66.5 typing-extensions==4.15.0
# Do not resolve upstream's conflicting optional pins again.
"${HOST_ENV_DIR}/bin/python" -m pip install --no-deps -e "${REPO_DIR}/policy_training"
"${HOST_ENV_DIR}/bin/python" -m pip freeze > "${HOST_DATA_ROOT}/logs/policy-freeze.txt"
"${HOST_ENV_DIR}/bin/python" "${REPO_DIR}/deploy/idc/check_dependencies.py"
