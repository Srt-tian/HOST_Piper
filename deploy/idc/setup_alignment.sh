#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_DATA_ROOT=/pfs/user/data/host_piper_paper
HOST_ENV_DIR="${HOST_DATA_ROOT}/envs/alignment"
test "$(hostname)" = dev-instance-shenrongtian
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${HOST_DATA_ROOT}/cache/pip"
export TMPDIR="${HOST_DATA_ROOT}/tmp"
export DS_BUILD_OPS=0
unset PYTHONPATH
if [ ! -f "${HOST_ENV_DIR}/pyvenv.cfg" ]; then
  python3 -m venv "${HOST_ENV_DIR}"
fi
"${HOST_ENV_DIR}/bin/python" -m pip install --no-index \
  "${HOST_DATA_ROOT}/cache/pip/pip-25.3-py3-none-any.whl"
"${HOST_ENV_DIR}/bin/python" -m pip install --retries 0 --timeout 60 \
  -r "${REPO_DIR}/deploy/idc/requirements-alignment.txt"
"${HOST_ENV_DIR}/bin/python" "${REPO_DIR}/deploy/idc/check_dependencies.py"
"${HOST_ENV_DIR}/bin/python" -m pip freeze > "${HOST_DATA_ROOT}/logs/alignment-freeze.txt"
