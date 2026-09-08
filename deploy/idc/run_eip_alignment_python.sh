#!/usr/bin/env bash
# Runs inside the pinned lingbot image, not on the development host directly.
# The image supplies Python3.10/CUDA; all Python packages come from HOST's own venv.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_DATA_ROOT=/pfs/user/data/host_piper_paper
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${HOST_DATA_ROOT}/envs/alignment/lib/python3.10/site-packages:${REPO_DIR}/alignment"
export HF_HOME="${HOST_DATA_ROOT}/cache/huggingface"
export TORCH_HOME="${HOST_DATA_ROOT}/cache/torch"
export TORCH_EXTENSIONS_DIR="${HOST_DATA_ROOT}/cache/torch_extensions_alignment_smoke"
export TRITON_CACHE_DIR="${HOST_DATA_ROOT}/cache/triton_alignment"
export XDG_CACHE_HOME="${HOST_DATA_ROOT}/cache/xdg"
export TMPDIR="${HOST_DATA_ROOT}/tmp"
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR="${HOST_DATA_ROOT}/logs"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
test -x /opt/venv/bin/python
test -x "${CUDA_HOME}/bin/nvcc"
exec /opt/venv/bin/python -S -u "$@"
