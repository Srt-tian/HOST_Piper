#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_DATA_ROOT=/pfs/user/data/host_piper_paper
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_DIR}/alignment"
export HF_HOME="${HOST_DATA_ROOT}/cache/huggingface"
export TORCH_HOME="${HOST_DATA_ROOT}/cache/torch"
export TORCH_EXTENSIONS_DIR="${HOST_DATA_ROOT}/cache/torch_extensions"
export TRITON_CACHE_DIR="${HOST_DATA_ROOT}/cache/triton_alignment"
export XDG_CACHE_HOME="${HOST_DATA_ROOT}/cache/xdg"
export WANDB_DIR="${HOST_DATA_ROOT}/logs"
export TMPDIR="${HOST_DATA_ROOT}/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
cd "${REPO_DIR}/alignment"
exec "${HOST_DATA_ROOT}/envs/alignment/bin/python" "$@"
