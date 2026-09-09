#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_ROOT=/pfs/user/data/host_piper_paper
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/root/host_piper_runtime/envs/alignment/lib/python3.10/site-packages:${REPO_DIR}/alignment"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="/root/host_piper_runtime/envs/alignment/bin:${CUDA_HOME}/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/root/host_piper_runtime/python310/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/root/host_piper_runtime/cache/huggingface
export TORCH_HOME=/root/host_piper_runtime/cache/torch
export TORCH_EXTENSIONS_DIR=/root/host_piper_runtime/cache/torch_extensions
export TRITON_CACHE_DIR=/root/host_piper_runtime/cache/triton
export XDG_CACHE_HOME=/root/host_piper_runtime/cache/xdg
export TMPDIR=/root/host_piper_runtime/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
export HOST_ALIGNMENT_LOGITS_TO_KEEP=1 HOST_VIDEO_DECODE_MODE=sequential
mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}"
exec /root/host_piper_runtime/python310/bin/python -S -u "$@"
