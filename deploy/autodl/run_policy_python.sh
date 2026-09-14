#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_ROOT=/root/autodl-tmp/host_piper
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${HOST_ROOT}/policy_runtime/site-packages:${REPO_DIR}/policy_training/src"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="${HOST_ROOT}/policy_runtime/site-packages/ninja/data/bin:${CUDA_HOME}/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="/root/host_piper_runtime/python310/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export HF_HOME="${HOST_ROOT}/policy_runtime/cache/huggingface"
export TORCH_HOME="${HOST_ROOT}/policy_runtime/cache/torch"
export TORCH_EXTENSIONS_DIR="${HOST_ROOT}/policy_runtime/cache/torch_extensions"
export TRITON_CACHE_DIR="${HOST_ROOT}/policy_runtime/cache/triton"
export TRITON_HOME="${HOST_ROOT}/policy_runtime/cache/triton_home"
export XDG_CACHE_HOME="${HOST_ROOT}/policy_runtime/cache/xdg"
export TMPDIR="${HOST_ROOT}/policy_setup_tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4 TOKENIZERS_PARALLELISM=false
export HOST_POLICY_VIDEO_DECODE=sequential HOST_POLICY_STRICT_LOAD=1
export HOST_POLICY_NATIVE_30HZ=1
export DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH="${HOST_ROOT}/policy_runtime/model_links"
export HOST_PREPARED_ROOT="${HOST_ROOT}/policy_candidates1500_20260914"
export HOST_DINO_REPO="${HOST_ROOT}/policy_runtime/dinov2"
export HOST_DINO_WEIGHTS=HOST_CHECKPOINT_STRICT HOST_SIGLIP_WEIGHTS=HOST_CHECKPOINT_STRICT
export HOST_INIT_CHECKPOINT="${HOST_INIT_CHECKPOINT:-${HOST_ROOT}/policy_weights/host_fcb38563ab7a/model.pt}"
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${REPO_DIR}/deploy/autodl/policy_zero2.json"
mkdir -p "${TMPDIR}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${TRITON_HOME}"
exec /root/host_piper_runtime/python310/bin/python -S -u "$@"
