#!/usr/bin/env bash
# CPU/import checks with CUDA build toolkit, no GPU allocation and no network.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_DATA_ROOT=/pfs/user/data/host_piper_paper
docker run --rm --network none \
  -v "${REPO_DIR}:${REPO_DIR}:ro" \
  -v "${HOST_DATA_ROOT}:${HOST_DATA_ROOT}" \
  -w "${REPO_DIR}/alignment" \
  -e "PYTHONPATH=${HOST_DATA_ROOT}/envs/alignment/lib/python3.10/site-packages:${REPO_DIR}/alignment" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e "HF_HOME=${HOST_DATA_ROOT}/cache/huggingface" \
  -e "TORCH_HOME=${HOST_DATA_ROOT}/cache/torch" \
  -e "TORCH_EXTENSIONS_DIR=${HOST_DATA_ROOT}/cache/torch_extensions" \
  -e "TRITON_CACHE_DIR=${HOST_DATA_ROOT}/cache/triton_alignment" \
  -e "WANDB_DIR=${HOST_DATA_ROOT}/logs" \
  -e WANDB_MODE=offline -e HF_HUB_OFFLINE=1 -e CUDA_HOME=/usr/local/cuda \
  --entrypoint /opt/venv/bin/python \
  lingbot-va:cu126-torch290-no-flashattn -S "$@"
# -S disables the image's site-packages: torch2.4 comes exclusively from the mounted
# alignment venv, not the image's torch2.9. This is not a GPU training entrypoint.
