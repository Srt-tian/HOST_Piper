#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_EXPECTED_COMMIT:?}"
: "${HOST_DTW_SHARD:?}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${HOST_EXPECTED_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
case "${HOST_DTW_SHARD}" in 0|1|2|3) ;; *) exit 2;; esac
export CUDA_VISIBLE_DEVICES="${HOST_DTW_SHARD}"
exec bash "${REPO_DIR}/deploy/autodl/run_python.sh" "${REPO_DIR}/alignment/piper_dtw_evaluate.py" \
 --root /pfs/user/data/host_piper_paper/alignment_visual_20260908 \
 --weights /root/host_piper_runtime/weights/qwen3_vl_embedding_8b_2c4565515e0f \
 --checkpoints /root/autodl-tmp/host_piper/runs/alignment_4task_weights_20260909/weights \
 --checkpoint-step 1500 --split both --per-task 0 --num-shards 4 --shard-index "${HOST_DTW_SHARD}" \
 --output "/root/autodl-tmp/host_piper/dtw_evaluations/full1500_20260914_shard${HOST_DTW_SHARD}" --device cuda
