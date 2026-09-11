#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_EXPECTED_COMMIT:?}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${HOST_EXPECTED_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
export CUDA_VISIBLE_DEVICES=0
RUNNER="${REPO_DIR}/deploy/autodl/run_python.sh"
OUTPUT=/root/autodl-tmp/host_piper/dtw_evaluations/pilot_serial_20260911
bash "${RUNNER}" "${REPO_DIR}/alignment/piper_dtw_evaluate.py" \
 --root /pfs/user/data/host_piper_paper/alignment_visual_20260908 \
 --weights /root/host_piper_runtime/weights/qwen3_vl_embedding_8b_2c4565515e0f \
 --checkpoints /root/autodl-tmp/host_piper/runs/alignment_4task_weights_20260909/weights \
 --output "${OUTPUT}" --per-task 2 --device cuda
bash "${RUNNER}" "${REPO_DIR}/deploy/autodl/render_dtw_review.py" --root "${OUTPUT}"
