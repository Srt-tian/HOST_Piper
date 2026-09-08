#!/usr/bin/env bash
# Single-node entrypoint; do not execute until the resolved EIP launch is approved.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_TRAIN_APPROVED:?Explicit resolved-resource launch approval is required}"
test "${HOST_TRAIN_APPROVED}" = 1
: "${HOST_PREPARED_ROOT:?}"
: "${HOST_INIT_CHECKPOINT:?}"
: "${HOST_RUN_DIR:?}"
: "${HOST_NUM_GPUS:?}"
: "${HOST_EXPECTED_COMMIT:?}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${HOST_EXPECTED_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
# The execution commit must be recoverable from the canonical remote before submission.
git -C "${REPO_DIR}" merge-base --is-ancestor HEAD '@{upstream}'
test -s "${HOST_INIT_CHECKPOINT}"
bash "${REPO_DIR}/deploy/idc/run_policy_python.sh" \
  "${REPO_DIR}/policy_training/scripts/piper_preflight.py" --root "${HOST_PREPARED_ROOT}"
cd "${REPO_DIR}/policy_training"
bash "${REPO_DIR}/deploy/idc/run_policy_python.sh" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${HOST_NUM_GPUS}" scripts/train.py \
  task="${HOST_PROFILE:-piper_3cam_paper_objective}" "$@"
