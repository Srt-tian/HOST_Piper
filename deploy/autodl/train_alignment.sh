#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_AUTODL_TRAIN_APPROVED:?}"
: "${HOST_EXPECTED_COMMIT:?}"
test "${HOST_AUTODL_TRAIN_APPROVED}" = 1
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${HOST_EXPECTED_COMMIT}"
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
RUNNER="${REPO_DIR}/deploy/autodl/run_python.sh"
export HOST_ALIGNMENT_MODEL_PATH=/root/host_piper_runtime/weights/qwen3_vl_embedding_8b_2c4565515e0f
bash "${RUNNER}" -c 'import torch,transformers,deepspeed; assert torch.__version__=="2.4.0+cu124"; assert transformers.__version__=="4.57.3"; assert deepspeed.__version__=="0.14.4"; assert torch.cuda.device_count()==4; print("Portable isolated environment verified")'
exec bash "${RUNNER}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 --max_restarts=0 \
 --no-python /bin/bash "${RUNNER}" "${REPO_DIR}/alignment/piper_autodl_train.py" \
 --root /pfs/user/data/host_piper_paper/alignment_visual_20260908 \
 --weights "${HOST_ALIGNMENT_MODEL_PATH}" \
 --output /root/autodl-tmp/host_piper/runs/alignment_4task_finalonly_20260909
