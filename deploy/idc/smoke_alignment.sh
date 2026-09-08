#!/usr/bin/env bash
# EIP entrypoint: two optimization steps then fresh-process ZeRO-3 reload.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_SMOKE_APPROVED:?Resolved EIP launch approval is required}"
test "${HOST_SMOKE_APPROVED}" = 1
: "${HOST_EXPECTED_COMMIT:?}"
: "${HOST_SMOKE_ROOT:?}"
: "${HOST_ALIGNMENT_MODEL_PATH:?}"
: "${HOST_SMOKE_OUTPUT:?}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${HOST_EXPECTED_COMMIT}"
test "$(git -C "${REPO_DIR}" branch --show-current)" = piper_paper
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
git -C "${REPO_DIR}" merge-base --is-ancestor HEAD '@{upstream}'
RUNNER="${REPO_DIR}/deploy/idc/run_eip_alignment_python.sh"
bash "${RUNNER}" -c 'import sys,torch,transformers; assert sys.version_info[:2] == (3,10); assert torch.__version__ == "2.4.0+cu124"; assert transformers.__version__ == "4.57.3"; assert torch.cuda.device_count() == 4; print("EIP smoke environment verified")'
bash "${RUNNER}" "${REPO_DIR}/deploy/idc/preflight_alignment_smoke.py" \
  --root "${HOST_SMOKE_ROOT}" --weights "${HOST_ALIGNMENT_MODEL_PATH}" --output "${HOST_SMOKE_OUTPUT}"
for HOST_SMOKE_PHASE in train reload; do
  bash "${RUNNER}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    --max_restarts=0 --no-python /bin/bash "${RUNNER}" "${REPO_DIR}/alignment/piper_smoke.py" \
    --mode "${HOST_SMOKE_PHASE}" --root "${HOST_SMOKE_ROOT}" --weights "${HOST_ALIGNMENT_MODEL_PATH}" \
    --output "${HOST_SMOKE_OUTPUT}" --anchors 4 --steps 2
done
test -f "${HOST_SMOKE_OUTPUT}/reload_complete.json"
printf 'Alignment smoke completed; this is NOT a production alignment checkpoint or DTW label set.\n'
