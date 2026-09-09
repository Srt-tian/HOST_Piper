#!/usr/bin/env bash
# Released training configuration; bounded at10 real optimizer updates, no retries.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_OFFICIAL_PROBE_APPROVED:?Resolved EIP launch approval required}"
test "${HOST_OFFICIAL_PROBE_APPROVED}" = 1
: "${HOST_EXPECTED_COMMIT:?}"
: "${HOST_SMOKE_ROOT:?}"
: "${HOST_ALIGNMENT_MODEL_PATH:?}"
: "${HOST_OFFICIAL_PROBE_OUTPUT:?}"
test "$(git -C "${REPO_DIR}" rev-parse HEAD)" = "${HOST_EXPECTED_COMMIT}"
test "$(git -C "${REPO_DIR}" branch --show-current)" = piper_paper
test -z "$(git -C "${REPO_DIR}" status --porcelain)"
git -C "${REPO_DIR}" merge-base --is-ancestor HEAD '@{upstream}'
RUNNER="${REPO_DIR}/deploy/idc/run_eip_alignment_python.sh"
bash "${RUNNER}" -c 'import sys,torch,transformers; assert sys.version_info[:2] == (3,10); assert torch.__version__ == "2.4.0+cu124"; assert transformers.__version__ == "4.57.3"; assert torch.cuda.device_count() == 4'
bash "${RUNNER}" "${REPO_DIR}/deploy/idc/preflight_alignment_smoke.py" \
  --root "${HOST_SMOKE_ROOT}" --weights "${HOST_ALIGNMENT_MODEL_PATH}" --output "${HOST_OFFICIAL_PROBE_OUTPUT}"
mkdir "${HOST_OFFICIAL_PROBE_OUTPUT}"
cd "${HOST_OFFICIAL_PROBE_OUTPUT}"
export WANDB_PROJECT=host_piper_official_probe
for HOST_PROBE_PHASE in train reload; do
  timeout --signal=TERM --kill-after=60s 2400s \
    bash "${RUNNER}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    --max_restarts=0 --no-python /bin/bash "${RUNNER}" "${REPO_DIR}/alignment/piper_official_probe.py" \
    --mode "${HOST_PROBE_PHASE}" --root "${HOST_SMOKE_ROOT}" --weights "${HOST_ALIGNMENT_MODEL_PATH}" \
    --output "${HOST_OFFICIAL_PROBE_OUTPUT}"
done
test -f "${HOST_OFFICIAL_PROBE_OUTPUT}/reload_complete.json"
printf 'Official-config10-update probe and full reload complete; no quality/DTW claim.\n'
