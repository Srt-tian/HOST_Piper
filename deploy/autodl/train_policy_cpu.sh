#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${HOST_RUN_DIR:?Set a dedicated run directory}"
: "${HOST_INIT_CHECKPOINT:?Set a verified weights-only initialization checkpoint}"
: "${HOST_EXPECTED_COMMIT:?Set the published execution commit}"
cd "$REPO_DIR"
test "$(git rev-parse HEAD)" = "$HOST_EXPECTED_COMMIT"
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = piper_paper
test "$(git rev-parse '@{upstream}')" = "$HOST_EXPECTED_COMMIT"
test -s "$HOST_INIT_CHECKPOINT"
test -s "${HOST_INIT_CHECKPOINT%.pt}.verified.json"
test ! -e "${HOST_RUN_DIR}/health_rank0.jsonl"
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '
  {n++; if ($1 < 70000) bad=1} END {if(n!=4 || bad) exit 1}'
test "$(df -B1 --output=avail /root/autodl-tmp | tail -n1)" -gt 26843545600
export HOST_POLICY_DS_CONFIG="${REPO_DIR}/deploy/autodl/policy_zero2_cpu.json"
exec bash deploy/autodl/run_policy_python.sh -m torch.distributed.run \
  --standalone --nproc_per_node=4 --no-python bash deploy/autodl/run_policy_python.sh \
  policy_training/scripts/train.py task=piper_autodl_cpu "$@"
