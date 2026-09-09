# Four-task alignment on AutoDL

User authorized this deployment and training, and requested **one final checkpoint only**.
This supersedes the earlier IDC-only workflow for this run, not for other projects.
No intermediate save, rotating save, automatic backup, or retry. Interruption loses this run's
unsaved progress. The final artifact is one complete sharded DeepSpeed checkpoint (multiple
files forming one checkpoint), not an additional consolidated Hugging Face copy.

## Isolation and storage

- Code: /pfs/user/code/host_piper_paper, branch piper_paper, clean published commit.
- AutoDL runtime: /root/host_piper_runtime/{python310,envs,weights,cache,tmp} on system disk.
- AutoDL data: /root/autodl-tmp/host_piper, aliased by /pfs/user/data/host_piper_paper.
- Base image Python/Torch remain untouched. Python3.10 -S imports copied alignment packages:
  torch2.4.0+cu124 / transformers4.57.3 / deepspeed0.14.4; CUDA toolkit12.4.
- The runtime and16.3GB initial weights use about22GiB of30GiB system disk.
  Raw videos/metadata and the265MiB pilot use data disk. Reserve110GiB before training
  and before the single final save; previously measured full ZeRO3 checkpoint is about98GiB.
- Transfer directly IDC to AutoDL, serial, no local bulk downloads or credential files.
- No full lossless cache: PyAV sequential decoding avoids the known HEVC random-seek failure.

## Effective training

Four A80080GB PCIe,480GiB cgroup RAM. All four tasks:796 training /88 held-out episodes,
2652 videos. The held-out split is NOT used for optimization; validation and DTW quality
evaluation are subsequent work, not claimed by completing this training loop.

Released alignment defaults:24 anchors/chunk, up to96 batch frames, M probabilities
0.1/0.2/0.7, microbatch4/rank, accumulation4, BF16 ZeRO3, AdamW1e-5,
official warmup/cosine100/3000. Run3000 actual optimizer updates.
Piper profile: three fixed camera identities, no random horizontal flip or joint inputs.
Two loader workers/rank, prefetch1. W&B disabled; no videos logged.

Memory changes are explicit: CPU optimizer offload and logits_to_keep=1/use_cache=False.
The latter removes unused all-token vocabulary logits; embedding loss uses hidden states.
A tiny random CPU Qwen test on real HOST inputs matched loss and100 gradient tensors;
816 sampled RGB frames across48 videos matched the certified cache exactly.
All2652 raw videos passed full sequential frame-count decode audit on IDC.
These CPU checks do NOT prove pretrained8B GPU peak memory or alignment quality.

After20 real updates, all-rank peak reserved memory must be <=65GiB; otherwise fail without
saving and review. Passing continues automatically to3000. This is a safety gate, not a
guarantee that later batches cannot OOM. No automatic parameter reduction or resubmission.

## Launch

First run verify_transfer.py with run_python.sh and verify all checksums, isolated imports,
CUDA/CPUAdam build, free GPUs, disk headroom, exact published clean commit. Then:

HOST_AUTODL_TRAIN_APPROVED=1 HOST_EXPECTED_COMMIT=<published-full-SHA> \
  bash deploy/autodl/train_alignment.sh

Entrypoint refuses a dirty or unexpected commit and never automatically restarts.
Output: /root/autodl-tmp/host_piper/runs/alignment_4task_finalonly_20260909.
Per-rank metrics report actual optimizer steps, loss, gradient norm and CUDA peaks.
checkpoint/final is written only after step3000, then train_complete.json is created.
No DTW labels or policy-training-ready marker are produced here.
Record actual commit, PID, output and verification evidence in a sanitized run ledger.
Freeze this checkout once training starts; use another checkout for concurrent development.
