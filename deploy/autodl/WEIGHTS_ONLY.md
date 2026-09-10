# Rolling model-only checkpoints

The user superseded the previous final-only request: intermediate checkpoints must contain
model weights only, with just the latest few retained. The user explicitly authorized a restart,
accepting loss of the previous unsaved steps. Old logs/data are preserved.

Use train_alignment_weights.sh, with the exact published clean commit approval.
New output: /root/autodl-tmp/host_piper/runs/alignment_4task_weights_20260909.

- Save step20, every100 optimizer steps, and final step3000.
- Keep latest3 complete checkpoints, including final; never create a duplicate final copy.
- One consolidated BF16 pytorch_model.bin contains the entire alignment model including its
  embedding/projection head, parameters and persistent model buffers. No optimizer, gradients,
  RNG state, scheduler state, activation cache, or separate Qwen copy.
- Expected16-18GiB each. Three retained plus one incoming is about64-72GiB, fitting the verified
  ~120GiB data-disk headroom. Enforce32GiB free before each save and40GiB before launch.
- All4 ranks participate in DeepSpeed layer-wise consolidation to rank0 CPU memory.
  Use the official save_16bit_model API with ZeRO3 gather enabled; no save_checkpoint call
  executes in weights-only mode.
- Write a new .incomplete directory, reload tensors with weights_only=True, verify exact key
  set/shapes/dtypes, finite values and SHA256, then rename atomically to a complete directory.
  Only after validation delete older run-owned complete checkpoints; never delete unknown
  files, symlinks, foreign checkpoints or the previous valid copy before successful save.
- Tiny tests are not real alignment weights. Full-model memory/save validation remains required
  when the first training checkpoint is produced.
- Models load for DTW/inference. They support warm-starting a new optimizer, NOT exact
  optimizer/scheduler/RNG-state continuation.
- Save step20 BEFORE the existing65GiB peak-reserved safety gate; if it fails, retain the
  weights and stop for review. The threshold is not relaxed or bypassed.
- Release unused CUDA allocator cache at optimizer-step boundaries and around consolidation
  to leave memory headroom. Live model tensors and mathematical training configuration unchanged.
- Still3000 steps, official24/96 anchors and micro4/accum4, CPU optimizer offload, four tasks.
  No automatic retry/resume. Do not hot-edit the running checkout.
Update after live memory review: the old micro4 run reached step5 with peak allocated~49GiB
but peak reserved~69GiB. The conservative restart uses microbatch2/accumulation8 (global64),
not the earlier micro4/accumulation4 noted above. Anchors24/96 and LR/schedule unchanged.
This gives additional activation headroom; no claim of zero OOM risk before measurement.
