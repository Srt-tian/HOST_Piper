# Held-out DTW pilot

User requested latest checkpoint DTW evaluation, and next-stage preparation only if quality is good.
Training remains on its frozen checkout/commit. Use a separate /pfs/user/code/host_piper_dtw_eval
checkout; do not edit the active training checkout or raw data.

- Select two non-self validation episodes per task by duration25th/75th quantiles, deterministic,
  using installed reviewed same-task/split canonical reference paths. Eight pairs total.
- Latest COMPLETE checkpoint is resolved at evaluation start, SHA256 checked and hard-linked
  into the evaluation artifact directory. This temporarily retains one extra weight snapshot
  after training rotation removes it; disk preflight includes that retention.
- One shared GPU only; require30GiB physical free before loading, cap evaluation Torch allocator
  at22GiB. Training measured historical reserve~46GiB leaves margin. Independent process may
  slow training temporarily. Do not catch-and-retry OOM or reduce evaluation sampling silently.
- Official96-frame4x evaluation,24 anchors/chunk and3 cameras, no augmentation, full pair mode.
  Preserve official forward-backward SmoothDTW argmax/coupling semantics (not hard-DTW repair).
- Inference-only hooks disable retaining all decoder hidden states and expose the identical last
  hidden state to the existing HOST feature extractor. Tiny exact-equivalence test required.
- Strict checkpoint load; no optimizer or training writes; exact existing coupling gates:
  loss<=0.15, coverage>80%, no backward indices, maximum jump<=7/24 reference anchors.
- Produce records, per-pair diagnostics, actual sampled RGB images, curves,8-sample contact
  sheets and an interactive HTML viewer against time-index baseline. Baseline is not GT.
- Passing numerical gates alone is NOT proof of semantic alignment. Review objects/phases,
  long plateaus, endpoints and task-order compatibility before enabling next-stage data work.
- Next safe stage if pilot passes: expand held-out review, then same-checkpoint full offline
  labels with canonical task/split references. No policy training or fabricated linear GT.
