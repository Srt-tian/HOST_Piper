# Piper paper-method adaptation

Authoritative checkout: IDC `/pfs/user/code/host_piper_paper`, branch `piper_paper`.
Upstream source: https://github.com/CGuangyan-BIT/HOST.git at
`9f3bba57792aa5053ac600b1b7a4625f96ea6662`.

## Scope and isolation

- All datasets, weights, virtual environments, caches and generated outputs live under
  `/pfs/user/data/host_piper_paper`. No system/KAI0 dependency changes.
- Do not launch training or submit EIP jobs without the user's resolved-resource confirmation.
- User is supplying multiple task roots. Pair episodes only within the same semantic task
  and split. Do not equate acquisition task IDs with distinct semantic tasks.
- User confirmed state_end left[0:6]/right[6:12], each in its own base coordinates,
  ordered xyz/RPY. User subsequently authorized m/rad as a working assumption; all-row
  range evidence supports it. RPY convention, TCP and command-target semantics remain unverified.
- Keep real_action/action_end separate from executed qpos/state_end. Conversion must require
  an explicit target-source selection and confirmed coordinate convention; never guess.

## Work and acceptance

1. Establish verified Git checkout and isolated policy environment; import/CPU tests.
2. Add manifest-driven multi-task LeRobot converter, finite/length/time checks, deterministic
   episode split, split-local same-task peers, train-only min/max normalization.
3. Preserve paper EEF physical20+aux2/proprio20, 3 x 224x224 vertical RGB concat;
   distinguish paper settings from the released 2-camera checkpoint compatibility profile.
4. Handle camera embeddings explicitly; reject other checkpoint shape mismatches.
5. Require real alignment/coupling output and text embeddings before policy training.
   No fabricated progress or zero text embeddings accepted for training.
6. Set up alignment in a separate environment (upstream transformer/torch versions differ).
7. Once task roots and semantics are resolved: pilot transfer, full download, convert,
   alignment/coupling, GPU smoke, measured memory and reproducible launch review.

## Known differences requiring evidence

Paper says 32-action horizon, 8:1 video sampling, 3 cameras, all velocity targets.
Released config uses dataset-dependent horizon, 2 cameras and sample targets for action/progress.
30Hz Piper must not be mislabeled 32Hz. First adaptation retains 30Hz with ratio8 and explicitly
records the resulting horizon30/reference21frames as a code-compatible deviation.
Exact 32/24-frame paper sampling is a separate sampler change needing joint video/action tests.
Public HOST weights omit the trained alignment network and optimizer state. GPU memory and
full checkpoint reload have not yet been validated.

## Progress-coordinate audit

The coupling script uses `(matched_ref_frame - ref_start + 1) / ref_length`, then interpolates
unmatched main frames. It is not main_index/main_length. It writes `ref_path`, but the policy
loader only consumes `aligned_progress` and ignores that coordinate provenance. Therefore,
independently normalizing every episode against a different reference can make labels
incomparable. Use a reviewed common reference per task/split (or explicitly rebase pairwise
alignments); the Piper preflight rejects inconsistent references. Canonical-reference selection
and model-derived alignments are still pending. Do not replace these with linear progress.

User explicitly confirmed (2026-09-08): train progress prediction with the offline visual/DTW
alignment-derived GT, converted to the sampled reference-window coordinate. No per-episode
linear-index labels as a substitute. This confirms the supervision method, not authorization
to submit an EIP training job or a resolution of the pending EEF conventions.

## Continuation, 2026-09-08

- Visual-only alignment staging completed: 884 episodes /509613 frames, train796 /val88,
  same episode split as policy conversion, 8 task/split anchor coordinates. No action assumptions.
- Added explicit Piper alignment profile, canonical eval-pair builder, strict three-camera
  progress exporter, temporal/seek audit, lossless seek-safe pilot and candidate review evidence.
- 25 policy-env CPU tests passed. Real alignment dataset/path grouping passed for18 sampled
  episodes across9 sources x2 splits with3-path and6-path groups; no model/GPU/GT validation.
- Critical video finding: 6/27 pilot HEVC videos disagree between Decord sequential and random
  access (max mean RGB error up to88/255). Six lossless all-intra H264 derivatives passed full
  sequential RGB digest equality to raw plus sampled forward/backward seek checks.357MiB cache.
  The rest of the2652 videos are NOT certified. Do not train directly on unresolved raw seeks.
- Canonical derived video PTS encodes row index at30fps without dropping/duplicating frames;
  actual capture time remains in parquet. This does not repair collection gaps.
-778/884 trajectories contain a gap>1.5 nominal periods; worst total index-clock drift0.3s.
  A gap-aware policy-window strategy remains to be implemented/validated; do not delete entire
  episodes or silently interpolate commands based only on these statistics.
- Reviewed8 candidate contact sheets at8 temporal samples/camera and installed explicit eval
  reference lists. Pilot screen only; manipulation order/objects vary across multi-object tasks.
  Prefer lemon for first end-to-end GPU/data pilot, then expand to all four tasks.
- No bulk weights, true DTW labels, text embeddings, GPU memory measurement or training job yet.
  Pending: EEF conventions/target choice, full decode-cache strategy and temporal-window handling,
  model artifacts/alignment training approval, GT quality review, canonical Git recovery and
  fully resolved EIP launch confirmation. Changes are still uncommitted/unpushed.

## Second continuation: verified pilot and alignment artifacts

- GOP16 lossless H264/no B frames passed full source-vs-cache RGB digests plus sampled
  seek checks for all6 known-problem videos. Cache185MiB versus357MiB all-intra.
- Built `lemon_seek_safe_pilot_20260908`:12train/4val,48 verified videos,3347 frames,
  ~265MiB. Deterministic subset preserves the existing split and its reviewed anchors.
  This is visual-only and contains NO policy actions, DTW labels or training-ready marker.
- Real public Qwen tokenizer/processor + native HOST collator passed with distinct Main/Ref,
  4 anchors each,2x977 input IDs and1680 video placeholder tokens. A RANDOM tiny Qwen model
  also passed the actual patched forward and HOST CLS extraction (8x128 finite features).
  No optimizer, pretrained8B forward, GPU or real progress labels were involved.
- Added gap-free candidate-window primitive and all-data audit, not yet wired into sampler.
  Retention for31-frame windows: bottles94.49%, pens95.62%, basket88.63%, lemon93.71%.
  For61 frames:89.70%,91.36%,79.86%,87.54%. Avoid whole-episode deletion for isolated gaps.
  Static-frame filtering/variable action spacing still need joint sampler integration tests.
- Qwen3-VL-Embedding-8B pinned at2c4565515e0f265c6511776e7193b22c0968ddc7;
  selected files total16,305,659,301 bytes. Small config/tokenizer files verified using Git
  blob hashes/LFS SHA256. Full weights download is still RUNNING, not yet verified complete.
  Source path: weights/qwen3_vl_embedding_8b_2c4565515e0f under the dedicated IDC data root.
- Download preflight checked hostname, data path,~8.6TiB free, unset proxies,1MiB HTTP206 pilot.
  HF metadata requires the host curl client (urllib403); large HF transfer was slow.
  Alternate public Qwen ModelScope1MiB prefix/size matches pinned HF artifact. Stopped only
  the owned slow downloader (PID3509776), preserved its partial, resumed serially with strict
  Content-Range checks and final pinned SHA256. No concurrent writers or automatic retries.
  Active log: logs/fetch_alignment_weights_20260908_modelscope.log. Inspect before restarting.
-29 CPU tests passed before additional resume-header negative tests were added; final rerun
  log is logs/piper_tests_20260908_preparation_r2.log. Check actual completion there.
- Developer container reports A800-SXM4-80GB but memory/utilization queries lack permission.
  This is NOT a verified free/usable GPU allocation or a memory measurement.
- A writable PRIVATE canonical Git remote is needed before EIP training submission. Only the
  official origin is configured; do not publish internal manifests/runbooks to a public fork
  without user direction. Still no EIP job or training started.

## IDC recovery and EIP smoke preparation

- User selected IDC for pipeline debugging; AutoDL remains out of the active workflow.
  User explicitly selected the IDC four-A800 debug queue for smoke. This is resource preference,
  not confirmation of a fully resolved submission payload. Do not silently submit.
- IDC SSH/PFS recovered after the earlier IDFS wait. Prior downloader/tests were no longer
  alive; first Qwen shard was partial (~2.68GB), final31-test log incomplete. Rechecked hostname,
  isolated path,~8TiB free, unset proxies and1MiB ModelScope range/prefix; serial resume started
  with no automatic retries. Current log: logs/fetch_alignment_weights_20260908_resume.log.
  A running transfer is not verified completion; check final marker and hashes before model use.
- New all-row pose audit:884 episodes/509613 rows, four tasks/nine sources. state_end xyz extrema
  across all sources -0.471764..0.632469; action_end -0.472536..0.636369. Both RPY fields lie
  within[-pi,pi] allowingfloat32 rounding. This supports m/rad; it does NOT identify rotation
  order/TCP or prove command semantics. inventory.confirmed staysfalse. Evidence and parquet
  SHA256s: logs/pose_ranges_20260908_resume.json.
-31 CPU tests passed after recovery, then36 passed after adding smoke/pose gates (19.58s).
  Final log: logs/piper_tests_20260908_smoke_gates.log. bash syntax and git diff checks passed.
- alignment/piper_smoke.py CPU RANDOM reduced-Qwen mode ran actual videos, HOST patched
  forward, TCC/SmoothDTW loss,2 optimizer steps and strict fresh-model checkpoint reload.
  Loss1.092342->1.035807; replay0.973729; both grad norms finite/nonzero. This is NOT an8B
  GPU test, semantic-quality result or production alignment checkpoint. Artifacts are confined
  to cpu_alignment_training_smoke_20260908; no progress labels exported.
- EIP smoke entrypoint deploy/idc/smoke_alignment.sh runs4-rank BF16 ZeRO-3, batch1/rank,
 4-anchor reduced length,2 steps then a fresh torchrun process for full checkpoint reload.
  Enforces approved flag, exact clean Git commit/upstream, complete pinned8B artifacts,
  48 certified video hashes and disk headroom. Current artifact preflight correctly rejects
  incomplete weights; video certificates pass. Full24/96-anchor memory remains a later test.
- Candidate immutable image from local inspect:
  mfyz.icompify.com:5000/magiclab/lingbot-va@sha256:341233831cea90ce9ce90fd8c2d69be02099a0c10101d63109b9dd831ee48e98.
  Image Python3.10/CUDA12.6 supplies runtime; -S and explicit HOST alignment site-packages
  keep torch2.4+cu124/transformers4.57.3/deepspeed0.14.4 isolated from image Python packages.
- EIP get_queue_specs for42147c08-9469-4012-80c0-890b988a07a8 returnedml.a800.4:
  Cpu120/MemoryMiB409600/GpuCount4/GpuTypeNVIDIA-A800-SXM4-80GB/GpuMemoryMiB81920/
  ShmSizeMiB204800. ZoneInfo wasempty; zone and live capacity must still be resolved/checked
  before payload confirmation. Do not infer current free GPUs from supported flavors.
- User has been asked for a writable PRIVATE canonical Git URL; no answer yet. The official
  origin cannot recover these uncommitted adaptations. No EIP submission until code recovery,
  artifacts and exact resource/image/command/data/output confirmation all satisfy the gates.

## Canonical publication decision

- User supplied https://github.com/Srt-tian/HOST_Piper and, after being told that it is public
  and that the adaptation contains internal path/task configuration, explicitly confirmed:
  "不用，就public就行". This supersedes the earlier private-remote prerequisite for this repo.
- Publish reviewed source/configuration/runbooks only, retain upstream licensing/attribution,
  and exclude credentials, raw datasets, videos, model weights and generated checkpoints.
- IDC /pfs/user/code/host_piper_paper on piper_paper remains the authoritative checkout.
  Local Git may act only as a code-object transport when IDC cannot reach GitHub; verify
  identical commits before/after transfer, never develop in that temporary transport copy.
- Keep the official source as upstream and use Srt-tian/HOST_Piper as canonical origin.
  Git recovery/clean-worktree and fully resolved EIP launch gates remain in force.

## Official-configuration probe preparation, 2026-09-09

- Task11534 successfully completed real pretrained8B four-A8002-step smoke and
 full fresh-process checkpoint reload; allocated39.061GiB/reserved49.453GiB peaks.
 Earlier "GPU not validated" entries above are historical, superseded for this
 short4-anchor smoke ONLY. Production quality/long lengths remain unverified.
- User replaced the proposed gradual length sweep with the released configuration.
 New wrapper runs original alignment/train.py with24/96 anchors, officialM mix,
 microbatch4/rank and single-node launcher accumulation4;10 actual engine updates
 with official3000-step/100-warmup schedule, then full independent-process reload.
 See deploy/idc/OFFICIAL_PROBE.md for exact adaptations, limits and acceptance.
- CPU official train sampler/real processor passed4 rank cases:8x4945 input IDs,
 24CLS/row;40 CPU tests passed. Certified12-train lemon dataset uses weighted
 replacement sampling perrank, not12/4 distributed slicing.
- New GPU probe is NOT submitted; await exact resource/code review and confirmation.
 No length-sweep code, formal DTW labels, production alignment or policy training.

## AutoDL final-only alignment, 2026-09-09

- IDC official probe11552 failed on an11.20GiB all-token vocabulary-logit allocation;
  one actual optimizer update completed, no checkpoint. See sanitized external run ledger.
- User authorized AutoDL training on the newly supplied host and explicitly requested
  only one final checkpoint. This supersedes the earlier IDC-only preference for this run.
  AutoDL runtime/weights use isolated /root/host_piper_runtime on system disk; data/output
  use /root/autodl-tmp/host_piper with IDC-compatible aliases. No base/KAI0 package changes.
- Added CPU optimizer offload, explicit logits_to_keep=1, raw PyAV sequential decoding,
  and a final-only3000-actual-update trainer retaining24/96 anchors and micro4/accum4.
  Twenty-update reserved-memory gate65GiB; not a guarantee against later OOM.
- Full sequential audit passed all2652 videos/884episodes. Exact resized RGB equality
  passed816 samples across48 pilot videos. Tiny CPU full-vs-one-token logits test matched
  loss and100 gradient tensors; not a real8B GPU verification.44 CPU tests passed.
- See deploy/autodl/README.md for storage, checksum/commit gates, command and limitations.
  No automatic intermediate checkpoint, backup or retry. No DTW labels produced.
  Deployment transfer in progress at preparation; actual launch is recorded separately.
