# IDC HOST / Piper workspace

Code: `/pfs/user/code/host_piper_paper` (`piper_paper`).
Data/weights/env/cache/logs: `/pfs/user/data/host_piper_paper`.
Upstream base commit: `9f3bba57792aa5053ac600b1b7a4625f96ea6662`.

## Isolation

`bash deploy/idc/setup_policy.sh` creates only a dedicated venv. No conda activation,
system pip install, KAI0 changes, Docker image modification or global proxy settings.
`bash deploy/idc/run_policy_python.sh ...` uses an explicit interpreter and isolated caches.
Model access is offline by default, so a missing artifact fails instead of downloading silently.
HOST's unused torchcodec 0.5 dependency was removed (not compatible with torch2.6);
hub pin raised to0.36 to satisfy diffusers0.36. All other package constraints remain inspectable.

CPU tests may first run in the existing immutable lingbot container with small dependencies
under `cache/cpu_test_deps`; this does NOT establish policy torch2.6 GPU compatibility.
The container is removed after the check; the image and host Python are not modified.

## Data pipeline

1. `python3 deploy/idc/inventory_tos.py --pilot`: serial metadata inventory + first-episode
   parquet/3-camera pilots for the four user-provided task roots. Generates `inventory.json`.
2. After hostname/disk/proxy/size/pilot review: `python3 deploy/idc/download_raw.py`.
   Uses eip-kit global auth and its TOSClient, 2 concurrent files, CRC checks; excludes annotation.
3. `data_preprocessing/piper/audit_raw.py --manifest .../inventory.json`: read-only full numeric
   audit and video container count/first-frame checks. This is not full quality/success labeling.
4. Confirm units, RPY, TCP and target-source fields in a NEW versioned manifest.
   `convert_lerobot.py --manifest ... --output /absolute/new/output` then writes HOST files,
   immutable video symlinks, split-local same-task peers and train-only normalization.
5. Run the upstream alignment -> coupling workflow to produce genuine `info_dtw.json`.
   A trained alignment checkpoint is not included in the public HOST policy weight artifact.
   `alignment/environment.yml` requires a separate environment (transformers4.57.3/torch2.4).
6. Precompute real T5 embeddings with the upstream script, then run `piper_preflight.py`.
   Missing progress/text, cross-task/split peers and degenerate progress are errors.

Raw action fields remain distinct: follow=state_end/qpos; master=action_end/real_action.
The name master is the HOST schema, not independent proof of collector semantics.
The converter refuses to select a target source without confirmation. No hidden +1 shift.
Both EEFs stay in their own arm-base coordinate systems.

## Profiles and initialization

- `piper_3cam_checkpoint_compat`: 20+2 action /20 proprio, 3 cameras, relative actions,
  released sample objectives, native30Hz, ratio8, no augmentation/drop for first validation.
- `piper_3cam_paper_objective`: absolute EEF and velocity objectives, paper optimizer settings,
  static filtering/augmentation. Still native30Hz: not exact paper32Hz temporal reproduction.
- Both retain upstream VAE-compatible video sampling: action30, agent5 rawframes,
  reference21 rawframes for6 keyframes. Do not describe this as paper32/24 sampling.
- Run `piper_camera_checkpoint.py` on CPU to explicitly adapt only camera_emb/pos_embed.
  Source camera identities must be known before copying specific indices. Otherwise initialize
  all three camera/position blocks (`[null,null,null]`) while preserving visual backbones.
  Training loads the adapted result with normal strict visual-encoder loading.

Required profile variables: HOST_PREPARED_ROOT, HOST_DINO_REPO, HOST_DINO_WEIGHTS,
HOST_SIGLIP_WEIGHTS, HOST_INIT_CHECKPOINT, HOST_RUN_DIR. Supply local VAE/text artifacts too.
No training was submitted. Before training: verify clean branch/commit, canonical recovery,
all artifacts, actual GPU access and peak memory, then obtain resolved EIP launch approval.

## Visual-only preparation and strict coupling

While EEF conventions are pending, use `prepare_alignment.py` to build video-only episodes.
It creates the same train/val partition as the policy converter, but no action JSON/GT/text.
The current output is `/pfs/user/data/host_piper_paper/alignment_visual_20260908`.

`run_alignment_container.sh alignment/piper_profile.py --root <visual-root> --output <new-json>`
is a CPU dataset smoke (use an absolute script path inside the container). `--context-frames 1`
uses3 paths/anchor;2 uses6. `--dataset-mode eval` reads explicit canonical eval-pair files.
This helper does NOT run the model or train. A later training entrypoint must call
`apply_profile()` and `enable_strict_loading()` before constructing upstream datasets.
Set `HOST_ALIGNMENT_MODEL_PATH` to the locally verified Qwen directory for both model/processor.

Screen reference candidates using `review_anchor_candidates.py`, then use
`prepare_eval_pairs.py --root <visual-root> --review <review-json>` to install canonical eval
references. No missing-file self-alignment fallback is allowed in the Piper profile.
`build_progress.py --root ... --records ... --anchors ... --output <new-dir>
--model-id <alignment-checkpoint-identifier> --context-frames 1` validates real evaluation
records and writes a separate label artifact. See PROGRESS.md for exact semantics/deviations.
Real model training/evaluation and label installation/rebasing remain pending.

Do not train on raw HEVC symlinks before resolving seek integrity. The27-video pilot found6
severe random-seek discrepancies. `cache_seek_safe.py` validated6 lossless derivatives with
full pixel digests and sampled seek comparisons. Canonical CFR video PTS is only a row-index
clock; raw capture timestamps still contain gaps and must guide future action-window checks.

## Verified lemon pilot and model preparation

`prepare_seek_safe_pilot.py` builds a bounded, new visual dataset from the existing split and
reviewed canonical anchors. Completed output: `lemon_seek_safe_pilot_20260908`,12train/4val,
48 verified videos/3347 frames/~265MiB. GOP16 remains lossless and passed the6-video regression
pilot. Raw datasets are untouched. This subset is for pipeline validation, not final evaluation.

`alignment/piper_processor_smoke.py --root <pilot> --weights <local-Qwen> --output <new-json>
--tiny-forward` tests the real processor and HOST collator, then optionally a RANDOM small
Transformer through the actual patched interface. This is strictly a CPU structural check;
it writes no features/GT and provides no8B memory or semantic-alignment evidence.

`time_windows.py` audits which raw contiguous windows avoid capture gaps>1.5/fps. Its primitive
is tested against an exhaustive implementation. It has NOT been integrated with the policy's
static filtering and variable-interval sampler; do not advertise the training sampler as fixed.

`fetch_alignment_weights.py plan` pins public Qwen artifacts to revision
`2c4565515e0f265c6511776e7193b22c0968ddc7`, checks IDC/storage/proxies and a1MiB pilot.
`download` verifies small files then transfers shards serially. `probe-modelscope` validates
the alternate public source's size/prefix; `download-modelscope --resume-partials` is an
EXPLICIT recovery operation after stopping the owned prior transfer. Existing final files are
hash-checked; resume responses must match the exact byte range before append. Full shard
SHA256 remains pinned to the HF revision even though the alternate publication uses master.
Inspect active processes/logs before any rerun; never start two writers on these artifacts.
`verified_artifacts.json` appears only after every selected file passes final checks.

## EIP alignment smoke (prepared, not yet GPU validated)

`alignment/piper_smoke.py --mode cpu-tiny --root <certified-pilot> --weights <Qwen-config-dir>
--output <new-dir>` runs two CPU optimization steps on a RANDOM reduced Qwen with real video
processor/collator, actual HOST alignment forward/loss and strict checkpoint replay. It does
not produce progress GT. This passed in the isolated existing container; it is not an8B test.

`preflight_alignment_smoke.py --root <pilot> --weights <Qwen-dir> --output <new-run-dir>`
re-hashes all derived video certificates and complete pinned model files, requires250GiB free,
and refuses incomplete weights or reused outputs. No downloads or GPU allocations occur.

Proposed EIP command: `bash /pfs/user/code/host_piper_paper/deploy/idc/smoke_alignment.sh`.
Required environment names: HOST_SMOKE_APPROVED, HOST_EXPECTED_COMMIT, HOST_SMOKE_ROOT,
HOST_ALIGNMENT_MODEL_PATH, HOST_SMOKE_OUTPUT. These contain no credentials. W&B is offline.
The wrapper uses `run_eip_alignment_python.sh` inside the inspected lingbot CUDA image;
torchrun children explicitly use the same -S isolated interpreter, avoiding inherited
image site-packages. It does NOT require nested Docker inside the EIP task.

Smoke:4 GPUs, BF16, ZeRO-3, microbatch1/rank, accumulation1,4 anchors,2 updates, AdamW lr1e-5,
wd1e-5, clip3. Then a NEW torchrun process reloads model/optimizer and verifies deterministic
loss replay. Peak allocated/reserved CUDA memory is recorded per rank, including save/load.
This small sequence only validates the plumbing, not paper24/96-anchor memory or label quality.
No retries or production GT export. GPU modes are unvalidated until an approved EIP task passes.

Do not run from dirty/unpushed code. Resolve a private canonical remote, commit/push, freeze
the checkout, verify live queue capacity and full resource tuple, then present all resolved
fields and obtain confirmation. A configured flavor alone does not prove free queue capacity.
