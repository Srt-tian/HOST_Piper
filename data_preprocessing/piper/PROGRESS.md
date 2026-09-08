# Progress is aligned reference position, not elapsed fraction

The upstream `coupling/progress_alignment/build_progress_info.py` consumes the trained
alignment module's records. For a Main-frame index i matched to reference-frame index j,
the non-segmented implementation stores p(i)=(j+1)/N_ref. Segment offsets are subtracted.
Sparse matched points are interpolated over all Main frames with `numpy.interp`;
outside matched support it holds the endpoint value. This is not i/N_main.

Pipeline:

1. Group by semantic task and split; review that manipulation procedure/order is compatible.
2. Train/evaluate the official visual alignment model (TCC + smooth-DTW objective).
3. Export real matching records; verify raw frame paths and their grouping. The upstream
   converter hard-codes `[3::4]`, which must agree with the alignment record's actual layout.
4. Filter bad matches: upstream skips loss>0.15, insufficient covered reference range,
   backward correspondence and excessive jumps. These heuristics are not substitutes for
   reviewing meaningful contact/grasp/release event alignment on sample videos.
5. Map matched reference indices to normalized progress, interpolate Main frames and store
   `info_dtw.json` with explicit `ref_path` provenance.
6. Ensure labels compared by the policy share a coordinate reference. The policy reads
   aligned_progress but ignores ref_path. Our preflight therefore requires a consistent
   reviewed anchor per task/split; alternatively a future adapter must explicitly rebase
   pairwise coordinates. Do not silently combine independently normalized references.

The reference itself may use its normalized timeline as the coordinate axis. Other episodes
must be mapped into that axis through alignment. This is different from linear progress for
every episode. A linear-only baseline can be a named ablation, never reported as HOST coupling.

Current status: no real progress labels or trained alignment checkpoint generated. Synthetic
unit-test fixtures are not exported as training labels. Public HOST policy weights do not
provide the trained alignment model. Inference uses HOST's predicted progress, not an external
human-provided per-frame index.

## Piper implementation and verified scope (2026-09-08)

`prepare_alignment.py` creates a visual-only dataset independently of pending EEF conventions,
with the SAME deterministic episode split as the policy converter. It makes NO actions or GT.
`alignment/piper_profile.py` explicitly disables joint tokens and horizontal flips, configures
the three-camera map, and rejects black-image substitution, random error retries, missing
evaluation reference lists and cross-task/split references. The upstream default actually has
USE_JOINTS=True despite the README example saying False; this override is intentional.

`prepare_eval_pairs.py` installs reviewed common references in `task_paths_eval.json`.
The upstream eval dataset silently falls back to self-alignment when this file is absent;
the Piper profile rejects that fallback. An anchor's explicit self-pair for defining its own
coordinate is allowed in evaluation, but self-pairing is forbidden for alignment training.

`build_progress.py` replaces the coupling script's fixed `[3::4]` extraction with validated
time-first groups of 3 cameras x 1 or 2 context frames. It checks camera identity, episode,
frame bounds, strict sample ordering and exact correspondence length. No min-length truncation.
Loss/range/causality/jump thresholds retain upstream values. Unlike the upstream converter,
repeated matched reference points are PRESERVED before interpolation so observed pauses remain
plateaus. This is an intentional adapter difference, not exact byte-for-byte reproduction.

At the pinned upstream commit, deterministic_alignment.py's `forward_argmax_indices` and
`forward_dtw_indices` BOTH come from argmax of the Smooth-DTW probability matrix `sim_mr`.
This is not an independently backtracked hard-DTW path; monotonicity is checked on export.
Do not describe the field as an ordinary raw-feature nearest neighbor or a hard-DTW guarantee.

Labels are written only to a NEW artifact directory with source-record hash/model ID, and
missing coverage is explicit. They are not automatically installed into the policy dataset.
The later visual->policy path rebasing, dataset completion and semantic event review remain
required. None of the real training labels have been generated yet.

Initial anchors for four tasks x two splits were screened from 8 sequentially decoded frames
per camera. See deploy/idc/anchor_pilot_review_20260908.json. Final sampled frames show completion,
but this is only a pilot screen, not full-motion review or proof all peers share action order.
Object order differs in pens/bottles, and basket object sets vary. Start with the short single-
object lemon task before scaling the coupling audit to all four tasks.
