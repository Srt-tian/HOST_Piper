# AutoDL policy launch preparation

The user authorized continuing until policy training is submitted/started, using the
existing four A80080GB AutoDL instance. This is not an EIP submission or a new instance.
Canonical development remains IDC branch piper_paper; deploy a clean, published commit.

`task=piper_autodl` builds on the native30Hz future executed-state profile. Observation t,
actions t+1..t+30,20 physical+2 auxiliary dimensions,3 vertically stacked224px cameras.
Use actual step1500 DTW labels and verified real T5 caches, not time-index peer labels.
The canonical self timeline is explicitly a reference coordinate axis.

Data preflight passed748train/76val candidate episodes. Full2652-video sequential audit,
all-episode temporal-window audit and label/text/split checks are recorded externally.
Semantic review is limited: lemon coarse phases look plausible but release timing differs;
pens insertion timing differs; basket object orders can differ and matched progress may
represent task fraction rather than the same object/subgoal. No dense semantic accuracy
or autonomous deployment claim. Full candidate metadata must remain marked as candidates;
any experimental full-data launch must record these limitations, not mark every label
semantically certified. The previous blanket pending-review note is not a requirement for
manual certification of every frame: numerical filtering plus sampled review supports an
experimental fine-tune, not a validated robot controller. Raw rejected trajectories stay excluded.

Strict init: all5 checkpoint modules must match. Only2->3 camera_emb/pos_embed may be
reinitialized (seed42); source camera identities are unknown. Visual backbone architecture
is loaded locally without downloading redundant weights, then restored from HOST.
Wan VAE comes from verified original local artifact; model downloads are disabled.
DINOv2 code pinned on AutoDL to7764ea0f912e53c92e82eb78a2a1631e92725fc8.

ZeRO2 BF16, gradient checkpointing, microbatch1 initially, accumulation4/global16,
AdamW lr1e-5,weight_decay.01,betas.9/.95,500warmup for10000step schedule,constant thereafter.
Visual encoder retains upstream10xLR. Test full actual forward/backward/optimizer before
long run; log per-rank loss/gradient norm/memory. Probe can override steps/accumulation.
Do not confuse a process launch, initialization or inference with a successful train step.

Model-only checkpoints every500 steps/final,2 retained at most. Save requires weight bytes
plus8GiB free, fsync+roundtrip equality+hash before publication. If necessary remove only older
verified run-owned checkpoint while preserving latest before saving. No optimizer state.
Saver intentionally rejects ZeRO3 (cannot save partitioned weights as full state).
Training stops on nonfinite/zero gradients, peak reserved>74GiB or disk reserve<8GiB.
These guards reduce risk, not a guarantee against all OOM/errors. W&B disabled unless separately
configured; metrics are available through logs/health_rankN.jsonl.

Runtime isolation via deploy/autodl/run_policy_python.sh: Python-S/explicit package path,
data-disk caches/temp, no base or alignment dependency writes. Decord's existing wheel has
cp36 metadata (pip check platform warning), but actual import+dataset import and dependency
relationships pass. Training explicitly uses PyAV sequential decoding, not Decord random seek.
