# Official-configuration Piper alignment probe (2026-09-09)

User superseded the gradual length sweep with "直接按官方的跑一下试试".
No length-sweep code was applied to the IDC repository or submitted.

## Code and scope

Use the released alignment/train.py loop, datasets/collator, Qwen8B forward,
TCC/SmoothDTW loss, optimizer and scheduler. Baseline upstream:
9f3bba57792aa5053ac600b1b7a4625f96ea6662. The wrapper does not modify those files.
Official defaults alone are insufficient: train_scripts/run_ds.sh overrides
scheduler total steps to3000 and, on one node, accumulation to4 (=4 /1 node).

-24 anchors/chunk; max96; M1/2/4 probabilities.1/.2/.7.
-4 input pairs/rank/microbatch;4 ranks; accumulation4. Effective64 chunk-pair
 equivalents/update, NOT necessarily64 distinct episode pairs: M4 uses one
 episode pair split into4 chunks/rank. M2 uses2 pairs, M1 uses4.
-3 camera views/context1;224 image resize; NUM_ALIGN_FRAMES24 (8 temporal
 overview positions x3 cameras per episode,48 combined main/ref overview slots).
-BF16 ZeRO3, noCPUoffload; official reduce/allgather200000000, overlap/contiguous,
 gather16bit-on-save=true; AdamWlr1e-5,betas.9,.999,eps1e-8,wd1e-5,clip3.
-WarmupCosineLR3000 total,100 warmup,min_ratio0,cos_min_ratio.3.
-Run only10 actual engine optimizer updates (=40 microsteps at accumulation4).
 The released loop remains configured3000; the wrapper stops at engine.global_steps10,
 saves all-rank full model/optimizer/scheduler checkpoint, then raises a handled
 local bounded-stop exception. It does not compress warmup/schedule into10 steps.
 Do not equate the loop's post-step accumulation label with actual engine updates.
-Train and reload are separate fresh torchrun processes. Reload explicitly loads
 model/optimizer/scheduler strictly and requires engine/client step10. Unlike the
 prior small smoke, this probe does not perform a replay-loss comparison.
-Any OOM, nonfinite loss/optimizer-boundary gradient, timeout or reload failure
 stops the job; no auto-retry, batch reduction, CPU offload or deletion.

## Explicit adaptation and operational differences

Piper profile selects own3-camera mapping/task paths, turns joint conditioning OFF,
turns random left/right flip OFF, and fails on decode/pairing errors instead of
black-frame or random-sample fallback. Brightness/contrast stay ON as released.
Image/model/library compatibility patches from the previously verified commit remain.
These are deliberate differences, not a claim of an untouched official environment.
Bounded termination, input/gradient/memory reports and additional checkpoint/reload
are diagnostic instrumentation, not a change to the learning objective.

Data: /pfs/user/data/host_piper_paper/lemon_seek_safe_pilot_20260908
12train/4val,48 certified videos,3347 frames, originating from user's TOS lemon task.
Train uses ONLY train split and official WeightedRandomSampler(replacement=True),
seed42+rank. Each rank draws12 entries per sampling epoch, so batch4 is nonempty;
do not misapply DistributedSampler's12/4=3 calculation to this code path.
Reference episodes remain distinct, same semantic task and same split.
No fabricated progress, actions, normstats or text embeddings are used.

Initialize pinned Qwen3-VL-Embedding-8B revision
2c4565515e0f265c6511776e7193b22c0968ddc7, not the2-step smoke checkpoint.
All weights/data/output remain IDC under /pfs/user/data/host_piper_paper.
No system/KAI0 modifications or new downloads. Network-disabled/noGPU CPU checks.
W&Boffline projecthost_piper_official_probe; noWANDB_API_KEY injection.

## Verified preparation (not a GPU result)

CPU same-task train sampler+real processor passed all4 rank cases, shapes8x4945
input IDs,8 rows each24CLS. Observed M4/M2 cases; paths/provenance/indices saved in:
 /pfs/user/data/host_piper_paper/official_probe_cpu_preflight_20260909
This is not a promise that actual multiworker stochastic batches will be identical.
40 CPU tests passed52.40s including4 new approval/config/output-gate tests.
Original train.py/config.py/scripts remain unchanged.

## Launch

bash /pfs/user/code/host_piper_paper/deploy/idc/probe_official_alignment.sh

EIPvariables: HOST_OFFICIAL_PROBE_APPROVED=1 onlyafter final confirmation;
HOST_EXPECTED_COMMIT=<published clean full SHA>;
HOST_SMOKE_ROOT=<certified lemon root>;
HOST_ALIGNMENT_MODEL_PATH=<pinned local Qwen weights>;
HOST_OFFICIAL_PROBE_OUTPUT=<new IDC HOST data directory>.
Image/wrapper same immutable lingbot digest as successful task11534.
Runtime checks4 GPUs and exactPython/Torch/Transformers versions. Certificate/hash
preflight and at least250GiB free for full checkpoint/temp required.
Shell imposes40min per phase; queue's60min total runtime limit remains applicable.
10updates may OOM; no duration/fit claim from previous4-anchor memory measurements.

Before submission: exact published Git SHA, clean upstream-tracking checkout,
queue live capacity/spec/zone evidence, resource/image/command/env structural diff
against11534, complete resolved launch review and explicit user confirmation.
No EIP task has been submitted for this probe during preparation.
