# HOST Piper adaptation

This repository contains an in-progress Piper adaptation of
[CGuangyan-BIT/HOST](https://github.com/CGuangyan-BIT/HOST), based on upstream commit
`9f3bba57792aa5053ac600b1b7a4625f96ea6662`. Preserve upstream notices and the existing
[policy license](policy_training/LICENSE); this adaptation does not grant rights to datasets
or pretrained weights distributed separately.

The active branch is `piper_paper`. See [the plan](PLANS.md),
[IDC runbook](deploy/idc/README.md), and [status](deploy/idc/STATUS_20260908.md).
Deployment paths and registry references are specific to the author's IDC environment;
they are not a portable install command for a fresh machine.

## Verified scope

- Manifest-driven four-task data preparation and three-camera policy adaptation scaffolding.
- Explicit same-task/split alignment pairing, reviewed pilot anchors, and strict DTW export gates.
- Lossless seek-safe pilot videos, raw timestamp/window auditing, and numerical pose ranges.
- Separate alignment/policy environments and 36 passing CPU regression tests.
- Actual HOST forward/loss/backward and checkpoint replay with real videos and a RANDOM
  reduced Qwen model on CPU. This is not pretrained8B or GPU validation.

The 4-A800 EIP alignment smoke entrypoint is prepared but has not been run successfully
on GPUs yet. Production alignment training, reliable DTW labels and policy fine-tuning
remain pending. No dataset, video, model weight, credential or training checkpoint is
included in this repository. Do not use random smoke artifacts as progress ground truth.
