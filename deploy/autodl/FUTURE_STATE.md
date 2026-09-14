# Native-row Piper policy sampling

Profile: `task=piper_3cam_future_state`, for both train and val. Enable strict video
decoding with `HOST_POLICY_VIDEO_DECODE=sequential` in the isolated policy runtime.

- Observation and proprioception use raw row t. Physical actions use executed
  follow-state rows t+1 through t+30, without duplicate/clamped samples.
- Agent endpoint is row t+30. Three views share the same sampled frame indices.
- Raw source rows must be contiguous; timestamps finite and strictly increasing.
  Exclude windows containing an adjacent gap greater than1.5/30 seconds.
  Do not delete episodes or interpolate state/commands to bridge gaps.
- Static-frame deletion and random action-span stretching are disabled in this
  profile. This is an explicit native30Hz adaptation, NOT exact paper sampling.
  The inherited velocity objectives and three-camera layout are unchanged.
- Reference-window progress retains upstream interpolation of DTW-derived endpoint
  coordinates. The current-time endpoint is excluded in future-state mode so that
  H progress slots match H future action slots. No main-index linear GT is created.
- Canonical-path rebasing, reviewed labels, genuine text embeddings and runtime
  preflight remain prerequisites. This sampler is NOT a training-ready marker.
- Existing profiles default to legacy sampling. Do not silently change old experiments.

Validation:72 CPU tests passed, including actual HOST video-sampler safe-start and
endpoint checks, next-state identity, final-window bounds, gap exclusion, and train/val
profile agreement. Full real converted-data window audit is recorded in the external run
ledger. No HOST policy GPU forward/backward or closed-loop validation claimed here.
