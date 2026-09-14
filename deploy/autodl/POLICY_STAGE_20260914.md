# Fixed-encoder policy preparation

User confirmed using latest alignment checkpoint1500 to prepare HOST policy fine-tuning,
not resuming encoder optimization. The user authorized retaining recorded per-arm state_end
xyz/RPY, m/rad, without adding a tool offset; exact deployment TCP naming is not an offline blocker.

URDF FK audit sampled up to32 rows from every one of884 episodes across9 sources.
For all18 source/side groups, extrinsic xyz (RzRyRx) is the lowest median rotation-error candidate.
Median flange-position error0.075..0.104mm; median rotation error0.0027..0.0037degrees.
The provided URDF TCP is135.8mm from link6; its median error is135.8mm.
Evidence supports the recorded flange-like endpoint, not replacing data poses with URDF FK.
Raw command correspondence and joint/pose timing are not certified by this audit.

Use conventions_executed_state.json for conversion. This chooses future executed state/qpos
as supervision; do not substitute action_end/real_action. Keep source manifest immutable.
All endpoint poses remain in their respective arm bases. No TCP/base registration correction.

Step1500 held-out pilot passed8/8 numeric gates. Object-set/order and contact-stage ambiguities
remain in visual review. Full DTW run produces CANDIDATE RECORDS, not approved policy labels.
Run four independently capped22GiB eval processes, fixed1500, both train and val, deterministic
disjoint shards. Excludes8 canonical self-pairs:876 nonself records expected across4x219 pairs.
Do not train on rejected or unreviewed records or manufacture linear progress for other episodes.

Next: filter/review labels and map canonical coordinates into policy paths, certify gap-aware
future-action sampling and text caches, deploy isolated policy dependencies and official HOST/Wan
artifacts with measured storage budget, then real policy GPU smoke before prolonged training.
No policy training has yet been started. Keep model-only rolling checkpoints and disk guards.
