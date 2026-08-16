# Package-parity proofs — 2026-08-03

Two end-to-end runs of the submission image with **no arguments**: the
entrypoint, the 24-attempt / 180-minute defaults and the accept ladder
exactly as `docker run ebim-task2-autonomy:latest` gives them. After each,
the official evaluate service was called once on whatever scene state the
container left behind, and the complete artifact set was kept.

| timestamp | image | accepted at | self-verdict | **official IoU** | case |
|---|---|---|---|---|---|
| `20260803_081723_814514` | ramp ladder + budget horizon | attempt 6/24 | 0.7295 | **0.7256** | `both_liner_dominant` |
| `20260803_092630_391117` | **shipped** (optimal-stopping ladder) | attempt 6/24 | 0.6995 | **0.6955** | `both_liner_dominant` |

`evidence_parity_final3_0803.json` is the shipped image's result and is the
one that certifies the submission; the 08:17 set is kept because it is an
independent second draw of the same policy.

Both passed the orientation gate. The 09:26 lay covers 81% of the target
bbox at 83% precision; the 08:17 one covers 75% at 96%.

## Self-verdict fidelity

The entry decides when to stop from its own replica of the evaluator, so
the replica's accuracy is load-bearing. Three independent measurements, all
on this same package:

| measurement | n | mean (official − self-verdict) |
|---|---|---|
| offline replay of saved evaluator artifacts (`../replay_parity_20260803.txt`) | 224 frames | −0.008 |
| live: one lay per iteration, officially evaluated (`../parity_0803_summary.txt`) | 10 lays | −0.005 |
| these two end-to-end runs | 2 runs | −0.004 |

The offline replay also reproduced the official `orientation_case` on
222/224 frames and the scored/not-scored gate on 224/224.

## Files

Per timestamp: `eval_camera_iou_*.json` (the verdict), both bbox streams and
all three label payloads (the evaluator's exact inputs), the RGB frame and
its tight/loose overlays. `run.log` is the full stdout of the 09:26 run,
including the per-attempt chain output. The `.npy` mask arrays are dropped
to keep the repo small — the `.png` renderings are kept.
