# Live re-validation on the resurrected rig — 2026-08-04 (GCP L4, driver 580.173.02)

Scene: barebone `--record` at eval-fix-preview generation; official eval
stack (`scripts/evaluation/task2`) live in its own container.

- `entry_run1.log` — the shipped image (live-evaluate commit 6f3bc5b) run with
  `EBIM_RUN_BUDGET_MIN=60`: startup probe found the organizers' evaluate
  service, warm-up verdict logged, attempt 1 aborted fail-closed, attempt 2
  **accepted on the official service's own verdict: `liner_only 0.7852`** —
  while the chain's internal crude estimate read 0.000 (the shed had towed
  strip and plate; the official metric scores the plate's FINAL pose).
- `eval_camera_iou_20260804_033226_140817.json` — the same lay evaluated by
  the **PR#59 (eval-fix-preview)** evaluator: `0.7852`, `liner_only`, correct.
- `eval_camera_iou_20260804_033209_471109.json` — the SAME lay, 17 s earlier,
  evaluated by the **origin/main @ e8c6235** evaluator: `0.0`,
  `no_target_bbox`, wrong — main's tight-stream target resolution collapses
  under a correctly covered target, exactly as PR#59's description says. Its
  artifact set has no loose files at all (main does not subscribe the stream).
- `eval_ab.log` — the switch script transcript (checkout main → restart eval
  node → evaluate → restore eval-fix → evaluate).

One placement, two verdicts: 0.7852 vs 0.0000, decided purely by which
evaluator version answers. This is the measured case for judging with the
LIVE service when it is reachable instead of betting on a version.

## Addendum — full no-arg certification run (36/240 defaults, image 3d44070)
`entry_final_36x240.log`: plain `docker run` (no args). Every judged attempt
was scored by the official service (14 live verdicts). A cold draw pool on
this rig — four consecutive `sideways`, 8/22 aborts, best gated draw 0.6492
rejected at the 0.68 hold — ended with the wall governor shrinking the
horizon to 30 ("wall budget binds before the 36-attempt cap") and the
relaxed ladder accepting `both_liner_dominant 0.3803` at attempt 22
(threshold there 0.68·8/15 = 0.363). Container exited 0, scene left scored.
Same image, same day, warmer pool: 0.7852 (the bounded run above). The L4
paces attempts at ~9 min, so 240 min fits ~26 attempts here; on a faster
evaluation rig the 36 cap re-engages. Both accept paths (hold-clear at
0.7852, wall-relaxed at 0.3803) are now live-exercised end to end.

## Addendum 2 — the 24 draws above are now IN the shipped ladder
The two runs in this directory draw lower than the first rig (valid-lay
mean 0.44 vs 0.55), so `MEASURED_GATED_IOUS` now pools all 56 banked
complete-run draws; under the pooled V-table the certification run's
attempt 13 (0.6492, rejected at 0.6635 above) would have been banked
(threshold 0.6459). Modelled deltas across both pace models:
`budget_matrix_20260804_pooled.txt`. `entry_smoke_pooled_ladder.log` is
the pooled-table image's live smoke (2-attempt bound): dirty scene from
the 0.3803 state detected and recovery-reset, one chain, official
verdict `both_liner_dominant 0.7008`, accepted, exit 0. That draw
arrived after the pool was frozen and stays out of it (exclusion is
time-based, not value-based).
