"""E[score] / P(run=0) for the POOLED-ladder entry across budget x cap.

Same machinery as budget_matrix_20260804.py, with two changes:

* The draw pool now also carries the 24 L4-rig attempts of 2026-08-04
  (entry_run1 + entry_final_36x240, evidence/live_reval_0804/), every one
  judged by the organizers' live evaluate service. The ladder the package
  ships (MEASURED_GATED_IOUS, 56 gated draws) is derived from the same
  pooled data.
* Every table is printed under BOTH measured pace models, because the two
  rigs paced differently and the organizers' hardware is unknown:
  first rig 415 s completed / 265 s abort; L4 rig 565 s / 355 s (fitted to
  the 22-attempt certification run's ~198 min wall).

The ladder/governor calls are the package's own functions, wired exactly
as official_run.main() wires them (effective-total relaxation +
last-startable lookahead at RESET_COST_S).
"""
import random, sys
sys.path.insert(0, "/mnt/t/code/robot/ebim-task2-autonomy/src")
from ebim_task2.official_run import (
    stop_threshold, should_start_attempt, attempts_left_in_budget,
    RESET_COST_S)

PHASE_A = [
    ("both_thermalpad_dominant", 0.3886), ("sideways", 0.0),
    ("both_liner_dominant", 0.3996), ("both_liner_dominant", 0.6157),
    ("sideways", 0.0), ("sideways", 0.0),
    ("both_liner_dominant", 0.5762), ("sideways", 0.0),
    ("abort", None), ("abort", None),
    ("both_liner_dominant", 0.5672), ("sideways", 0.0),
    ("both_liner_dominant", 0.7327), ("abort", None),
    ("sideways", 0.0), ("both_liner_dominant", 0.5951),
    ("abort", None), ("both_thermalpad_dominant", 0.6689),
    ("both_liner_dominant", 0.6197), ("abort", None), ("abort", None),
    ("both_liner_dominant", 0.3406), ("sideways", 0.0), ("sideways", 0.0),
]
PARITY = [
    ("both_thermalpad_dominant", 0.7268), ("both_liner_dominant", 0.2037),
    ("sideways", 0.0), ("both_thermalpad_dominant", 0.7636),
    ("abort", None), ("abort", None), ("sideways", 0.0),
    ("both_thermalpad_dominant", 0.5154), ("liner_only", 0.8020),
    ("both_liner_dominant", 0.5275),
]
# 2026-08-04, L4 rig, all officially judged (evidence/live_reval_0804/).
L4_RUN1 = [("abort", None), ("liner_only", 0.7852)]
L4_FINAL = [
    ("both_thermalpad_dominant", 0.6464), ("abort", None),
    ("both_liner_dominant", 0.3506), ("abort", None),
    ("sideways", 0.0), ("sideways", 0.0), ("sideways", 0.0),
    ("sideways", 0.0), ("abort", None),
    ("both_liner_dominant", 0.3023),
    ("both_thermalpad_dominant", 0.5049), ("abort", None),
    ("both_liner_dominant", 0.6492), ("abort", None),
    ("both_liner_dominant", 0.2738), ("both_liner_dominant", 0.3037),
    ("liner_only", 0.4155), ("both_liner_dominant", 0.3441),
    ("abort", None), ("abort", None), ("abort", None),
    ("both_liner_dominant", 0.3803),
]
DRAWS = PHASE_A + PARITY + L4_RUN1 + L4_FINAL      # 58 labeled draws
PACES = (("first rig 415/265", 415.0, 265.0),
         ("L4 rig    565/355", 565.0, 355.0))
RESET_S = 40.0
VALID = {"liner_only", "both_liner_dominant"}

def gated(d):
    case, iou = d
    return iou if (iou is not None and case in VALID) else 0.0

def run_once(rng, cap, budget_s, completed_s, abort_s, kill_s=None):
    t, durs, final = 0.0, [], 0.0
    for attempt in range(1, cap + 1):
        avg = sum(durs) / len(durs) if durs else None
        if not should_start_attempt(attempt, t, budget_s, avg):
            break
        draw = rng.choice(DRAWS)
        dur = abort_s if draw[1] is None else completed_s
        if kill_s is not None and t + dur > kill_s:
            return 0.0, t              # killed mid-chain: partial lay ~ 0
        t += dur; durs.append(dur)
        avg = sum(durs) / len(durs)
        g = gated(draw)
        eff = min(cap, attempt + attempts_left_in_budget(
            t, budget_s, avg, cap - attempt))
        thr = stop_threshold(attempt, eff)
        if not should_start_attempt(attempt + 1, t + RESET_COST_S,
                                    budget_s, avg):
            thr = 0.0
        if draw[1] is not None and g > 0 and g >= thr:
            return g, t
        final = g
        if attempt < cap:
            if kill_s is not None and t + RESET_S > kill_s:
                return 0.0, t
            t += RESET_S
            final = 0.0
    return final, t

def cell(cap, budget_min, completed_s, abort_s, kill_min=None,
         n=100_000, seed=7):
    rng = random.Random(seed)
    bs = budget_min * 60.0
    ks = None if kill_min is None else kill_min * 60.0
    tot = zeros = 0.0; tmax = 0.0
    for _ in range(n):
        s, t = run_once(rng, cap, bs, completed_s, abort_s, kill_s=ks)
        tot += s; zeros += (s == 0.0); tmax = max(tmax, t)
    return tot / n, zeros / n, tmax / 60.0

for pace_name, cs, as_ in PACES:
    print(f"pace {pace_name}:")
    print(f"{'cap':>4} {'budget':>7} {'E[score]':>9} {'P(0)':>7} "
          f"{'max wall(min)':>13}")
    for cap in (24, 36, 48):
        for bm in (120, 180, 240, 300, 360):
            e, p0, tm = cell(cap, bm, cs, as_)
            print(f"{cap:>4} {bm:>7} {e:>9.4f} {p0:>6.1%} {tm:>13.0f}")
    print()

for pace_name, cs, as_ in PACES:
    print(f"pace {pace_name}, organizer kills the container early:")
    for kill in (180, 120):
        print(f"  kill at {kill} min:")
        for cap, bm in ((24, 180), (36, 240), (48, 360)):
            e, p0, _ = cell(cap, bm, cs, as_, kill_min=kill)
            print(f"     declared {cap}/{bm:<3} -> E {e:.4f}  P(0) {p0:.1%}")
    print()
