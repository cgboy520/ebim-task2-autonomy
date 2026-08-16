#!/usr/bin/env python3
"""Where do RUN-level zeros come from, and which lever actually removes them?

Two different things are called "a zero" and they want opposite treatment:

  per-ATTEMPT zero  — 68% of draws. Cheap: the ladder just resets and
                      redraws. Reducing it matters only through how many
                      good draws fit in the budget.
  per-RUN zero      — the official evaluate on the FINAL scene state reads
                      0. This is the submission score, and it has one
                      dominant cause: the accept ladder held out, nothing
                      ever cleared it, and the run ended standing on an
                      unconditioned draw.

Empirical inputs are the 34 real attempts measured on 2026-08-03 with the
frozen recipe: rehearsal Phase A (24, never-accept) + the post-fix parity
run (10, never-accept, each one officially evaluated).
"""
import random
from collections import Counter

# (case, raw_iou or None for a chain abort)
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
DRAWS = PHASE_A + PARITY

# Durations measured on the parity run's timestamps (start -> verdict).
COMPLETED_S = 415.0      # mean of the 8 completed lays
ABORT_S = 265.0          # mean of the 2 aborts
RESET_S = 40.0           # ack + settle, measured 27-28 s + the settle sleep

VALID = {"liner_only", "both_liner_dominant"}


def gated(draw):
    case, iou = draw
    if iou is None or case not in VALID:
        return 0.0
    return iou


def ladder_shipped(idx, cap):
    rem = cap - idx
    if rem >= 3:
        return 0.68
    if rem == 2:
        return 0.55
    if rem == 1:
        return 0.45
    return 0.0


def make_ladder(hold, relax_last):
    """Hold out for `hold`, then accept ANY valid lay over the last
    `relax_last` attempts (a straight ramp between the two)."""
    def f(idx, cap):
        rem = cap - idx
        if rem >= relax_last:
            return hold
        if rem <= 0:
            return 0.0
        return hold * rem / relax_last
    return f


def should_start(idx, elapsed, avg, budget_s):
    if idx <= 1:
        return True
    rem = budget_s - elapsed
    if rem <= 0:
        return False
    if avg is None:
        return True
    return rem > 0.8 * avg


def run_once(ladder, rng, cap, budget_s, pool):
    t = 0.0
    durs = []
    final = 0.0
    why = "budget/cap with a reset last"
    for idx in range(1, cap + 1):
        avg = sum(durs) / len(durs) if durs else None
        if not should_start(idx, t, avg, budget_s):
            break
        draw = rng.choice(pool)
        dur = ABORT_S if draw[1] is None else COMPLETED_S
        t += dur
        durs.append(dur)
        avg = sum(durs) / len(durs)
        g = gated(draw)
        thr = ladder(idx, cap)
        if not should_start(idx + 1, t + RESET_S, avg, budget_s):
            thr = 0.0
        if g > 0 and g >= thr:
            return g, "accepted"
        final, why = g, ("ended on " + draw[0])
        if idx < cap:
            if should_start(idx + 1, t, avg, budget_s):
                t += RESET_S
                final, why = 0.0, "reset destroyed the last lay"
            else:
                break
    return final, why


def stats(ladder, cap, budget_s, pool, n=120_000, seed=11):
    rng = random.Random(seed)
    out = [run_once(ladder, rng, cap, budget_s, pool) for _ in range(n)]
    scores = [s for s, _ in out]
    e = sum(scores) / n
    pz = sum(1 for s in scores if s <= 0) / n
    causes = Counter(w for s, w in out if s <= 0)
    return e, pz, sum(1 for s in scores if s >= 0.5) / n, causes


def show_pool(pool, label):
    c = Counter(d[0] for d in pool)
    sc = [gated(d) for d in pool if gated(d) > 0]
    print(f"{label}: n={len(pool)}  {dict(c)}")
    print(f"   scored {len(sc)}/{len(pool)} ({len(sc)/len(pool):.0%}), "
          f"mean {sum(sc)/len(sc):.3f}, "
          f">=0.68 {sum(1 for x in sc if x >= 0.68)}/{len(pool)} "
          f"({sum(1 for x in sc if x >= 0.68)/len(pool):.1%})")


CAP, BUDGET = 24, 180 * 60.0

print("=" * 74)
print("EMPIRICAL DRAW POOL (34 real attempts, frozen recipe, 2026-08-03)")
show_pool(DRAWS, "all")
print()
print("per-attempt zero decomposition:")
tot = len(DRAWS)
for case, k in Counter(d[0] for d in DRAWS).most_common():
    tag = "SCORES" if case in VALID else "zero"
    print(f"   {case:<26} {k:>2}/{tot}  {k/tot:5.1%}   {tag}")

print("\n" + "=" * 74)
print(f"RUN-LEVEL outcome, shipped ladder, {CAP} attempts / "
      f"{BUDGET/60:.0f} min")
e, pz, p5, causes = stats(ladder_shipped, CAP, BUDGET, DRAWS)
print(f"   E[score] {e:.4f}   P(run scores 0) {pz:.1%}   P(>=0.5) {p5:.1%}")
print("   what the zero runs were standing on:")
for w, k in causes.most_common():
    print(f"      {w:<34} {k/ (pz*120_000):6.1%} of the zeros")

print("\n" + "=" * 74)
print("LEVER 1 — ladder shape (same draws, same budget)")
print(f"{'hold / relax over last N':<32}{'E[score]':>10}{'P(zero)':>10}{'P>=0.5':>9}")
print(f"{'shipped (0.68/0.55/0.45/0)':<32}{e:>10.4f}{pz:>10.1%}{p5:>9.1%}")
for hold in (0.68, 0.60, 0.55, 0.50):
    for relax in (3, 5, 8, 12):
        lad = make_ladder(hold, relax)
        e2, pz2, p52, _ = stats(lad, CAP, BUDGET, DRAWS)
        print(f"{'hold %.2f, relax over last %2d' % (hold, relax):<32}"
              f"{e2:>10.4f}{pz2:>10.1%}{p52:>9.1%}")

print("\n" + "=" * 74)
print("LEVER 2 — fix a failure mode at the source (shipped ladder)")
print("   each row replaces that share of draws with a redraw from the rest")


def without(case_name, keep_fraction):
    """Pool with `case_name` draws thinned to keep_fraction of their rate."""
    keep = [d for d in DRAWS if d[0] != case_name]
    n_case = sum(1 for d in DRAWS if d[0] == case_name)
    return keep + [d for d in DRAWS if d[0] == case_name][
        :max(0, int(round(n_case * keep_fraction)))]


print(f"{'scenario':<32}{'E[score]':>10}{'P(zero)':>10}{'P>=0.5':>9}")
print(f"{'baseline':<32}{e:>10.4f}{pz:>10.1%}{p5:>9.1%}")
for case in ("abort", "sideways", "both_thermalpad_dominant"):
    for frac, tag in ((0.5, "halved"), (0.0, "eliminated")):
        pool = without(case, frac)
        e2, pz2, p52, _ = stats(ladder_shipped, CAP, BUDGET, pool)
        print(f"{case + ' ' + tag:<32}{e2:>10.4f}{pz2:>10.1%}{p52:>9.1%}")

print("\n" + "=" * 74)
print("LEVER 3 — more attempts / more budget (shipped ladder)")
print(f"{'cap / budget':<32}{'E[score]':>10}{'P(zero)':>10}{'P>=0.5':>9}")
for cap, mins in ((24, 180), (30, 240), (36, 300), (48, 400), (60, 500)):
    e2, pz2, p52, _ = stats(ladder_shipped, cap, mins * 60.0, DRAWS)
    print(f"{'%d att / %d min' % (cap, mins):<32}"
          f"{e2:>10.4f}{pz2:>10.1%}{p52:>9.1%}")


print("\n" + "=" * 74)
print("LEVER 1b — the two aborts the RESET-ACK FIX already removes")
print("   Phase A attempts 10 and 14 aborted because the previous reset's")
print("   ack was lost and the chain ran on a dirty scene ('only -4.2 mm of")
print("   -x overhang', 'ids not inferable' -> 'plate not found'). The")
print("   handshake fix measured 0/10 lost acks, so those draws are gone.")
FIXED = [d for i, d in enumerate(DRAWS)
         if i not in (9, 13)]          # 0-based: Phase A attempts 10 and 14
show_pool(FIXED, "post-fix pool")
print(f"{'ladder':<32}{'E[score]':>10}{'P(zero)':>10}{'P>=0.5':>9}")
for name, lad in (("shipped", ladder_shipped),
                  ("hold 0.68, relax last 12", make_ladder(0.68, 12))):
    e2, pz2, p52, _ = stats(lad, CAP, BUDGET, FIXED)
    print(f"{name:<32}{e2:>10.4f}{pz2:>10.1%}{p52:>9.1%}")

print("\n" + "=" * 74)
print("COMBINED — best ladder x budget, on the post-fix pool")
print(f"{'cap / budget':<32}{'E[score]':>10}{'P(zero)':>10}{'P>=0.5':>9}")
best = make_ladder(0.68, 12)
for cap, mins in ((24, 180), (30, 240), (36, 300), (48, 400)):
    lad = make_ladder(0.68, max(3, cap // 2))
    e2, pz2, p52, _ = stats(lad, cap, mins * 60.0, FIXED)
    print(f"{'%d att / %d min (relax last %d)' % (cap, mins, max(3, cap//2)):<32}"
          f"{e2:>10.4f}{pz2:>10.1%}{p52:>9.1%}")

print("\n" + "=" * 74)
print("MEAN RUN LENGTH (does a better ladder cost wall time?)")


def mean_len(ladder, cap, budget_s, pool, n=40_000, seed=5):
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(n):
        t = 0.0
        durs = []
        for idx in range(1, cap + 1):
            avg = sum(durs) / len(durs) if durs else None
            if not should_start(idx, t, avg, budget_s):
                break
            draw = rng.choice(pool)
            dur = ABORT_S if draw[1] is None else COMPLETED_S
            t += dur
            durs.append(dur)
            avg = sum(durs) / len(durs)
            g = gated(draw)
            thr = ladder(idx, cap)
            if not should_start(idx + 1, t + RESET_S, avg, budget_s):
                thr = 0.0
            if g > 0 and g >= thr:
                break
            if idx < cap and should_start(idx + 1, t, avg, budget_s):
                t += RESET_S
        tot += t
    return tot / n / 60.0


for name, lad in (("shipped", ladder_shipped),
                  ("hold 0.68, relax last 12", make_ladder(0.68, 12))):
    print(f"   {name:<32} {mean_len(lad, CAP, BUDGET, FIXED):5.1f} min mean")
