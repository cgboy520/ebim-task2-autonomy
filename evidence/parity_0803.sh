#!/bin/bash
# parity_0803.sh — post-fix validation of the official entry.
#
# One loop iteration = one honest sample of the two things the entry's whole
# accept decision rests on:
#
#   1. PARITY. official_run's self-verdict is a REPLICA of the organizers'
#      evaluator (perception.predict_official_verdict). Nothing so far has
#      measured the replica against the real thing on the SAME scene state —
#      the banked evidence records official IoUs, the rehearsal records
#      predicted IoUs, and the two never touch. Each iteration lays once
#      (never-accept, so the lay survives), then calls the official evaluate
#      on that exact state: (predicted, official) pair.
#
#   2. RESET RELIABILITY. The 08-03 rehearsal lost the reset ack on 2 of 14
#      resets, and the attempt after each loss ran on a dirty scene and was
#      wasted. The reset between iterations goes through the FIXED
#      official_run.scene_reset (endpoint-matched handshake) and its rc is
#      recorded, so the loop measures the ack-loss rate directly.
#
# Deliberately NOT using official_run's internal loop: an internal reset
# destroys the lay before the official evaluate can see it.
#
# Usage:  setsid nohup bash /root/ebim/parity_0803.sh [ITERS] < /dev/null &
# Log:    /root/ebim/parity_0803.log      Pairs: /root/ebim/parity_0803.tsv
set -uo pipefail
. /root/ebim/scene_lib.sh
ITERS=${1:-10}
L=/root/ebim/parity_0803.log
T=/root/ebim/parity_0803.tsv
: > "$L"
[ -s "$T" ] || printf 'iter\tcase\tpredicted\tofficial\treset_rc\tsnapshot\n' > "$T"
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$L"; }

if pgrep -af "lay_[d]own.py|hunt[.]sh|record_[u]ntil|run_vla[2]|rehearsal_[0-9]" \
     | grep -vw $$ >/dev/null; then
  say "!! another probe/recording job is running; refusing to start"; exit 1
fi
docker image inspect ebim-task2-autonomy:latest >/dev/null 2>&1 \
  || { say "!! image missing"; exit 1; }

DOCKER_RUN=(docker run --rm --network host --ipc host
            -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp
            -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4
            ebim-task2-autonomy:latest)

say "== parity run: $ITERS iterations =="
scene_restart_lib parity "$L" || exit 1

for i in $(seq 1 "$ITERS"); do
  say "-- iteration $i/$ITERS: one never-accept lay --"
  OUT=$(mktemp)
  timeout 3600 "${DOCKER_RUN[@]}" \
    python3 -u -m ebim_task2.official_run \
      --attempts 1 --stop-iou 0.99 --budget-min 120 2>&1 | tee -a "$L" > "$OUT"
  # official_run prints exactly one verdict line per attempt.
  CASE=$(grep -o "predicted verdict [a-z_]*" "$OUT" | tail -1 | awk '{print $3}')
  PRED=$(grep -o "raw IoU [0-9.]*" "$OUT" | tail -1 | awk '{print $3}')
  rm -f "$OUT"
  [ -z "$CASE" ] && CASE=aborted
  [ -z "$PRED" ] && PRED=-

  read -r IOU SNAP <<< "$(official_eval_lib)"
  say "-- iteration $i: case=$CASE predicted=$PRED official=$IOU"

  # Fixed handshake; rc 0 = ack seen, 1 = ack lost (the bug this measures).
  timeout 300 "${DOCKER_RUN[@]}" python3 -c \
    'import sys; from ebim_task2.official_run import scene_reset; sys.exit(0 if scene_reset() else 1)' \
    >>"$L" 2>&1
  RRC=$?
  say "-- iteration $i: scene_reset rc=$RRC"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$CASE" "$PRED" "$IOU" "$RRC" "$SNAP" >> "$T"
done

say "== parity summary =="
python3 - "$T" <<'PY' 2>&1 | tee -a "$L"
import json
import sys
from collections import Counter

rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]


def official_case(snap):
    """The decisive field: an IoU of 0 could be a bad lay OR a failed gate."""
    if not snap or snap == "-":
        return "-"
    try:
        return json.load(open(snap))["orientation_case"]
    except Exception:
        return "?"


print(f"iterations      : {len(rows)}")
print(f"reset ack lost  : {sum(1 for r in rows if r[4] != '0')} / {len(rows)}")
print(f"replica cases   : {Counter(r[1] for r in rows).most_common()}")
print(f"official cases  : {Counter(official_case(r[5]) for r in rows).most_common()}")

# official_eval_lib already applies the orientation gate to its IoU, so the
# replica's RAW IoU has to be gated the same way before the two are
# comparable — a wrong-face lay carries a healthy raw IoU and still scores 0.
VALID = {"liner_only", "both_liner_dominant"}
pairs = [(float(p) if c in VALID else 0.0, float(o), c, official_case(s))
         for _, c, p, o, _, s in rows
         if p not in ("-", "") and o not in ("-", "")]
if pairs:
    print(f"\nparity pairs    : {len(pairs)}  (replica IoU shown GATED)")
    for p, o, pc, oc in pairs:
        agree = "case+gate agree" if (pc == oc and (p > 0) == (o > 0)) else (
            "gate agrees, case differs" if (p > 0) == (o > 0)
            else "<-- GATE DISAGREES")
        print(f"   replica {p:.4f} ({pc:<24}) official {o:.4f} "
              f"({oc:<24}) {agree}")
    cases = sum(1 for _, _, pc, oc in pairs if pc == oc)
    print(f"case agreement  : {cases}/{len(pairs)}")
    agree = sum(1 for p, o, _, _ in pairs if (p > 0) == (o > 0))
    print(f"gate agreement  : {agree}/{len(pairs)}")
    both = [(p, o) for p, o, _, _ in pairs if p > 0 and o > 0]
    if both:
        err = [o - p for p, o in both]
        print(f"both scored     : n={len(both)}  mean official-minus-replica "
              f"{sum(err)/len(err):+.4f}  max |delta| "
              f"{max(abs(e) for e in err):.4f}")
    # The number the accept ladder actually rides on.
    scored = [o for _, o, _, _ in pairs if o > 0]
    print(f"official scored>0: {len(scored)}/{len(pairs)}"
          + (f"   IoUs {[f'{x:.3f}' for x in sorted(scored)]}" if scored else ""))
PY
say "PARITY_DONE"
