#!/usr/bin/env python3
"""Offline parity: our self-verdict replica vs the REAL official evaluator,
replayed over every artifact set the eval container has ever written.

Why offline. The entry's whole accept decision rests on
``perception.predict_official_verdict`` being a faithful replica of
``scripts/evaluation/task2/evaluation.py``. Nothing had ever measured the two
against each other on the SAME frame: the banked evidence holds official
IoUs, the rehearsal logs hold predicted IoUs, and they never touch. Every
evaluate call saves the complete evaluator input set (mask .npy + the three
label payloads + both bbox JSONs), so the comparison needs no rig time and
no new lays — just a replay.

Self-check first: the official function is re-run on the saved inputs and
compared with the saved verdict. A row that fails that check is dropped, so
a reconstruction bug can never masquerade as a replica disagreement.

Usage (inside ebim-task2-autonomy:latest, which has our package + numpy):
    python3 replay_parity.py [EVAL_DIR] [--limit N]
"""
from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter
from types import SimpleNamespace

import numpy as np

EVAL_DIR = "/root/docker/ebim-challenge/eval-task2/evaluate"
UPSTREAM = "/root/ebim/ebim-benchmark/scripts/evaluation/task2"

# image_utils imports cv2 for the overlay drawing; the two helpers
# evaluation.py needs from it are pure. A stub keeps the OFFICIAL source
# importable verbatim instead of copying its logic (which would defeat the
# point of the comparison).
if "cv2" not in sys.modules:
    try:
        import cv2  # noqa: F401
    except ImportError:
        stub = types.ModuleType("cv2")
        stub.FONT_HERSHEY_SIMPLEX = 0
        stub.LINE_AA = 16
        stub.rectangle = lambda *a, **k: None
        stub.putText = lambda *a, **k: None
        sys.modules["cv2"] = stub

sys.path.insert(0, UPSTREAM)
from config import SEMANTIC_RAW_ID_NAME_HINTS  # noqa: E402
from evaluation import (  # noqa: E402
    evaluate_thermalpad_target_iou,
    hints_from_label_payload,
)

from ebim_task2.official_run import (  # noqa: E402
    parse_loose_label_map,
    select_target_bbox_px,
)
from ebim_task2.perception import (  # noqa: E402
    infer_semantic_ids,
    predict_official_verdict,
)

VALID = {"liner_only", "both_liner_dominant"}


def bbox_stub(path: str):
    """Rebuild a Detection2DArray-shaped duck from a saved bbox JSON."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    raw = payload.get("detections", payload) if isinstance(payload, dict) else payload
    dets = []
    for det in raw or []:
        box = det.get("bbox") or {}
        centre = box.get("center") or {}
        results = [
            SimpleNamespace(hypothesis=SimpleNamespace(
                class_id=str(r.get("class_id", "")), score=r.get("score")))
            for r in (det.get("results") or [])
        ]
        dets.append(SimpleNamespace(
            bbox=SimpleNamespace(
                center=SimpleNamespace(position=SimpleNamespace(
                    x=float(centre.get("x", 0.0)),
                    y=float(centre.get("y", 0.0)))),
                size_x=float(box.get("size_x", 0.0)),
                size_y=float(box.get("size_y", 0.0))),
            results=results))
    return SimpleNamespace(detections=dets)


def read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def main() -> int:
    eval_dir = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
        else EVAL_DIR
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    stamps = sorted(
        f[len("eval_camera_iou_"):-len(".json")]
        for f in os.listdir(eval_dir)
        if f.startswith("eval_camera_iou_") and f.endswith(".json"))
    if limit:
        stamps = stamps[-limit:]

    rows = []
    skipped = Counter()
    for ts in stamps:
        def p(kind: str, ext: str) -> str:
            return os.path.join(eval_dir, f"eval_camera_{kind}_{ts}.{ext}")

        saved = json.load(open(p("iou", "json"), encoding="utf-8"))
        sem_labels = read(p("semantic_labels", "txt"))
        tight_labels = read(p("bbox_tight_labels", "txt"))
        loose_labels = read(p("bbox_loose_labels", "txt"))
        if not (sem_labels and tight_labels) or not os.path.exists(
                p("semantic_segmentation", "npy")):
            skipped["missing artifacts"] += 1
            continue
        mask = np.load(p("semantic_segmentation", "npy"))
        tight = bbox_stub(p("bbox2d_tight", "json"))
        loose = (bbox_stub(p("bbox2d_loose", "json"))
                 if os.path.exists(p("bbox2d_loose", "json")) else None)

        hints = hints_from_label_payload(sem_labels) or SEMANTIC_RAW_ID_NAME_HINTS
        try:
            official = evaluate_thermalpad_target_iou(
                tight, tight_labels,
                thermalpad_label="thermalpad", liner_label="liner",
                target_label="target", semantic_hints=hints, label_array=mask,
                target_bbox_msg=loose,
                target_labels_payload=loose_labels if loose is not None else None)
        except Exception as exc:  # noqa: BLE001
            skipped[f"official raised: {type(exc).__name__}"] += 1
            continue

        # Self-check: the replay must reproduce what the container saved.
        if (official["orientation_case"] != saved["orientation_case"]
                or abs(official["iou_thermalpad_vs_target_current"]
                       - saved["iou_thermalpad_vs_target_current"]) > 1e-6):
            skipped["replay != saved verdict"] += 1
            continue

        # --- the replica ---
        # Two id sources, because they are not the same experiment:
        #   inferred  = the production path (geometry, official_run locks it
        #               on the clean pre-chain view; here it must work on the
        #               end state, so it fails more often than it does live);
        #   payload   = the raw ids the OFFICIAL evaluator itself resolves
        #               through (hints_from_label_payload). Comparing this
        #               one isolates the DECISION logic from id inference.
        target_bbox = None
        if loose is not None and loose_labels:
            target_bbox = select_target_bbox_px(
                loose, parse_loose_label_map(loose_labels).get("target"))

        def verdict(ids):
            if not ids:
                return None
            return predict_official_verdict(
                mask, thermalpad_id=ids["thermalpad"],
                target_id=ids["target"], liner_id=ids["liner"],
                target_bbox=target_bbox)

        inferred = infer_semantic_ids(mask)
        # The NEW production path, if the package under test exposes it:
        # live payload first, geometric lock only as a fallback.
        try:
            from ebim_task2.official_run import ids_from_semantic_payload
            prod_ids = ids_from_semantic_payload(sem_labels) or inferred
        except ImportError:
            prod_ids = inferred
        by_name = {v: k for k, v in (hints or {}).items()}
        payload_ids = None
        if {"liner", "target"} <= by_name.keys():
            payload_ids = {
                "liner": by_name["liner"],
                "target": by_name["target"],
                # No 'thermalpad' in the map is the DEGRADED regime the
                # official evaluator itself lands in: it counts 0 pad
                # pixels. -1 never matches a mask value, reproducing that.
                "thermalpad": by_name.get("thermalpad", -1),
            }
        if inferred is None and payload_ids is None:
            skipped["replica: no id source"] += 1
            continue

        v_inf = verdict(inferred)
        v_pay = verdict(payload_ids)
        v_prod = verdict(prod_ids)
        rows.append({
            "ts": ts,
            "off_case": official["orientation_case"],
            "off_iou": official["iou_thermalpad_vs_target_current"],
            "off_gated": (official["iou_thermalpad_vs_target_current"]
                          if official["is_orientation_correct"] else 0.0),
            "inf_case": v_inf[0] if v_inf else None,
            "inf_gated": (v_inf[1] if v_inf and v_inf[0] in VALID else
                          (0.0 if v_inf else None)),
            "prod_case": v_prod[0] if v_prod else None,
            "prod_gated": (v_prod[1] if v_prod and v_prod[0] in VALID else
                           (0.0 if v_prod else None)),
            "pay_case": v_pay[0] if v_pay else None,
            "pay_gated": (v_pay[1] if v_pay and v_pay[0] in VALID else
                          (0.0 if v_pay else None)),
            "sem_has_pad": "thermalpad" in (hints or {}).values(),
            "tight_pad_id": official["thermalpad_label_id"],
        })

    print(f"snapshots examined : {len(stamps)}")
    print(f"usable rows        : {len(rows)}")
    for k, v in skipped.most_common():
        print(f"   skipped: {k}: {v}")
    if not rows:
        return 1

    for tag, ck, gk in (("INFERRED ids (production path)", "inf_case", "inf_gated"),
                        ("PAYLOAD ids (evaluator's own map)", "pay_case", "pay_gated"),
                        ("NEW production path (payload, else geometry)",
                         "prod_case", "prod_gated")):
        sub = [r for r in rows if r[ck] is not None]
        print(f"\n{'=' * 70}\n== {tag}: n={len(sub)}")
        if not sub:
            continue
        conf = Counter((r[ck], r["off_case"]) for r in sub)
        preds = sorted({r[ck] for r in sub})
        offs = sorted({r["off_case"] for r in sub})
        w = max(len(c) for c in preds) + 2
        print("  replica \\ official".ljust(w) +
              "".join(f"{c[:22]:>24}" for c in offs))
        for pc in preds:
            print(f"{pc:<{w}}" + "".join(
                f"{conf.get((pc, oc), 0):>24}" for oc in offs))

        agree = sum(1 for r in sub if r[ck] == r["off_case"])
        gate = sum(1 for r in sub if (r[gk] > 0) == (r["off_gated"] > 0))
        print(f"  exact case agreement : {agree}/{len(sub)} "
              f"({agree/len(sub):.1%})")
        print(f"  gate agreement (>0)  : {gate}/{len(sub)} "
              f"({gate/len(sub):.1%})")

        fn = [r for r in sub if r[gk] == 0 and r["off_gated"] > 0]
        fp = [r for r in sub if r[gk] > 0 and r["off_gated"] == 0]
        print(f"  replica ZERO / official SCORED : {len(fn)}  (lays thrown away)")
        if fn:
            v = sorted(r["off_gated"] for r in fn)
            print(f"     official IoUs min {v[0]:.4f} med {v[len(v)//2]:.4f} "
                  f"max {v[-1]:.4f} mean {sum(v)/len(v):.4f}")
            print(f"     replica cases {Counter(r[ck] for r in fn).most_common()}")
            print(f"     of those, semantic map still named 'thermalpad': "
                  f"{sum(1 for r in fn if r['sem_has_pad'])}/{len(fn)}")
        print(f"  replica SCORED / official ZERO : {len(fp)}  (false accepts)")
        if fp:
            print(f"     official cases "
                  f"{Counter(r['off_case'] for r in fp).most_common()}")
            v = sorted(r[gk] for r in fp)
            print(f"     replica IoUs min {v[0]:.4f} max {v[-1]:.4f}")

        both = [r for r in sub if r[gk] > 0 and r["off_gated"] > 0]
        if both:
            err = [r["off_gated"] - r[gk] for r in both]
            print(f"  both scored (n={len(both)}): mean official-minus-replica "
                  f"{sum(err)/len(err):+.4f}, "
                  f"max |err| {max(abs(e) for e in err):.4f}")
            for r in sorted(both, key=lambda r: -abs(r["off_gated"] - r[gk]))[:4]:
                print(f"     {r['ts']}  replica {r[gk]:.4f} "
                      f"official {r['off_gated']:.4f}  "
                      f"({r[ck]} / {r['off_case']})")

    print("\n== evaluator regime (does the LIVE semantic map still name "
          "'thermalpad'?) ==")
    for has in (True, False):
        sub = [r for r in rows if r["sem_has_pad"] is has]
        if not sub:
            continue
        cases = Counter(r["off_case"] for r in sub)
        scored = sum(1 for r in sub if r["off_gated"] > 0)
        print(f"   sem_has_thermalpad={has}: n={len(sub)}  "
              f"official scored>0 {scored}/{len(sub)} ({scored/len(sub):.1%})")
        print(f"      official cases: {cases.most_common()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
