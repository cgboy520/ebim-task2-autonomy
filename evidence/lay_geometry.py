#!/usr/bin/env python3
"""Turn an evaluator IoU artifact into the lay geometry that produced it.

The official metric is an IoU of AXIS-ALIGNED boxes over two identical
0.12 x 0.02 m footprints, which makes the raw number hard to read: a
short lay, a laterally offset lay and a spun PLATE all lower it for
different reasons and want different fixes. This prints, in millimetres
at the plate plane, what each box actually is and what the score would
have been with each defect removed one at a time.

Usage: lay_geometry.py <eval_camera_iou_*.json> [...]
"""
import json
import math
import sys

PX_PER_M = 626.0        # room plate plane; barebone 625/635
PAD_L, PAD_W = 0.120, 0.020


def rect(b):
    return (b["x2"] - b["x1"], b["y2"] - b["y1"],
            (b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


def iou_of(w1, h1, cx1, cy1, w2, h2, cx2, cy2):
    ix = max(0.0, min(cx1 + w1 / 2, cx2 + w2 / 2)
             - max(cx1 - w1 / 2, cx2 - w2 / 2))
    iy = max(0.0, min(cy1 + h1 / 2, cy2 + h2 / 2)
             - max(cy1 - h1 / 2, cy2 - h2 / 2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def aabb_of_rotated(length, width, deg):
    c, s = abs(math.cos(math.radians(deg))), abs(math.sin(math.radians(deg)))
    return length * c + width * s, length * s + width * c


def plate_tilt(w, h):
    """Degrees of in-image rotation implied by a 120x20 mm plate's AABB."""
    best, bd = 0.0, 1e9
    for deg in [d / 4.0 for d in range(0, 361)]:
        aw, ah = aabb_of_rotated(PAD_L * PX_PER_M, PAD_W * PX_PER_M, deg)
        d = abs(aw - w) + abs(ah - h)
        if d < bd:
            best, bd = deg, d
    return best, bd


def report(path):
    r = json.load(open(path))
    r = r.get("result", r)
    iou = r.get("iou_thermalpad_vs_target_current")
    if iou is None:
        print(f"{path}: no verdict")
        return
    print(f"\n== {path.split('/')[-1]}")
    print(f"   case {r.get('orientation_case')}  IoU {iou:.4f}  "
          f"coverage {r.get('coverage_on_target', 0):.3f}  "
          f"precision {r.get('precision_on_pad', 0):.3f}")
    pb, tb = r.get("pad_bbox"), r.get("target_bbox")
    if not (pb and tb):
        print("   no boxes (zero verdict)")
        return
    pw, ph, pcx, pcy = rect(pb)
    tw, th, tcx, tcy = rect(tb)
    k = 1000.0 / PX_PER_M
    tilt, resid = plate_tilt(tw, th)
    print(f"   pad    {pw:5.1f} x {ph:5.1f} px = {pw * k:6.1f} x {ph * k:5.1f} mm")
    print(f"   target {tw:5.1f} x {th:5.1f} px = {tw * k:6.1f} x {th * k:5.1f} mm"
          f"   -> plate turned ~{min(tilt, 90 - tilt):.0f} deg off axis "
          f"(fit residual {resid:.1f} px)")
    print(f"   centre offset {(pcx - tcx) * k:+.1f} , {(pcy - tcy) * k:+.1f} mm")
    print(f"   check IoU {iou_of(pw, ph, pcx, pcy, tw, th, tcx, tcy):.4f}")
    # counterfactuals, one defect at a time
    ideal_w, ideal_h = aabb_of_rotated(PAD_L * PX_PER_M, PAD_W * PX_PER_M, 0.0)
    print("   if the pad were perfectly centred:      "
          f"{iou_of(pw, ph, tcx, tcy, tw, th, tcx, tcy):.4f}")
    print("   if the plate had not turned:            "
          f"{iou_of(pw, ph, pcx, pcy, ideal_w, ideal_h, tcx, tcy):.4f}")
    print("   if the pad were a flat full-length lay: "
          f"{iou_of(ideal_w, ideal_h, pcx, pcy, tw, th, tcx, tcy):.4f}")
    print("   both:                                   "
          f"{iou_of(ideal_w, ideal_h, tcx, tcy, ideal_w, ideal_h, tcx, tcy):.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
