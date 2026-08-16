"""Perception from the semantic-segmentation camera.

Three capabilities:

1. :func:`observe_placement` — the scaffold's local development success gate.
   It computes bbox IoU between ``(thermalpad ∪ liner)`` and ``target`` pixels
   plus the ``liner_dominance_ratio``. The official evaluator is authoritative;
   this is only a fast local proxy.

2. :func:`estimate_target_pose` — the geometric pose estimator used by the
   perception-driven policy. It turns raw semantic IDs into an SE(2) placement
   target (``x, y, yaw``) in pixel space, using mask centroids and the liner's
   principal axis from PCA. Pure numpy, no GPU.

3. :func:`estimate_pad_pose` — the grasp-side counterpart of (2): centroid +
   principal-axis yaw of the ``thermalpad ∪ liner`` blob, i.e. where the pad
   actually is so the gripper can pick it up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementObservation:
    """Local-development success signals derived from a semantic mask."""

    iou: float
    liner_dominance_ratio: float
    pad_pixels: int
    target_pixels: int
    #: Axis-aligned ``(y0, y1, x0, x1)`` pixel boxes the IoU was computed
    #: from (None when the blob is not visible); the runner reports the
    #: placement residual in millimetres from these.
    placed_bbox: tuple[int, int, int, int] | None = None
    target_bbox: tuple[int, int, int, int] | None = None


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Axis-aligned bounding box ``(y0, y1, x0, x1)`` of True pixels, or None."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    inter_h = max(0, iy1 - iy0 + 1)
    inter_w = max(0, ix1 - ix0 + 1)
    inter = inter_h * inter_w
    area_a = (ay1 - ay0 + 1) * (ax1 - ax0 + 1)
    area_b = (by1 - by0 + 1) * (bx1 - bx0 + 1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def observe_placement(
    mask: np.ndarray,
    *,
    thermalpad_id: int,
    target_id: int,
    liner_id: int,
) -> PlacementObservation:
    """Compute local success signals from an ``int32`` semantic-ID mask.

    Returns safe zeros when the relevant objects are not visible.
    """
    mask = np.asarray(mask)
    pad = mask == thermalpad_id
    tgt = mask == target_id
    liner = mask == liner_id

    pad_px = int(pad.sum())
    target_px = int(tgt.sum())
    liner_px = int(liner.sum())

    # liner_dominance_ratio = liner / (liner + pad): how much of the pad+liner
    # blob is the correctly-oriented liner.
    denom = liner_px + pad_px
    liner_dominance = float(liner_px) / float(denom) if denom > 0 else 0.0

    placed = pad | liner
    placed_bbox = _bbox(placed)
    target_bbox = _bbox(tgt)
    if placed_bbox is None or target_bbox is None:
        iou = 0.0
    else:
        iou = _iou(placed_bbox, target_bbox)

    return PlacementObservation(
        iou=iou,
        liner_dominance_ratio=liner_dominance,
        pad_pixels=pad_px,
        target_pixels=target_px,
        placed_bbox=placed_bbox,
        target_bbox=target_bbox,
    )


@dataclass(frozen=True)
class TargetPose:
    """SE(2) placement target in image/pixel coordinates."""

    x: float
    y: float
    yaw: float
    visible: bool


def _centroid_axis(mask: np.ndarray) -> tuple[float, float, float] | None:
    """Return ``(cx, cy, yaw)`` of a mask's principal axis, or None if empty.

    ``yaw`` is the angle of the dominant PCA eigenvector in image space
    (radians), measured so that it is stable under reflection by taking the
    axis with the largest spread.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        if xs.size == 1:
            return float(xs[0]), float(ys[0]), 0.0
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    # 2x2 covariance of pixel coordinates
    sxx = float(np.mean(dx * dx))
    syy = float(np.mean(dy * dy))
    sxy = float(np.mean(dx * dy))
    # eigen-decomposition of [[sxx, sxy],[sxy, syy]]
    trace = sxx + syy
    diff = sxx - syy
    # largest eigenvalue eigenvector direction. arctan2 handles diff == 0
    # correctly (diagonal blob -> +/- pi/4); for a fully isotropic blob this
    # yields 0.0, a sane default orientation.
    yaw = float(0.5 * np.arctan2(2.0 * sxy, diff))
    return cx, cy, yaw


def estimate_target_pose(
    mask: np.ndarray,
    *,
    target_id: int,
    liner_id: int,
) -> TargetPose:
    """Estimate the placement target pose (pixel-space SE(2)).

    The target position is the centroid of the ``target`` mask. The desired
    orientation is taken from the ``liner`` mask's principal axis so the pad is
    laid down aligned with the liner. If the liner is not visible the orientation
    defaults to 0 (any rotation that satisfies the dominance gate still passes).
    """
    mask = np.asarray(mask)
    tgt = mask == target_id
    if tgt.sum() == 0:
        return TargetPose(0.0, 0.0, 0.0, visible=False)

    axis = _centroid_axis(tgt)
    if axis is None:
        return TargetPose(0.0, 0.0, 0.0, visible=False)
    cx, cy, _ = axis

    liner = mask == liner_id
    liner_axis = _centroid_axis(liner)
    yaw = liner_axis[2] if liner_axis is not None else 0.0
    return TargetPose(x=cx, y=cy, yaw=yaw, visible=True)


def infer_semantic_ids(mask: np.ndarray,
                       drop_raster_noise: bool = False
                       ) -> dict[str, int] | None:
    """Geometric semantic-id inference from the raw mask alone.

    Scene-instance raw ids drift, and after an in-place ``scene_reset`` the
    ``semantic_labels`` JSON goes stale (timestamp frozen at scene load, the
    class set can lose ``thermalpad``) — so ids must be recovered from blob
    geometry:

    * background — the id with the most pixels;
    * target — a slat-shaped blob (w >= 2.5 h) with a well-filled bbox AND a
      real slat width (w >= 30 px): the highlighted DIMM slot;
    * board rack — one id spanning several disjoint slats: low bbox fill or
      a tall bbox (h > 100 px). Excluded;
    * pad/liner — the remaining small blobs (n <= 6000 px), no fill/aspect
      requirement (arm occlusion leaves ragged low-fill fragments): the
      larger is ``liner`` (the film covers the pad
      from the top-down camera), the smaller ``thermalpad``; a single blob
      serves as both.

    Returns None when the scene is not unambiguously classifiable (e.g. the
    arm occludes the pad) — callers should keep the previous mapping.
    """
    mask = np.asarray(mask)
    ids, counts = np.unique(mask, return_counts=True)
    if len(ids) < 3:
        return None
    bg = int(ids[int(np.argmax(counts))])

    slats = []   # (id, fill, bbox) — bbox is (x0, y0, x1, y1)
    racks = []   # rack bboxes
    small = []   # (n_pixels, id)
    for i, n in zip(ids, counts):
        if int(i) == bg:
            continue
        ys, xs = np.nonzero(mask == i)
        w = int(xs.max()) - int(xs.min()) + 1
        h = int(ys.max()) - int(ys.min()) + 1
        fill = float(n) / float(w * h)
        if drop_raster_noise and int(n) < 20 and fill < 0.3:
            # drop scattered raster noise (tiny, low-fill fragments);
            # off by default.
            continue
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if w >= 2.5 * h and fill > 0.5 and w >= 30:
            slats.append((int(i), fill, box))
        elif h > 100 or fill <= 0.15:
            racks.append(box)                             # board rack column
        elif n <= 6000:
            small.append((int(n), int(i)))
    if not slats:
        return None
    # The DIMM slot sits inside the board-rack column; a flat unoccluded liner
    # can be slat-shaped with a high fill too, but lies away from the rack
    # (same x band, well south). Prefer slat candidates whose bbox intersects
    # a rack bbox in BOTH axes, then higher fill.
    def _overlaps_rack(b) -> bool:
        return any(
            b[0] <= r[2] and r[0] <= b[2] and b[1] <= r[3] and r[1] <= b[3]
            for r in racks
        )

    slats.sort(key=lambda s: (_overlaps_rack(s[2]), s[1]), reverse=True)
    target_id = slats[0][0]
    # A liner misfiled as a slat candidate must still be classifiable as the
    # liner when a rack-overlapping slot won the target slot.
    for sid, _fill, box in slats[1:]:
        if not _overlaps_rack(box):
            small.append((int((mask == sid).sum()), sid))
    if not small:
        return None
    small.sort(reverse=True)
    liner_id = small[0][1]
    pad_id = small[1][1] if len(small) > 1 else liner_id
    return {"thermalpad": pad_id, "target": target_id, "liner": liner_id}


def estimate_pad_pose(
    mask: np.ndarray,
    *,
    thermalpad_id: int,
    liner_id: int,
) -> TargetPose:
    """Estimate the grasp target pose (pixel-space SE(2)).

    The grasp point is the centroid of the ``thermalpad ∪ liner`` blob and the
    approach yaw its principal axis — where the pad actually is, as opposed to
    :func:`estimate_target_pose`, which is where the pad must end up. Returns
    an invisible pose when the blob is empty.
    """
    mask = np.asarray(mask)
    blob = (mask == thermalpad_id) | (mask == liner_id)
    axis = _centroid_axis(blob)
    if axis is None:
        return TargetPose(0.0, 0.0, 0.0, visible=False)
    cx, cy, yaw = axis
    return TargetPose(x=cx, y=cy, yaw=yaw, visible=True)


def predict_official_verdict(
    mask: np.ndarray,
    *,
    thermalpad_id: int,
    target_id: int,
    liner_id: int,
    target_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[str, float]:
    """Predict the official dev evaluator's verdict for this frame.

    Replica of the upstream dev evaluator (loose-bbox target, tight pad
    stream):

    * TARGET bbox = loose annotator (full extent regardless of occlusion).
      Callers that track the plate pass its full-extent pixel bbox as
      ``target_bbox`` (``(y0, y1, x0, x1)`` inclusive, as ``_bbox``);
      without it the visible-pixel bbox is used — an under-estimate on
      good lays (the cover hides the plate).
    * PAD = tight stream: liner only -> ``liner_only`` (correct);
      thermalpad only -> ``thermalpad_only`` (wrong face); both ->
      pixel-count tie-break at strictly >0.9 dominance via the LIVE ids.

    Returns ``(orientation_case, predicted_iou)``.
    """
    mask = np.asarray(mask)
    pad_px = int((mask == thermalpad_id).sum())
    liner_px = int((mask == liner_id).sum())
    if target_bbox is None:
        target_bbox = _bbox(mask == target_id)
    if target_bbox is None:
        return "no_target_bbox", 0.0
    if pad_px == 0 and liner_px == 0:
        return "neither_pad_present", 0.0
    if pad_px == 0:
        b = _bbox(mask == liner_id)
        return "liner_only", (_iou(b, target_bbox) if b else 0.0)
    if liner_px == 0:
        b = _bbox(mask == thermalpad_id)
        return "thermalpad_only", (_iou(b, target_bbox) if b else 0.0)

    total = pad_px + liner_px
    if liner_px / total > 0.9:
        b = _bbox(mask == liner_id)
        return "both_liner_dominant", (_iou(b, target_bbox) if b else 0.0)
    if pad_px / total > 0.9:
        b = _bbox(mask == thermalpad_id)
        return "both_thermalpad_dominant", (_iou(b, target_bbox) if b else 0.0)
    return "sideways", 0.0


@dataclass(frozen=True)
class PadGeometry:
    """Pad geometry recovered from the eval camera alone (no ground truth).

    All heights are metres above the floor plane; xy are world metres.
    """

    centroid_xy: tuple[float, float]
    top_xy: tuple[float, float]
    z_top: float
    z_bottom: float
    pixels: int


def estimate_pad_geometry_from_depth(
    mask: np.ndarray,
    depth: np.ndarray,
    *,
    thermalpad_id: int,
    liner_id: int,
    camera_height_m: float,
    focal_px: float,
    cx: float,
    cy: float,
    origin_xy: tuple[float, float],
    flip_y: bool = True,
    top_band_m: float = 0.008,
    surround_px: int = 6,
) -> PadGeometry | None:
    """Replace the ``/isaac/task2/pad_points`` ground truth with camera data.

    The scripted policy needs four numbers about the pad: where it is, where
    its top cluster is, how high its crest is, and what it is resting on (the
    grasp clamps against that). All four are available from the top-down eval
    camera without ground truth:

    * heights come from depth — ``z = camera_height - depth``; the crest is a
      high percentile (robust to single noisy pixels);
    * the support surface is the depth of a ring just *outside* the pad blob,
      which is the tray (or floor) the pad sits on;
    * pixel→world uses the pinhole scale **at the pad's own measured height**,
      ``focal_px / (camera_height - z)`` — one flat pixels-per-metre is only
      correct at the height it was calibrated at.

    Returns None when the pad is not visible or its depth is unusable, so the
    caller keeps its previous estimate rather than acting on garbage.
    """
    mask = np.asarray(mask)
    depth = np.asarray(depth, dtype=np.float64)
    if mask.shape != depth.shape:
        return None
    blob = (mask == thermalpad_id) | (mask == liner_id)
    n = int(blob.sum())
    if n == 0:
        return None
    finite = np.isfinite(depth) & (depth > 0.0)
    sel = blob & finite
    if int(sel.sum()) < max(8, n // 8):
        return None

    z = camera_height_m - depth[sel]
    z_top = float(np.percentile(z, 99.0))

    # Support surface: a ring just outside the blob. Dilate by a few pixels
    # with a cheap max-filter over shifts, then take the ring's median height.
    ring = np.zeros_like(blob)
    for dy in range(-surround_px, surround_px + 1):
        for dx in range(-surround_px, surround_px + 1):
            ring |= np.roll(np.roll(blob, dy, axis=0), dx, axis=1)
    ring &= ~blob
    ring &= finite
    if int(ring.sum()) >= 8:
        z_bottom = float(np.median(camera_height_m - depth[ring]))
    else:
        z_bottom = 0.0

    ys, xs = np.nonzero(sel)

    def to_world(px: np.ndarray, py: np.ndarray, plane_z: float):
        scale = focal_px / max(camera_height_m - plane_z, 1e-6)
        wx = origin_xy[0] + (px.mean() - cx) / scale
        dy = (py.mean() - cy) / scale
        return float(wx), float(origin_xy[1] + (-dy if flip_y else dy))

    centroid_xy = to_world(xs.astype(np.float64), ys.astype(np.float64), z_top)

    top = z >= z_top - top_band_m
    if int(top.sum()) >= 4:
        top_xy = to_world(xs[top].astype(np.float64), ys[top].astype(np.float64), z_top)
    else:
        top_xy = centroid_xy

    return PadGeometry(
        centroid_xy=centroid_xy,
        top_xy=top_xy,
        z_top=z_top,
        z_bottom=z_bottom,
        pixels=n,
    )
