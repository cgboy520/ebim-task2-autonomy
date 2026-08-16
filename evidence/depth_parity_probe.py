"""depth-vs-GT pad geometry residual probe (runner A/B definitions)."""
import sys, time
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

from ebim_task2.perception import estimate_pad_geometry_from_depth
from ebim_task2.official_run import ids_from_semantic_payload
from ebim_task2.runner import _decode_semantic

CAM = dict(camera_height_m=1.95, focal_px=1221.665, cx=640.0, cy=360.0,
           origin_xy=(0.837, -0.065), flip_y=True)


def depth_decode(msg):
    arr = np.frombuffer(msg.data, dtype=np.float32)
    return arr.reshape(msg.height, msg.width).astype(np.float64)


def main(n=12):
    rclpy.init()
    node = rclpy.create_node("depth_probe")
    latest = {}
    node.create_subscription(
        Image, "/isaac/eval_camera/semantic_segmentation",
        lambda m: latest.__setitem__("mask", m), qos_profile_sensor_data)
    node.create_subscription(
        Image, "/isaac/eval_camera/depth",
        lambda m: latest.__setitem__("depth", m), qos_profile_sensor_data)
    node.create_subscription(
        String, "/isaac/eval_camera/semantic_labels",
        lambda m: latest.__setitem__("labels", m), qos_profile_sensor_data)
    node.create_subscription(
        Float32MultiArray, "/isaac/task2/pad_points",
        lambda m: latest.__setitem__("pad", m), qos_profile_sensor_data)
    rows = []
    t_end = time.monotonic() + 150
    while time.monotonic() < t_end and len(rows) < n:
        rclpy.spin_once(node, timeout_sec=0.3)
        if not all(k in latest for k in ("mask", "depth", "labels", "pad")):
            continue
        ids = ids_from_semantic_payload(latest["labels"].data)
        if ids is None or ids.get("thermalpad", -1) < 0:
            latest.pop("labels", None)
            continue
        mask = _decode_semantic(latest.pop("mask"))
        depth = depth_decode(latest.pop("depth"))
        if mask is None:
            continue
        d = list(latest["pad"].data)
        pts = d[2:]
        n3 = (len(pts) // 3) * 3
        xs, ys, zs = pts[0:n3:3], pts[1:n3:3], pts[2:n3:3]
        if not xs:
            continue
        gt_c = (sum(xs) / len(xs), sum(ys) / len(ys))
        gt_ztop = float(np.percentile(zs, 99.0))
        g = estimate_pad_geometry_from_depth(
            mask, depth, thermalpad_id=ids["thermalpad"],
            liner_id=ids["liner"], **CAM)
        if g is None:
            print("depth estimate: None (pad invisible/unusable)")
            time.sleep(1.0)
            continue
        dx = (g.centroid_xy[0] - gt_c[0]) * 1000
        dy = (g.centroid_xy[1] - gt_c[1]) * 1000
        dz = (g.z_top - gt_ztop) * 1000
        rows.append((dx, dy, dz))
        print(f"sample {len(rows)}: d_xy=({dx:+.1f},{dy:+.1f}) mm  "
              f"d_ztop={dz:+.1f} mm  z_bot={g.z_bottom:.4f}  px={g.pixels}",
              flush=True)
        time.sleep(1.0)
    if rows:
        a = np.array(rows)
        print(f"\nn={len(rows)}  |d_xy| mean={np.hypot(a[:,0],a[:,1]).mean():.1f} mm  "
              f"max={np.hypot(a[:,0],a[:,1]).max():.1f} mm  "
              f"d_ztop mean={a[:,2].mean():+.1f} sd={a[:,2].std():.1f} mm")
    else:
        print("NO SAMPLES — streams incomplete or pad class absent")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
