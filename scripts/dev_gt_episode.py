#!/usr/bin/env python3
"""DEV-ONLY ground-truth episode runner for the EBiM Task 2 sim.

This script is NOT part of the competition policy. It runs on hosts without
Isaac Sim's RTX renderer, where the eval camera (and therefore the official
scorer) is unavailable. What it validates is the entire mechanical stack
that the competition policy depends on:

  1. pedal-token base driving (/pedal/state, body-frame, 1 s watchdog)
  2. runtime arm-base self-calibration (/isaac/*_ee_pose x FK^-1)
  3. camera-free world->arm-base goal transform + DLS IK + joint interpolation
  4. gripper driver semantics (0.0 open / 0.8 closed)
  5. pick-and-place mechanics against the (CPU-fallback) scene physics

It uses /isaac/task2/object_poses (ground truth) instead of the semantic
camera. On a native-Linux host the *competition* path (PerceptionPolicy, fed
by /isaac/eval_camera/semantic_segmentation) replaces the GT lookup; the
base-approach, calibration, and motion layers are shared.

Usage (inside the sidecar image, host network):
  python3 /opt/ebim-task2/scripts/dev_gt_episode.py [--arm right]
      [--park-distance 0.62] [--skip-drive] [--cycles 1]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

sys.path.insert(0, "/opt/ebim-task2/src")
from ebim_task2.calibration import estimate_world_to_arm_base, pose_msg_to_matrix  # noqa: E402
from ebim_task2.motion import franka_fk, interpolate_waypoints, solve_ik  # noqa: E402

JOINTS = [f"right_fr3v2_joint{i + 1}" for i in range(7)]
HOME = np.array([0.0, -0.4, 0.0, -2.4, 0.0, 1.6, 0.8])
GRIPPER_JOINT = "right_right_finger_joint"

PEDAL_HZ = 5.0           # > 1 Hz watchdog
DRIVE_LIN = 0.5          # m/s (bridge --pedal-linear-speed)
DRIVE_ANG = 1.2          # rad/s (bridge --pedal-angular-speed)


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class GTEpisode(Node):
    def __init__(self, arm: str) -> None:
        super().__init__("dev_gt_episode")
        self.arm = arm
        self.joints: list[float] | None = None
        self.ee_world: np.ndarray | None = None      # 4x4
        self.base_xyy: tuple[float, float, float] | None = None
        self.objects: dict[str, list[float]] = {}
        self._t_bw: np.ndarray | None = None         # world -> arm-base (calibrated)

        qos = QoSProfile(depth=10)
        self.create_subscription(JointState, f"/isaac/{arm}_joint_states", self._on_joints, qos)
        self.create_subscription(PoseStamped, f"/isaac/{arm}_ee_pose", self._on_ee, qos)
        self.create_subscription(Odometry, "/isaac/odom", self._on_odom, qos)
        self.create_subscription(String, "/isaac/task2/object_poses", self._on_obj, qos)

        self.arm_pub = self.create_publisher(JointState, f"/isaac/{arm}_joint_commands", 10)
        self.grip_pub = self.create_publisher(JointState, f"/isaac/{arm}_robotiq_joint_commands", 10)
        self.pedal_pub = self.create_publisher(String, "/pedal/state", 10)

    # -- subscribers -------------------------------------------------------
    def _on_joints(self, msg: JointState) -> None:
        self.joints = [float(p) for p in msg.position[:7]]

    def _on_ee(self, msg: PoseStamped) -> None:
        self.ee_world = pose_msg_to_matrix(
            (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z),
            (msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w),
        )

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self.base_xyy = (float(p.x), float(p.y), yaw_of(msg.pose.pose.orientation))

    def _on_obj(self, msg: String) -> None:
        try:
            self.objects = json.loads(msg.data)["objects"]
        except Exception as e:
            self.get_logger().warning(f"object_poses parse failed: {e}")

    # -- primitives --------------------------------------------------------
    def spin_until(self, pred, timeout_s: float, what: str) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.05)
            if pred():
                return True
        self.get_logger().error(f"timeout waiting for {what}")
        return False

    def publish_gripper(self, open_fraction: float) -> None:
        msg = JointState()
        msg.name = [GRIPPER_JOINT]
        msg.position = [float(0.8 * (1.0 - open_fraction))]  # 0 open / 0.8 closed
        self.grip_pub.publish(msg)

    def publish_arm(self, q: np.ndarray) -> None:
        msg = JointState()
        msg.name = JOINTS
        msg.position = [float(x) for x in q]
        self.arm_pub.publish(msg)

    def pedal(self, token: str) -> None:
        self.pedal_pub.publish(String(data=token))

    # -- high level steps --------------------------------------------------
    def calibrate_arm_base(self, samples: int = 20) -> np.ndarray:
        # Wait for the arm to be stationary before sampling.
        last_q = None
        still_since = time.time()
        while time.time() - still_since < 1.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joints is None:
                continue
            if last_q is not None and np.max(np.abs(np.asarray(self.joints) - last_q)) < 0.002:
                pass
            else:
                still_since = time.time()
            last_q = list(self.joints)
            if time.time() - still_since > 8.0:
                break
        got: list[tuple[list[float], np.ndarray]] = []
        t0 = time.time()
        while len(got) < samples and time.time() - t0 < 15.0:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joints is not None and self.ee_world is not None:
                got.append((list(self.joints), self.ee_world.copy()))
                time.sleep(0.02)
        t_wb, spread = estimate_world_to_arm_base(got)
        self.get_logger().info(
            f"arm-base calibration: spread={spread * 1000:.2f} mm, "
            f"T_world_armbase xyz={np.round(t_wb[:3, 3], 3).tolist()}"
        )
        return t_wb

    def drive_base(self, goal_xyy: tuple[float, float, float], tol_xy=0.08, tol_yaw=0.08) -> bool:
        """Closed-loop pedal driving in SE(2). Body-frame: FWD/BACK = +-x,
        A/B = +-y (lateral), A+C/B+C = +-yaw."""
        gx, gy, gyaw = goal_xyy
        t0 = time.time()
        period = 1.0 / PEDAL_HZ
        last_log = 0.0
        rotating = False
        last_pose: tuple[float, float, float] | None = None
        last_progress = time.time()
        recover_until = 0.0
        while time.time() - t0 < 600.0:
            rclpy.spin_once(self, timeout_sec=period)
            if self.base_xyy is None:
                continue
            x, y, yaw = self.base_xyy

            # --- stall detection + recovery ------------------------------
            if last_pose is None or math.hypot(x - last_pose[0], y - last_pose[1]) > 0.02 \
                    or abs(wrap(yaw - last_pose[2])) > 0.02:
                last_pose = (x, y, yaw)
                last_progress = time.time()
            now = time.time()
            if now < recover_until:
                self.pedal("BACK")
                continue
            if now - last_progress > 8.0:
                self.get_logger().warning("drive stalled; backing up to recover")
                recover_until = now + 2.5
                last_progress = now + 3.0
                self.pedal("BACK")
                continue

            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
            heading_err = wrap(math.atan2(dy, dx) - yaw)
            yaw_err = wrap(gyaw - yaw)
            if now - last_log > 10.0:
                last_log = now
                self.get_logger().info(
                    f"driving: base=({x:.2f},{y:.2f},{yaw:.2f}) dist={dist:.2f} heading_err={heading_err:.2f}"
                )

            # rotation hysteresis to avoid oscillation
            if rotating and abs(heading_err) < 0.15:
                rotating = False
            elif not rotating and abs(heading_err) > 0.35 and dist > tol_xy:
                rotating = True

            token = "NONE"
            if rotating:
                token = "A+C" if heading_err > 0 else "B+C"
            elif dist > tol_xy:
                # aligned enough: drive forward, strafe residual lateral
                fwd = math.cos(heading_err) * dist
                lat = math.sin(heading_err) * dist
                if abs(lat) > abs(fwd) * 0.6 and abs(lat) > 0.10:
                    token = "A" if lat > 0 else "B"
                else:
                    token = "FWD" if fwd > 0 else "BACK"
            elif abs(yaw_err) > tol_yaw:
                token = "A+C" if yaw_err > 0 else "B+C"
            else:
                self.pedal("NONE")
                self.get_logger().info(f"parked at ({x:.3f}, {y:.3f}, {yaw:.3f})")
                return True
            self.pedal(token)
        self.pedal("NONE")
        self.get_logger().error("base drive timeout")
        return False

    def move_to_world(self, world_goal: np.ndarray, *, fine_iters: int = 3,
                      fine_tol_m: float = 0.005, settle_tol: float = 0.15) -> bool:
        """World-frame move with ee-pose servo refinement.

        ``world_goal`` is the desired TCP pose in the WORLD frame. The coarse
        phase runs the usual IK+interpolation; then up to ``fine_iters``
        corrections re-anchor on the MEASURED flange pose (/isaac/*_ee_pose):
        the measured TCP is flange @ Tr(0,0,tcp_z), the residual between
        desired and measured TCP is added onto the current flange pose and
        re-solved. This cancels FK/calibration error at the workspace.
        """
        if not self.move_to(self._t_bw @ world_goal, settle_tol=settle_tol):
            return False
        tcp_off = np.array([0.0, 0.0, 0.15])
        from ebim_task2.motion import _jacobian
        for it in range(fine_iters):
            if self.ee_world is None:
                return True  # cannot refine without measurements
            tcp_meas = self.ee_world[:3, 3] + self.ee_world[:3, :3] @ tcp_off
            err = world_goal[:3, 3] - tcp_meas
            if float(np.linalg.norm(err)) <= fine_tol_m:
                self.get_logger().info(f"servo converged after {it} iters")
                return True
            self.get_logger().info(f"servo iter {it}: tcp err {np.round(err, 4).tolist()}")
            # Cartesian correction via the position Jacobian (no re-IK, so no
            # branch flips): dq = pinv(J) @ dp, clipped to small safe steps.
            cur = np.asarray(self.joints, dtype=np.float64)
            dp_base = self._t_bw[:3, :3] @ err
            jac = _jacobian(cur)[:3, :]
            dq = np.linalg.pinv(jac) @ dp_base
            dq = np.clip(dq, -0.15, 0.15)
            goal_q = cur + dq
            t_end = time.time() + 12.0
            while time.time() < t_end:
                self.publish_arm(goal_q)
                rclpy.spin_once(self, timeout_sec=0.05)
        if self.ee_world is not None:
            tcp_meas = self.ee_world[:3, 3] + self.ee_world[:3, :3] @ tcp_off
            final_err = float(np.linalg.norm(world_goal[:3, 3] - tcp_meas))
            self.get_logger().info(f"servo final tcp err: {final_err * 1000:.1f} mm")
        return True

    def move_to(self, goal_pose: np.ndarray, max_step=0.08, settle_tol=0.15,
                timeout_s=30.0) -> bool:
        """IK + interpolated joint streaming; the sim's PD chases latched
        position targets with visible lag/oscillation, so intermediate points
        are streamed open-loop and only the final goal is waited out (loose
        tolerance)."""
        if self.joints is None:
            return False
        cur = np.asarray(self.joints, dtype=np.float64)
        res = solve_ik(goal_pose, cur, max_iters=300)
        if not res.success:
            # Coarse phases tolerate a near-solution (the ee servo refines the
            # final pose anyway); refuse only genuinely unreachable goals.
            if res.pos_error > 0.02:
                self.get_logger().error(f"IK failed: pos_error={res.pos_error:.4f}")
                return False
            self.get_logger().warning(f"IK near-solution accepted: err={res.pos_error:.4f}")
        traj = interpolate_waypoints(cur, res.q, max_joint_step=max_step)
        for wp in traj:
            for _ in range(4):  # ~0.2 s of streaming per waypoint
                self.publish_arm(wp)
                rclpy.spin_once(self, timeout_sec=0.05)
        goal_q = traj[-1]
        t_deadline = time.time() + timeout_s
        best = float("inf")
        while time.time() < t_deadline:
            self.publish_arm(goal_q)
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.joints is not None:
                err = float(np.max(np.abs(np.asarray(self.joints) - goal_q)))
                best = min(best, err)
                if err <= settle_tol:
                    return True
        self.get_logger().error(f"joint convergence timeout (best err {best:.3f} rad)")
        return False

    def raise_arms_travel(self) -> None:
        """Bring both arms to the scene ready pose for driving. With the arm
        hanging low, +x (FWD) base motion is blocked."""
        ready = [0.0, -1.5, 0.0, -2.2, 0.0, 1.5, 0.785]
        for side in ("right", "left"):
            pub = self.create_publisher(JointState, f"/isaac/{side}_joint_commands", 10)
            msg = JointState()
            msg.name = [f"{side}_fr3v2_joint{i + 1}" for i in range(7)]
            msg.position = list(ready)
            for _ in range(40):
                pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.05)

    # -- episode -----------------------------------------------------------
    def run(self, park_distance: float, skip_drive: bool) -> bool:
        if not self.spin_until(
            lambda: self.joints is not None and self.ee_world is not None
            and self.base_xyy is not None and bool(self.objects),
            30.0, "initial topics"):
            return False

        self.raise_arms_travel()
        t_wb = self.calibrate_arm_base()
        self._t_bw = np.linalg.inv(t_wb)

        pad_w = self.objects["thermalpad"][:3]
        tgt_w = self.objects["board_target"][:3]
        self.get_logger().info(f"GT pad={np.round(pad_w,3).tolist()} target={np.round(tgt_w,3).tolist()}")

        if not skip_drive:
            # The south-side park (1.75, 1.00, yaw +pi/2) is the only single
            # spot where BOTH the pad (0.58 m) and the target (0.72 m) are
            # IK-reachable for the FR3. Route stays south of the table edge
            # (y=1.60); the straight east approach collides.
            route = [
                (self.base_xyy[0], 0.90, None),     # slide south first
                (1.75, 0.90, None),                 # cruise west along the lane
                (1.75, 1.15, math.pi / 2),          # final park, face the table
            ]
            for gx, gy, gyaw in route:
                if gyaw is None:
                    self.get_logger().info(f"waypoint: ({gx:.2f}, {gy:.2f})")
                    if not self.drive_base((gx, gy, self.base_xyy[2]), tol_xy=0.12, tol_yaw=999.0):
                        return False
                else:
                    self.get_logger().info(f"park: ({gx:.2f}, {gy:.2f}, yaw {gyaw:.2f})")
                    if not self.drive_base((gx, gy, gyaw)):
                        return False
            t_wb = self.calibrate_arm_base()  # re-verify after the drive
            self._t_bw = np.linalg.inv(t_wb)

        def world_goal(world_xyz, z) -> np.ndarray:
            w = np.eye(4)
            w[:3, 3] = [world_xyz[0], world_xyz[1], z]
            # top-down grasp frame (tool z down), yaw 0
            w[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
            return w

        grasp_z = pad_w[2] + 0.005
        place_z = 0.78

        # Adaptive clearance: this dev path does not command the spine (the
        # scoring chain does, see mirror_lay.SPINE_TARGET); the arm-base height
        # is scene-dependent (0.42..0.69) and FR3 reach is 0.855 m, so clamp
        # the clearance goal within `reach` of the arm base.
        ab = np.linalg.inv(self._t_bw)[:3, 3] if self._t_bw is not None else None
        reach = 0.76

        def clear_goal(world_xyz) -> np.ndarray:
            z = world_xyz[2] + 0.22
            if ab is not None:
                dz2 = reach * reach - (world_xyz[0] - ab[0]) ** 2 - (world_xyz[1] - ab[1]) ** 2
                z = min(z, ab[2] + math.sqrt(max(dz2, 0.01)))
            return world_goal(world_xyz, z)

        # Phase 1: rough approach to pregrasp using the spawn calibration.
        if not self.move_to(self._t_bw @ clear_goal(pad_w)):
            self.get_logger().error("failed at pregrasp")
            return False
        self.publish_gripper(1.0)

        # Phase 2: LOCAL recalibration at the pregrasp configuration, then
        # ee-pose-servoed moves (cancels the remaining cm-class error).
        self.get_logger().info("local recalibration at pregrasp")
        t_wb = self.calibrate_arm_base()
        self._t_bw = np.linalg.inv(t_wb)
        ab = np.linalg.inv(self._t_bw)[:3, 3]

        steps = [
            ("approach", clear_goal(pad_w), 1.0),
            ("descend", world_goal(pad_w, grasp_z), 1.0),
            ("grasp", None, 0.0),
            ("lift", clear_goal(pad_w), 0.0),
            ("preplace", clear_goal(tgt_w), 0.0),
            ("place", world_goal(tgt_w, place_z), 0.0),
            ("release", None, 1.0),
            ("retreat", clear_goal(tgt_w), 1.0),
        ]
        for name, goal, grip in steps:
            self.get_logger().info(f"step: {name}")
            if goal is not None:
                fine = 3 if name in ("descend", "place") else 0
                if not self.move_to_world(goal, fine_iters=fine, settle_tol=0.25 if name == "place" else 0.15):
                    self.get_logger().error(f"failed at {name}")
                    return False
            self.publish_gripper(grip)
            time.sleep(0.6 if name in ("grasp", "release") else 0.2)
            if name in ("grasp", "lift", "preplace"):
                rclpy.spin_once(self, timeout_sec=0.1)
                pp = self.objects.get("thermalpad", [0, 0, 0])[:3]
                self.get_logger().info(f"  pad GT after {name}: {np.round(pp, 3).tolist()}")

        # outcome check via GT
        self.spin_until(lambda: bool(self.objects), 5.0, "final GT")
        pad_now = self.objects.get("thermalpad", [0, 0, 0])[:3]
        err = math.hypot(pad_now[0] - tgt_w[0], pad_now[1] - tgt_w[1])
        self.get_logger().info(
            f"RESULT pad_now={np.round(pad_now,3).tolist()} target={np.round(tgt_w,3).tolist()} xy_err={err:.3f} m"
        )
        return err < 0.08


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="right")
    ap.add_argument("--park-distance", type=float, default=0.72)
    ap.add_argument("--skip-drive", action="store_true")
    ap.add_argument("--cycles", type=int, default=1)
    args = ap.parse_args()

    rclpy.init()
    node = GTEpisode(args.arm)
    ok = False
    try:
        for i in range(args.cycles):
            node.get_logger().info(f"=== GT episode cycle {i + 1}/{args.cycles} ===")
            ok = node.run(args.park_distance, args.skip_drive)
            if not ok:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.pedal("NONE")
        node.destroy_node()
        rclpy.shutdown()
    print(f"GT_EPISODE_RESULT={'SUCCESS' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
