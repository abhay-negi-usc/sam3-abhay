#!/usr/bin/env python3
"""ROS 2 node: fuse a history of cable-connector neck detections (in image
coordinates) with the camera pose at each frame to estimate the 3D pose of the
connector.

Connector pose convention (output):
  * origin = the neck 3D point where the cable and connector interface.
  * z axis = along the connector axis (direction it points, away from the cable).
  * x, y axes = arbitrary (a stable frame completed from z).

Method (multi-view fusion; single connector):
  For every neck message we look up the camera pose from TF at the image stamp and
  store (neck pixel p, image-direction d, camera center C, camera rotation R_wc).
  With >= min_views and enough parallax we compute:
    * origin P  : least-squares triangulation of the 3D point closest to all the
                  back-projected neck rays. Metric because the camera poses are metric.
    * z axis a  : the connector axis is a 3D line; each view's image line (through p
                  along d) back-projects to a plane with world normal n_k, and the
                  axis is orthogonal to every n_k. a = smallest singular vector of
                  the stacked n_k (SVD). Sign resolved to match the observed 2D arrow.

Neck diameter (config, meters): the physical connector diameter at the neck. With
metric camera poses the position is already metric, so the diameter is used as a
consistency check (expected neck pixel radius per view is logged) and as the length
for the axis/TF visualization. It is exposed in config as requested and can serve as
a metric scale prior if you later fuse with weak/unscaled camera poses.

Inputs:
  ~necks (geometry_msgs/PoseArray, from cable_neck_ros_node): position.x/y = neck
    pixel (u, v); orientation = yaw about +z with yaw = atan2(dy, dx) in the pixel
    frame. header.stamp/frame_id identify the camera frame + time.
  <camera_info_topic> (sensor_msgs/CameraInfo): intrinsics K.
  TF: world_frame <- image header.frame_id (camera optical frame) at each stamp.

Outputs:
  ~connector_pose (geometry_msgs/PoseStamped) in world_frame.
  TF broadcast world_frame -> connector_frame.

Depends only on numpy + rclpy + tf2_ros (no torch/SAM3), so it can run in a plain
rclpy environment separate from the detector node.
"""
import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


def quat_to_rotmat(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return x, y, z, w


def frame_from_z(a):
    """Return a 3x3 rotation whose 3rd column is unit vector a (z), x/y arbitrary."""
    z = a / (np.linalg.norm(a) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


class ConnectorPoseNode(Node):
    def __init__(self):
        super().__init__("connector_pose_estimator")
        p = self.declare_parameter
        self.neck_diameter = float(p("neck_diameter", 0.02).value)   # meters
        self.world_frame = p("world_frame", "map").value
        self.connector_frame = p("connector_frame", "connector").value
        self.camera_frame_override = p("camera_frame", "").value      # if image lacks frame_id
        self.necks_topic = p("necks_topic", "/cable_neck_detector/necks").value
        self.camera_info_topic = p("camera_info_topic", "/camera/color/camera_info").value
        self.max_history = int(p("max_history", 60).value)
        self.min_views = int(p("min_views", 2).value)
        self.min_parallax_deg = float(p("min_parallax_deg", 3.0).value)
        self.tf_timeout = float(p("tf_timeout", 0.2).value)

        self.K = None
        self.Kinv = None
        self.history = deque(maxlen=self.max_history)   # dicts: p,d,C,R_wc
        self.last_pixel = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_caminfo, 1)
        self.create_subscription(PoseArray, self.necks_topic, self.on_necks, 10)
        self.pub_pose = self.create_publisher(PoseStamped, "~/connector_pose", 10)
        self.get_logger().info(
            f"Waiting for CameraInfo on '{self.camera_info_topic}' and necks on "
            f"'{self.necks_topic}'. world='{self.world_frame}', neck_diameter="
            f"{self.neck_diameter} m.")

    # ---------------------------------------------------------------- callbacks
    def on_caminfo(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=float).reshape(3, 3)
            self.Kinv = np.linalg.inv(self.K)
            self.get_logger().info(f"Got intrinsics: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} "
                                   f"cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}")

    def on_necks(self, msg: PoseArray):
        if self.K is None:
            self.get_logger().warn("No CameraInfo yet; skipping neck message.", throttle_duration_sec=5.0)
            return
        if not msg.poses:
            return

        # single-connector association: nearest pixel to the last accepted one
        def pixel(pose):
            return np.array([pose.position.x, pose.position.y])
        if self.last_pixel is None:
            pose = msg.poses[0]
        else:
            pose = min(msg.poses, key=lambda ps: np.linalg.norm(pixel(ps) - self.last_pixel))
        u, v = pose.position.x, pose.position.y
        yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)  # pure-z quat
        d = np.array([math.cos(yaw), math.sin(yaw)])                    # pixel-frame dir

        src = msg.header.frame_id or self.camera_frame_override
        if not src:
            self.get_logger().warn("Neck message has no frame_id and no camera_frame param set.")
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, src, msg.header.stamp,
                timeout=Duration(seconds=self.tf_timeout))
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF {self.world_frame}<-{src} unavailable: {e}",
                                   throttle_duration_sec=2.0)
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        C = np.array([t.x, t.y, t.z])
        R_wc = quat_to_rotmat(q.x, q.y, q.z, q.w)

        self.last_pixel = np.array([u, v])
        self.history.append(dict(p=np.array([u, v]), d=d, C=C, R_wc=R_wc))
        self.estimate_and_publish(msg.header.stamp)

    # ---------------------------------------------------------------- estimation
    def estimate_and_publish(self, stamp):
        if len(self.history) < self.min_views:
            self.get_logger().info(f"accumulating views: {len(self.history)}/{self.min_views}",
                                   throttle_duration_sec=2.0)
            return

        rays, centers, normals, samples = [], [], [], []
        for h in self.history:
            u, v = h["p"]
            r_cam = self.Kinv @ np.array([u, v, 1.0])
            g = h["R_wc"] @ r_cam
            g /= (np.linalg.norm(g) + 1e-12)
            rays.append(g); centers.append(h["C"])
            # back-projected image-line plane normal for the axis constraint
            dx, dy = h["d"]
            l = np.array([dy, -dx, dx * v - dy * u])           # image line through p along d
            m_cam = self.K.T @ l
            n = h["R_wc"] @ m_cam
            n /= (np.linalg.norm(n) + 1e-12)
            normals.append(n)
            samples.append(h)

        # parallax gate: need spread in ray directions to triangulate a point
        G = np.array(rays)
        max_ang = 0.0
        for i in range(len(G)):
            for j in range(i + 1, len(G)):
                c = np.clip(G[i] @ G[j], -1, 1)
                max_ang = max(max_ang, math.degrees(math.acos(c)))
        if max_ang < self.min_parallax_deg:
            self.get_logger().info(f"insufficient parallax ({max_ang:.1f} deg); moving the camera "
                                   f"more will improve the estimate.", throttle_duration_sec=2.0)
            return

        # --- origin: least-squares closest point to all rays ---
        A = np.zeros((3, 3)); b = np.zeros(3)
        for g, C in zip(rays, centers):
            Pperp = np.eye(3) - np.outer(g, g)
            A += Pperp; b += Pperp @ C
        try:
            P = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.get_logger().warn("degenerate triangulation.", throttle_duration_sec=2.0)
            return

        # --- z axis: null space of the stacked image-line-plane normals ---
        N = np.array(normals)
        _, _, Vt = np.linalg.svd(N)
        a = Vt[-1]
        a /= (np.linalg.norm(a) + 1e-12)

        # resolve axis sign to match the observed 2D arrows (majority vote)
        vote = 0.0
        for h in samples:
            Rcw = h["R_wc"].T
            Xc = Rcw @ (P - h["C"])
            Xc2 = Rcw @ (P + 0.01 * a - h["C"])
            if Xc[2] <= 1e-6 or Xc2[2] <= 1e-6:
                continue
            px1 = self.K @ (Xc / Xc[2]); px2 = self.K @ (Xc2 / Xc2[2])
            vote += float((px2 - px1)[:2] @ h["d"])
        if vote < 0:
            a = -a

        # consistency: expected neck pixel radius from the diameter, latest view
        h = samples[-1]
        Xc = h["R_wc"].T @ (P - h["C"])
        depth = float(Xc[2])
        exp_r = (self.K[0, 0] * (self.neck_diameter / 2.0) / depth) if depth > 1e-6 else float("nan")

        R = frame_from_z(a)
        qx, qy, qz, qw = rotmat_to_quat(R)

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, P)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        self.pub_pose.publish(ps)

        tfm = TransformStamped()
        tfm.header.stamp = stamp
        tfm.header.frame_id = self.world_frame
        tfm.child_frame_id = self.connector_frame
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, P)
        tfm.transform.rotation.x, tfm.transform.rotation.y = qx, qy
        tfm.transform.rotation.z, tfm.transform.rotation.w = qz, qw
        self.tf_broadcaster.sendTransform(tfm)

        self.get_logger().info(
            f"connector pose | P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f}) "
            f"z=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) views={len(self.history)} "
            f"parallax={max_ang:.1f}deg depth={depth:.2f}m exp_neck_r={exp_r:.1f}px")


def main():
    rclpy.init()
    node = ConnectorPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
