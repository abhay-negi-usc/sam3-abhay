#!/usr/bin/env python3
"""ROS 2 node: fuse a history of cable-connector neck detections (in image
coordinates) with the camera pose at each frame to estimate the 3D pose of the
connector.

Connector pose convention (output):
  * origin = the neck 3D point where the cable and connector interface.
  * x axis = along the connector AXIS (the direction it points, away from the cable).
  * z axis = `up_axis` (a world-frame direction, default +Z), re-orthogonalized perpendicular to x.
    This is the "the connector's z is coincident with the world z" assumption -- it pins down the
    roll about the axis, which the measurement itself cannot determine.
  * y axis = z x x  (completes a right-handed frame; horizontal).

  EVERY axis is meaningful, so the published TF can be read directly in RViz / tf2_echo. (This node
  used to put the axis in Z with ARBITRARY x/y, which looked wrong on inspection and forced every
  consumer to rebuild the frame themselves.)

Method (multi-view fusion; single connector):
  For every neck message we look up the camera pose from TF at the image stamp and
  store (neck pixel p, image-direction d, camera center C, camera rotation R_wc).
  With >= min_views and enough parallax we compute:
    * origin P  : least-squares triangulation of the 3D point closest to all the
                  back-projected neck rays. Metric because the camera poses are metric.
    * axis a    : the connector axis is a 3D line; each view's image line (through p
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
from rclpy.time import Time
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


def frame_from_axis(a, up):
    """Connector frame as a 3x3 rotation (columns = x, y, z), in world coordinates.

    x = the connector AXIS `a` (the one rotational DOF the multi-view fusion actually measures),
    z = `up` re-orthogonalized perpendicular to x  (the assumption that pins down roll about the
        axis, which the measurement cannot determine),
    y = z x x.

    Unlike an arbitrary completion of the axis, every column here means something, so the resulting
    TF is directly interpretable.
    """
    x = np.asarray(a, dtype=float)
    x = x / (np.linalg.norm(x) + 1e-12)
    u = np.asarray(up, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)
    y = np.cross(u, x)
    if np.linalg.norm(y) < 1e-6:          # axis is ~parallel to `up`: fall back to another reference
        alt = np.array([1.0, 0.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y = np.cross(alt, x)
    y /= (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z /= (np.linalg.norm(z) + 1e-12)
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
        # The published connector frame's z is aligned to this world-frame direction (re-orthogonalized
        # perpendicular to the measured axis). It fixes the roll about the axis, which the fusion
        # cannot measure. Default +Z = the "connector z is coincident with the base z" assumption.
        self.up_axis = [float(v) for v in p("up_axis", [0.0, 0.0, 1.0]).value]
        self.tf_timeout = float(p("tf_timeout", 0.2).value)
        # TF history to keep. Necks are stamped with the IMAGE time, and SAM3 inference can take
        # seconds, so by the time a neck arrives its stamp is already seconds old. The tf2 default
        # cache is only 10 s, which makes that lookup fail with "extrapolation into the past".
        # Keep well more history than the worst-case detector latency.
        self.tf_cache_s = float(p("tf_cache_s", 60.0).value)

        self.K = None
        self.Kinv = None
        self.history = deque(maxlen=self.max_history)   # dicts: p,d,C,R_wc
        self.last_pixel = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=self.tf_cache_s))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_caminfo, 1)
        self.create_subscription(PoseArray, self.necks_topic, self.on_necks, 10)
        self.pub_pose = self.create_publisher(PoseStamped, "~/connector_pose", 10)
        self.get_logger().info(
            f"Waiting for CameraInfo on '{self.camera_info_topic}' and necks on "
            f"'{self.necks_topic}'. world='{self.world_frame}', neck_diameter="
            f"{self.neck_diameter} m, tf_cache={self.tf_cache_s}s, tf_timeout={self.tf_timeout}s.")

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
            # Report HOW STALE the neck is: it carries the IMAGE stamp, so it lags by the detector's
            # inference time. If age > tf_cache the lookup falls off the back of the buffer.
            age = (self.get_clock().now() - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
            self.get_logger().warn(
                f"TF {self.world_frame}<-{src} unavailable (neck stamp is {age:.1f}s old; "
                f"tf_cache={self.tf_cache_s}s): {e}", throttle_duration_sec=2.0)
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

        # --- connector axis: null space of the stacked image-line-plane normals ---
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

        # Build the CONNECTOR frame: x = the measured axis, z = up_axis, y = z x x. Every axis is
        # meaningful, so consumers (and RViz) can use this TF directly -- no rebuilding needed.
        R = frame_from_axis(a, self.up_axis)
        qx, qy, qz, qw = rotmat_to_quat(R)

        # Stamp the OUTPUT with NOW -- not with the image time.
        # The connector is a STATIC object pose in world_frame: this is our current best estimate of
        # where the cable IS, not a claim about where it was. Stamping it with the (seconds-old) image
        # time forces every consumer into a time-travel lookup and trips "Lookup would require
        # extrapolation into the past" as soon as the detector is slow -- it makes the frame unusable
        # in tf2_echo, RViz and the demo alike. The image stamp is still used INTERNALLY (above) to
        # fetch the camera pose at CAPTURE time, which is the part that must be time-accurate.
        now = self.get_clock().now().to_msg()

        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, P)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        self.pub_pose.publish(ps)

        tfm = TransformStamped()
        tfm.header.stamp = now
        tfm.header.frame_id = self.world_frame
        tfm.child_frame_id = self.connector_frame
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = map(float, P)
        tfm.transform.rotation.x, tfm.transform.rotation.y = qx, qy
        tfm.transform.rotation.z, tfm.transform.rotation.w = qz, qw
        self.tf_broadcaster.sendTransform(tfm)

        self.get_logger().info(
            f"connector pose | P=({P[0]:.3f},{P[1]:.3f},{P[2]:.3f}) "
            f"axis=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}) views={len(self.history)} "
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
