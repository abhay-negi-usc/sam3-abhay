#!/usr/bin/env python3
"""ROS 2 node: subscribe to a camera image, run SAM3 cable-neck detection, and
publish the cable-connector neck origin + orientation in IMAGE (pixel) coordinates.

Published:
  ~/necks   (geometry_msgs/PoseArray) - one pose per detected connector neck.
            position.x = u (px), position.y = v (px), position.z = 0.
            orientation = yaw quaternion about +z, yaw = atan2(dy, dx) in the pixel
            frame (u right, v down); recover direction as (cos yaw, sin yaw).
            header is copied from the input image (stamp + frame_id) so downstream
            nodes can time-sync with the camera pose.
  ~/debug_image (sensor_msgs/Image, bgr8) - annotated overlay (if publish_debug).

Parameters:
  image_topic (str)         input sensor_msgs/Image topic
  cable_prompt (str)        SAM3 text prompt for cable      (default "cable")
  connector_prompt (str)    SAM3 text prompt for connector  (default "connector")
  threshold (float)         SAM3 confidence threshold       (default 0.5)
  mislabel_overlap (float)  drop cable masks overlapping a connector by > this
  publish_debug (bool)      publish the annotated overlay   (default True)

Run in an environment where BOTH `sam3` (torch/CUDA) and `rclpy` import, e.g. a
python 3.10 conda env with `source /opt/ros/humble/setup.bash`.
"""
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cable_neck_core import NeckDetector, render_overlay  # noqa: E402


def imgmsg_to_rgb(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image to an HxWx3 uint8 RGB array (no cv_bridge)."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()
    h, w, step = msg.height, msg.width, msg.step
    if enc in ("rgb8", "bgr8"):
        rows = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return rows[:, :, ::-1] if enc == "bgr8" else rows
    if enc in ("mono8", "8uc1"):
        gray = buf.reshape(h, step)[:, :w]
        return np.repeat(gray[:, :, None], 3, axis=2)
    if enc in ("rgba8", "bgra8"):
        rows = buf.reshape(h, step)[:, : w * 4].reshape(h, w, 4)[:, :, :3]
        return rows[:, :, ::-1] if enc == "bgra8" else rows
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def rgb_to_imgmsg(rgb_or_bgr: np.ndarray, header, encoding="bgr8") -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = rgb_or_bgr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = rgb_or_bgr.shape[1] * 3
    msg.data = np.ascontiguousarray(rgb_or_bgr).tobytes()
    return msg


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))  # x, y, z, w


class CableNeckNode(Node):
    def __init__(self):
        super().__init__("cable_neck_detector")
        p = self.declare_parameter
        self.image_topic = p("image_topic", "/camera/color/image_raw").value
        cable_prompt = p("cable_prompt", "cable").value
        connector_prompt = p("connector_prompt", "connector").value
        threshold = p("threshold", 0.5).value
        # Separate, usually LOWER bar for the connector. It is the harder class -- SAM3 tends to label
        # the whole assembly "cable", leaving no connector mask, and compute_necks iterates over
        # CONNECTORS, so zero connector masks => zero necks regardless of the cable. -1 = use `threshold`.
        connector_threshold = p("connector_threshold", -1.0).value
        connector_threshold = None if connector_threshold < 0 else connector_threshold
        mislabel_overlap = p("mislabel_overlap", 0.6).value
        self.publish_debug = p("publish_debug", True).value

        # ADAPTIVE THRESHOLD (recommended). Rather than committing to one threshold, run SAM3 once at
        # `confidence_floor` and then search the threshold PAIR per image, keeping the MOST CONFIDENT
        # masks that still yield a valid neck -- i.e. a cable and connector that actually TOUCH. This
        # is the direct answer to SAM3 flip-flopping between labelling the connector "cable" and vice
        # versa: whichever class comes up empty at a fixed threshold kills the neck, because a neck IS
        # the cable/connector contact. Costs no extra inference -- the threshold is only a filter on
        # per-mask scores, so the forward pass is identical. See NeckDetector.detect_adaptive.
        self.adaptive = bool(p("adaptive", True).value)
        self.confidence_floor = float(p("confidence_floor", 0.05).value)

        self.get_logger().info("Loading SAM3 model (first run downloads the checkpoint)...")
        self.detector = NeckDetector(cable_prompt=cable_prompt,
                                     connector_prompt=connector_prompt,
                                     threshold=threshold,
                                     connector_threshold=connector_threshold,
                                     mislabel_overlap=mislabel_overlap)
        self.get_logger().info(
            f"Model ready on {self.detector.device}. "
            f"prompts: cable='{cable_prompt}' (thr {self.detector.cable_threshold:.2f}), "
            f"connector='{connector_prompt}' (thr {self.detector.connector_threshold:.2f}).")

        # keep only the latest frame; SAM3 inference is ~1-2 s/frame
        sub_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub_necks = self.create_publisher(PoseArray, "~/necks", 10)
        self.pub_debug = (self.create_publisher(Image, "~/debug_image", 1)
                          if self.publish_debug else None)
        self.sub = self.create_subscription(Image, self.image_topic,
                                            self.on_image, sub_qos)
        self._busy = False
        self.get_logger().info(f"Subscribed to '{self.image_topic}'. Waiting for images.")

    def on_image(self, msg: Image):
        if self._busy:                     # drop frames while an inference is running
            return
        self._busy = True
        try:
            rgb = imgmsg_to_rgb(msg)
            res = (self.detector.detect_adaptive(PILImage.fromarray(rgb),
                                                 floor=self.confidence_floor)
                   if self.adaptive else self.detector.detect(PILImage.fromarray(rgb)))
            necks = res["necks"]

            pa = PoseArray()
            pa.header = msg.header
            for n in necks:
                u, v = n["neck"]
                dx, dy = n["direction"]
                yaw = math.atan2(dy, dx)            # pixel-frame direction angle
                qx, qy, qz, qw = yaw_to_quat(yaw)
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = float(u), float(v), 0.0
                pose.orientation.x, pose.orientation.y = qx, qy
                pose.orientation.z, pose.orientation.w = qz, qw
                pa.poses.append(pose)
            self.pub_necks.publish(pa)

            if self.pub_debug is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                vis = render_overlay(bgr, res["cleaned_cables"], res["conn_masks"], necks)
                self.pub_debug.publish(rgb_to_imgmsg(vis, msg.header, "bgr8"))

            # Report the RAW per-prompt mask counts, not just the necks. Without these, a total
            # failure is indistinguishable from a near miss: `connectors=0` means the connector prompt
            # found nothing, so compute_necks (which loops over connectors) can never emit a neck --
            # lower `connector_threshold` or reword `connector_prompt`. `dropped` counts cable masks
            # thrown away as mislabelled connectors (mislabel_overlap).
            # In adaptive mode, report the thresholds the SEARCH actually chose for this frame -- that
            # is the diagnostic. `thr=-` means no admissible threshold pair produced a cable/connector
            # contact, i.e. lowering `confidence_floor` (or rewording a prompt) is what's needed.
            if self.adaptive:
                if res.get("eff_conf") is not None:
                    thr = (f"thr: cable={res['thr_cable']:.2f} conn={res['thr_conn']:.2f} "
                           f"(eff {res['eff_conf']:.2f}, {res.get('combos_tried', 0)} combos)")
                else:
                    thr = (f"thr=- (no admissible pair gave a neck; {res.get('combos_tried', 0)} "
                           f"combos tried above floor {self.confidence_floor:.2f})")
            else:
                thr = "thr=fixed"
            self.get_logger().info(
                f"necks={len(necks)} | raw: cables={res['cables_raw']} "
                f"connectors={res['connectors_raw']} dropped={res.get('n_dropped', 0)} | {thr} "
                + " ".join(f"[C{n['connector']} @({n['neck'][0]:.0f},{n['neck'][1]:.0f}) "
                           f"{n['angle_deg']:+.0f}deg]" for n in necks))
        except Exception as e:  # noqa: BLE001 - keep the node alive on bad frames
            import traceback
            self.get_logger().error(
                f"detection failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = CableNeckNode()
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
