#!/usr/bin/env python3
"""ROS 2 node: subscribe to a camera image, run SAM3 cable-connector JUNCTION detection
(the diameter-profiling method), and publish the junction origin + orientation in IMAGE
(pixel) coordinates.

WHICH METHOD IS THIS? -- read the term.
  This node uses the JUNCTION method (cable_neck_diameter.py / JunctionDetector): it unions
  the "cable" and "connector" masks into one assembly, TRACES the cable's length, and puts
  the junction where the CONSTANT-diameter cable run ends. Everything here says "junction".

  Its sibling is cable_neck_ros_node.py, which uses the NECK method (cable_neck_core.py /
  NeckDetector / compute_necks): it pairs a connector mask with the cable pixels touching
  it and takes the contact centroid. Everything there says "neck".

  Both name the SAME physical point (cable meets connector) but locate it differently. Run
  ONE node at a time; the topic name (~/junctions vs ~/necks) and the log word ("junction"
  vs "neck") tell you unambiguously which method is producing the output. To swap methods,
  swap nodes -- nothing downstream has to guess.

Published:
  ~/junctions   (geometry_msgs/PoseArray) - one pose per detected junction (0 or 1; the
                diameter method reports a single junction per frame).
                position.x = u (px), position.y = v (px), position.z = 0.
                orientation = yaw quaternion about +z, yaw = atan2(dy, dx) in the pixel
                frame (u right, v down); recover direction as (cos yaw, sin yaw). The
                direction is the CONNECTOR's principal axis (same rule as the neck node),
                signed to point from the junction INTO the connector.
                header is copied from the input image (stamp + frame_id) so downstream
                nodes can time-sync with the camera pose.
  ~/debug_image (sensor_msgs/Image, bgr8) - annotated overlay (if publish_debug):
                traced centreline, both cable-outline rails, the junction dot + diameter
                chord, and the connector-axis arrow.

Parameters:
  image_topic (str)         input sensor_msgs/Image topic  (default /camera/color/image_raw)
  cable_prompt (str)        SAM3 text prompt for cable      (default "cable")
  connector_prompt (str)    SAM3 text prompt for connector  (default "connector")
  threshold (float)         SAM3 confidence threshold, cable prompt   (default 0.5)
  connector_threshold (float) connector prompt threshold; -1 = use `threshold` (default 0.3)
  work_dim (int)            downscale max-dimension for the geometry  (default 1024)
  min_contrast (float)      drop a junction whose connector/cable diameter ratio is below
                            this (a weak/absent diameter step is not a real junction; 0 =
                            publish anything)                          (default 1.3)
  publish_debug (bool)      publish the annotated overlay              (default True)

NOTE: the junction method is single-junction per frame and has NO adaptive-threshold mode
(unlike the neck node): it is classification-free, so it never gets "starved" by SAM3
labelling the whole assembly one class. A lower `connector_threshold` still helps SAM3
include the connector body in the assembly, which sharpens the diameter step.

Run in an environment where BOTH `sam3` (torch/CUDA) and `rclpy` import (see
ROS_NODES_README.md for the Jazzy / Python 3.12 setup).
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
from cable_neck_diameter import (  # noqa: E402
    JunctionDetector, largest_component, render_overlay)


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


class CableJunctionNode(Node):
    def __init__(self):
        super().__init__("cable_junction_detector")
        p = self.declare_parameter
        self.image_topic = p("image_topic", "/camera/color/image_raw").value
        cable_prompt = p("cable_prompt", "cable").value
        connector_prompt = p("connector_prompt", "connector").value
        threshold = p("threshold", 0.5).value
        # Lower default bar for the connector prompt so SAM3 includes the connector body in
        # the assembly -- that is what makes the diameter STEP sharp. Unlike the neck node,
        # an empty connector mask is NOT fatal here (the method is classification-free and
        # works on the unioned shape), it only softens the step. -1 = use `threshold`.
        connector_threshold = p("connector_threshold", 0.3).value
        connector_threshold = None if connector_threshold < 0 else connector_threshold
        self.work_dim = int(p("work_dim", 1024).value)
        # A junction IS a diameter change; contrast = connector/cable diameter ratio. Reject
        # junctions below this (a flat profile means no real cable->connector step this frame).
        self.min_contrast = float(p("min_contrast", 1.3).value)
        self.publish_debug = p("publish_debug", True).value

        self.get_logger().info("Loading SAM3 model (first run downloads the checkpoint)...")
        self.detector = JunctionDetector(cable_prompt=cable_prompt,
                                         connector_prompt=connector_prompt,
                                         threshold=threshold,
                                         connector_threshold=connector_threshold,
                                         work_dim=self.work_dim)
        self.get_logger().info(
            f"Model ready on {self.detector.device}. JUNCTION method (diameter profiling). "
            f"prompts: cable='{cable_prompt}' (thr {self.detector.cable_threshold:.2f}), "
            f"connector='{connector_prompt}' (thr {self.detector.connector_threshold:.2f}). "
            f"min_contrast={self.min_contrast:.2f}")

        # keep only the latest frame; SAM3 inference is ~1-2 s/frame
        sub_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub_junctions = self.create_publisher(PoseArray, "~/junctions", 10)
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
            out = self.detector.detect(PILImage.fromarray(rgb))
            # Keep only junctions with a real diameter step (contrast >= min_contrast).
            junctions = [j for j in out["junctions"]
                         if j.get("contrast", 0.0) >= self.min_contrast]

            pa = PoseArray()
            pa.header = msg.header
            for j in junctions:
                u, v = j["junction"]
                dx, dy = j["direction"]
                yaw = math.atan2(dy, dx)            # pixel-frame direction angle
                qx, qy, qz, qw = yaw_to_quat(yaw)
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = float(u), float(v), 0.0
                pose.orientation.x, pose.orientation.y = qx, qy
                pose.orientation.z, pose.orientation.w = qz, qw
                pa.poses.append(pose)
            self.pub_junctions.publish(pa)

            if self.pub_debug is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                # draw the accepted junction (or None -> a NO JUNCTION overlay)
                res = junctions[0] if junctions else None
                vis = render_overlay(bgr, largest_component(out["assembly"]), res)
                self.pub_debug.publish(rgb_to_imgmsg(vis, msg.header, "bgr8"))

            # Report RAW per-prompt mask counts alongside the result: connectors=0 does NOT
            # kill the junction here (classification-free), but it does flatten the step, so
            # a low-contrast/rejected frame with connectors=0 means "reword connector_prompt
            # or lower connector_threshold". `contrast` is the diameter step this frame.
            det = out["result"]
            if not junctions:
                why = (f"contrast {det['contrast']:.1f} < {self.min_contrast:.1f}"
                       if det is not None else "no traceable assembly")
                self.get_logger().info(
                    f"junctions=0 ({why}) | raw: cables={out['cables_raw']} "
                    f"connectors={out['connectors_raw']}")
            else:
                self.get_logger().info(
                    f"junctions={len(junctions)} | raw: cables={out['cables_raw']} "
                    f"connectors={out['connectors_raw']} | "
                    + " ".join(f"[@({j['junction'][0]:.0f},{j['junction'][1]:.0f}) "
                               f"{j['angle_deg']:+.0f}deg x{j['contrast']:.1f} "
                               f"dia {j['cable_diameter_px']:.0f}->"
                               f"{j['connector_diameter_px']:.0f}px]" for j in junctions))
        except Exception as e:  # noqa: BLE001 - keep the node alive on bad frames
            import traceback
            self.get_logger().error(
                f"detection failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = CableJunctionNode()
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
