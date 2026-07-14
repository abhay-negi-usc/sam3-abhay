#!/usr/bin/env python3
"""ROS 2 node: publish the CONNECTOR TIP (image coords) from the COMBINED cable+connector segmentation.

WHY THIS EXISTS
---------------
SAM3 frequently labels the whole assembly "cable" and returns NO connector mask at all. That starves
cable_neck_ros_node completely: compute_necks() iterates over CONNECTOR masks, so zero connector masks
=> zero necks, no matter how good the cable mask is.

This node refuses to depend on that classification. It UNIONS both prompts into a single
'cable_and_connector' object and works purely on the SHAPE:
  * find the object's two ends (its geodesic diameter -- geodesic, not Euclidean, because a cable is a
    CURVE and can double back on itself),
  * pick the CONNECTOR end: the end nearest the connector mask if SAM3 produced one, otherwise the
    THICKER end (a connector is fatter than the cable it terminates),
  * walk BACK along the curve by a predefined length (`curve_px`) and take tip - back as the axis. A
    fixed arc length is far more stable than the local tangent at the very tip, which is dominated by
    mask noise.
See compute_tip() in cable_neck_core.py.

TOPICS
------
  ~/tips         geometry_msgs/PoseArray -- position.x/y = tip pixel (u, v); orientation = yaw about
                 +z with yaw = atan2(dy, dx) in the PIXEL frame. This is the SAME convention as
                 cable_neck_ros_node's ~/necks, deliberately: connector_pose_node can fuse these
                 UNCHANGED. Its RANSAC/consensus outlier rejection then applies to tips for free.
  ~/debug_image  sensor_msgs/Image (bgr8) -- combined mask + tip + walked-back segment + axis arrow.

Fuse to a 3D tip pose with the existing estimator (no changes needed to it):

    python connector_pose_node.py --ros-args \
      -p necks_topic:=/cable_tip_detector/tips \
      -p connector_frame:=connector_tip \
      -p world_frame:=base_link \
      -p camera_info_topic:=/camera/camera1/color/camera_info \
      -p tf_cache_s:=60.0
"""
import math

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from cable_neck_core import NeckDetector, render_tip_overlay
from cable_neck_ros_node import imgmsg_to_rgb, rgb_to_imgmsg, yaw_to_quat


class CableTipNode(Node):
    def __init__(self):
        super().__init__("cable_tip_detector")
        p = self.declare_parameter
        self.image_topic = p("image_topic", "/camera/color/image_raw").value
        cable_prompt = p("cable_prompt", "cable").value
        connector_prompt = p("connector_prompt", "connector").value
        threshold = p("threshold", 0.5).value
        # Separate, usually LOWER bar for the connector -- it is the harder class. Note this node does
        # not NEED a connector mask (that is the whole point), but when one exists it is the best
        # evidence for WHICH end is the connector, so it is still worth detecting when possible.
        connector_threshold = p("connector_threshold", -1.0).value
        connector_threshold = None if connector_threshold < 0 else connector_threshold
        # The predefined curve length: how far back along the cable, in PIXELS, to walk from the tip
        # when measuring the axis. Longer = smoother/more stable direction but assumes the connector
        # is straight over that span; shorter = follows a curved cable more closely but is noisier.
        self.curve_px = int(p("curve_px", 40).value)
        self.publish_debug = p("publish_debug", True).value

        self.get_logger().info("Loading SAM3 model (first run downloads the checkpoint)...")
        self.detector = NeckDetector(cable_prompt=cable_prompt,
                                     connector_prompt=connector_prompt,
                                     threshold=threshold,
                                     connector_threshold=connector_threshold)
        self.get_logger().info(
            f"Model ready on {self.detector.device}. TIP mode (classification-free: cable+connector "
            f"masks are UNIONED). prompts: cable='{cable_prompt}' "
            f"(thr {self.detector.cable_threshold:.2f}), connector='{connector_prompt}' "
            f"(thr {self.detector.connector_threshold:.2f}), curve_px={self.curve_px}.")

        sub_qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1,
                             reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub_tips = self.create_publisher(PoseArray, "~/tips", 10)
        self.pub_debug = (self.create_publisher(Image, "~/debug_image", 1)
                          if self.publish_debug else None)
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, sub_qos)
        self._busy = False
        self.get_logger().info(f"Subscribed to '{self.image_topic}'. Waiting for images.")

    def on_image(self, msg: Image):
        if self._busy:                     # drop frames while an inference is running
            return
        self._busy = True
        try:
            rgb = imgmsg_to_rgb(msg)
            res = self.detector.detect_tip(PILImage.fromarray(rgb), curve_px=self.curve_px)

            # Publish even when empty: downstream (connector_pose_node) ignores empty arrays, and a
            # steady tick makes "detector alive but seeing nothing" distinguishable from "detector dead".
            pa = PoseArray()
            pa.header = msg.header
            if res["tip"] is not None:
                u, v = res["tip"]
                dx, dy = res["direction"]
                qx, qy, qz, qw = yaw_to_quat(math.atan2(dy, dx))     # pixel-frame direction
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = float(u), float(v), 0.0
                pose.orientation.x, pose.orientation.y = qx, qy
                pose.orientation.z, pose.orientation.w = qz, qw
                pa.poses.append(pose)
            self.pub_tips.publish(pa)

            if self.pub_debug is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.pub_debug.publish(
                    rgb_to_imgmsg(render_tip_overlay(bgr, res), msg.header, "bgr8"))

            # Log the RAW per-prompt counts too: `connectors=0` is expected here (and harmless -- that
            # is exactly the failure this node is built to survive), but it tells you WHICH rule chose
            # the end -- the connector mask, or the fall-back thickness test.
            if res["tip"] is not None:
                src = "conn-mask" if res["used_connector"] else "thicker-end"
                self.get_logger().info(
                    f"tip=1 | raw: cables={res['cables_raw']} connectors={res['connectors_raw']} | "
                    f"@({res['tip'][0]:.0f},{res['tip'][1]:.0f}) {res['angle_deg']:+.0f}deg "
                    f"thick={res['thickness_px']:.0f}px [{src}]")
            else:
                self.get_logger().info(
                    f"tip=0 | raw: cables={res['cables_raw']} connectors={res['connectors_raw']} | "
                    "no cable/connector mask big enough to have ends")
        except Exception as e:  # noqa: BLE001 - keep the node alive on bad frames
            import traceback
            self.get_logger().error(
                f"detection failed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = CableTipNode()
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
