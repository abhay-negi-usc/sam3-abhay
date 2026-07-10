# Cable-neck ROS 2 nodes (target: Jazzy)

> Deploy target is **ROS 2 Jazzy** (Ubuntu 24.04, Python 3.12). The node code uses
> only rclpy/tf2_ros API that is stable across Humble→Jazzy; it was logic-validated
> locally under Humble (rclpy launch + synthetic fusion), and the geometry was
> validated end-to-end via the offline CLI. On Jazzy, node 1 can run in a **single
> Python 3.12 env** (torch+SAM3+rclpy together) because Jazzy's Python is 3.12 —
> matching the `sam3` env — so no split-env workaround is needed.

Two nodes plus a shared core:

| File | Role |
|------|------|
| `cable_neck_core.py` | SAM3 detection + neck geometry (shared by CLI and node 1). |
| `cable_neck_ros_node.py` | **Node 1** — camera image → neck origin + direction (image coords). |
| `connector_pose_node.py` | **Node 2** — fuse neck history + camera-pose history → 3D connector pose. |
| `cable_neck_orientation.py` | Offline batch CLI (same core), for folders of images. |

## Data flow

```
camera image ──▶ [node 1: cable_neck_detector] ──▶ /cable_neck_detector/necks (PoseArray, pixel coords)
                                                          │
CameraInfo ───────────────────────────────────┐          │
TF world←camera_optical ───────────────────────┼────▶ [node 2: connector_pose_estimator] ──▶ /connector_pose_estimator/connector_pose (PoseStamped)
                                                                                            └▶ TF: world → connector
```

## Node 1 — `cable_neck_ros_node.py`

Subscribes a `sensor_msgs/Image`, runs SAM3, publishes one pose per connector neck.

**Publishes**
- `~/necks` (`geometry_msgs/PoseArray`): `position.x=u`, `position.y=v` (pixels), `position.z=0`;
  `orientation` = yaw quaternion about +z, `yaw = atan2(dy, dx)` in the pixel frame
  (recover direction as `(cos yaw, sin yaw)`). Header copied from the input image.
- `~/debug_image` (`sensor_msgs/Image`, bgr8): annotated overlay (if `publish_debug`).

**Parameters** — `image_topic` (`/camera/color/image_raw`), `cable_prompt` (`cable`),
`connector_prompt` (`connector`), `threshold` (0.5), `mislabel_overlap` (0.6), `publish_debug` (True).

**Environment (Jazzy / Python 3.12)** — needs BOTH `sam3` (torch/CUDA) **and** `rclpy`.
Jazzy's `rclpy` is Python 3.12, which matches SAM3's requirement, so use one env:

Option A — system Jazzy python (simplest on the robot):
```bash
source /opt/ros/jazzy/setup.bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .                 # repo root: installs sam3 + deps (numpy<2)
pip install "setuptools<81"      # pkg_resources for model_builder
```

Option B — conda python 3.12 env (a conda 3.12 env can load Jazzy's rclpy, same as the
3.10/Humble check done here):
```bash
conda create -n sam3_ros -c conda-forge python=3.12 -y && conda activate sam3_ros
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e . && pip install "setuptools<81"
source /opt/ros/jazzy/setup.bash
```

> numpy note: SAM3 pins `numpy<2` (→1.26). Jazzy packages are generally numpy-2-built;
> rclpy runtime is numpy-version tolerant, but if you hit an ABI warning from a ROS
> python package, prefer the conda env (Option B) to isolate numpy 1.26.

Run:
```bash
source /opt/ros/jazzy/setup.bash            # (+ conda activate sam3_ros for Option B)
python scripts/cable_neck_ros_node.py --ros-args \
  -p image_topic:=/camera/color/image_raw -p publish_debug:=true
```

## Node 2 — `connector_pose_node.py`

Fuses a history of neck detections + camera poses into the connector's 3D pose:
origin at the neck (cable↔connector interface), **z axis along the connector axis**, x/y arbitrary.

- **Origin**: least-squares triangulation of the 3D point closest to all back-projected neck rays.
- **z axis**: the connector axis is a 3D line; each view's image line back-projects to a plane
  with world normal `n_k`, and the axis is ⟂ every `n_k` → smallest singular vector (SVD).
  Sign resolved to match the observed 2D arrows.
- **neck_diameter** (config, m): with metric camera poses the position is already metric, so the
  diameter is used as a consistency check (expected neck pixel radius per view is logged) and as a
  metric scale prior. Exposed in config as requested.

**Subscribes** — `necks_topic` (`/cable_neck_detector/necks`), `camera_info_topic`
(`/camera/color/camera_info`); looks up TF `world_frame ← image.header.frame_id` at each stamp.

**Publishes** — `~/connector_pose` (`geometry_msgs/PoseStamped`, in `world_frame`) and a TF
broadcast `world_frame → connector_frame`.

**Parameters** — `neck_diameter` (0.02 m), `world_frame` (`map`), `connector_frame` (`connector`),
`camera_frame` (fallback if image has no frame_id), `necks_topic`, `camera_info_topic`,
`max_history` (60), `min_views` (2), `min_parallax_deg` (3.0), `tf_timeout` (0.2 s).

Node 2 depends only on numpy + rclpy + tf2_ros (no torch), so it runs in a plain rclpy env:
```bash
source /opt/ros/jazzy/setup.bash
python scripts/connector_pose_node.py --ros-args -p neck_diameter:=0.018 -p world_frame:=base_link
```

## Notes
- SAM3 inference is ~1–2 s/frame; node 1 keeps only the latest frame (BEST_EFFORT, depth 1).
- Move the camera between views — node 2 needs parallax (>`min_parallax_deg`) to triangulate.
- Single-connector assumption: node 2 tracks the neck nearest the previous observation.
