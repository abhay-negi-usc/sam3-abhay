# Cable-neck ROS 2 nodes (target: Jazzy)

> Deploy target is **ROS 2 Jazzy** (Ubuntu 24.04, Python 3.12). The node code uses
> only rclpy/tf2_ros API that is stable across Humble→Jazzy; it was logic-validated
> locally under Humble (rclpy launch + synthetic fusion), and the geometry was
> validated end-to-end via the offline CLI. On Jazzy, node 1 can run in a **single
> Python 3.12 env** (torch+SAM3+rclpy together) because Jazzy's Python is 3.12 —
> matching the `sam3` env — so no split-env workaround is needed.

Nodes plus shared cores:

| File | Role |
|------|------|
| `cable_neck_core.py` | SAM3 detection + **neck** geometry (shared by CLI and node 1a). |
| `cable_neck_ros_node.py` | **Node 1a (NECK method)** — camera image → **neck** origin + direction. |
| `cable_neck_diameter.py` | SAM3 seg + **junction** geometry (`JunctionDetector`, shared by CLI and node 1b). |
| `cable_junction_ros_node.py` | **Node 1b (JUNCTION method)** — camera image → **junction** origin + direction. |
| `connector_pose_node.py` | **Node 2** — fuse a history of image detections + camera poses → 3D connector pose. |
| `cable_neck_orientation.py` | Offline batch CLI for the **neck** method (folders of images). |

## Neck vs. junction — two interchangeable detectors for the SAME point

Node 1 comes in two variants. Both find the point where the cable meets the connector and
publish it with an **identical message schema** (a `PoseArray` of pixel `(u, v)` + a yaw
about +z). They differ only in HOW they locate it — and in the WORD they use, so you always
know which one is live:

| | **NECK method** (node 1a) | **JUNCTION method** (node 1b) |
|---|---|---|
| File | `cable_neck_ros_node.py` | `cable_junction_ros_node.py` |
| Node name | `cable_neck_detector` | `cable_junction_detector` |
| Core | `cable_neck_core.py` (`NeckDetector` / `compute_necks`) | `cable_neck_diameter.py` (`JunctionDetector` / `compute_junction`) |
| Topic | `~/necks` | `~/junctions` |
| Log word | `necks=…` | `junctions=…` |
| How | pairs a **connector mask** with the cable pixels touching it; origin = contact centroid; direction = connector PCA axis | unions both masks, **traces the cable's length**, origin = end of the **constant-diameter** run; direction = connector PCA axis (same rule) |
| Poses/frame | 0..N (one per connector) | 0 or 1 (the single most prominent junction) |
| Fails when | SAM3 emits no connector mask → 0 necks (mitigate: `adaptive:=true`, lower `connector_threshold`) | cable/connector strands merge in the mask (coils, parallel runs) → wrong diameter step |
| Confidence | per-frame chosen thresholds (adaptive) | `contrast` = connector/cable diameter ratio (`min_contrast` gate) |

**To swap methods, run the other node** — nothing downstream changes except the topic name.
Node 2 consumes whichever you point its `necks_topic` at:
`-p necks_topic:=/cable_junction_detector/junctions` (junction) or
`-p necks_topic:=/cable_neck_detector/necks` (neck). Do **not** run both node-1 variants at
once unless you want two independent estimates on two topics.

## Data flow

Pick ONE node-1 variant; its output topic feeds node 2 (repoint `necks_topic` accordingly):

```
                ┌─ [node 1a: cable_neck_detector]     ──▶ /cable_neck_detector/necks ──────┐  (PoseArray,
camera image ──▶┤   (NECK method)                                                          │   pixel coords)
                └─ [node 1b: cable_junction_detector] ──▶ /cable_junction_detector/junctions┘
                    (JUNCTION method)                                                       │
CameraInfo ───────────────────────────────────┐                                            │
TF world←camera_optical ───────────────────────┼────▶ [node 2: connector_pose_estimator] ──▶ /connector_pose_estimator/connector_pose (PoseStamped)
                                                        (necks_topic := the chosen topic)  └▶ TF: world → connector
```

## Node 1a — `cable_neck_ros_node.py` (NECK method)

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
source /opt/sam3_venv/bin/activate && source /opt/ros/jazzy/setup.bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/cable_neck_ros_node.py --ros-args \
  -p image_topic:=/camera/camera1/color/image_raw -p publish_debug:=true \
  -p adaptive:=true -p confidence_floor:=0.2
```

> **`adaptive:=true` is the important flag.** With a fixed threshold SAM3 flips between labelling the
> connector "cable" and the cable "connector"; whichever class comes up empty kills the neck outright,
> because `compute_necks` iterates over CONNECTOR masks — a neck **is** the cable/connector contact.
> Adaptive mode runs SAM3 **once** at `confidence_floor`, then searches the threshold *pair* in
> software, keeping the most confident masks that still yield a valid neck. It costs **no extra
> inference**: the threshold is only a filter on per‑mask scores (`keep = out_probs > threshold`), so
> the forward pass is identical. The log reports what it chose per frame:
> ```
> necks=1 | raw: cables=3 connectors=2 dropped=1 | thr: cable=0.91 conn=0.18 (eff 0.18, 1 combos)
> necks=0 | raw: cables=3 connectors=0 dropped=0 | thr=- (no admissible pair gave a neck)
> ```
> The second line means **no connector mask at all, even at the floor** — lower `confidence_floor` or
> reword `connector_prompt`. (If SAM3 *never* produces a connector mask, use `cable_tip_ros_node.py`
> instead: it unions the masks and works on shape, so it cannot be starved.)

> **Note on the topic name:** realsense2_camera nests topics under *camera_namespace* **and**
> *camera_name*, so with `camera_name:=camera1` the image is on `/camera/camera1/color/image_raw` —
> **not** `/camera1/...`. The image **frame** is still `camera1_color_optical_frame`.

## Node 1b — `cable_junction_ros_node.py` (JUNCTION method)

Same job and message schema as node 1a, but the origin is found by the **diameter-profiling
junction** method (`JunctionDetector` in `cable_neck_diameter.py`): union the cable+connector
masks, trace the cable's length, and place the junction where the **constant-diameter cable
run ends** and the profile climbs into the connector. Direction is the **connector's PCA
axis** (identical rule to node 1a). Everything the node prints/publishes says **"junction"**.

**Publishes**
- `~/junctions` (`geometry_msgs/PoseArray`): **0 or 1** pose — `position.x=u`, `position.y=v`
  (pixels), `position.z=0`; `orientation` = yaw quaternion about +z, `yaw = atan2(dy, dx)`
  in the pixel frame. Same convention as node 1a's `~/necks`, so node 2 reads it unchanged.
- `~/debug_image` (`sensor_msgs/Image`, bgr8): overlay with the traced centreline, both
  cable-outline rails, the junction dot + diameter chord, and the connector-axis arrow.

**Parameters** — `image_topic` (`/camera/color/image_raw`), `cable_prompt` (`cable`),
`connector_prompt` (`connector`), `threshold` (0.5), `connector_threshold` (0.3; `-1`=use
`threshold`), `work_dim` (1024; geometry downscale), `min_contrast` (1.3; reject a junction
whose connector/cable diameter ratio is below this), `publish_debug` (True).

**No adaptive mode.** The junction method is classification-free (it works on the unioned
shape), so an empty connector mask does **not** zero the output the way it does for the neck
node — it only softens the diameter step. Lower `connector_threshold` to pull the connector
body into the assembly and sharpen the step; use `min_contrast` to reject flat-profile frames.

Run (same env as node 1a):
```bash
source /opt/sam3_venv/bin/activate && source /opt/ros/jazzy/setup.bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/cable_junction_ros_node.py --ros-args \
  -p image_topic:=/camera/camera1/color/image_raw -p publish_debug:=true \
  -p connector_threshold:=0.3 -p min_contrast:=1.5
```
Per-frame log:
```
junctions=1 | raw: cables=3 connectors=2 | [@(1244,1512) +81deg x5.4 dia 115->620px]
junctions=0 (contrast 1.1 < 1.5) | raw: cables=2 connectors=0
```
The second line = a real assembly was traced but its profile was too flat to be a junction
(here the connector prompt found nothing) — lower `connector_threshold` / reword the prompt,
or lower `min_contrast`.

### Offline CLI (junction method)
`cable_neck_diameter.py` is the batch/offline counterpart (same geometry). It writes, per
image, `<stem>_junction.png` (overlay), `<stem>_profile.png` (diameter-vs-arclength plot),
`<stem>_junction.json`, and caches `<stem>_assembly.png`. `--render-only` re-runs just the
geometry/overlay from the cached masks (seconds, no SAM3):
```bash
python scripts/cable_neck_diameter.py --input data/cable_test_images \
  --output data/cable_neck_diameter_results          # full SAM3 pass, caches masks
python scripts/cable_neck_diameter.py --render-only \
  --input data/cable_test_images --output data/cable_neck_diameter_results  # re-render only
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
Node 2 is **method-agnostic** — it only needs the `PoseArray` schema, so point `necks_topic`
at whichever node-1 variant is running: `/cable_neck_detector/necks` (neck) **or**
`/cable_junction_detector/junctions` (junction).

**Publishes** — `~/connector_pose` (`geometry_msgs/PoseStamped`, in `world_frame`) and a TF
broadcast `world_frame → connector_frame`.

**Parameters** — `neck_diameter` (0.02 m), `world_frame` (`map`), `connector_frame` (`connector`),
`camera_frame` (fallback if image has no frame_id), `necks_topic`, `camera_info_topic`,
`max_history` (60), `min_views` (2), `min_parallax_deg` (3.0), `tf_timeout` (0.2 s).

Node 2 depends only on numpy + rclpy + tf2_ros (no torch), so it runs in a plain rclpy env:
```bash
source /opt/ros/jazzy/setup.bash
python3 scripts/connector_pose_node.py --ros-args -p neck_diameter:=0.018 -p world_frame:=base_link \
  -p camera_info_topic:=/camera/camera1/color/camera_info -p tf_cache_s:=60.0
```

## Notes
- SAM3 inference is ~1–2 s/frame; node 1 keeps only the latest frame (BEST_EFFORT, depth 1).
- Move the camera between views — node 2 needs parallax (>`min_parallax_deg`) to triangulate.
- Single-connector assumption: node 2 tracks the neck nearest the previous observation.
