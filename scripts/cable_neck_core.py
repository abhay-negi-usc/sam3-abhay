"""Shared core for cable-neck + connector-orientation detection.

Two layers:
  * pure-geometry helpers (`compute_necks`, `render_overlay`) — no torch, easy to
    test on precomputed masks.
  * `NeckDetector` — wraps SAM3 text-prompt segmentation and runs the geometry.

Neck convention (image/pixel coordinates):
  * origin  = (u, v), u right, v down, top-left = (0, 0)
  * direction = unit vector (dx, dy) in the SAME pixel frame (dy > 0 points down),
    pointing from the neck into the connector body.
  * angle_deg = atan2(-dy, dx): 0deg = +u (right), +90deg = up.
"""
import math
import os

import cv2
import numpy as np
from scipy import ndimage

# BGR colors for cv2 drawing
COL_CABLE = (60, 60, 231)     # red-ish
COL_CONN = (113, 204, 46)     # green-ish
COL_NECK = (0, 215, 255)      # amber dot
COL_ARROW = (255, 200, 0)     # cyan/blue arrow


def masks_from_output(out):
    """Return list of boolean HxW masks from a Sam3 processor output."""
    return [m.squeeze().cpu().numpy() > 0.5 for m in out["masks"]]


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])  # (x, y)


def pca_axis(mask):
    """Return (major_unit_vec (x,y), elongation) for a boolean mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        return np.array([1.0, 0.0]), 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    major = evecs[:, 0]
    elong = math.sqrt(max(evals[0], 1e-9) / max(evals[1], 1e-9))
    return major / (np.linalg.norm(major) + 1e-9), elong


def merge_overlapping(masks, iou_thr=0.25, ov_thr=0.55):
    """Greedily union masks that are duplicate detections of the same object
    (high IoU, or one largely contained in the other). Iterates to convergence
    so chains of overlapping detections collapse into one."""
    merged = [m.copy() for m in masks if m.sum() > 0]
    changed = True
    while changed:
        changed = False
        out = []
        for m in merged:
            hit = None
            for i, mm in enumerate(out):
                inter = int((m & mm).sum())
                if inter == 0:
                    continue
                iou = inter / int((m | mm).sum())
                ov = inter / min(int(m.sum()), int(mm.sum()))
                if iou > iou_thr or ov > ov_thr:
                    hit = i
                    break
            if hit is None:
                out.append(m)
            else:
                out[hit] = out[hit] | m
                changed = True
        merged = out
    return merged


def compute_necks(cable_masks, conn_masks_raw, H, W, mislabel_overlap=0.6):
    """Geometry only. Given boolean cable/connector masks, return a dict with the
    cleaned masks and one neck per connector.

    Returns dict(conn_masks, cleaned_cables, necks, n_dropped). Each neck is a
    dict(connector, cables, neck=[u,v], direction=[dx,dy], angle_deg, length, tip=[u,v]).
    """
    diag = math.hypot(W, H)
    band = max(4, int(0.012 * diag))                 # contact band width (px)
    min_contact = max(30, int((0.004 * diag) ** 2 / 4))
    min_cable_area = int(0.0003 * H * W)
    max_gap = 0.06 * diag                             # bridge cable<->connector gaps

    # merge duplicate detections of the same physical connector
    conn_masks = merge_overlapping(conn_masks_raw)
    conn_union = np.zeros((H, W), dtype=bool)
    for cm in conn_masks:
        conn_union |= cm

    # de-duplicate mislabels + subtract connector pixels from cables
    cleaned_cables = []
    n_dropped = 0
    for cm in cable_masks:
        area = int(cm.sum())
        if area == 0:
            continue
        if (cm & conn_union).sum() / area > mislabel_overlap:
            n_dropped += 1                            # this "cable" is a connector
            continue
        stub = cm & ~conn_union
        if stub.sum() >= min_cable_area:
            cleaned_cables.append(stub)

    cables_union = np.zeros((H, W), dtype=bool)
    for cable in cleaned_cables:
        cables_union |= cable

    necks = []
    for cj, conn in enumerate(conn_masks):
        if conn.sum() < 20:
            continue
        dist_to_conn = ndimage.distance_transform_edt(~conn)
        conn_c = centroid(conn)
        major, elong = pca_axis(conn)

        contact = cables_union & (dist_to_conn <= band)
        contact_cables = [ci for ci, c in enumerate(cleaned_cables)
                          if (c & (dist_to_conn <= band)).sum() >= min_contact]

        if contact.sum() < min_contact:
            # fallback: bridge a small gap (intermediate hardware in between)
            if cables_union.any():
                dmin = float(dist_to_conn[cables_union].min())
                if dmin <= max_gap:
                    contact = cables_union & (dist_to_conn <= dmin + band)
                    contact_cables = [ci for ci, c in enumerate(cleaned_cables)
                                      if (c & (dist_to_conn <= dmin + band)).sum() > 0]
            if contact.sum() == 0:
                continue

        neck = centroid(contact)  # (x, y)
        v_body = conn_c - neck
        if np.linalg.norm(v_body) < 1e-6:
            continue
        if elong > 1.3:
            direction = major.copy()
            if np.dot(direction, v_body) < 0:
                direction = -direction
        else:
            direction = v_body / np.linalg.norm(v_body)

        ys, xs = np.nonzero(conn)
        proj = (xs - neck[0]) * direction[0] + (ys - neck[1]) * direction[1]
        L = max(float(np.percentile(proj, 95)), 0.15 * diag)
        tip = neck + direction * L
        angle = math.degrees(math.atan2(-direction[1], direction[0]))

        necks.append(dict(
            connector=cj,
            cables=contact_cables,
            neck=[float(neck[0]), float(neck[1])],
            direction=[float(direction[0]), float(direction[1])],
            angle_deg=round(angle, 2),
            length=float(L),
            tip=[float(tip[0]), float(tip[1])],
        ))

    return dict(conn_masks=conn_masks, cleaned_cables=cleaned_cables,
                necks=necks, n_dropped=n_dropped)


def render_overlay(bgr, cleaned_cables, conn_masks, necks, alpha=0.35):
    """Return an annotated BGR image: faint masks + neck dot + orientation arrow."""
    H, W = bgr.shape[:2]
    diag = math.hypot(W, H)
    thick = max(2, int(0.0035 * diag))
    dot_r = max(4, int(0.007 * diag))

    vis = bgr.astype(np.float32)
    for cm in cleaned_cables:
        vis[cm] = (1 - alpha) * vis[cm] + alpha * np.array(COL_CABLE, np.float32)
    for cm in conn_masks:
        vis[cm] = (1 - alpha) * vis[cm] + alpha * np.array(COL_CONN, np.float32)
    vis = vis.clip(0, 255).astype(np.uint8)

    for nk in necks:
        u, v = nk["neck"]
        dx, dy = nk["direction"]
        L = nk["length"]
        p0 = (int(round(u)), int(round(v)))
        p1 = (int(round(u + dx * L)), int(round(v + dy * L)))
        cv2.arrowedLine(vis, p0, p1, COL_ARROW, thick, tipLength=0.18,
                        line_type=cv2.LINE_AA)
        cv2.circle(vis, p0, dot_r, COL_NECK, -1, cv2.LINE_AA)
        cv2.circle(vis, p0, dot_r, (0, 0, 0), max(1, thick // 2), cv2.LINE_AA)
        lx = int(u + dx * L * 0.62) + dot_r
        ly = int(v + dy * L * 0.62)
        lbl = f"C{nk['connector']}:{nk['angle_deg']:+.0f}deg"
        fscale = 0.0009 * diag
        fthick = max(2, thick // 2)
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fscale, fthick)
        cv2.rectangle(vis, (lx - 4, ly - th - 6), (lx + tw + 4, ly + bl + 2),
                      (0, 0, 0), -1)
        cv2.putText(vis, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, fscale,
                    COL_ARROW, fthick, cv2.LINE_AA)
    return vis


class NeckDetector:
    """SAM3-backed cable-neck detector. Loads the model once; reuse across frames."""

    def __init__(self, cable_prompt="cable", connector_prompt="connector",
                 threshold=0.5, connector_threshold=None, mislabel_overlap=0.6):
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._torch = torch
        self.cable_prompt = cable_prompt
        self.connector_prompt = connector_prompt
        self.mislabel_overlap = mislabel_overlap
        # SEPARATE thresholds per prompt. The connector is the harder class: it is small, and SAM3
        # happily labels the whole assembly "cable", which leaves conn_masks EMPTY -- and since
        # compute_necks iterates over CONNECTORS, that yields ZERO necks no matter how good the cable
        # mask is. A lower threshold for the connector alone lets marginal detections through WITHOUT
        # flooding the cable side with spurious masks; one shared threshold cannot do both.
        self.cable_threshold = float(threshold)
        self.connector_threshold = (float(threshold) if connector_threshold is None
                                    else float(connector_threshold))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # SDPA backend -- this is what makes SAM3 fit on a small, pre-Ampere GPU.
        # SAM3's get_sdpa_settings() disables Flash Attention on any GPU below Ampere and falls back
        # to the MATH kernel, which MATERIALIZES the full NxN attention matrix (~822 MB at 1008px)
        # and OOMs a 6 GB card. The cutlass MEMORY-EFFICIENT backend runs on Pascal and never
        # materializes it. It is only eligible for fp16 (bf16 needs sm_80), which is why detect()
        # autocasts to float16 on CUDA. Weights stay fp32 -- autocast casts per op, so there is no
        # fp16/fp32 mismatch (unlike .half()'ing the model, which SAM3 does not support).
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)         # needs Ampere; unavailable here
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # the one that saves the memory
            torch.backends.cuda.enable_math_sdp(True)           # keep only as a last-resort fallback
        model = build_sam3_image_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Do NOT .half() this model. SAM3 creates fp32 tensors internally all over its graph (decoder
        # queries, text embeddings, ...), and autocast doesn't cover those paths -- fp16 weights then
        # collide with them ("mat1 and mat2 must have the same dtype, but got Float and Half") in one
        # op after another. Keep fp32 weights + autocast (see detect); to fit a small GPU, free other
        # VRAM (close GPU-accelerated apps) rather than changing the model's precision.
        # RESOLUTION MUST STAY 1008: the ViTDet backbone's RoPE `freqs_cis` buffer is baked for that
        # grid, so any other value trips the assert in vitdet.reshape_for_broadcast.
        resolution = int(os.environ.get("SAM3_RESOLUTION", "1008"))
        self.processor = Sam3Processor(model, resolution=resolution,
                                       confidence_threshold=threshold)

    def detect(self, pil_rgb):
        """Run SAM3 + geometry on a PIL RGB image. Returns the compute_necks dict
        plus the raw cable/connector counts."""
        torch = self._torch
        W, H = pil_rgb.size
        # float16 on CUDA: REQUIRED for the memory-efficient SDPA backend on a pre-Ampere GPU (bf16
        # needs sm_80; without fp16 the attention silently falls back to the math kernel, which
        # materializes the full NxN matrix and OOMs). bfloat16 on CPU.
        ac_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        autocast = torch.autocast(self.device, dtype=ac_dtype)
        with torch.inference_mode(), autocast:
            state = self.processor.set_image(pil_rgb)
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(self.cable_threshold)
            cable_masks = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.cable_prompt))
            self.processor.reset_all_prompts(state)
            # Lower bar for the connector -- see __init__: no connector mask means no neck at all.
            self.processor.set_confidence_threshold(self.connector_threshold)
            conn_masks_raw = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.connector_prompt))
        result = compute_necks(cable_masks, conn_masks_raw, H, W, self.mislabel_overlap)
        result["cables_raw"] = len(cable_masks)
        result["connectors_raw"] = len(conn_masks_raw)
        return result
