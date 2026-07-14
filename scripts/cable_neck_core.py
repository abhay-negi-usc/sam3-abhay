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


def masks_scores_from_output(out):
    """Boolean masks + their confidence scores, sorted by score DESCENDING.

    The scores are what make an adaptive threshold possible: Sam3Processor applies the threshold as a
    plain filter (`keep = out_probs > confidence_threshold`), so the forward pass does not depend on it
    at all. Run once at a low floor, keep the scores, and any threshold can then be evaluated in
    software for free. See NeckDetector.detect_adaptive.
    """
    masks = [m.squeeze().cpu().numpy() > 0.5 for m in out["masks"]]
    if "scores" in out and out["scores"] is not None and len(out["scores"]) == len(masks):
        scores = [float(s) for s in out["scores"].detach().cpu().numpy().reshape(-1)]
    else:                                     # shouldn't happen, but never lose the masks over it
        scores = [1.0] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: -scores[i])
    return [masks[i] for i in order], [scores[i] for i in order]


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


# ===================================================================================================
# CONNECTOR TIP -- classification-free
# ===================================================================================================
# SAM3 routinely labels the WHOLE assembly "cable" and returns no connector mask at all. That starves
# any connector-first pipeline: compute_necks() iterates over CONNECTOR masks, so zero connectors =>
# zero necks, no matter how good the cable mask is. The tip pipeline below therefore does not trust
# the cable/connector classification AT ALL -- it unions both prompts into one 'cable_and_connector'
# object and works purely on that object's SHAPE.

def _geodesic_bfs(mask, seed):
    """Geodesic (within-mask) pixel distance from `seed`; -1 where unreachable.

    Geodesic, NOT Euclidean: a cable is a CURVE, so two points can be adjacent in the image yet far
    apart along the cable (think of a bend that doubles back). Euclidean distance would happily jump
    the gap. Implemented as a layered dilation on the mask's bounding box -- cheap, and vectorised so
    it costs a small fraction of one SAM3 inference.
    """
    out = np.full(mask.shape, -1, dtype=np.int32)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = mask[y0:y1, x0:x1]
    dist = np.full(sub.shape, -1, dtype=np.int32)
    cur = np.zeros(sub.shape, dtype=bool)
    sy, sx = int(seed[0]) - y0, int(seed[1]) - x0
    if not (0 <= sy < sub.shape[0] and 0 <= sx < sub.shape[1] and sub[sy, sx]):
        return out
    cur[sy, sx] = True
    dist[cur] = 0
    k = np.ones((3, 3), np.uint8)
    d = 0
    while cur.any():
        d += 1
        grown = cv2.dilate(cur.astype(np.uint8), k).astype(bool)
        nxt = grown & sub & (dist < 0)
        if not nxt.any():
            break
        dist[nxt] = d
        cur = nxt
    out[y0:y1, x0:x1] = dist
    return out


def compute_tip(cable_masks, conn_masks_raw, H, W, curve_px=40, min_area_frac=0.0004):
    """Locate the CONNECTOR TIP from the UNION of the cable and connector segmentations.

    Steps:
      1. union both prompts -> one 'cable_and_connector' mask; keep its largest connected component,
      2. find that shape's two ENDS as its geodesic diameter (BFS from any seed -> farthest = A;
         BFS from A -> farthest = B),
      3. decide which end is the connector:
           * if SAM3 DID produce a connector mask, take the end nearest it (strongest evidence), else
           * take the THICKER end -- a connector is fatter than the cable it terminates. Thickness is
             the mean distance-transform value over the first `curve_px` of curve from that end,
      4. walk BACK along the curve from the tip by `curve_px` (the predefined curve length) and take
         `tip - back` as the connector AXIS. A fixed arc length gives a far more stable direction than
         the local tangent at the very tip, which is dominated by mask noise.

    Returns dict(mask, tip=[u,v], back=[u,v], direction=[dx,dy] (pixel frame, tip-ward), angle_deg,
    thickness_px, used_connector) -- or None if nothing usable was found.
    """
    combined = np.zeros((H, W), dtype=bool)
    for m in cable_masks:
        combined |= m
    conn_union = np.zeros((H, W), dtype=bool)
    for m in conn_masks_raw:
        conn_union |= m
    combined |= conn_union                       # <-- the classification is deliberately ignored here

    min_area = max(64, int(min_area_frac * H * W))
    if combined.sum() < min_area:
        return None

    lbl, n = ndimage.label(combined)
    if n == 0:
        return None
    sizes = ndimage.sum(combined, lbl, index=list(range(1, n + 1)))
    comp = (lbl == (int(np.argmax(sizes)) + 1))   # the cable assembly; drops unrelated blobs
    if comp.sum() < min_area:
        return None

    # --- the two ends: geodesic diameter ---
    ys, xs = np.nonzero(comp)
    dA = _geodesic_bfs(comp, (ys[0], xs[0]))
    A = np.unravel_index(int(np.argmax(np.where(dA >= 0, dA, -1))), comp.shape)
    dB = _geodesic_bfs(comp, A)
    B = np.unravel_index(int(np.argmax(np.where(dB >= 0, dB, -1))), comp.shape)

    edt = ndimage.distance_transform_edt(comp)

    def thickness_of(end):
        de = _geodesic_bfs(comp, end)
        sel = (de >= 0) & (de <= curve_px)
        return float(edt[sel].mean()) if sel.any() else 0.0

    # --- which end is the connector? ---
    if conn_union.any():
        cy, cx = np.nonzero(conn_union)
        cc = np.array([cy.mean(), cx.mean()])
        tip = A if (np.linalg.norm(np.array(A) - cc)
                    <= np.linalg.norm(np.array(B) - cc)) else B
        used_connector = True
    else:
        tip = A if thickness_of(A) >= thickness_of(B) else B
        used_connector = False

    # --- axis: walk back along the curve by the predefined length ---
    dT = _geodesic_bfs(comp, tip)
    reach = dT[dT >= 0]
    if reach.size == 0:
        return None
    lo, hi = max(1, curve_px - 3), curve_px + 3
    band = (dT >= lo) & (dT <= hi)
    if not band.any():                            # cable shorter than curve_px: use its far end
        band = (dT == int(reach.max()))
    by, bx = np.nonzero(band)
    back = np.array([by.mean(), bx.mean()])       # (y, x) centroid of the band -> stable centreline pt

    v = np.array(tip, dtype=float) - back         # (dy, dx), pointing from the body OUT to the tip
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-6:
        return None
    dy, dx = v / nrm

    return dict(
        mask=comp,
        tip=[float(tip[1]), float(tip[0])],       # [u, v]
        back=[float(back[1]), float(back[0])],    # [u, v]
        direction=[float(dx), float(dy)],         # pixel frame (x right, y down), tip-ward
        angle_deg=float(math.degrees(math.atan2(-dy, dx))),
        thickness_px=float(thickness_of(tip)),
        used_connector=used_connector,
    )


def render_tip_overlay(bgr, res, alpha=0.35):
    """Overlay the combined mask, the walked-back curve segment, the tip and the axis arrow."""
    vis = bgr.copy()
    if res.get("mask") is not None:
        ov = vis.copy()
        ov[res["mask"]] = COL_CABLE
        vis = cv2.addWeighted(ov, alpha, vis, 1.0 - alpha, 0)
    if not res.get("tip"):
        cv2.putText(vis, "NO TIP", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                    cv2.LINE_AA)
        return vis
    u, v = (int(round(c)) for c in res["tip"])
    bu, bv = (int(round(c)) for c in res["back"])
    dx, dy = res["direction"]
    cv2.line(vis, (bu, bv), (u, v), COL_CONN, 2, cv2.LINE_AA)          # the curve segment measured
    cv2.arrowedLine(vis, (u, v), (int(u + 70 * dx), int(v + 70 * dy)),
                    COL_ARROW, 3, cv2.LINE_AA, tipLength=0.25)         # the connector axis
    cv2.circle(vis, (bu, bv), 4, COL_CONN, -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 7, COL_NECK, -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 9, (0, 0, 0), 2, cv2.LINE_AA)
    src = "conn-mask" if res.get("used_connector") else "thicker-end"
    cv2.putText(vis, f"TIP {res['angle_deg']:+.0f}deg  thick={res.get('thickness_px', 0):.0f}px  [{src}]",
                (u + 14, v - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ARROW, 2, cv2.LINE_AA)
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

    def _segment_both(self, pil_rgb):
        """Run BOTH prompts once and return (cable_masks, conn_masks_raw). Shared by detect/detect_tip."""
        torch = self._torch
        ac_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        with torch.inference_mode(), torch.autocast(self.device, dtype=ac_dtype):
            state = self.processor.set_image(pil_rgb)
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(self.cable_threshold)
            cable_masks = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.cable_prompt))
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(self.connector_threshold)
            conn_masks_raw = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.connector_prompt))
        return cable_masks, conn_masks_raw

    def _segment_both_scored(self, pil_rgb, floor):
        """Run BOTH prompts ONCE at the confidence FLOOR; return (cable_masks, cable_scores,
        conn_masks, conn_scores), each sorted by score descending. One inference -- the threshold
        search then happens purely in software."""
        torch = self._torch
        ac_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        with torch.inference_mode(), torch.autocast(self.device, dtype=ac_dtype):
            state = self.processor.set_image(pil_rgb)
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(floor)
            cable_m, cable_s = masks_scores_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.cable_prompt))
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(floor)
            conn_m, conn_s = masks_scores_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.connector_prompt))
        return cable_m, cable_s, conn_m, conn_s

    def detect_adaptive(self, pil_rgb, floor=0.05, mislabel_overlap=None):
        """ADAPTIVE-THRESHOLD neck detection: tune the thresholds per image, subject to a confidence
        floor, until the cable and connector actually TOUCH (which is what defines the neck).

        The problem with a fixed threshold. SAM3 puts the connector in the "cable" bucket (or the cable
        in the "connector" bucket) depending on the frame. Whichever class comes up empty, compute_necks
        emits nothing -- it needs BOTH, because a neck IS the cable/connector contact. One global
        threshold cannot be right for every frame.

        The fix. Do not commit to a threshold. Get every candidate mask WITH ITS SCORE, then search the
        threshold PAIR, keeping the MOST CONFIDENT masks that still yield a valid neck. The acceptance
        test is GEOMETRIC, not confidence-based -- which is the whole point: a mislabel only matters if
        it destroys the cable/connector contact.

        Why this is nearly free:
          * The threshold is only a FILTER on per-mask scores (`keep = out_probs > threshold`) -- the
            forward pass is identical. So sweeping it costs NO extra inference: one run at `floor`.
          * Only the actual mask SCORES are distinguishable thresholds. Keeping the top-k of a class is
            therefore the complete set of reachable subsets, so the search is exactly
            len(cable) x len(connector) cheap geometry calls -- typically well under 25.

        Preference order: the combination whose LOWEST admitted score is HIGHEST -- i.e. the highest
        thresholds that still produce a neck. Never admits anything below `floor`.

        Returns the compute_necks dict plus: cables_raw, connectors_raw, thr_cable, thr_conn, eff_conf
        (the chosen effective confidence), combos_tried. thr_* are None if nothing worked.
        """
        W, H = pil_rgb.size
        mo = self.mislabel_overlap if mislabel_overlap is None else mislabel_overlap
        cable_m, cable_s, conn_m, conn_s = self._segment_both_scored(pil_rgb, floor)

        # Every reachable (cable, connector) subset pair, ranked by effective confidence.
        cands = []
        for kc in range(1, len(cable_m) + 1):
            for kk in range(1, len(conn_m) + 1):
                eff = min(cable_s[kc - 1], conn_s[kk - 1])   # the weakest mask this pair admits
                if eff < floor:
                    continue
                cands.append((eff, kc, kk))
        cands.sort(key=lambda t: -t[0])                      # most confident first

        for tried, (eff, kc, kk) in enumerate(cands, start=1):
            res = compute_necks(cable_m[:kc], conn_m[:kk], H, W, mo)
            if res["necks"]:                                 # <-- the geometric acceptance test
                res.update(cables_raw=len(cable_m), connectors_raw=len(conn_m),
                           thr_cable=cable_s[kc - 1], thr_conn=conn_s[kk - 1],
                           eff_conf=eff, combos_tried=tried)
                return res

        # No admissible threshold pair yields a neck. Return the most permissive result so the debug
        # overlay still shows what SAM3 actually saw (which is how you diagnose it).
        res = compute_necks(cable_m, conn_m, H, W, mo)
        res.update(cables_raw=len(cable_m), connectors_raw=len(conn_m),
                   thr_cable=None, thr_conn=None, eff_conf=None, combos_tried=len(cands))
        return res

    def detect_tip(self, pil_rgb, curve_px=40):
        """Classification-free CONNECTOR TIP detection: union both prompts, then use the SHAPE.

        Unlike detect() (which needs a connector mask to exist, since compute_necks iterates over
        connectors), this survives SAM3 labelling the whole assembly "cable" -- the common failure. See
        compute_tip. Always returns a dict; res['tip'] is None when nothing usable was found."""
        W, H = pil_rgb.size
        cable_masks, conn_masks_raw = self._segment_both(pil_rgb)
        res = compute_tip(cable_masks, conn_masks_raw, H, W, curve_px=curve_px)
        if res is None:
            res = dict(mask=None, tip=None, back=None, direction=None, angle_deg=0.0,
                       thickness_px=0.0, used_connector=False)
        res["cables_raw"] = len(cable_masks)
        res["connectors_raw"] = len(conn_masks_raw)
        return res
