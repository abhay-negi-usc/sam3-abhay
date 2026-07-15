#!/usr/bin/env python
"""Run SAM3 text-prompted segmentation over an image directory.

Edit CONFIG below and run:
    python scripts/run_sam3_predictions.py

For each image, writes to <image_dir>/predictions:
    <stem>_overlay.png   masks + boxes + scores drawn on the image
    <stem>_mask.png      binary union of all instance masks
    <stem>.json          boxes (xyxy), scores, per-instance mask areas
plus a top-level summary.json with the config and per-image detection counts.
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass  # HEIC/HEIF support unavailable; install `pillow-heif` to enable it.

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CONFIG = {
    "prompt": "receptacle",
    "image_dir": "/home/rp/abhay_ws/sam3-abhay/data/socket_test_images/",
    "threshold": 0.10,
    # rendering
    "alpha": 0.5,
    "color": (46, 204, 113),
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".heic", ".heif")


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main(config=CONFIG):
    prompt = config["prompt"]
    image_dir = config["image_dir"]
    threshold = config["threshold"]
    color = tuple(config.get("color", (46, 204, 113)))
    alpha = config.get("alpha", 0.5)

    out_dir = os.path.join(image_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    )
    if not files:
        raise SystemExit(f"No images found in {os.path.abspath(image_dir)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Loading SAM3 model on {device}...")
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)

    print(f"{len(files)} images | prompt={prompt!r} | threshold={threshold}")

    font = _font(40)
    summary = {"config": {**config, "color": list(color)}, "images": {}}

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for fn in files:
            image = Image.open(os.path.join(image_dir, fn)).convert("RGB")
            stem = os.path.splitext(fn)[0]

            state = processor.set_image(image)
            out = processor.set_text_prompt(state=state, prompt=prompt)
            masks, boxes, scores = out["masks"], out["boxes"], out["scores"]

            base = np.asarray(image, dtype=np.float32)
            overlay = base.copy()
            union = np.zeros(base.shape[:2], dtype=bool)
            instances = []

            for mask, box, score in zip(masks, boxes, scores):
                m = mask.squeeze().float().cpu().numpy() > 0.5
                union |= m
                instances.append({
                    "box_xyxy": [float(v) for v in box.float().cpu().tolist()],
                    "score": float(score),
                    "mask_area": int(m.sum()),
                })

            if union.any():
                overlay[union] = (
                    (1 - alpha) * overlay[union]
                    + alpha * np.array(color, dtype=np.float32)
                )

            out_img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(out_img)
            for inst in instances:
                x0, y0, x1, y1 = inst["box_xyxy"]
                draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
                draw.text((x0, max(0, y0 - 44)), f"{inst['score']:.2f}",
                          fill=color, font=font)
            draw.text((20, 20), f"{prompt} ({len(instances)})",
                      fill=color, font=font)

            out_img.save(os.path.join(out_dir, f"{stem}_overlay.png"))
            Image.fromarray((union * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"{stem}_mask.png")
            )
            with open(os.path.join(out_dir, f"{stem}.json"), "w") as f:
                json.dump({
                    "image": fn,
                    "prompt": prompt,
                    "threshold": threshold,
                    "instances": instances,
                }, f, indent=2)

            summary["images"][fn] = len(instances)
            print(f"  {fn}: {len(instances)} detections")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    total = sum(summary["images"].values())
    print(f"\n{total} detections across {len(files)} images")
    print(f"Predictions saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
