#!/usr/bin/env python
"""Run SAM3 text-prompted segmentation over a folder of images and save overlays.

Supports one or more text prompts (comma-separated). Each prompt-class gets a
distinct color; every instance of a class shares that class color. Saves a
combined overlay per image plus a per-class binary mask.

Usage:
    python scripts/segment_cable_images.py \
        --input cable_test_images --output cable_test_results \
        --prompts "cable,connector" --threshold 0.5
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# One distinct color per prompt-class (RGB, 0-255).
CLASS_COLORS = [
    (231, 76, 60),    # red
    (46, 204, 113),   # green
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (26, 188, 156),   # teal
]


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="cable_test_images")
    ap.add_argument("--output", default="cable_test_results")
    ap.add_argument("--prompts", default="cable,connector",
                    help="comma-separated text prompts")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    os.makedirs(args.output, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading SAM3 model...")
    model = build_sam3_image_model()  # bpe_path=None -> uses sam3/assets default
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    files = sorted(
        f for f in os.listdir(args.input)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"Found {len(files)} images. Prompts={prompts}, thresh={args.threshold}")

    # summary[fn] = {prompt: count}
    summary = {}
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else torch.autocast("cpu", dtype=torch.bfloat16))

    with torch.inference_mode(), autocast:
        for fn in files:
            path = os.path.join(args.input, fn)
            image = Image.open(path).convert("RGB")
            stem = os.path.splitext(fn)[0]
            state = processor.set_image(image)

            base = np.asarray(image, dtype=np.float32)
            overlay = base.copy()
            box_draw = []  # (box, score, color)
            counts = {}

            for pi, prompt in enumerate(prompts):
                color = CLASS_COLORS[pi % len(CLASS_COLORS)]
                processor.reset_all_prompts(state)
                out = processor.set_text_prompt(state=state, prompt=prompt)
                masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
                n = len(scores)
                counts[prompt] = n

                if n > 0:
                    h, w = masks[0].squeeze().shape[-2:]
                    class_mask = np.zeros((h, w), dtype=bool)
                    for m in masks:
                        class_mask |= (m.squeeze().cpu().numpy() > 0.5)
                    overlay[class_mask] = (
                        (1 - args.alpha) * overlay[class_mask]
                        + args.alpha * np.array(color, dtype=np.float32)
                    )
                    Image.fromarray((class_mask * 255).astype(np.uint8)).save(
                        os.path.join(args.output, f"{stem}_{prompt.replace(' ', '_')}_mask.png")
                    )
                    for box, score in zip(boxes, scores):
                        box_draw.append(
                            ([float(v) for v in box.cpu().tolist()], score.item(), color)
                        )

            # draw boxes + a legend on top of the combined overlay
            out_img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(out_img)
            font = _font(40)
            for box, score, color in box_draw:
                x0, y0, x1, y1 = box
                draw.rectangle([x0, y0, x1, y1], outline=color, width=5)
                draw.text((x0, max(0, y0 - 44)), f"{score:.2f}", fill=color, font=font)

            # legend
            ly = 20
            for pi, prompt in enumerate(prompts):
                color = CLASS_COLORS[pi % len(CLASS_COLORS)]
                draw.rectangle([20, ly, 70, ly + 40], fill=color)
                draw.text((80, ly), f"{prompt} ({counts.get(prompt, 0)})",
                          fill=color, font=font)
                ly += 56

            out_img.save(os.path.join(args.output, f"{stem}_seg.png"))
            summary[fn] = counts
            print(f"  {fn}: " + ", ".join(f"{p}={counts[p]}" for p in prompts))

    print("\n=== Summary (instances per prompt) ===")
    for fn in files:
        c = summary.get(fn, {})
        print(f"{fn}: " + ", ".join(f"{p}={c.get(p, 0)}" for p in prompts))
    print(f"\nResults saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
