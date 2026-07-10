#!/usr/bin/env python
"""Detect the cable *neck* (where a cable meets a connector) and the connector's
orientation at that neck, for each cable-connector pair, over a folder of images.

Pipeline per image (see cable_neck_core.py for the implementation):
  1. SAM3 text-prompt segmentation for "cable" and "connector".
  2. De-duplicate mislabels: drop cable masks that mostly overlap a connector, and
     merge duplicate connector detections; subtract connector pixels from cables.
  3. Pair each connector with cables that touch it (within a small band).
  4. Neck origin  = centroid of the cable pixels hugging that connector.
  5. Orientation  = connector principal axis (PCA), signed into the connector body.

Writes an annotated PNG and a JSON (neck origin + direction) per image.

Usage:
    python scripts/cable_neck_orientation.py \
        --input data/cable_test_images --output data/cable_neck_results
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

from cable_neck_core import NeckDetector, render_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/cable_test_images")
    ap.add_argument("--output", default="data/cable_neck_results")
    ap.add_argument("--cable-prompt", default="cable")
    ap.add_argument("--connector-prompt", default="connector")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mislabel-overlap", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading SAM3 model...")
    detector = NeckDetector(cable_prompt=args.cable_prompt,
                            connector_prompt=args.connector_prompt,
                            threshold=args.threshold,
                            mislabel_overlap=args.mislabel_overlap)

    files = sorted(f for f in os.listdir(args.input)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"Found {len(files)} images.")

    summary = {}
    for fn in files:
        image = Image.open(os.path.join(args.input, fn)).convert("RGB")
        res = detector.detect(image)
        necks = res["necks"]

        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        vis = render_overlay(bgr, res["cleaned_cables"], res["conn_masks"],
                             necks, alpha=args.alpha)

        stem = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(args.output, f"{stem}_neck.png"), vis)
        record = dict(
            image=fn,
            connectors_raw=res["connectors_raw"],
            connectors_merged=len(res["conn_masks"]),
            cables_raw=res["cables_raw"],
            cables_mislabeled_dropped=res["n_dropped"],
            cables_kept=len(res["cleaned_cables"]),
            necks=[dict(connector=n["connector"], cables=n["cables"],
                        neck=[round(n["neck"][0], 1), round(n["neck"][1], 1)],
                        angle_deg=n["angle_deg"],
                        direction=[round(n["direction"][0], 4),
                                   round(n["direction"][1], 4)])
                   for n in necks],
        )
        with open(os.path.join(args.output, f"{stem}_neck.json"), "w") as jf:
            json.dump(record, jf, indent=2)
        summary[fn] = record
        print(f"  {fn}: connectors={res['connectors_raw']}->{len(res['conn_masks'])} "
              f"cables_raw={res['cables_raw']} dropped_as_connector={res['n_dropped']} "
              f"necks={len(necks)}")
        for n in necks:
            print(f"      neck@({n['neck'][0]:.0f},{n['neck'][1]:.0f}) "
                  f"connector{n['connector']} cables{n['cables']} "
                  f"orient={n['angle_deg']:+.1f}deg")

    print("\n=== Summary ===")
    for fn in files:
        s = summary[fn]
        print(f"{fn}: {len(s['necks'])} neck(s); "
              f"dropped {s['cables_mislabeled_dropped']} mislabeled cable->connector")
    print(f"\nResults saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
