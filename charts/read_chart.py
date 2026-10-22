"""Read a camera conversion of a synthetic chart back into per-patch colours.

Maps each patch from sensor coordinates to the output image, samples a central window of each patch,
and reports the mean colour, its spread, and the anchors' brightness by position. Without a geometry
fit the mapping is the active area scaled to the output's size, and the window is the central quarter
of the patch's side, so small shifts from the camera's lens correction do not reach the patch edges.
With --geometry, positions follow the camera's distortion correction (geometry_map.py), and --window
sets the window's side as a fraction of the patch's.

    python charts/read_chart.py LAYOUT.json OUTPUT.jpg|.png [--csv OUT.csv]
        [--geometry GEOMETRY.json] [--window 0.25]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
from aggregate_repeats import read_png16  # noqa: E402
from geometry_map import Geometry  # noqa: E402


def load(path):
    if path.lower().endswith(".png"):
        return read_png16(path).astype(np.float64)
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float64)


def read(layout, img, geometry=None, window=0.25):
    x0, y0, W, H = layout["active_area"]
    h, w = img.shape[:2]
    sx, sy = w / W, h / H
    out = []
    for p in layout["patches"]:
        px, py = p["x"] + p["w"] / 2, p["y"] + p["h"] / 2
        if geometry is None:
            cx, cy, k = (px - x0) * sx, (py - y0) * sy, 1.0
        else:
            cx, cy = geometry.to_output(px, py, sx)
            k = float(geometry.stretch(np.hypot(px - geometry.c_s[0], py - geometry.c_s[1]) / geometry.half))
        hw, hh = p["w"] * sx * k * window / 2, p["h"] * sy * k * window / 2
        win = img[int(cy - hh):int(cy + hh), int(cx - hw):int(cx + hw)]
        mean = win.reshape(-1, 3).mean(axis=0)
        sd = win.reshape(-1, 3).std(axis=0)
        out.append({"name": p["name"], "design": p["rgb"], "mean": mean.tolist(), "sd": sd.tolist(),
                    "cx": cx, "cy": cy, "window_px": win.shape[0] * win.shape[1]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("image")
    ap.add_argument("--csv")
    ap.add_argument("--geometry")
    ap.add_argument("--window", type=float, default=0.25)
    a = ap.parse_args()
    layout = json.load(open(a.layout))
    img = load(a.image)
    rows = read(layout, img, Geometry(a.geometry) if a.geometry else None, a.window)
    print(f"{os.path.basename(a.image)}: {img.shape[1]} x {img.shape[0]}")
    for r in rows:
        m, s = r["mean"], r["sd"]
        print(f"  {r['name']:12s} design {r['design']}  out R {m[0]:6.1f} G {m[1]:6.1f} B {m[2]:6.1f}  "
              f"spread {max(s):4.1f}")
    anchors = [r for r in rows if r["name"].startswith("anchor_")]
    if anchors:
        lum = [0.2126 * r["mean"][0] + 0.7152 * r["mean"][1] + 0.0722 * r["mean"][2] for r in anchors]
        print(f"  anchor luma: min {min(lum):.1f}, max {max(lum):.1f}, spread {max(lum) - min(lum):.1f} levels")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["name", "design_r", "design_g", "design_b", "out_r", "out_g", "out_b", "sd_max", "cx", "cy"])
            for r in rows:
                wr.writerow([r["name"], *r["design"], *[f"{v:.3f}" for v in r["mean"]], f"{max(r['sd']):.3f}",
                             f"{r['cx']:.1f}", f"{r['cy']:.1f}"])


if __name__ == "__main__":
    main()
