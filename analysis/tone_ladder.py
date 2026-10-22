"""Measure each look's tone response from the tone-ladder chart.

Reads the trimmed-mean conversion of the ladder in each look, takes each neutral patch's output, and
writes one CSV per look (design level, output R, G, B, and luma) plus a summary: the output for black
and the darkest step, the levels where output reaches middle grey (sRGB 118) and the start of
clipping (254), the span in stops between output 5 and output 250, and the largest colour cast along
the ladder.

    python analysis/tone_ladder.py LAYOUT.json AGGREGATE_DIR OUT_DIR
"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
from read_chart import load, read  # noqa: E402

LOOKS = ["provia", "velvia", "astia", "pro_neg_hi", "pro_neg_std", "classic_chrome", "eterna", "classic_neg",
         "eterna_bleach_bypass"]


def luma(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def crossing(levels, out, target):
    """The design level where output first reaches target, by interpolation in log2 level."""
    for (l0, o0), (l1, o1) in zip(zip(levels, out), zip(levels[1:], out[1:])):
        if l0 > 0 and o0 < target <= o1:
            t = (target - o0) / (o1 - o0)
            return 2 ** (math.log2(l0) + t * (math.log2(l1) - math.log2(l0)))
    return None


def curve(layout, img):
    rows = read(layout, img)
    by_name = {p["name"]: p["level"] for p in layout["patches"]}
    pts = sorted((by_name[r["name"]], r["mean"]) for r in rows)
    return pts


def main(layout_path, agg_dir, out_dir):
    layout = json.load(open(layout_path))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{'look':22s} {'black':>6s} {'2^-13':>6s} {'mid-grey at':>12s} {'clip from':>10s} {'span 5-250':>11s} {'largest cast':>13s}")
    for look in LOOKS:
        path = os.path.join(agg_dir, look, f"tone_ladder__{look}.png")
        if not os.path.exists(path):
            print(f"{look:22s} missing")
            continue
        pts = curve(layout, load(path))
        with open(os.path.join(out_dir, f"tone_{look}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["level", "out_r", "out_g", "out_b", "luma"])
            for lv, m in pts:
                w.writerow([lv, *[f"{v:.3f}" for v in m], f"{luma(m):.3f}"])
        levels = [lv for lv, _ in pts]
        lum = [luma(m) for _, m in pts]
        black, darkest = lum[0], lum[1]
        mid = crossing(levels, lum, 118.0)
        clip = crossing(levels, lum, 254.0)
        lo, hi = crossing(levels, lum, 5.0), crossing(levels, lum, 250.0)
        span = math.log2(hi / lo) if lo and hi else float("nan")
        cast = max(max(abs(m[0] - m[1]), abs(m[2] - m[1])) for _, m in pts)
        fmt = lambda v: f"{v:.4f}" if v else "  none"
        mid_s = f"{fmt(mid)} ({math.log2(mid):+.2f} st)" if mid else "none"
        print(f"{look:22s} {black:6.1f} {darkest:6.1f} {mid_s:>18s} {fmt(clip):>10s} {span:8.2f} st {cast:9.1f} lv")


if __name__ == "__main__":
    main(*sys.argv[1:4])
