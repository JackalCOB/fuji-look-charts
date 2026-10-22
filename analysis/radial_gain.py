"""Measure the camera's radial brightening from flat-field charts, as a gain on sensor level.

Each flat field's trimmed-mean conversion is sampled on a grid of windows. Each window's output luma is
converted back to an effective sensor level through the same look's tone ladder (tone_<look>.csv, by
interpolation in log2 level), and divided by the effective level near the centre. The result is the
gain the camera applied at that distance from the centre, independent of the tone curve, if the
brightening acts on sensor level. The vignetting table the container records (0xf010) is printed
beside it as the gain it would imply, 100 / V.

    python analysis/radial_gain.py AGGREGATE_DIR TONE_DIR [LOOK ...]
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
from read_chart import load  # noqa: E402

LEVELS = [0.005, 0.015, 0.04, 0.1, 0.25, 0.5]
BINS = np.arange(0.0, 1.11, 0.1)
# The container's VignettingParams (0xf010): knots as a fraction of the half diagonal, and the
# illumination there in percent, read by charts/make_chart.py --show-lens.
VIG_KNOTS = [0.3535, 0.5, 0.6127, 0.707, 0.7908, 0.8662, 0.9352, 1.0, 1.0606]
VIG_PCT = [98.4614, 96.8779, 94.9023, 90.9814, 84.7329, 78.437, 72.9604, 67.457, 67.457]


def inverse_tone(tone_csv):
    rows = [r for r in csv.DictReader(open(tone_csv)) if float(r["level"]) > 0]
    lv = np.array([float(r["level"]) for r in rows])
    lu = np.array([float(r["luma"]) for r in rows])
    order = np.argsort(lu)
    lu, lv = np.maximum.accumulate(lu[order]), lv[order]
    return lambda y: 2 ** np.interp(y, lu, np.log2(lv))


def window_grid(img, size=40, step=60):
    h, w = img.shape[:2]
    cy, cx, half = h / 2, w / 2, math.hypot(w / 2, h / 2)
    pts = []
    for y in range(size, h - size, step):
        for x in range(size, w - size, step):
            win = img[y - size // 2:y + size // 2, x - size // 2:x + size // 2].reshape(-1, 3).mean(axis=0)
            pts.append((math.hypot(x - cx, y - cy) / half, 0.2126 * win[0] + 0.7152 * win[1] + 0.0722 * win[2]))
    return np.array(pts)


def main(agg_dir, tone_dir, looks):
    looks = looks or ["provia"]
    table_gain = {k: 100.0 / v for k, v in zip(VIG_KNOTS, VIG_PCT)}
    print("radius bin:            " + " ".join(f"{b:5.1f}" for b in BINS[:-1]))
    print("table gain at knots:   " + ", ".join(f"r {k:.3f} x{g:.3f}" for k, g in list(table_gain.items())[:8]))
    for look in looks:
        inv = inverse_tone(os.path.join(tone_dir, f"tone_{look}.csv"))
        for level in LEVELS:
            path = os.path.join(agg_dir, look, f"flat_{level:g}__{look}.png")
            if not os.path.exists(path):
                continue
            pts = window_grid(load(path))
            eff = inv(pts[:, 1])
            centre = np.median(eff[pts[:, 0] < 0.1])
            gain = eff / centre
            cells = []
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                sel = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
                cells.append(f"{np.median(gain[sel]):5.3f}" if sel.any() else "    -")
            out_c = np.median(pts[pts[:, 0] < 0.1, 1])
            print(f"{look:14s} flat {level:<6g} (centre output {out_c:5.1f}): " + " ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
