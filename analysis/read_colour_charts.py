"""Read every colour chart conversion into one table of patches per look.

For each chart, the input side is taken from the chart raw itself: each patch's stored sensor samples
turned back into white-balanced camera RGB and averaged per channel (verify_chart.py checks the same
values against the design). That is what the camera was given, including the rounding of dark
channels to whole sensor steps, so fits use it rather than the design.

The output side is the trimmed mean of the look's five conversions, read over the central part of
each patch (WINDOW of its side), with positions mapped through the geometry fit.

Writes OUT/patches_<look>.csv with the catalogue fields, the stored camera RGB, and the output's mean
and spread. Figures only.

    python analysis/read_colour_charts.py CHARTS_DIR LAYOUTS_DIR AGGREGATE_DIR GEOMETRY.json OUT_DIR
        RADIAL_COEF.npy [LOOK ...]
"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
from geometry_map import Geometry  # noqa: E402
from make_chart import WHITE, radial_gain, records  # noqa: E402
from read_chart import load, read  # noqa: E402
from verify_chart import stored  # noqa: E402

WINDOW = 0.35
LOOKS = ["provia", "velvia", "astia", "pro_neg_hi", "pro_neg_std", "classic_chrome", "eterna", "classic_neg",
         "eterna_bleach_bypass"]
FIELDS = ["kind", "colour_id", "copy", "anchor", "family", "x", "y", "r", "level", "stops", "hue", "fraction",
          "chroma"]


def stored_rgb(chart, layout, radial, geometry):
    """Per patch, the stored samples as white-balanced camera RGB, averaged per channel."""
    b = open(chart, "rb").read()
    s, t = stored(b)
    cfa = np.array(list(records(b)[0x0131]))[::-1].reshape(6, 6)
    black = np.array(t[0xF00A], dtype=np.float64)
    black_site = black.reshape(6, 6) if black.size == 36 else np.full((6, 6), black[0])
    g, r, bl = t[0xF00E][:3]
    gain = np.array([r / g, 1.0, bl / g])
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    colour = cfa[yy % 6, xx % 6]
    blk = black_site[yy % 6, xx % 6]
    val = (s - blk) / (WHITE - blk) * gain[colour]
    val *= radial_gain(radial, s.shape[0], s.shape[1], layout["active_area"], geometry)
    out = {}
    for p in layout["patches"]:
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        v, c = val[y:y + h, x:x + w], colour[y:y + h, x:x + w]
        out[p["name"]] = [float(v[c == ch].mean()) for ch in range(3)]
    return out


def main(charts_dir, layouts_dir, agg_dir, geometry_path, out_dir, radial_path, *looks):
    looks = list(looks) or LOOKS
    geo = Geometry(geometry_path)
    radial = np.load(radial_path)
    os.makedirs(out_dir, exist_ok=True)
    writers, files = {}, []
    for look in looks:
        f = open(os.path.join(out_dir, f"patches_{look}.csv"), "w", newline="")
        files.append(f)
        w = csv.writer(f)
        w.writerow(["chart", "name", *FIELDS, "lab_l", "lab_a", "lab_b", "design_r", "design_g", "design_b",
                    "cam_r", "cam_g", "cam_b", "out_r", "out_g", "out_b", "sd_max", "window_px"])
        writers[look] = w
    for lp in sorted(glob.glob(os.path.join(layouts_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(lp))[0]
        layout = json.load(open(lp))
        cam = stored_rgb(os.path.join(charts_dir, f"{name}.raf"), layout, radial, geo)
        meta = {p["name"]: p for p in layout["patches"]}
        done = []
        for look in looks:
            path = os.path.join(agg_dir, look, f"{name}__{look}.png")
            if not os.path.exists(path):
                continue
            for r in read(layout, load(path), geo, WINDOW):
                p = meta[r["name"]]
                writers[look].writerow([name, r["name"], *[p.get(k, "") for k in FIELDS],
                                        *[f"{v:.3f}" for v in p["lab"]], *[f"{v:.6f}" for v in p["rgb"]],
                                        *[f"{v:.6f}" for v in cam[r["name"]]], *[f"{v:.3f}" for v in r["mean"]],
                                        f"{max(r['sd']):.3f}", r["window_px"]])
            done.append(look)
        print(f"{name}: {len(layout['patches'])} patches, read in {len(done)} looks")
    for f in files:
        f.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
