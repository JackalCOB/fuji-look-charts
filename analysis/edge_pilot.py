"""Measure how far the camera carries colour across an edge, from the edge pilot chart.

For each quadrant block in the layout, averages output rows across the vertical edge (top half) and
output columns across the horizontal edge (left half), locates the edge where the channel with the
largest step crosses its midpoint, and reports how far from the edge each side settles to within 0.5
and 1.0 levels of its plateau. The plateau is the median 60 to 85 output pixels from the edge.

For each strip patch, reads a central 4 x 4 output-pixel window and compares it with the plateau of
the same colour in a block, so the smallest patch that still reads the colour of a large area shows.

Figures only. Values are 8-bit output levels, as read_chart.load returns them from the 16-bit
trimmed means.

With --geometry, strip patch positions follow the camera's distortion correction.

    python analysis/edge_pilot.py LAYOUT.json AGGREGATE.png [...] [--geometry GEOMETRY.json]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
from read_chart import load  # noqa: E402
from geometry_map import Geometry  # noqa: E402

REACH, PLATEAU = 85, (60, 85)


def to_out(layout, img, geometry=None):
    x0, y0, W, H = layout["active_area"]
    h, w = img.shape[:2]
    sx, sy = w / W, h / H
    if geometry is not None:
        return lambda x, y: geometry.to_output(x, y, sx)
    return lambda x, y: ((x - x0) * sx, (y - y0) * sy)


def settle(profile, plateau, tol):
    """Smallest distance from the edge beyond which every channel stays within tol of its plateau."""
    dev = np.abs(profile - plateau).max(axis=1)
    bad = np.nonzero(dev > tol)[0]
    return 0 if bad.size == 0 else int(bad.max()) + 1


def edge(prof, design_at):
    """Locate the edge in a profile (n x 3, centred on the designed edge) and measure each side."""
    n = prof.shape[0]
    xs = np.arange(n) - n // 2
    left = np.median(prof[(xs <= -PLATEAU[0]) & (xs >= -PLATEAU[1])], axis=0)
    right = np.median(prof[(xs >= PLATEAU[0]) & (xs <= PLATEAU[1])], axis=0)
    c = int(np.argmax(np.abs(right - left)))
    mid = (left[c] + right[c]) / 2
    sign = np.sign(right[c] - left[c])
    above = np.nonzero(sign * (prof[:, c] - mid) > 0)[0]
    above = above[above > n // 2 - 40]
    k = above[0]
    frac = (mid - prof[k - 1, c]) / (prof[k, c] - prof[k - 1, c])
    found = xs[k - 1] + frac
    e = int(round(found)) + n // 2
    out = {"offset": found, "left": left, "right": right}
    for tol in (0.5, 1.0):
        out[f"left_{tol}"] = settle(prof[:e][::-1][:REACH], left, tol)
        out[f"right_{tol}"] = settle(prof[e:][:REACH], right, tol)
    return out


def blocks(layout, img):
    f = to_out(layout, img)
    rows = []
    for b in layout["blocks"]:
        ox, oy = f(b["x"], b["y"])
        size = b["size"] * img.shape[1] / layout["active_area"][2]
        half = size / 2
        ex, ey = int(round(ox + half)), int(round(oy + half))
        r0, r1 = int(oy + 0.15 * size), int(oy + 0.4 * size)
        vert = img[r0:r1, ex - 100:ex + 100].mean(axis=0)
        c0, c1 = int(ox + 0.15 * size), int(ox + 0.4 * size)
        horiz = img[ey - 100:ey + 100, c0:c1].mean(axis=1)
        cen_a = img[int(oy + half / 2) - 10:int(oy + half / 2) + 10, int(ox + half / 2) - 10:int(ox + half / 2) + 10]
        cen_b = img[int(oy + half / 2) - 10:int(oy + half / 2) + 10, int(ox + 1.5 * half) - 10:int(ox + 1.5 * half) + 10]
        rows.append({"name": b["name"], "v": edge(vert, ex), "h": edge(horiz, ey),
                     "a": cen_a.reshape(-1, 3).mean(axis=0), "b": cen_b.reshape(-1, 3).mean(axis=0)})
    return rows


def strips(layout, img, plateaus, geometry=None):
    f = to_out(layout, img, geometry)
    out = []
    for s in layout["strips"]:
        ref = plateaus.get(tuple(s["colour"]))
        res = []
        for p in s["patches"]:
            cx, cy = f(p["x"] + p["size"] / 2, p["y"] + p["size"] / 2)
            win = img[int(cy) - 2:int(cy) + 2, int(cx) - 2:int(cx) + 2].reshape(-1, 3).mean(axis=0)
            res.append((p["size"], win, None if ref is None else np.abs(win - ref).max()))
        out.append((s["name"], res))
    return out


def main(layout_path, images, geometry=None):
    layout = json.load(open(layout_path))
    for path in images:
        img = load(path)
        print(f"\n{os.path.basename(path)}")
        print("  block            dir  edge offset px   settle to 0.5 (A side, B side)   to 1.0       step (max channel)")
        res = blocks(layout, img)
        plateaus = {}
        for r, b in zip(res, layout["blocks"]):
            plateaus.setdefault(tuple(b["a"]), r["a"])
            for d in ("v", "h"):
                e = r[d]
                step = np.abs(e["right"] - e["left"]).max()
                print(f"  {r['name']:16s} {d}    {e['offset']:+6.2f}        {e['left_0.5']:4d} {e['right_0.5']:4d}"
                      f"                      {e['left_1.0']:4d} {e['right_1.0']:4d}     {step:6.1f}")
        print("  context: the same colour beside a different neighbour, block centres")
        for i, b in enumerate(layout["blocks"]):
            for j, c in enumerate(layout["blocks"]):
                if j > i and b["a"] == c["a"]:
                    d = res[i]["a"] - res[j]["a"]
                    print(f"    {b['name']} vs {c['name']}: A differs by R {d[0]:+.2f} G {d[1]:+.2f} B {d[2]:+.2f}")
        for name, rows in strips(layout, img, plateaus, geometry):
            cells = "  ".join(f"{s}: {'-' if d is None else f'{d:4.1f}'}" for s, _, d in rows)
            print(f"  {name:16s} centre vs block plateau, max channel: {cells}")


if __name__ == "__main__":
    args = sys.argv[1:]
    geo = None
    if "--geometry" in args:
        i = args.index("--geometry")
        geo = Geometry(args[i + 1])
        del args[i:i + 2]
    main(args[0], args[1:], geo)
