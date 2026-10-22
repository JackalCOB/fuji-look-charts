"""Write the pilot layouts: the edge pilot and the geometry grid.

The geometry grid places dark markers across the whole frame, to fit where each sensor position lands
in the camera's output (geometry()).

The edge pilot measures how far the camera's processing carries colour across a patch edge.

Six blocks of four quadrants, the two colours of a pair placed diagonally, so each block has vertical
and horizontal edges between them. The camera's JPEGs subsample chroma horizontally (4:2:2), so the
two directions can differ. Below them, three strips hold square patches from 24 to 192 sensor pixels
on a contrasting field or on the grey background, to find the smallest patch whose centre still reads
the colour of a large area.

Values are white-balanced camera RGB, 0 to 1 of the range from black to white, kept near the 0.09
middle grey so nothing clips. Everything sits on 6-pixel boundaries and within 0.6 of the half
diagonal.

    python charts/layouts_pilot.py charts/layouts
"""
import json
import os
import sys

ACTIVE = (6, 13, 6240, 4160)
BACKGROUND = [0.09, 0.09, 0.09]

RED = [0.20, 0.025, 0.025]
CYAN = [0.025, 0.11, 0.11]
GREEN = [0.03, 0.15, 0.03]
MAGENTA = [0.12, 0.03, 0.12]
BLUE = [0.03, 0.04, 0.22]
YELLOW = [0.16, 0.14, 0.02]
LIGHT = [0.30, 0.30, 0.30]
DARK = [0.02, 0.02, 0.02]
GREY = BACKGROUND

PAIRS = [("red_cyan", RED, CYAN), ("green_magenta", GREEN, MAGENTA), ("blue_yellow", BLUE, YELLOW),
         ("light_dark", LIGHT, DARK), ("red_grey", RED, GREY), ("blue_grey", BLUE, GREY)]
STRIPS = [("red_on_cyan", RED, CYAN), ("blue_on_yellow", BLUE, YELLOW), ("red_on_grey", RED, None)]
SIZES = [24, 36, 48, 72, 96, 144, 192]

BLOCK, GAP, STRIP_H, STRIP_GAP, PITCH = 720, 120, 300, 60, 330


def rect(name, x, y, w, h, rgb, **extra):
    return {"name": name, "x": x, "y": y, "w": w, "h": h, "rgb": list(rgb), **extra}


def layout():
    x0, y0, W, H = ACTIVE
    width = 3 * BLOCK + 2 * GAP
    left = (x0 + W // 2 - width // 2) // 6 * 6
    top = 1008
    q = BLOCK // 2
    patches, blocks = [], []
    for i, (name, a, b) in enumerate(PAIRS):
        bx = left + (i % 3) * (BLOCK + GAP)
        by = top + (i // 3) * (BLOCK + GAP)
        blocks.append({"name": name, "x": bx, "y": by, "size": BLOCK, "a": a, "b": b})
        patches += [rect(f"{name}_a_tl", bx, by, q, q, a), rect(f"{name}_b_tr", bx + q, by, q, q, b),
                    rect(f"{name}_b_bl", bx, by + q, q, q, b), rect(f"{name}_a_br", bx + q, by + q, q, q, a)]
    strips = []
    sy = top + 2 * BLOCK + GAP + GAP
    for name, colour, field in STRIPS:
        if field is not None:
            patches.append(rect(f"{name}_field", left, sy, width, STRIP_H, field))
        cy = sy + STRIP_H // 2
        entries = []
        for k, s in enumerate(SIZES):
            cx = left + PITCH // 2 + k * PITCH
            px, py = (cx - s // 2) // 6 * 6, (cy - s // 2) // 6 * 6
            patches.append(rect(f"{name}_{s}", px, py, s, s, colour, size=s))
            entries.append({"size": s, "x": px, "y": py})
        strips.append({"name": name, "colour": colour, "field": field or BACKGROUND, "y": sy, "patches": entries})
        sy += STRIP_H + STRIP_GAP
    return {"name": "edge_pilot", "active_area": list(ACTIVE), "background": BACKGROUND,
            "description": "Edge pilot: six quadrant blocks of contrasting pairs, and patches of 24 to 192 "
                           "sensor pixels on contrasting fields, on a 0.09 background",
            "blocks": blocks, "strips": strips, "patches": patches}


def geometry():
    """Dark 36-pixel markers every 240 sensor pixels across the whole active area, on a 0.25 grey.

    Rows and columns are symmetric about the centre of the active area, with an extra row or column
    60 pixels in from each edge, so the mapping from sensor to output position can be fitted to the
    corners.
    """
    x0, y0, W, H = ACTIVE
    cx, cy, m = x0 + W // 2, y0 + H // 2, 36
    xs = sorted({cx + k * 240 for k in range(-12, 13)} | {x0 + 60 + m // 2, x0 + W - 60 - m // 2})
    ys = sorted({cy + k * 240 for k in range(-8, 9)} | {y0 + 60 + m // 2, y0 + H - 60 - m // 2})
    patches = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            px, py = (x - m // 2) // 6 * 6, (y - m // 2) // 6 * 6
            patches.append(rect(f"marker_{j:02d}_{i:02d}", px, py, m, m, [0.0, 0.0, 0.0]))
    return {"name": "geometry_grid", "active_area": list(ACTIVE), "background": [0.25, 0.25, 0.25],
            "description": f"Geometry grid: {len(patches)} dark 36-pixel markers every 240 sensor pixels on "
                           "a 0.25 background, to fit where each sensor position lands in the output",
            "patches": patches}


def main(out_dir):
    geo = geometry()
    os.makedirs(out_dir, exist_ok=True)
    json.dump(geo, open(os.path.join(out_dir, "geometry_grid.json"), "w"), indent=1)
    print(f"geometry grid: {len(geo['patches'])} markers")
    lay = layout()
    x0, y0, W, H = ACTIVE
    cx, cy, half_diag = x0 + W / 2, y0 + H / 2, (W ** 2 + H ** 2) ** 0.5 / 2
    r_max = 0.0
    for p in lay["patches"]:
        assert p["x"] % 6 == 0 and p["y"] % 6 == 0 and p["w"] % 6 == 0 and p["h"] % 6 == 0, p
        assert x0 <= p["x"] and p["x"] + p["w"] <= x0 + W and y0 <= p["y"] and p["y"] + p["h"] <= y0 + H, p
        for px, py in [(p["x"], p["y"]), (p["x"] + p["w"], p["y"] + p["h"]), (p["x"], p["y"] + p["h"]),
                       (p["x"] + p["w"], p["y"])]:
            r_max = max(r_max, ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 / half_diag)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(lay, open(os.path.join(out_dir, "edge_pilot.json"), "w"), indent=1)
    print(f"edge pilot: {len(lay['patches'])} rectangles, farthest corner at {r_max:.3f} of the half diagonal")


if __name__ == "__main__":
    main(sys.argv[1])
