"""Write the colour chart layouts: the training set and the validation set, from separate seeds.

Every chart is a grid of abutting 120-pixel patches (20 X-Trans cells a side) on a 0.09 grey, covering
the usable frame: one cell in from each edge of the active area, centres within 0.9 of the half
diagonal, and wholly inside the camera's output once its distortion correction has moved them. The
edge pilot of 14 October 2026 found that the camera's processing settles within about 10 output
pixels of an edge, so a 120-pixel patch, about 58 output pixels, leaves a clean centre to read.

Colours are chosen in CIELAB and written as white-balanced camera RGB (colour_space.py). A colour is
a lightness level, given as camera luminance Y in stops, a hue angle, and a chroma as a fraction of
the most the gamut allows at that lightness and hue.

**Anchors.** Six colours at eight cells each, the same cells in every chart of both sets: three greys
and three mid-level colours, spread over eight sectors and six radial bands. They show whether a
colour renders the same in every chart, which it would not if the camera adapted to a chart's
content, and whether any effect of position remains after compensation.

**Training** (seed 20261014). A regular grid: 18 levels from 2^-9 to 2^-0.5 in half stops, and at each
36 hues every 10 degrees at six chroma fractions, plus the neutral: 3,906 colours. Each is placed
twice, in different charts, at random cells, so it meets different neighbours and positions. Spare
cells take third copies of random grid colours.

**Validation** (seed 20261015). Colours the training never saw, and layouts it never saw:
- `random` – continuous random levels, hues, and chroma fractions, off the training grid, with 40
  colours each placed at eight cells across the two charts for the consistency check
- `reordered` – training grid colours in new cells beside new neighbours
- `natural` – colours drawn from chosen ranges for skin, foliage, sky, earth, and near-neutrals
- `low_key` and `high_key` – charts of only dark or only bright colours, to test whether the camera
  adapts tone to a chart's overall brightness

Writes one layout per chart and a catalogue CSV per set with every patch's design.

    python charts/layouts_colour.py DATA/charts/colour/layouts DATA/results/geometry/geometry_fit.json

The layouts run to about 11 MB, so they live with the charts in your data folder rather than in this repository.
The seeds reproduce them exactly.
"""
import csv
import json
import math
import os
import random
import sys

import numpy as np

import colour_space as cs
from geometry_map import Geometry

ACTIVE = (6, 13, 6240, 4160)
CELL = 120
ORIGIN = (6, 54)
BACKGROUND = 0.09
R_MAX = 0.9
OUT_W, OUT_H, OUT_SCALE, OUT_MARGIN = 3120, 2080, 0.5, 4
SENSOR_MAX = 0.75

TRAIN_SEED = 20261014
VALID_SEED = 20261015
ANCHOR_SEED = 20261016

LEVELS = [2 ** (-9 + 0.5 * k) for k in range(18)]
HUES = [5 + 10 * k for k in range(36)]
FRACTIONS = [0.16, 0.32, 0.48, 0.64, 0.8, 0.95]
TRAIN_CHARTS = 6

ANCHORS = [("grey_dark", 0.0225, 0, 0.0), ("grey_mid", 0.09, 0, 0.0), ("grey_light", 0.36, 0, 0.0),
           ("red", 0.09, 40, 0.6), ("green", 0.09, 140, 0.6), ("blue", 0.09, 280, 0.6)]

# Chosen ranges for natural colours: hue degrees, chroma in CIELAB units, levels in stops of Y.
NATURAL = {"skin": ((25, 65), (12, 38), (-4.5, -1.5)), "foliage": ((95, 150), (15, 55), (-6.5, -2.5)),
           "sky": ((220, 270), (10, 45), (-3.5, -1.0)), "earth": ((45, 90), (8, 40), (-6.0, -2.0)),
           "near_neutral": ((0, 360), (0, 6), (-8.0, -0.8))}


def cells(geo):
    """Usable cells as (x, y, r): whole, a cell in from each edge, r <= R_MAX, inside the output."""
    x0, y0, W, H = ACTIVE
    out = []
    cols = (x0 + W - ORIGIN[0]) // CELL
    rows = (y0 + H - ORIGIN[1]) // CELL
    for j in range(1, rows - 1):
        for i in range(1, cols - 1):
            x, y = ORIGIN[0] + i * CELL, ORIGIN[1] + j * CELL
            cx, cy = x + CELL / 2, y + CELL / 2
            r = math.hypot(cx - geo.c_s[0], cy - geo.c_s[1]) / geo.half
            if r > R_MAX:
                continue
            ox, oy = geo.to_output(np.array([x, x + CELL, x, x + CELL]), np.array([y, y, y + CELL, y + CELL]),
                                   OUT_SCALE)
            if ox.min() < OUT_MARGIN or oy.min() < OUT_MARGIN or ox.max() > OUT_W - OUT_MARGIN \
                    or oy.max() > OUT_H - OUT_MARGIN:
                continue
            out.append((x, y, r, math.degrees(math.atan2(cy - geo.c_s[1], cx - geo.c_s[0])) % 360))
    return out


def colour(level, hue, fraction, chroma=None):
    L = float(cs.lightness(level))
    if chroma is None:
        cmax = float(cs.max_chroma(np.array([L]), np.array([hue]), SENSOR_MAX)[0]) if fraction > 0 else 0.0
        chroma = fraction * cmax
    else:
        cmax = float(cs.max_chroma(np.array([L]), np.array([hue]), SENSOR_MAX)[0])
        chroma = min(chroma, 0.95 * cmax)
        fraction = chroma / cmax if cmax > 0 else 0.0
    cam = cs.lch_to_cam(np.array([L]), np.array([chroma]), np.array([hue]))[0]
    lab = cs.cam_to_lab(cam)
    return {"level": level, "stops": math.log2(level), "hue": hue, "fraction": fraction, "chroma": chroma,
            "lab": lab.tolist(), "rgb": [float(v) for v in np.clip(cam, 0, None)]}


def anchor_cells(usable):
    """48 cells: one in each of 8 sectors x 6 radial bands, colour a taking band (sector + a) mod 6.

    The frame is wider than tall, so a sector's reach in radius depends on its angle. Bands are equal
    shares of each sector's cells in order of radius, from the centre to the sector's edge.
    """
    rng = random.Random(ANCHOR_SEED)
    chosen = {}
    for s in range(8):
        sector = sorted((c for c in usable if s * 45 <= c[3] < (s + 1) * 45 and c[2] >= 0.08), key=lambda c: c[2])
        for b in range(6):
            pool = sector[b * len(sector) // 6:(b + 1) * len(sector) // 6]
            chosen[(s, b)] = rng.choice(pool)
    out = []
    for a, (name, level, hue, frac) in enumerate(ANCHORS):
        col = colour(level, hue, frac)
        for s in range(8):
            out.append((chosen[(s, (s + a) % 6)], f"anchor_{name}_s{s}", col, name))
    return out


def patch(name, cell, col, **meta):
    x, y = cell[0], cell[1]
    return {"name": name, "x": x, "y": y, "w": CELL, "h": CELL, "rgb": col["rgb"], "r": round(cell[2], 4),
            **{k: col[k] for k in ("level", "stops", "hue", "fraction", "chroma", "lab")}, **meta}


def fill(name, seed, free, entries, anchors, description, extra=None):
    """Place entries (name, colour, meta) into free cells in a seeded random order."""
    rng = random.Random(seed)
    order = list(free)
    rng.shuffle(order)
    if len(entries) > len(order):
        raise SystemExit(f"{name}: {len(entries)} patches for {len(order)} cells")
    patches = [patch(n, c, col, kind="anchor", anchor=a) for c, n, col, a in anchors]
    for cell, (pname, col, meta) in zip(order, entries):
        patches.append(patch(pname, cell, col, **meta))
    return {"name": name, "seed": seed, "active_area": list(ACTIVE), "background": [BACKGROUND] * 3,
            "cell": CELL, "description": description, **(extra or {}), "patches": patches}


def training(free, anchors):
    grid = []
    for li, level in enumerate(LEVELS):
        grid.append((f"g_l{li:02d}_n", colour(level, 0, 0.0), {"kind": "grid", "level_index": li}))
        for hue in HUES:
            for fi, f in enumerate(FRACTIONS):
                grid.append((f"g_l{li:02d}_h{hue:03d}_f{fi}", colour(level, hue, f),
                             {"kind": "grid", "level_index": li, "fraction_index": fi}))
    rng = random.Random(TRAIN_SEED)
    n, cap = TRAIN_CHARTS, len(free)
    first = list(range(len(grid)))
    rng.shuffle(first)
    charts = [[] for _ in range(n)]
    home = {}
    for k, j in enumerate(first):
        home[j] = k % n
        charts[k % n].append((j, 1))
    for k, j in enumerate(first):
        charts[(home[j] + 1 + k % (n - 1)) % n].append((j, 2))
    for c in range(n):
        spare = cap - len(charts[c])
        have = {j for j, _ in charts[c]}
        pool = [j for j in range(len(grid)) if j not in have]
        for j in rng.sample(pool, spare):
            charts[c].append((j, 3))
    layouts = []
    for c in range(n):
        entries = [(f"{grid[j][0]}_c{copy}", grid[j][1], {**grid[j][2], "colour_id": grid[j][0], "copy": copy})
                   for j, copy in charts[c]]
        layouts.append(fill(f"train_{c + 1:02d}", TRAIN_SEED + c + 1, free, entries, anchors,
                            f"Training chart {c + 1} of {n}: the regular grid, each colour twice across the set"))
    return layouts, len(grid)


def rand_colour(rng, lo=-9.0, hi=-0.5):
    level = 2 ** rng.uniform(lo, hi)
    if rng.random() < 0.03:
        return colour(level, 0, 0.0)
    return colour(level, rng.uniform(0, 360), rng.uniform(0.0, 0.95))


def validation(free, anchors, grid_count):
    rng = random.Random(VALID_SEED)
    cap = len(free)
    layouts = []
    repeats = [(f"v_rep{k:02d}", rand_colour(rng)) for k in range(40)]
    for c in range(2):
        entries = []
        for k, (rid, col) in enumerate(repeats):
            for copy in range(4):
                entries.append((f"{rid}_c{c * 4 + copy + 1}", col, {"kind": "repeat", "colour_id": rid}))
        while len(entries) < cap:
            entries.append((f"v_rand{c + 1}_{len(entries):04d}", rand_colour(rng), {"kind": "random"}))
        layouts.append(fill(f"valid_random_{c + 1:02d}", VALID_SEED + 10 + c, free, entries, anchors,
                            "Validation: random colours off the training grid, and 40 colours repeated at "
                            "eight cells across the two random charts"))
    grid_ids = rng.sample(range(grid_count), cap)
    entries = []
    for j in grid_ids:
        li, rest = divmod(j, 1 + len(HUES) * len(FRACTIONS))
        level = LEVELS[li]
        if rest == 0:
            gid, col = f"g_l{li:02d}_n", colour(level, 0, 0.0)
        else:
            hi, fi = divmod(rest - 1, len(FRACTIONS))
            gid, col = f"g_l{li:02d}_h{HUES[hi]:03d}_f{fi}", colour(level, HUES[hi], FRACTIONS[fi])
        entries.append((f"{gid}_v", col, {"kind": "reordered", "colour_id": gid}))
    layouts.append(fill("valid_reordered", VALID_SEED + 20, free, entries, anchors,
                        "Validation: training grid colours in new cells beside new neighbours"))
    entries = []
    families = list(NATURAL)
    while len(entries) < cap:
        fam = families[len(entries) % len(families)]
        (h0, h1), (c0, c1), (s0, s1) = NATURAL[fam]
        col = colour(2 ** rng.uniform(s0, s1), rng.uniform(h0, h1) % 360, 0.0, chroma=rng.uniform(c0, c1))
        entries.append((f"v_nat_{fam}_{len(entries):04d}", col, {"kind": "natural", "family": fam}))
    layouts.append(fill("valid_natural", VALID_SEED + 30, free, entries, anchors,
                        "Validation: colours from chosen ranges for skin, foliage, sky, earth, and near-neutrals"))
    for name, lo, hi in [("low_key", -9.0, -5.0), ("high_key", -3.5, -0.5)]:
        entries = [(f"v_{name}_{k:04d}", rand_colour(rng, lo, hi), {"kind": name}) for k in range(cap)]
        layouts.append(fill(f"valid_{name}", VALID_SEED + (40 if name == "low_key" else 50), free, entries,
                            anchors, f"Validation: only {'dark' if name == 'low_key' else 'bright'} colours, "
                                     f"Y from 2^{lo:g} to 2^{hi:g}"))
    return layouts


def catalogue(path, layouts):
    cols = ["chart", "name", "kind", "colour_id", "copy", "anchor", "family", "x", "y", "w", "h", "r", "level",
            "stops", "hue", "fraction", "chroma", "lab_l", "lab_a", "lab_b", "cam_r", "cam_g", "cam_b"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for lay in layouts:
            for p in lay["patches"]:
                wr.writerow([lay["name"], p["name"], p.get("kind", ""), p.get("colour_id", ""), p.get("copy", ""),
                             p.get("anchor", ""), p.get("family", ""), p["x"], p["y"], p["w"], p["h"], p["r"],
                             f"{p['level']:.6f}", f"{p['stops']:.4f}", f"{p['hue']:.3f}", f"{p['fraction']:.4f}",
                             f"{p['chroma']:.3f}", *[f"{v:.3f}" for v in p["lab"]],
                             *[f"{v:.6f}" for v in p["rgb"]]])


def main(out_dir, geometry_path):
    geo = Geometry(geometry_path)
    usable = cells(geo)
    anchors = anchor_cells(usable)
    taken = {(c[0], c[1]) for c, _, _, _ in anchors}
    free = [c for c in usable if (c[0], c[1]) not in taken]
    train, grid_count = training(free, anchors)
    valid = validation(free, anchors, grid_count)
    os.makedirs(out_dir, exist_ok=True)
    for lay in train + valid:
        for p in lay["patches"]:
            assert p["x"] % 6 == 0 and p["y"] % 6 == 0, p
        json.dump(lay, open(os.path.join(out_dir, f"{lay['name']}.json"), "w"), indent=0)
    catalogue(os.path.join(out_dir, "catalogue_training.csv"), train)
    catalogue(os.path.join(out_dir, "catalogue_validation.csv"), valid)
    print(f"usable cells {len(usable)}, anchors {len(anchors)}, free {len(free)} per chart")
    print(f"training: {len(train)} charts, {grid_count} grid colours, "
          f"{sum(len(l['patches']) for l in train)} patches with anchors")
    for lay in valid:
        print(f"validation {lay['name']}: {len(lay['patches'])} patches")
    rgb = np.array([p["rgb"] for lay in train + valid for p in lay["patches"]])
    print(f"camera RGB range: min {rgb.min(axis=0).round(5).tolist()}, max {rgb.max(axis=0).round(4).tolist()}, "
          f"largest sensor fraction {(rgb / cs.WB_GAIN).max():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
