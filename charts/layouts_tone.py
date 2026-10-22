"""Write the tone-ladder and flat-field chart layouts.

The ladder: 41 neutral patches, full scale down in third-stop steps across 13 stops plus one at zero,
shuffled with a recorded seed onto a 7 x 6 grid inside the central third of the frame, where the
camera's radial brightening is smallest. Shuffling keeps a patch's level independent of its
position. The background is a neutral 0.05.

The flat fields: whole-frame neutral greys at six levels, with no patches, to measure the radial
brightening at every position and level.

Values are white-balanced camera RGB, 0 to 1 of the range from black to white, as make_chart.py
expects. Every patch sits on a 6-pixel boundary inside the active area.

    python charts/layouts_tone.py charts/layouts
"""
import json
import os
import random
import sys

ACTIVE = (6, 13, 6240, 4160)
SEED = 20261013
FLAT_LEVELS = [0.005, 0.015, 0.04, 0.1, 0.25, 0.5]


def snap(v):
    return v // 6 * 6


def ladder():
    x0, y0, W, H = ACTIVE
    levels = [round(2 ** (-k / 3), 6) for k in range(40)] + [0.0]
    order = list(range(len(levels)))
    random.Random(SEED).shuffle(order)
    cols, rows, size = 7, 6, 240
    span_x, span_y = 2200, 1600
    cell_x, cell_y = span_x // cols, span_y // rows
    left = x0 + W // 2 - span_x // 2
    top = y0 + H // 2 - span_y // 2
    patches = []
    for slot, li in enumerate(order):
        c, r = slot % cols, slot // cols
        x = snap(left + c * cell_x + (cell_x - size) // 2)
        y = snap(top + r * cell_y + (cell_y - size) // 2)
        v = levels[li]
        patches.append({"name": f"level_{li:02d}", "level": v, "x": x, "y": y, "w": size, "h": size,
                        "rgb": [v, v, v]})
    return {"name": "tone_ladder", "seed": SEED, "active_area": list(ACTIVE), "background": [0.05, 0.05, 0.05],
            "description": "Tone ladder: 41 neutral patches from full scale down 13 stops in thirds, and zero, "
                           "shuffled inside the central third on a 0.05 background", "patches": patches}


def flat(level):
    return {"name": f"flat_{level:g}", "active_area": list(ACTIVE), "background": [level, level, level],
            "description": f"Flat field: the whole frame at {level:g} of full scale", "patches": []}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lay = ladder()
    x0, y0, W, H = ACTIVE
    for p in lay["patches"]:
        assert p["x"] % 6 == 0 and p["y"] % 6 == 0, p
        assert x0 <= p["x"] and p["x"] + p["w"] <= x0 + W and y0 <= p["y"] and p["y"] + p["h"] <= y0 + H, p
    json.dump(lay, open(os.path.join(out_dir, "tone_ladder.json"), "w"), indent=1)
    for level in FLAT_LEVELS:
        json.dump(flat(level), open(os.path.join(out_dir, f"flat_{level:g}.json"), "w"), indent=1)
    cx, cy, half_diag = x0 + W / 2, y0 + H / 2, (W ** 2 + H ** 2) ** 0.5 / 2
    r_max = max(((p["x"] + p["w"] / 2 - cx) ** 2 + (p["y"] + p["h"] / 2 - cy) ** 2) ** 0.5 for p in lay["patches"])
    print(f"ladder: {len(lay['patches'])} patches, farthest centre at {r_max / half_diag:.3f} of the half diagonal")
    print(f"flat fields: {len(FLAT_LEVELS)} levels {FLAT_LEVELS}")


if __name__ == "__main__":
    main(sys.argv[1])
