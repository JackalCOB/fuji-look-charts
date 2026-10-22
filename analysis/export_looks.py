"""Write the fitted lookups (fit_lut.py) as LightCraft's measured-look asset, assets/camera-looks/X-T4.json.

One file per camera model holds every look. Each look's grid nodes are its Adobe RGB encoded output (gamma 563/256)
at CIELAB nodes of white-balanced camera RGB through the camera's forward matrix: 17 lightness nodes from 0 to 100
and 29 a* and b* nodes from -168 to 168, index (l * 29 + a) * 29 + b. Values are stored as unsigned 16-bit numbers,
value = n * NODE_STEP + NODE_OFFSET, written as 12 hexadecimal digits per node (three channels, big-endian). The
nodes reach about -2.8 and 3.3 where no colour lands (the fit's smoothness carries them out), so the range is -4 to 4
and a step is 1/8192, a thirtieth of an 8-bit level.

The script checks the round trip: every stored node within half a step of the fitted one.

    python analysis/export_looks.py FITS_DIR OUT_JSON
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
import colour_space as cs  # noqa: E402

NODE_OFFSET = -4.0
NODE_STEP = 1.0 / 8192
LOOKS = [  # (fit name, id, display name, film simulation as the camera names it)
    ("provia", "provia", "Standard (PROVIA)", "PROVIA/STANDARD"),
    ("velvia", "velvia", "Vivid (Velvia)", "Velvia/VIVID"),
    ("astia", "astia", "Soft (ASTIA)", "ASTIA/SOFT"),
    ("classic_chrome", "classic-chrome", "Muted (Classic Chrome)", "CLASSIC CHROME"),
    ("pro_neg_hi", "pro-neg-hi", "Portrait Hi (PRO Neg. Hi)", "PRO Neg. Hi"),
    ("pro_neg_std", "pro-neg-std", "Portrait (PRO Neg. Std)", "PRO Neg. Std"),
    ("eterna", "eterna", "Cinema (ETERNA)", "ETERNA/CINEMA"),
    ("classic_neg", "classic-neg", "Negative (Classic Neg.)", "CLASSIC Neg."),
    ("eterna_bleach_bypass", "eterna-bleach-bypass", "Bleach Bypass (ETERNA Bleach Bypass)", "ETERNA BLEACH BYPASS"),
]


def encode(nodes):
    q = np.round((nodes - NODE_OFFSET) / NODE_STEP)
    if q.min() < 0 or q.max() > 65535:
        raise SystemExit(f"a node is outside {NODE_OFFSET} to {NODE_OFFSET + 65535 * NODE_STEP}")
    q = q.astype(np.uint16)
    back = q.astype(np.float64) * NODE_STEP + NODE_OFFSET
    err = np.abs(back - nodes).max()
    if err > NODE_STEP / 2 + 1e-12:
        raise SystemExit(f"round trip off by {err}")
    return q.astype(">u2").tobytes().hex(), err


def main(fits, out):
    grid_l = grid_ab = None
    looks = []
    for fit, ident, name, camera_name in LOOKS:
        z = np.load(os.path.join(fits, f"lut_{fit}.npz"))
        if grid_l is None:
            grid_l, grid_ab = z["grid_l"], z["grid_ab"]
        if not (np.array_equal(z["grid_l"], grid_l) and np.array_equal(z["grid_ab"], grid_ab)):
            raise SystemExit(f"{fit}: a different grid")
        nodes = z["nodes"]
        if nodes.shape != (len(grid_l) * len(grid_ab) ** 2, 3):
            raise SystemExit(f"{fit}: {nodes.shape} nodes")
        hexed, err = encode(nodes)
        print(f"{fit:22s} nodes {nodes.min():+.3f} to {nodes.max():+.3f}, round trip within {err:.2e}")
        looks.append({
            "id": ident,
            "name": name,
            "description": f"Measured from the X-T4's {camera_name} film simulation. Not made or endorsed by Fujifilm.",
            "nodes": hexed,
        })
    doc = {
        "version": 1,
        "model": "X-T4",
        "group": "X-T4 looks, measured",
        "input": {
            "forward_matrix": cs.FORWARD_D65.tolist(),
            "white": cs.D50.tolist(),
            "lightness": [float(grid_l[0]), float(grid_l[-1]), len(grid_l)],
            "ab": [float(grid_ab[0]), float(grid_ab[-1]), len(grid_ab)],
        },
        "output": "Adobe RGB (1998), encoded with gamma 563/256",
        "node_offset": NODE_OFFSET,
        "node_step": NODE_STEP,
        "looks": looks,
    }
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {out}: {os.path.getsize(out):,} bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
