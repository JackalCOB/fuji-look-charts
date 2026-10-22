"""Check a chart raw two ways: LightCraft decodes exactly what was stored, and what was stored is the design.

1. The sensor samples stored in the chart's strip are compared with LightCraft's decode of the file,
   written by the clone's `rawdump` example (u32 width, u32 height, then u16 samples). They must be
   identical.
2. Each patch's stored samples are turned back into white-balanced camera RGB, undoing the black level,
   white balance, and any radial compensation, and averaged per channel over the patch. The result must
   match the patch's design to within the rounding of a 14-bit sample.

    python charts/verify_chart.py CHART.raf LAYOUT.json DUMP.bin [--radial=COEF.npy]
        [--geometry=GEOMETRY.json]
"""
import json
import struct
import sys

import numpy as np

from geometry_map import Geometry
from make_chart import WHITE, radial_gain, raw_ifd, records


def stored(b):
    base, t, _ = raw_ifd(b)
    width, height = t[0xF001][0], t[0xF002][0]
    offset = base + t[0xF007][0]
    stride = t.get(0xF005, [0])[0] * 4 or width * 2
    rows = [np.frombuffer(b, "<u2", count=width, offset=offset + y * stride) for y in range(height)]
    return np.stack(rows), t


def main(chart, layout_path, dump, *flags):
    b = open(chart, "rb").read()
    s, t = stored(b)
    d = open(dump, "rb").read()
    w, h = struct.unpack_from("<II", d, 0)
    dec = np.frombuffer(d, "<u2", offset=8, count=w * h).reshape(h, w)
    if dec.shape != s.shape:
        print(f"decode shape {dec.shape} differs from stored {s.shape}")
        sys.exit(1)
    diff = np.abs(dec.astype(np.int32) - s.astype(np.int32))
    print(f"decode vs stored: {s.shape[1]} x {s.shape[0]}, largest difference {diff.max()}, "
          f"samples differing {(diff > 0).sum()}")
    layout = json.load(open(layout_path))
    cfa = np.array(list(records(b)[0x0131]))[::-1].reshape(6, 6)
    black = np.array(t[0xF00A], dtype=np.float64)
    black_site = black.reshape(6, 6) if black.size == 36 else np.full((6, 6), black[0])
    g, r, bl = t[0xF00E][:3]
    gain = np.array([r / g, 1.0, bl / g])
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    colour = cfa[yy % 6, xx % 6]
    blk = black_site[yy % 6, xx % 6]
    val = (s - blk) / (WHITE - blk) * gain[colour]
    radial = [f.split("=", 1)[1] for f in flags if f.startswith("--radial=")]
    geometry = [f.split("=", 1)[1] for f in flags if f.startswith("--geometry=")]
    if radial:
        val *= radial_gain(np.load(radial[0]), s.shape[0], s.shape[1], layout["active_area"],
                           Geometry(geometry[0]) if geometry else None)
    worst, worst_rel = 0.0, 0.0
    for p in layout["patches"]:
        x, y, pw, ph = p["x"], p["y"], p["w"], p["h"]
        v, c = val[y:y + ph, x:x + pw], colour[y:y + ph, x:x + pw]
        for ch in range(3):
            m = v[c == ch].mean()
            err = abs(m - p["rgb"][ch])
            worst = max(worst, err)
            worst_rel = max(worst_rel, err / max(p["rgb"][ch], 1e-4))
    step = gain.max() / (WHITE - blk.mean())
    print(f"stored vs design, {len(layout['patches'])} patches: largest error {worst:.2e} "
          f"(one sample step is up to {step:.2e}), largest relative error {worst_rel:.2%}")
    ok = diff.max() == 0 and worst <= step
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:])
