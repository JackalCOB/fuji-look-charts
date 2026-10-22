"""Write a synthetic X-T4 chart: a copy of an uncompressed RAF with designed patches in its sensor data.

Only the sensor samples change. The header, metadata, embedded preview and every other byte are kept,
so the camera treats the file as its own. The layout read here is the one LightCraft documents in
crates/raw/src/vendor/raf.rs: big-endian pointers at byte 84 (embedded JPEG, header records, raw
block), the X-Trans layout in header record 0x0131 stored in reverse, and a little-endian TIFF raw
block whose 0xf000 IFD gives width (0xf001), height (0xf002), bits (0xf003), row stride in 32-bit
words (0xf005, when non-zero), strip offset (0xf007), byte count (0xf008), black levels (0xf00a) and
white balance (0xf00e, G R B). Uncompressed files store each sample as a 16-bit little-endian word.

Patch values are given as white-balanced camera RGB, 0 to 1 of the range from black to white. Each
sample is written as black + value * (white - black) / gain, so the camera's own as-shot white
balance returns the designed value.

With --radial=COEF.npy, every sample is also divided by the camera's radial brightening at its
position, gain(r) = 1 + c1 r^2 + c2 r^4 + ..., with r the distance from the centre of the active area
as a fraction of its half diagonal (see the README). The camera's brightening then
returns the designed value everywhere in the frame. The gain was measured against output position,
and the camera's distortion correction moves positions, so with --geometry=GEOMETRY.json each sample's
gain is taken at the output radius it lands on (charts/geometry_map.py).

    python charts/make_chart.py CONTAINER.raf LAYOUT.json OUT.raf [--radial=COEF.npy]
        [--geometry=GEOMETRY.json]
"""
import json
import struct
import sys

import numpy as np

from geometry_map import Geometry

WHITE = 16383


def be32(b, at):
    return struct.unpack_from(">I", b, at)[0]


def records(b):
    off, length = be32(b, 92), be32(b, 96)
    d = b[off:off + length]
    n, p, out = be32(d, 0), 4, {}
    for _ in range(min(n, 4096)):
        tag, size = struct.unpack_from(">HH", d, p)
        out[tag] = d[p + 4:p + 4 + size]
        p += 4 + size
    return out


def raw_ifd(b):
    base, length = be32(b, 100), be32(b, 104)
    raw = b[base:base + length]
    if raw[:4] != b"II*\x00":
        raise SystemExit("raw block is not a little-endian TIFF")
    ifd0 = struct.unpack_from("<I", raw, 4)[0]
    sub = None
    for i in range(struct.unpack_from("<H", raw, ifd0)[0]):
        tag, typ, cnt, val = struct.unpack_from("<HHII", raw, ifd0 + 2 + 12 * i)
        if tag == 0xF000:
            sub = val
    if sub is None:
        raise SystemExit("no 0xf000 raw IFD")
    tags, where = {}, {}
    for i in range(struct.unpack_from("<H", raw, sub)[0]):
        tag, typ, cnt, val = struct.unpack_from("<HHII", raw, sub + 2 + 12 * i)
        if typ in (5, 10):  # RATIONAL, SRATIONAL: pairs of 32-bit numerator and denominator
            fmt = "<" + ("ii" if typ == 10 else "II") * cnt
            pairs = struct.unpack_from(fmt, raw, val)
            tags[tag] = [pairs[2 * k] / pairs[2 * k + 1] if pairs[2 * k + 1] else 0.0 for k in range(cnt)]
            where[tag] = (base + val, typ, cnt)
            continue
        size = 2 if typ == 3 else 4
        if cnt * size <= 4:
            vals = struct.unpack_from("<" + ("H" if size == 2 else "I") * cnt, raw, sub + 2 + 12 * i + 8)
        else:
            vals = struct.unpack_from("<" + ("H" if size == 2 else "I") * cnt, raw, val)
        tags[tag] = list(vals)
    return base, tags, where


def neutralise_lens(b, tags, where):
    """Rewrite the lens-correction tables so they correct nothing, keeping their scale and knots.

    Layouts from LightCraft's docs/fuji-lens-corrections.md: distortion (0xf00b) and vignetting
    (0xf010) are [scale, knot x n, value x n], CA (0xf00f) is [scale, knot x n, red x n, blue x n,
    scale]. Distortion values become 0%, vignetting values 100%, and CA shifts 0.
    """
    changed = []
    for tag, neutral, layout in [(0xF00B, 0, "values"), (0xF010, 100, "values"), (0xF00F, 0, "ca")]:
        if tag not in where:
            continue
        off, typ, cnt = where[tag]
        if layout == "values":
            n = (cnt - 1) // 2
            idx = range(1 + n, 1 + 2 * n)
        else:
            n = (cnt - 2) // 3
            idx = range(1 + n, 1 + 3 * n)
        for k in idx:
            struct.pack_into("<ii", b, off + 8 * k, int(neutral * 100), 100)
        changed.append(f"0x{tag:04x}: {len(idx)} values set to {neutral}")
    return changed


def layout_mask(layout, height, width):
    """Per-pixel white-balanced RGB from the layout: a background and a list of rectangles."""
    img = np.empty((height, width, 3), np.float64)
    img[:] = layout["background"]
    for p in layout["patches"]:
        x, y, w, h = p["x"], p["y"], p["w"], p["h"]
        img[y:y + h, x:x + w] = p["rgb"]
    return img


def radial_gain(coef, height, width, active, geometry=None):
    """The camera's radial brightening at every sensor pixel, from the fitted polynomial in r^2.

    Without a geometry the gain is taken at the sensor radius, as if output positions were a plain
    scaling of sensor positions. With one, it is taken at the output radius the pixel lands on.
    """
    x0, y0, W, H = active
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    r = np.hypot(xx + 0.5 - (x0 + W / 2), yy + 0.5 - (y0 + H / 2)) / np.hypot(W / 2, H / 2)
    if geometry is not None:
        r = geometry.output_radius(r)
    r2 = r ** 2
    g = np.ones_like(r2)
    for k, c in enumerate(coef):
        g += c * r2 ** (k + 1)
    return g


def main(container, layout_path, out_path, *flags):
    b = bytearray(open(container, "rb").read())
    base, t, where = raw_ifd(b)
    if "--show-lens" in flags:
        for tag in (0xF00B, 0xF00F, 0xF010):
            if tag in t:
                print(f"0x{tag:04x} ({len(t[tag])} values): {[round(v, 4) for v in t[tag]]}")
        return
    if "--neutral-lens" in flags:
        for line in neutralise_lens(b, t, where):
            print(line)
    width, height, bits = t[0xF001][0], t[0xF002][0], t[0xF003][0]
    offset, count = base + t[0xF007][0], t[0xF008][0]
    stride = t.get(0xF005, [0])[0] * 4 or width * 2
    if count < stride * (height - 1) + width * 2:
        raise SystemExit("strip too short for 16-bit words: not an uncompressed container")
    rec = records(bytes(b))
    cfa = np.array(list(rec[0x0131]))[::-1].reshape(6, 6)
    black = np.array(t[0xF00A], dtype=np.float64)
    if black.size == 36:
        black_site = black.reshape(6, 6)
    else:
        black_site = np.full((6, 6), black[0])
    g, r, bl = t[0xF00E][:3]
    gain = np.array([r / g, 1.0, bl / g])
    layout = json.load(open(layout_path))
    rgb = layout_mask(layout, height, width)
    yy, xx = np.mgrid[0:height, 0:width]
    colour = cfa[yy % 6, xx % 6]
    blk = black_site[yy % 6, xx % 6]
    val = np.take_along_axis(rgb, colour[..., None], axis=2)[..., 0] / gain[colour]
    radial = [f.split("=", 1)[1] for f in flags if f.startswith("--radial=")]
    geometry = [f.split("=", 1)[1] for f in flags if f.startswith("--geometry=")]
    if radial:
        geo = Geometry(geometry[0]) if geometry else None
        val /= radial_gain(np.load(radial[0]), height, width, layout["active_area"], geo)
    samples = np.clip(np.rint(blk + val * (WHITE - blk)), 0, WHITE).astype("<u2")
    for y in range(height):
        start = offset + y * stride
        b[start:start + width * 2] = samples[y].tobytes()
    open(out_path, "wb").write(b)
    print(f"{width} x {height}, {bits} bits, stride {stride}, black {black_site.mean():.1f}, "
          f"gain R {gain[0]:.4f} B {gain[2]:.4f}, {len(layout['patches'])} patches"
          f"{', radial compensation' if radial else ''}{' at output radius' if radial and geometry else ''}"
          f" -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
