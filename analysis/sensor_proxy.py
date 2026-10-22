"""A small camera-RGB image of a raw, framed like the camera's own conversion of it.

The raw is decoded by LightCraft, through its `rawdump` example (see the README), whose output matched the
sensor sums upstream's corpus test asserts for the public lossless and compressed X-T4 raws. Each
6 x 6 X-Trans cell of the active area is averaged per colour, so every proxy pixel has all three
channels without demosaicing. Black level and the as-shot white balance come from the raw's own IFD,
giving white-balanced camera RGB, 0 to 1 of the range from black to white.

The camera corrects its lens's distortion when it renders, so the proxy is resampled through the
raw's own distortion table (0xf00b), read at the plain radius with the frame-filling zoom, which
matched the geometry measured on the chart grid to within 1.2 pixels at 3,120 x 2,080, out to 0.92 of
the half diagonal.
The camera's radial brightening is applied too, so the proxy carries the same corner gain as the
camera's output. With `radial="table"` it comes from the raw's own vignetting table (0xf010), as the
gain 100 / V at the plain sensor radius: on the chart container that reading matched the measured gain
within 1.2% out to 0.9 of the half diagonal, and it follows whichever lens took the photo. With the
fitted polynomial (`radial=COEF`), it is the chart container's lens, which suits charts only.

    python analysis/sensor_proxy.py RAF OUT.npy [--size 624x416] [--radial COEF.npy|table]
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
from make_chart import raw_ifd, records  # noqa: E402

# LightCraft's rawdump example (crates/raw/examples/rawdump.rs), built with
# `cargo build --release -p lightcraft-raw --example rawdump`
RAWDUMP = os.environ.get("LIGHTCRAFT_RAWDUMP", "rawdump")
ACTIVE = (6, 13, 6240, 4160)
WHITE = 16383


def decode(raf):
    """LightCraft's decode of the raw's sensor samples, as an array (height, width)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dump.bin")
        subprocess.run([RAWDUMP, raf, out], check=True, capture_output=True)
        b = open(out, "rb").read()
    w, h = struct.unpack_from("<II", b, 0)
    return np.frombuffer(b, "<u2", offset=8, count=w * h).reshape(h, w).astype(np.float64)


def lens_tables(b):
    _, t, _ = raw_ifd(b)
    return t


def cell_peak(s, b, t):
    """The brightest sample of each 6 x 6 X-Trans cell, as a fraction of the range from black to white."""
    black = np.array(t[0xF00A], dtype=np.float64)
    black_site = black.reshape(6, 6) if black.size == 36 else np.full((6, 6), black[0])
    x0, y0, W, H = ACTIVE
    rows, cols = H // 6, W // 6
    a = s[y0:y0 + rows * 6, x0:x0 + cols * 6]
    yy, xx = np.mgrid[y0:y0 + rows * 6, x0:x0 + cols * 6]
    v = (a - black_site[yy % 6, xx % 6]) / (WHITE - black_site[yy % 6, xx % 6])
    return v.reshape(rows, 6, cols, 6).max(axis=(1, 3))


def cell_rgb(s, b, t):
    """White-balanced camera RGB per 6 x 6 X-Trans cell of the active area."""
    cfa = np.array(list(records(b)[0x0131]))[::-1].reshape(6, 6)
    black = np.array(t[0xF00A], dtype=np.float64)
    black_site = black.reshape(6, 6) if black.size == 36 else np.full((6, 6), black[0])
    g, r, bl = t[0xF00E][:3]
    gain = np.array([r / g, 1.0, bl / g])
    x0, y0, W, H = ACTIVE
    rows, cols = H // 6, W // 6
    a = s[y0:y0 + rows * 6, x0:x0 + cols * 6]
    # cells aligned to the CFA's period: x0 and y0 are not multiples of 6, so index the pattern by
    # absolute position
    yy, xx = np.mgrid[y0:y0 + rows * 6, x0:x0 + cols * 6]
    colour = cfa[yy % 6, xx % 6]
    blk = black_site[yy % 6, xx % 6]
    v = (a - blk) / (WHITE - blk)
    out = np.zeros((rows, cols, 3))
    for ch in range(3):
        m = (colour == ch).astype(np.float64)
        num = (v * m).reshape(rows, 6, cols, 6).sum(axis=(1, 3))
        den = m.reshape(rows, 6, cols, 6).sum(axis=(1, 3))
        out[..., ch] = num / den * gain[ch]
    return out


def vignetting(t):
    """The raw's vignetting table as knots and percent illumination, anchored at 100% on the axis."""
    v = t.get(0xF010)
    if not v or len(v) < 3:
        return None
    n = (len(v) - 1) // 2
    knots, vals = np.array(v[1:1 + n]), np.array(v[1 + n:1 + 2 * n])
    if knots[0] > 0:
        knots, vals = np.r_[0.0, knots], np.r_[100.0, vals]
    if not np.all(vals > 0):
        return None
    return knots, vals


def distortion(t):
    """The raw's distortion table as knots and percent, anchored at zero."""
    d = t.get(0xF00B)
    if not d or len(d) < 3:
        return None
    n = (len(d) - 1) // 2
    knots, vals = np.array(d[1:1 + n]), np.array(d[1 + n:1 + 2 * n])
    if knots[0] > 0:
        knots, vals = np.r_[0.0, knots], np.r_[0.0, vals]
    return knots, vals


def source_radius(r, z, knots, vals):
    s = z * r
    for _ in range(40):
        s = z * r * (1 + np.interp(s, knots, vals) / 100)
    return s


def zoom(knots, vals, W, H):
    half = np.hypot(W / 2, H / 2)
    ang = np.linspace(1e-4, np.pi / 2, 400)
    rb = np.minimum((W / 2) / np.cos(ang), (H / 2) / np.sin(ang)) / half
    z = 1.0
    for _ in range(40):
        z = 1 / max(1 + np.interp(source_radius(r, z, knots, vals), knots, vals) / 100 for r in rb)
    return z


def resample(cells, size, table, radial=None, vig=None):
    """Sample the cell image on the output grid, through the distortion and the radial gain: the fitted
    polynomial at the output radius (`radial`), or the raw's own vignetting table at the source radius
    (`vig`)."""
    rows, cols = cells.shape[:2]
    ow, oh = size
    W, H = cols * 6.0, rows * 6.0
    half = np.hypot(W / 2, H / 2)
    v, u = np.mgrid[0:oh, 0:ow]
    dx = ((u + 0.5) / ow - 0.5) * W
    dy = ((v + 0.5) / oh - 0.5) * H
    r = np.hypot(dx, dy) / half
    if table is not None:
        knots, vals = table
        z = zoom(knots, vals, W, H)
        f = np.where(r > 0, source_radius(r, z, knots, vals) / np.maximum(r, 1e-9), z)
    else:
        f = np.ones_like(r)
    sx = (W / 2 + dx * f) / 6.0 - 0.5
    sy = (H / 2 + dy * f) / 6.0 - 0.5
    x0 = np.clip(np.floor(sx).astype(int), 0, cols - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, rows - 2)
    fx, fy = np.clip(sx - x0, 0, 1)[..., None], np.clip(sy - y0, 0, 1)[..., None]
    img = (cells[y0, x0] * (1 - fx) * (1 - fy) + cells[y0, x0 + 1] * fx * (1 - fy)
           + cells[y0 + 1, x0] * (1 - fx) * fy + cells[y0 + 1, x0 + 1] * fx * fy)
    if vig is not None:
        knots, vals = vig
        img = img * (100.0 / np.interp(r * f, knots, vals))[..., None]
    elif radial is not None:
        r2 = r ** 2
        g = 1 + sum(c * r2 ** (k + 1) for k, c in enumerate(radial))
        img = img * g[..., None]
    return img


def proxy(raf, size=(624, 416), radial=None, geometry=True, clip=None):
    """`radial` is the fitted polynomial, or "table" for the raw's own vignetting table.

    With `clip` (a fraction of white), also returns a mask of the frame's pixels that draw on any cell
    whose brightest sample reaches it, mapped through the same geometry.
    """
    b = open(raf, "rb").read()
    t = lens_tables(b)
    s = decode(raf)
    cells = cell_rgb(s, b, t)
    table = distortion(t) if geometry else None
    if isinstance(radial, str) and radial == "table":
        vig = vignetting(t)
        if vig is None:
            raise ValueError(f"{os.path.basename(raf)}: no usable vignetting table")
        img = resample(cells, size, table, None, vig)
    else:
        img = resample(cells, size, table, radial)
    if clip is None:
        return img
    flag = (cell_peak(s, b, t) >= clip).astype(np.float64)
    mask = resample(np.repeat(flag[..., None], 3, axis=2), size, table)[..., 0] > 0
    return img, mask


def main():
    args = sys.argv[1:]
    size, radial = (624, 416), None
    if "--size" in args:
        i = args.index("--size")
        size = tuple(int(v) for v in args[i + 1].split("x"))
        del args[i:i + 2]
    if "--radial" in args:
        i = args.index("--radial")
        radial = "table" if args[i + 1] == "table" else np.load(args[i + 1])
        del args[i:i + 2]
    img = proxy(args[0], size, radial)
    np.save(args[1], img.astype(np.float32))
    print(f"{os.path.basename(args[0])}: proxy {size[0]} x {size[1]}, camera RGB mean {img.reshape(-1, 3).mean(0).round(4)}")


if __name__ == "__main__":
    main()
