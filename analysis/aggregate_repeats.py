"""Combine each set of repeat conversions into one image, and check the repeats against each other.

For every raw and look under a conversions folder, reads <look>/<stem>__<look>__r1..rN.jpg and writes
_aggregate/<look>/<stem>__<look>.png: the per-pixel trimmed mean of the repeats (the highest and
lowest value at each pixel dropped, the rest averaged), as 16-bit PNG scaled by 257 so 8-bit levels
map exactly onto the 16-bit range and the sub-level precision the averaging gains is kept.

Writes a QC table, _aggregate/qc.csv: per raw and look, each repeat's mean absolute deviation from the
trimmed mean, and a flag on any repeat more than three times the median deviation of its set.
Prints aggregate figures only. Images are never displayed.

    python analysis/aggregate_repeats.py DATA/conversions/charts
"""
import collections
import csv
import glob
import os
import re
import sys

import numpy as np
from PIL import Image

NAME = re.compile(r"^(?P<stem>.+)__(?P<look>[a-z_]+)__r(?P<r>\d+)\.jpg$")


def trimmed_mean(stack):
    """Per-pixel mean after dropping the highest and lowest of each pixel's values."""
    if stack.shape[0] < 3:
        return stack.mean(axis=0)
    s = np.sort(stack, axis=0)
    return s[1:-1].mean(axis=0)


def _png16(path, img):
    """Write a float image in 8-bit levels as 16-bit RGB PNG, scaled by 257.

    Pillow has no 16-bit RGB mode, so the PNG is encoded here: each row uses PNG filter type 1 (Sub),
    with 6 bytes per pixel, and the whole image is zlib-compressed into one IDAT chunk.
    """
    import struct
    import zlib
    arr = np.clip(np.rint(img * 257.0), 0, 65535).astype(np.uint16)
    h, w, c = arr.shape
    big = arr.astype(">u2")
    raw = bytearray()
    row_bytes = w * c * 2
    flat = big.reshape(h, row_bytes // 2).view(np.uint8).reshape(h, row_bytes)
    for y in range(h):
        row = flat[y]
        sub = row.copy()
        sub[6:] = (row[6:].astype(np.int16) - row[:-6].astype(np.int16)).astype(np.uint8)
        raw.append(1)
        raw += sub.tobytes()

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")
    with open(path + ".part", "wb") as f:
        f.write(png)
    os.replace(path + ".part", path)


def read_png16(path):
    """Read a 16-bit RGB PNG written by _png16, returning float32 in 8-bit levels.

    Only filter types 0 and 1 are handled, which is all _png16 writes. Pillow cannot be used, because
    it reduces 16-bit RGB to 8 bits on loading.
    """
    import struct
    import zlib
    data = open(path, "rb").read()
    pos, idat, w = 8, b"", None
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
            if (depth, ctype) != (16, 2):
                raise ValueError(f"{path}: not 16-bit RGB")
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 6 + 1)
    rows = raw[:, 1:].reshape(h, w, 6).astype(np.uint16)
    sub = raw[:, 0] == 1
    if not np.all((raw[:, 0] == 0) | sub):
        raise ValueError(f"{path}: unsupported PNG filter")
    rows[sub] = np.cumsum(rows[sub], axis=1) % 256
    vals = rows.astype(np.uint8).reshape(h, w * 6).view(">u2").reshape(h, w, 3)
    return vals.astype(np.float32) / 257.0


def main(root, looks=()):
    sets = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(root, "*", "*.jpg")):
        m = NAME.match(os.path.basename(f))
        if m and (not looks or m["look"] in looks):
            sets[(m["stem"], m["look"])][int(m["r"])] = f
    out_root = os.path.join(root, "_aggregate")
    qc_rows, flagged, made, skipped = [], [], 0, 0
    devs_all = []
    for (stem, look), reps in sorted(sets.items()):
        out_dir = os.path.join(out_root, look)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{stem}__{look}.png")
        stack = np.stack([np.asarray(Image.open(reps[r]).convert("RGB")).astype(np.float32)
                          for r in sorted(reps)])
        agg = trimmed_mean(stack)
        devs = [float(np.abs(stack[i] - agg).mean()) for i in range(stack.shape[0])]
        med = float(np.median(devs))
        for r, d in zip(sorted(reps), devs):
            flag = med > 0 and d > 3 * med
            qc_rows.append({"stem": stem, "look": look, "repeat": r, "mean_abs_dev": f"{d:.4f}",
                            "flag": "yes" if flag else ""})
            if flag:
                flagged.append(f"{stem} {look} r{r} ({d:.3f} against median {med:.3f})")
        devs_all += devs
        if os.path.exists(out):
            skipped += 1
        else:
            _png16(out, agg)
            made += 1
    qc_name = "qc.csv" if not looks else "qc_" + "_".join(sorted(looks)) + ".csv"
    with open(os.path.join(out_root, qc_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "look", "repeat", "mean_abs_dev", "flag"])
        w.writeheader()
        w.writerows(qc_rows)
    counts = collections.Counter(len(v) for v in sets.values())
    print(f"sets {len(sets)} (repeats per set: {dict(counts)}), aggregates written {made}, already present {skipped}")
    d = np.array(devs_all)
    print(f"repeat deviation from the trimmed mean: median {np.median(d):.4f}, p95 {np.percentile(d, 95):.4f}, "
          f"max {d.max():.4f} levels")
    print(f"flagged repeats: {len(flagged)}")
    for line in flagged[:20]:
        print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2:]))
