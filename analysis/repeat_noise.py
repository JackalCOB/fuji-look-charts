"""Characterise the difference between two camera conversions of the same raw with the same profile.

Used on the public raw only. Reports whether the JPEG quantisation tables match, how the difference
splits between luma and chroma, whether it follows the 8 x 8 JPEG block grid, whether it tracks
image detail, and how it shrinks when both images are averaged over larger blocks.

    python analysis/repeat_noise.py A.jpg B.jpg
"""
import sys

import numpy as np
from PIL import Image


def box(img, k):
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return img[:h, :w].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3))


def main(a_path, b_path):
    A, B = Image.open(a_path), Image.open(b_path)
    print(f"quantisation tables identical: {A.quantization == B.quantization}")
    print(f"chroma subsampling identical: {A.layer == B.layer if hasattr(A, 'layer') else 'n/a'}")
    a = np.asarray(A.convert("RGB")).astype(np.float64)
    b = np.asarray(B.convert("RGB")).astype(np.float64)
    d = a - b
    print(f"size {a.shape[1]} x {a.shape[0]}")
    print(f"RGB: mean |diff| {np.abs(d).mean():.3f}, RMS {np.sqrt((d ** 2).mean()):.3f}, "
          f"max {np.abs(d).max():.0f}, mean signed {d.mean():+.4f}, pixels differing {100 * (np.abs(d).max(axis=2) > 0).mean():.1f}%")

    ya = np.asarray(A.convert("YCbCr")).astype(np.float64)
    yb = np.asarray(B.convert("YCbCr")).astype(np.float64)
    yd = ya - yb
    for i, name in enumerate(["Y (luma)", "Cb", "Cr"]):
        print(f"  {name}: RMS {np.sqrt((yd[..., i] ** 2).mean()):.3f}")

    # JPEG block grid: difference energy on 8-pixel block borders against block interiors.
    L = yd[..., 0]
    col_border = np.zeros(L.shape[1], bool)
    col_border[7::8] = True
    col_border[8::8] = True
    print(f"luma diff RMS on 8-px block borders {np.sqrt((L[:, col_border] ** 2).mean()):.3f}, "
          f"inside blocks {np.sqrt((L[:, ~col_border] ** 2).mean()):.3f}")

    # Neighbour correlation: white noise is near 0, smooth or blocky differences are high.
    lag1 = np.corrcoef(L[:, :-1].ravel(), L[:, 1:].ravel())[0, 1]
    lag8 = np.corrcoef(L[:, :-8].ravel(), L[:, 8:].ravel())[0, 1]
    print(f"luma diff correlation with the next pixel {lag1:+.3f}, with the pixel 8 along {lag8:+.3f}")

    # Detail: local gradient of the image against the size of the difference.
    g = np.hypot(*np.gradient(ya[..., 0]))
    q = np.quantile(g, [0.25, 0.5, 0.75, 0.95])
    bands = [g <= q[0], (g > q[0]) & (g <= q[1]), (g > q[1]) & (g <= q[2]), (g > q[2]) & (g <= q[3]), g > q[3]]
    names = ["flattest 25%", "25-50%", "50-75%", "75-95%", "most detailed 5%"]
    print("luma diff RMS by image detail: " + ", ".join(
        f"{n} {np.sqrt((L[m] ** 2).mean()):.3f}" for n, m in zip(names, bands)))

    # Brightness: is the difference larger in shadows, mid-tones or highlights?
    yv = ya[..., 0]
    for lo, hi in [(0, 32), (32, 96), (96, 160), (160, 224), (224, 256)]:
        m = (yv >= lo) & (yv < hi)
        if m.any():
            print(f"  luma {lo:3d}-{hi:3d}: share {100 * m.mean():5.1f}%, diff RMS {np.sqrt((L[m] ** 2).mean()):.3f}")

    # Aggregation: the same difference after averaging both images over k x k blocks.
    print("after averaging over k x k blocks (k, mean |diff|, max |diff| in 8-bit levels):")
    for k in [1, 2, 4, 8, 16, 32]:
        dk = np.abs(box(a, k) - box(b, k))
        print(f"  k={k:2d}: mean {dk.mean():.4f}, max {dk.max():.3f}")
    k = int(np.ceil(max(a.shape[:2]) / 192))
    dk = np.abs(box(a, k) - box(b, k))
    print(f"  at a 192-pixel proxy (k={k}): mean {dk.mean():.4f}, max {dk.max():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
