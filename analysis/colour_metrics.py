"""Colour difference for comparing renders: CIELAB from 8-bit sRGB, ΔE76, and ΔE2000.

CIELAB is taken from each image's own colour space, so sRGB values are decoded with the sRGB transfer
function, converted to XYZ with the sRGB primaries, and referred to the sRGB white, D65. ΔE2000
follows Sharma, Wu, and Dalal, "The CIEDE2000 color-difference formula: implementation notes,
supplementary test data, and mathematical observations", Color Research and Application 30(1), 2005,
including their notes on the hue angle and its mean when a chroma is zero. `python colour_metrics.py`
checks it against the paper's 34 published test pairs (testdata/ciede2000_sharma2005.txt), which it
must reproduce to four decimal places.
"""
import os
import sys

import numpy as np

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
D65 = SRGB_TO_XYZ @ np.ones(3)


def srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def xyz_to_lab(xyz, white=D65):
    t = np.asarray(xyz, dtype=np.float64) / white
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def srgb_to_lab(rgb):
    """CIELAB (D65) of 8-bit sRGB levels, which may be fractional."""
    return xyz_to_lab(srgb_to_linear(rgb) @ SRGB_TO_XYZ.T)


def delta_e76(lab1, lab2):
    return np.linalg.norm(np.asarray(lab1, dtype=np.float64) - np.asarray(lab2, dtype=np.float64), axis=-1)


def delta_e2000(lab1, lab2, kl=1.0, kc=1.0, kh=1.0):
    lab1, lab2 = np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cbar7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(cbar7 / (cbar7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where((a1p == 0) & (b1 == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = np.where((a2p == 0) & (b2 == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp = L2 - L1
    dCp = c2p - c1p
    zero = (c1p * c2p) == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero, 0.0, dh)
    dHp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
    Lbar = (L1 + L2) / 2
    Cbar = (c1p + c2p) / 2
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2))
    hbar = np.where(zero, hsum, hbar)
    t = (1 - 0.17 * np.cos(np.radians(hbar - 30)) + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6)) - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    dtheta = 30 * np.exp(-(((hbar - 275) / 25) ** 2))
    cbar7p = Cbar ** 7
    rc = 2 * np.sqrt(cbar7p / (cbar7p + 25.0 ** 7))
    sl = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    sc = 1 + 0.045 * Cbar
    sh = 1 + 0.015 * Cbar * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc
    return np.sqrt((dLp / (kl * sl)) ** 2 + (dCp / (kc * sc)) ** 2 + (dHp / (kh * sh)) ** 2
                   + rt * (dCp / (kc * sc)) * (dHp / (kh * sh)))


def self_test():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "ciede2000_sharma2005.txt")
    data = np.loadtxt(path)
    got = delta_e2000(data[:, 0:3], data[:, 3:6])
    worst = np.abs(np.round(got, 4) - data[:, 6]).max()
    swapped = np.abs(delta_e2000(data[:, 3:6], data[:, 0:3]) - got).max()
    white = srgb_to_lab([255, 255, 255])
    print(f"CIEDE2000: {len(data)} published pairs, largest difference after rounding to 4 places {worst:.4f}, "
          f"symmetry {swapped:.1e}; sRGB white is L* {white[0]:.4f} a* {white[1]:.4f} b* {white[2]:.4f}")
    ok = worst == 0 and swapped < 1e-9 and np.abs(white - [100, 0, 0]).max() < 1e-3
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if self_test() else 1)
