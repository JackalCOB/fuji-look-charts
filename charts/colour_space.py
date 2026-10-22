"""Colour conversions the chart generator uses to place samples.

Chart colours are chosen in CIELAB (D50), so they spread evenly to the eye, and written as
white-balanced X-T4 camera RGB. The bridge is the X-T4's D65 forward matrix from LightCraft's own
spectral table (crates/raw/src/spectral_table.rs, generated there from rawtoaces-data, Apache-2.0).
It only decides where samples sit. No profile is fitted through it, so its accuracy limits how evenly
the samples spread, not what a profile learns.

The outer limit for a colour at a given lightness and hue is the most chroma that keeps every camera
channel at or above zero, keeps the colour inside Rec. 2020 (which holds almost every real surface
colour), and keeps every sensor channel at or below a chosen fraction of full scale, so nothing clips
on the sensor once the camera's corner brightening is compensated.
"""
import numpy as np

# White-balanced X-T4 camera RGB -> XYZ (D50), LightCraft's ForwardMatrix2 (D65) for the X-T4.
FORWARD_D65 = np.array([[0.675162, 0.317733, -0.028599],
                        [0.248581, 1.05006, -0.298641],
                        [0.04702, -0.219918, 0.998002]])
CAM_FROM_XYZ = np.linalg.inv(FORWARD_D65)

D50 = np.array([0.96422, 1.0, 0.82521])
D65 = np.array([0.95047, 1.0, 1.08883])
BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])
REC2020_PRIMARIES = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)]
D65_XY = (0.3127, 0.3290)

# The public container's as-shot white balance, R and B over G (make_chart.py reads it from the file).
WB_GAIN = np.array([1.9139073, 1.0, 2.1887417])


def adapt(src, dst):
    """Bradford chromatic adaptation matrix from white src to white dst (XYZ)."""
    s, d = BRADFORD @ src, BRADFORD @ dst
    return np.linalg.inv(BRADFORD) @ np.diag(d / s) @ BRADFORD


def rgb_to_xyz(primaries, white_xy):
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = white_xy
    w = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    return xyz * np.linalg.solve(xyz, w)


REC2020_FROM_XYZ_D50 = np.linalg.inv(rgb_to_xyz(REC2020_PRIMARIES, D65_XY)) @ adapt(D50, D65)


def _f(t):
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)


def _finv(t):
    d = 6 / 29
    return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))


def lab_to_xyz(lab, white=D50):
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    return np.stack([_finv(fy + a / 500), _finv(fy), _finv(fy - b / 200)], axis=-1) * white


def xyz_to_lab(xyz, white=D50):
    f = _f(np.asarray(xyz, dtype=np.float64) / white)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lightness(y):
    return 116 * _f(np.asarray(y, dtype=np.float64)) - 16


def lch_to_cam(L, c, h):
    """White-balanced camera RGB of CIELAB lightness L, chroma c, hue angle h in degrees."""
    hr = np.radians(h)
    lab = np.stack(np.broadcast_arrays(L, c * np.cos(hr), c * np.sin(hr)), axis=-1)
    return lab_to_xyz(lab) @ CAM_FROM_XYZ.T


def cam_to_lab(cam):
    return xyz_to_lab(np.asarray(cam, dtype=np.float64) @ FORWARD_D65.T)


def feasible(L, c, h, sensor_max):
    cam = lch_to_cam(L, c, h)
    rec = lab_to_xyz(np.stack(np.broadcast_arrays(L, c * np.cos(np.radians(h)), c * np.sin(np.radians(h))),
                              axis=-1)) @ REC2020_FROM_XYZ_D50.T
    return (cam.min(axis=-1) >= 0) & (rec.min(axis=-1) >= 0) & ((cam / WB_GAIN).max(axis=-1) <= sensor_max)


def max_chroma(L, h, sensor_max, hi=300.0, steps=40):
    """The most chroma at lightness L and hue h that is feasible, by bisection along the hue ray."""
    L, h = np.broadcast_arrays(np.asarray(L, dtype=np.float64), np.asarray(h, dtype=np.float64))
    lo_c, hi_c = np.zeros_like(L), np.full_like(L, hi)
    ok0 = feasible(L, np.zeros_like(L), h, sensor_max)
    for _ in range(steps):
        mid = (lo_c + hi_c) / 2
        ok = feasible(L, mid, h, sensor_max)
        lo_c, hi_c = np.where(ok, mid, lo_c), np.where(ok, hi_c, mid)
    return np.where(ok0, lo_c, 0.0)
