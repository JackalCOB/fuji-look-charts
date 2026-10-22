"""Fit where each sensor position lands in the camera's output, from the geometry grid chart.

Each dark marker's centroid is found in the output near its predicted position, weighting each pixel
by how far it falls below the local background. Predictions start from a plain half-scale mapping and
are refined from the markers found so far, working outward from the centre.

Two mappings are fitted to the centroids. A similarity (scale and shift) shows how far a plain
scaling falls short. The radial model is the one used downstream:

    o = c_o + (s - c_s) * (out_scale) * (b0 + b1 r^2 + b2 r^4 + b3 r^6)

with s a sensor position, c_s the centre of the active area, r the distance from it as a fraction of
the active area's half diagonal, and c_o the output centre, fitted. out_scale is the output size over
the active area's. The fit is written as JSON with its residuals.

    python analysis/geometry.py LAYOUT.json AGGREGATE.png [...] --out GEOMETRY.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
from read_chart import load  # noqa: E402

SEARCH = 34


def centroid(lum, px, py):
    x, y = int(round(px)), int(round(py))
    h, w = lum.shape
    if not (SEARCH <= x < w - SEARCH and SEARCH <= y < h - SEARCH):
        return None
    win = lum[y - SEARCH:y + SEARCH + 1, x - SEARCH:x + SEARCH + 1]
    border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
    bg = np.median(border)
    depth = bg - win
    if depth.max() < 40:
        return None
    wgt = np.where(depth > 0.25 * depth.max(), depth, 0.0)
    yy, xx = np.mgrid[-SEARCH:SEARCH + 1, -SEARCH:SEARCH + 1]
    cx, cy = (wgt * xx).sum() / wgt.sum(), (wgt * yy).sum() / wgt.sum()
    if max(abs(cx), abs(cy)) > SEARCH - 12:
        return None
    return x + cx, y + cy


def design(layout, k=None):
    x0, y0, W, H = layout["active_area"]
    cs = np.array([x0 + W / 2, y0 + H / 2])
    half = np.hypot(W / 2, H / 2)
    pts = np.array([[p["x"] + p["w"] / 2, p["y"] + p["h"] / 2] for p in layout["patches"]])
    return pts, cs, half


def basis(s, cs, half, order):
    d = s - cs
    r2 = (d ** 2).sum(axis=1) / half ** 2
    return d, np.stack([r2 ** k for k in range(order + 1)], axis=1)


def fit_radial(s, o, cs, half, scale, order=3):
    """Least squares for c_o (2) and b0..b_order, linear once c_s is fixed."""
    d, rb = basis(s, cs, half, order)
    n = len(s)
    A = np.zeros((2 * n, 2 + order + 1))
    A[:n, 0], A[n:, 1] = 1, 1
    A[:n, 2:] = scale * d[:, :1] * rb
    A[n:, 2:] = scale * d[:, 1:] * rb
    y = np.concatenate([o[:, 0], o[:, 1]])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ sol
    res = np.hypot(y[:n] - pred[:n], y[n:] - pred[n:])
    return sol, res


def predict(sol, s, cs, half, scale, order):
    d, rb = basis(s, cs, half, order)
    f = rb @ sol[2:]
    return np.stack([sol[0] + scale * d[:, 0] * f, sol[1] + scale * d[:, 1] * f], axis=1)


def find_all(lum, s, cs, half, scale, out_centre):
    """Find markers outward from the centre, refitting the prediction as they accumulate."""
    r = np.hypot(*(s - cs).T) / half
    order_idx = np.argsort(r)
    found = {}
    sol, order = np.array([*out_centre, 1.0]), 0
    for band in np.arange(0.2, 1.21, 0.1):
        for i in order_idx[(r[order_idx] < band)]:
            if i in found:
                continue
            guess = predict(sol, s[i:i + 1], cs, half, scale, order)[0]
            c = centroid(lum, *guess)
            if c is not None:
                found[i] = c
        idx = sorted(found)
        order = 1 if len(idx) < 40 else 3
        sol, _ = fit_radial(s[idx], np.array([found[i] for i in idx]), cs, half, scale, order)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    layout = json.load(open(a.layout))
    s_all, cs, half = design(layout)
    x0, y0, W, H = layout["active_area"]
    results = {}
    for path in a.images:
        img = load(path)
        h, w = img.shape[:2]
        scale = w / W
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        found = find_all(lum, s_all, cs, half, scale, ((cs[0] - x0) * scale, (cs[1] - y0) * scale))
        idx = sorted(found)
        s, o = s_all[idx], np.array([found[i] for i in idx])
        r = np.hypot(*(s - cs).T) / half
        # Similarity: o = t + k * scale * (s - cs)
        n = len(s)
        A = np.zeros((2 * n, 3))
        A[:n, 0], A[n:, 1] = 1, 1
        A[:n, 2], A[n:, 2] = scale * (s[:, 0] - cs[0]), scale * (s[:, 1] - cs[1])
        y = np.concatenate([o[:, 0], o[:, 1]])
        sim, *_ = np.linalg.lstsq(A, y, rcond=None)
        p = A @ sim
        sim_res = np.hypot(y[:n] - p[:n], y[n:] - p[n:])
        sol, res = fit_radial(s, o, cs, half, scale, 3)
        name = os.path.basename(path)
        print(f"\n{name}: {len(idx)} of {len(s_all)} markers found, out to r = {r.max():.3f}")
        print(f"  plain half scale would put the centre at ({(cs[0] - x0) * scale:.1f}, {(cs[1] - y0) * scale:.1f})")
        print(f"  similarity: centre ({sim[0]:.2f}, {sim[1]:.2f}), scale x{sim[2]:.5f}, residual RMS "
              f"{np.sqrt((sim_res ** 2).mean()):.2f} px, max {sim_res.max():.2f}")
        print(f"  radial: centre ({sol[0]:.2f}, {sol[1]:.2f}), b = {np.round(sol[2:], 5).tolist()}, residual RMS "
              f"{np.sqrt((res ** 2).mean()):.3f} px, p95 {np.percentile(res, 95):.3f}, max {res.max():.3f}")
        for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
            sel = (r >= lo) & (r < hi)
            if sel.any():
                pr = predict(np.array([*((cs - [x0, y0]) * scale), 1.0]), s[sel], cs, half, scale, 0)
                shift = np.hypot(*(o[sel] - pr).T)
                print(f"    r {lo:.2f}-{hi:.2f}: {sel.sum():3d} markers, shift from plain half scale median "
                      f"{np.median(shift):5.1f} px, max {shift.max():5.1f}; radial residual max {res[sel].max():.2f}")
        rr = np.linspace(0, 1, 11)
        f = sum(sol[2 + k] * rr ** (2 * k) for k in range(4))
        print("  output radius over sensor radius, by sensor r: " +
              " ".join(f"{x:.1f}:{v:.4f}" for x, v in zip(rr, f)))
        results[name] = {"markers": len(idx), "centre_out": sol[:2].tolist(), "b": sol[2:].tolist(),
                         "rms_px": float(np.sqrt((res ** 2).mean())), "max_px": float(res.max()),
                         "similarity": {"centre": sim[:2].tolist(), "scale": float(sim[2]),
                                        "rms_px": float(np.sqrt((sim_res ** 2).mean()))}}
    if a.out:
        out = {"model": "o = c_o + (s - c_s) * out_scale * (b0 + b1 r^2 + b2 r^4 + b3 r^6)",
               "active_area": layout["active_area"], "c_s": cs.tolist(), "half_diagonal": float(half),
               "fits": results}
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
