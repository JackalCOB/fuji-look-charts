"""Fit a flexible three-dimensional lookup per look from the training charts, as a reference model.

The lookup maps white-balanced camera RGB to the camera's Adobe RGB output. Its input is the camera
colour expressed in CIELAB through the X-T4 forward matrix (charts/colour_space.py), so the
grid's nodes spread evenly to the eye, and each output channel is a trilinear interpolation of node
values. The node values are the least-squares fit to the training patches with a second-difference
smoothness penalty along each axis, so nodes the charts do not reach follow their neighbours.

Clipped outputs are bounds, not values: a channel the camera wrote as 0 or 255 only says the true value
was at or beyond it. After each solve, such observations whose prediction is already beyond the bound
are dropped, and the fit is solved again, three times.

It is a reference, not a profile LightCraft can use: it shows how closely a global model can follow
each look, which the profile forms are then measured against. The smoothness weight is chosen by
holding out every fourth hue of the training grid, so the choice tests colours between the grid's
hues without touching the validation charts. Figures are ΔE2000, on patches with no channel clipped in
the camera's output.

    python analysis/fit_lut.py ADOBE_PATCHES_DIR SRGB_PATCHES_DIR OUT_DIR [LOOK ...]
"""
import csv
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
sys.path.insert(0, os.path.dirname(__file__))
import colour_space as cs  # noqa: E402
from colour_metrics import SRGB_TO_XYZ, delta_e2000, xyz_to_lab  # noqa: E402

LOOKS = ["provia", "velvia", "astia", "pro_neg_hi", "pro_neg_std", "classic_chrome", "eterna", "classic_neg",
         "eterna_bleach_bypass"]
GRID_L = np.linspace(0, 100, 17)
GRID_AB = np.linspace(-168, 168, 29)
LAMBDAS = [1e-2, 1e-1, 1.0]
ADOBE_TO_XYZ = np.array([[0.5767309, 0.1855540, 0.1881852],
                         [0.2973769, 0.6273491, 0.0752741],
                         [0.0270343, 0.0706872, 0.9911085]])
ADOBE_GAMMA = 563 / 256
LO, HI = 0.5 / 255, 254.5 / 255


def load(adobe_dir, srgb_dir, look):
    a = list(csv.DictReader(open(os.path.join(adobe_dir, f"patches_{look}.csv"))))
    s = {(r["chart"], r["name"]): r for r in csv.DictReader(open(os.path.join(srgb_dir, f"patches_{look}.csv")))}
    cam = np.array([[float(r["cam_r"]), float(r["cam_g"]), float(r["cam_b"])] for r in a])
    out = np.array([[float(r["out_r"]), float(r["out_g"]), float(r["out_b"])] for r in a]) / 255.0
    srgb = np.array([[float(s[(r["chart"], r["name"])][c]) for c in ("out_r", "out_g", "out_b")] for r in a]) / 255.0
    return a, cam, out, srgb


def encode_input(cam):
    return cs.cam_to_lab(np.clip(cam, 0, None))


def weights(lab):
    """Sparse trilinear weights of each sample on the (L, a, b) grid."""
    n = len(lab)
    nl, na = len(GRID_L), len(GRID_AB)
    idx, w = [], []
    coords = []
    for axis, grid in ((0, GRID_L), (1, GRID_AB), (2, GRID_AB)):
        x = np.clip(lab[:, axis], grid[0], grid[-1])
        f = (x - grid[0]) / (grid[1] - grid[0])
        i0 = np.clip(np.floor(f).astype(int), 0, len(grid) - 2)
        coords.append((i0, f - i0))
    rows = np.repeat(np.arange(n), 8)
    cols, vals = np.empty(n * 8, int), np.empty(n * 8)
    k = 0
    for dl in (0, 1):
        for da in (0, 1):
            for db in (0, 1):
                (il, tl), (ia, ta), (ib, tb) = coords
                c = ((il + dl) * na + (ia + da)) * na + (ib + db)
                v = (tl if dl else 1 - tl) * (ta if da else 1 - ta) * (tb if db else 1 - tb)
                cols[k::8], vals[k::8] = c, v
                k += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, nl * na * na))


def smoothness():
    nl, na = len(GRID_L), len(GRID_AB)

    def d2(m):
        return sp.diags([np.ones(m - 2), -2 * np.ones(m - 2), np.ones(m - 2)], [0, 1, 2], shape=(m - 2, m))

    il, ia = sp.identity(nl), sp.identity(na)
    return sp.vstack([sp.kron(sp.kron(d2(nl), ia), ia), sp.kron(sp.kron(il, d2(na)), ia),
                      sp.kron(sp.kron(il, ia), d2(na))]).tocsr()


def solve(W, y, D, lam):
    """Least squares for each output channel with the smoothness penalty, dropping satisfied bounds."""
    keep = np.ones(y.shape, bool)
    x = np.full((W.shape[1], 3), 0.5)
    DtD = (D.T @ D) * lam * W.shape[0] / D.shape[0]
    for _ in range(3):
        for ch in range(3):
            m = keep[:, ch]
            Wm = W[m]
            A = (Wm.T @ Wm + DtD).tocsr()
            pre = sp.diags(1.0 / np.maximum(A.diagonal(), 1e-12))
            x[:, ch], _ = spla.cg(A, Wm.T @ y[m, ch], x0=x[:, ch], rtol=1e-7, maxiter=3000, M=pre)
        p = W @ x
        keep = ~(((y <= LO) & (p <= y)) | ((y >= HI) & (p >= y)))
    return x


def adobe_lab(enc):
    return xyz_to_lab((np.clip(enc, 0, 1) ** ADOBE_GAMMA) @ ADOBE_TO_XYZ.T)


def srgb_from_adobe(enc):
    lin = (np.clip(enc, 0, 1) ** ADOBE_GAMMA) @ (np.linalg.inv(SRGB_TO_XYZ) @ ADOBE_TO_XYZ).T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def srgb_lab(enc):
    v = np.clip(enc, 0, 1)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return xyz_to_lab(lin @ SRGB_TO_XYZ.T)


def scores(pred, out, srgb, sel):
    """ΔE2000 in Adobe RGB and, through clipping, in sRGB, on patches with nothing clipped."""
    ok_a = sel & ~((out <= LO) | (out >= HI)).any(axis=1)
    ok_s = sel & ~((srgb <= LO) | (srgb >= HI)).any(axis=1)
    ea = delta_e2000(adobe_lab(pred[ok_a]), adobe_lab(out[ok_a]))
    es = delta_e2000(srgb_lab(srgb_from_adobe(pred[ok_s])), srgb_lab(srgb[ok_s]))
    return ea, es


def main(adobe_dir, srgb_dir, out_dir, looks):
    looks = looks or LOOKS
    os.makedirs(out_dir, exist_ok=True)
    D = smoothness()
    for look in looks:
        rows, cam, out, srgb = load(adobe_dir, srgb_dir, look)
        train = np.array([r["chart"].startswith("train_") for r in rows])
        hue = np.array([float(r["hue"]) for r in rows])
        kind = np.array([r["kind"] for r in rows])
        held = train & (kind == "grid") & np.isin(np.round(hue).astype(int) % 40, [35]) & \
            (np.array([float(r["fraction"]) for r in rows]) > 0)
        fit_set = train & ~held
        lab = encode_input(cam)
        W = weights(lab)
        best = None
        line = f"{look:22s} hold out every 4th hue ({held.sum()} patches):"
        for lam in LAMBDAS:
            x = solve(W[fit_set], out[fit_set], D, lam)
            ea, es = scores(W @ x, out, srgb, held)
            line += f"  lambda {lam:g}: {ea.mean():.2f}/{np.percentile(ea, 95):.2f}"
            if best is None or ea.mean() < best[0]:
                best = (ea.mean(), lam)
        print(line + "  (Adobe RGB mean/p95)")
        lam = best[1]
        x = solve(W[train], out[train], D, lam)
        pred = W @ x
        ea, es = scores(pred, out, srgb, train)
        print(f"{'':22s} all training, lambda {lam:g}: Adobe RGB mean {ea.mean():.2f} median {np.median(ea):.2f} "
              f"p95 {np.percentile(ea, 95):.2f} (n {len(ea)}); sRGB mean {es.mean():.2f} p95 {np.percentile(es, 95):.2f} "
              f"(n {len(es)})")
        np.savez_compressed(os.path.join(out_dir, f"lut_{look}.npz"), nodes=x, grid_l=GRID_L, grid_ab=GRID_AB,
                            lam=lam)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
