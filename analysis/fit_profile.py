"""Fit a look's profile in the form LightCraft applies one, from the training charts.

The forward model follows upstream's `crates/engine/src/camera_preview.rs` and `crates/raw/src/profile.rs`
at `b3d392b4`, as read on 14 October 2026:

1. `scene = M · camera`, white-balanced camera RGB to linear Rec. 2020 (the profile's matrix)
2. a hue/saturation table, 72 hues x 5 saturations x 5 values, applied in HSV of linear ProPhoto RGB
   with the value axis sRGB-encoded, saturation 0 left as identity, and the Rec. 2020 luminance
   restored afterwards (`HueSat::apply`)
3. the finish stage's tone and chroma curves (`displayed`): `o = tone(luma)`, `k = chroma(o)`, and
   each channel `o + (v · o / luma - o) · k`, with luma the Rec. 2020 luminance
4. optionally, per-channel curves on the encoded display values, which upstream's model lacks and a
   look that tints neutrals needs (Classic Neg.)

The display-linear Rec. 2020 result is converted to the camera's output space, clipped, and encoded.

Two forms:
- **a** – the matrix is fitted with the rest
- **b** – the matrix is fixed to the X-T4's measured D65 forward matrix (LightCraft's spectral table),
  and only the look on top is fitted

The parameters are fitted in turn (matrix, tone, table, chroma, channel curves), each by least squares
on the encoded output with clipped observations treated as bounds, for several rounds. The table's
smoothness weight is chosen from REG_WEIGHTS by holding out every fourth hue, then the profile is
refitted on every training patch with it. Figures are
ΔE2000 on patches with no channel clipped, on the training charts and on hues held out of the fit as
fit_lut.py holds them out.

    python analysis/fit_profile.py ADOBE_PATCHES_DIR SRGB_PATCHES_DIR OUT_DIR FORM LOOK [LOOK ...]
        FORM is a or b, then optionally v (the table also scales value, with no luminance restore, as a
        DNG look table does) and + (per-channel curves): a, av, a+, av+, b, bv, b+, bv+
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "charts"))
sys.path.insert(0, os.path.dirname(__file__))
import colour_space as cs  # noqa: E402
from colour_metrics import delta_e2000  # noqa: E402
from fit_lut import (ADOBE_GAMMA, ADOBE_TO_XYZ, HI, LO, adobe_lab, load, srgb_from_adobe,  # noqa: E402
                     srgb_lab)

HUES, SATS, VALS = 72, 5, 5
TONE_N = 32
TONE_X = 2.0 ** np.linspace(-12.0, 0.6, TONE_N)
CHROMA_N = 8
CURVE_N = 12
TABLE_NFEV = 60
REG_WEIGHTS = [0.03, 0.1, 0.3, 1.0]
LUMA = np.array([0.2627, 0.6780, 0.0593])
REC2020 = [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)]
PROPHOTO = [(0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)]
D50_XY = (0.3457, 0.3585)

REC2020_TO_XYZ = cs.rgb_to_xyz(REC2020, cs.D65_XY)
TO_PRO = np.linalg.inv(cs.rgb_to_xyz(PROPHOTO, D50_XY)) @ cs.adapt(cs.D65, cs.D50) @ REC2020_TO_XYZ
FROM_PRO = np.linalg.inv(TO_PRO)
REC_TO_ADOBE = np.linalg.inv(ADOBE_TO_XYZ) @ REC2020_TO_XYZ
# Form b's fixed matrix: camera -> XYZ D50 (forward matrix) -> XYZ D65 -> linear Rec. 2020
SPECTRAL = np.linalg.inv(REC2020_TO_XYZ) @ cs.adapt(cs.D50, cs.D65) @ cs.FORWARD_D65


def srgb_encode(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def srgb_decode(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


class Profile:
    def __init__(self, form, reg=0.1):
        self.form = form
        self.reg = reg
        self.M = SPECTRAL.copy()
        self.tone_theta = np.zeros(TONE_N)
        self.value = "v" in form  # DNG look-table semantics: value scale active, no luminance restore
        self.table = np.zeros((VALS, HUES, SATS, 3))  # hue shift (degrees), log saturation, log value
        self.chroma = np.ones(CHROMA_N)
        self.curves = np.tile(np.linspace(0, 1, CURVE_N), (3, 1))

    # tone: knot outputs increasing in (0, 1)
    def tone_knots(self, theta=None):
        theta = self.tone_theta if theta is None else theta
        return 0.9995 * (1 - np.exp(-np.cumsum(np.exp(theta))))

    def tone(self, y, theta=None):
        k = self.tone_knots(theta)
        xs, ys = np.r_[0.0, TONE_X], np.r_[0.0, k]
        out = np.interp(y, xs, ys)
        a, b = (TONE_X[-2], k[-2]), (TONE_X[-1], k[-1])
        slope = np.clip((b[1] - a[1]) / (b[0] - a[0]), 0.1, 16.0)
        over = y > b[0]
        out[over] = 1 - (1 - b[1]) * np.exp(-(y[over] - b[0]) * slope / max(1 - b[1], 0.01))
        return np.where(y > 0, out, 0.0)

    def table_nodes(self, pro):
        """For ProPhoto colours: hue, saturation, value index, and the trilinear node weights."""
        mx, mn = pro.max(axis=1), pro.min(axis=1)
        d = mx - mn
        safe = np.where(d > 0, d, 1)
        r, g, b = pro[:, 0], pro[:, 1], pro[:, 2]
        h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) * 60
        h = np.where(d > 0, h, 0.0)
        s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0.0)
        v = srgb_encode(np.minimum(mx, 1.0))
        hf = (h % 360) / 360 * HUES
        h0 = np.floor(hf).astype(int) % HUES
        th = hf - np.floor(hf)
        sf = np.clip(s, 0, 1) * (SATS - 1)
        s0 = np.minimum(np.floor(sf).astype(int), SATS - 2)
        ts = sf - s0
        vf = np.clip(v, 0, 1) * (VALS - 1)
        v0 = np.minimum(np.floor(vf).astype(int), VALS - 2)
        tv = vf - v0
        idx, w = [], []
        for dv in (0, 1):
            for dh in (0, 1):
                for ds in (0, 1):
                    idx.append(((v0 + dv) * HUES + (h0 + dh) % HUES) * SATS + s0 + ds)
                    w.append((tv if dv else 1 - tv) * (th if dh else 1 - th) * (ts if ds else 1 - ts))
        return h, s, mx, np.stack(idx, 1), np.stack(w, 1)

    def apply_table(self, rec, nodes=None, table=None):
        table = self.table if table is None else table
        pro = rec @ TO_PRO.T
        h, s, mx, idx, w = self.table_nodes(pro) if nodes is None else nodes
        flat = table.reshape(-1, 3)
        sat0 = (idx % SATS) == 0
        shift = (flat[idx, 0] * np.where(sat0, 0, 1) * w).sum(1)
        logs = (flat[idx, 1] * np.where(sat0, 0, 1) * w).sum(1)
        s2 = np.minimum(s * np.exp(logs), 1.0)
        h2 = (h + shift) % 360
        if self.value:
            logv = (flat[idx, 2] * w).sum(1)
            venc = srgb_encode(np.minimum(mx, 1.0))
            v2 = np.where(mx <= 1.0, srgb_decode(np.minimum(venc * np.exp(logv), 1.0)), mx * np.minimum(np.exp(logv), 1))
            return hsv_to_rgb(h2, s2, v2) @ FROM_PRO.T
        out_pro = hsv_to_rgb(h2, s2, mx)
        out = out_pro @ FROM_PRO.T
        before, after = rec @ LUMA, out @ LUMA
        ok = (before > 0) & (after > 0) & np.isfinite(out).all(1)
        share = np.where(ok, after / np.where(ok, before, 1), 0)
        t = np.clip((share - 0.15) / (0.25 - 0.15), 0, 1)
        wgt = np.where(ok, t * t * (3 - 2 * t), 0)[:, None]
        restored = out * np.where(ok, before / np.where(ok, after, 1), 1)[:, None]
        return rec + (restored - rec) * wgt

    def display(self, scene, theta=None, chroma=None):
        chroma = self.chroma if chroma is None else chroma
        scene = np.maximum(scene, 0)
        y = scene @ LUMA
        o = self.tone(y, theta)
        k = np.interp(o, np.linspace(0, 1, CHROMA_N), chroma)
        ys = np.where(y > 0, y, 1)
        out = o[:, None] + (scene * (o / ys)[:, None] - o[:, None]) * k[:, None]
        return np.where((y > 0)[:, None], out, 0.0)

    def finish(self, disp, curves=None):
        curves = self.curves if curves is None else curves
        adobe = np.clip(disp @ REC_TO_ADOBE.T, 0, 1) ** (1 / ADOBE_GAMMA)
        if not self.form.endswith("+"):
            return adobe
        xs = np.linspace(0, 1, CURVE_N)
        return np.stack([np.interp(adobe[:, c], xs, curves[c]) for c in range(3)], 1)

    def predict(self, cam):
        scene = cam @ self.M.T
        return self.finish(self.display(self.apply_table(scene)))

    def to_json(self):
        return {"form": self.form, "matrix": self.M.tolist(),
                "tone_knots": [[float(x), float(y)] for x, y in zip(TONE_X, self.tone_knots())],
                "hue_sat": {"hue_divisions": HUES, "sat_divisions": SATS, "val_divisions": VALS,
                            "data": [[float(sh), float(np.exp(ls)), float(np.exp(lv))]
                                     for sh, ls, lv in self.table.reshape(-1, 3)],
                            "luminance_restored": not self.value,
                            "srgb_value": True},
                "chroma": self.chroma.tolist(),
                "channel_curves": self.curves.tolist() if self.form.endswith("+") else None}


def hsv_to_rgb(h, s, v):
    c = v * s
    hp = (h % 360) / 60
    x = c * (1 - np.abs(hp % 2 - 1))
    z = np.zeros_like(h)
    i = np.floor(hp).astype(int) % 6
    r = np.choose(i, [c, x, z, z, x, c])
    g = np.choose(i, [x, c, c, x, z, z])
    b = np.choose(i, [z, z, x, c, c, x])
    m = v - c
    return np.stack([r + m, g + m, b + m], 1)


def censored(pred, obs):
    """Residuals, zero where a clipped observation is already met or passed."""
    r = pred - obs
    r[(obs <= LO) & (pred <= obs)] = 0
    r[(obs >= HI) & (pred >= obs)] = 0
    return r


def fit(prof, cam, obs, rounds=4, verbose=False):
    # starting tone from the neutral patches: scene luma against observed luminance, monotone
    for rnd in range(rounds):
        if prof.form.startswith("a"):
            def f_m(p):
                prof.M = p.reshape(3, 3)
                return censored(prof.predict(cam), obs).ravel()
            prof.M = least_squares(f_m, prof.M.ravel(), method="trf", x_scale=0.1, max_nfev=60).x.reshape(3, 3)

        scene_t = prof.apply_table(cam @ prof.M.T)

        def f_tone(p):
            return censored(prof.finish(prof.display(scene_t, theta=p)), obs).ravel()
        res = least_squares(f_tone, prof.tone_theta, method="trf", max_nfev=200)
        prof.tone_theta = res.x

        scene = cam @ prof.M.T
        nodes = prof.table_nodes(scene @ TO_PRO.T)
        idx = nodes[3]
        n = len(cam)
        rows = np.repeat(np.arange(n), 8)
        cols = idx.ravel()
        S = sp.csr_matrix((np.ones(n * 8), (rows, cols)), shape=(n, VALS * HUES * SATS)).astype(bool)
        Sfull = sp.vstack([S] * 3).tocsr()  # residual layout: channel-major via .T.ravel()
        planes = 3 if prof.value else 2
        jac_sp = sp.hstack([Sfull] * planes).tocsr()

        R = REG

        m = VALS * HUES * SATS

        def unpack(p):
            t = np.zeros((VALS, HUES, SATS, 3))
            for k in range(planes):
                t[..., k] = p[k * m:(k + 1) * m].reshape(VALS, HUES, SATS)
            return t

        def f_table(p):
            out = prof.finish(prof.display(prof.apply_table(scene, nodes, unpack(p))))
            reg = prof.reg * np.concatenate([R @ (p[k * m:(k + 1) * m] / (10 if k == 0 else 1)) for k in range(planes)])
            return np.r_[censored(out, obs).T.ravel(), reg]

        t0 = np.concatenate([prof.table[..., k].ravel() for k in range(planes)])
        jac_full = sp.vstack([jac_sp, sp.block_diag([R != 0] * planes)]).tocsr()
        res = least_squares(f_table, t0, method="trf", jac_sparsity=jac_full, max_nfev=TABLE_NFEV, x_scale=0.1)
        prof.table = unpack(res.x)

        scene_t = prof.apply_table(cam @ prof.M.T, None)

        def f_chroma(p):
            return censored(prof.finish(prof.display(scene_t, chroma=p)), obs).ravel()
        prof.chroma = least_squares(f_chroma, prof.chroma, method="trf", bounds=(0, 4), max_nfev=100).x

        if prof.form.endswith("+"):
            disp = prof.display(scene_t)

            def f_curves(p):
                return censored(prof.finish(disp, curves=p.reshape(3, CURVE_N)), obs).ravel()
            prof.curves = least_squares(f_curves, prof.curves.ravel(), method="trf", max_nfev=100).x.reshape(3, CURVE_N)
        if verbose:
            r = censored(prof.predict(cam), obs)
            print(f"    round {rnd + 1}: RMS {np.sqrt((r ** 2).mean()) * 255:.3f} levels", flush=True)
    return prof


def smoothness_operator():
    """Second differences of a table plane along hue (wrapping), saturation, and value, as a sparse
    matrix on the plane flattened value-major, hue-middle, saturation-minor."""
    def at(v, h, s):
        return (v * HUES + h % HUES) * SATS + s
    triples = []
    for v in range(VALS):
        for h in range(HUES):
            for s in range(SATS):
                triples.append((at(v, h - 1, s), at(v, h, s), at(v, h + 1, s)))
                if 1 <= s <= SATS - 2:
                    triples.append((at(v, h, s - 1), at(v, h, s), at(v, h, s + 1)))
                if 1 <= v <= VALS - 2:
                    triples.append((at(v - 1, h, s), at(v, h, s), at(v + 1, h, s)))
    rows = np.repeat(np.arange(len(triples)), 3)
    cols = np.array(triples).ravel()
    vals = np.tile([1.0, -2.0, 1.0], len(triples))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(triples), VALS * HUES * SATS))


REG = smoothness_operator()


def init_tone(prof, cam, obs):
    """Start the tone curve near the neutrals' observed luminance."""
    neutral = (np.abs(cam - cam.mean(1, keepdims=True)).max(1) < 1e-3) & (cam.mean(1) > 0)
    y = (cam[neutral] @ prof.M.T) @ LUMA
    lin = obs[neutral] ** ADOBE_GAMMA @ np.linalg.inv(REC_TO_ADOBE).T @ LUMA
    order = np.argsort(y)
    target = np.interp(TONE_X, y[order], np.maximum.accumulate(lin[order]))
    target = np.clip(target, 1e-5, 0.999)
    inc = np.diff(np.r_[0.0, -np.log(1 - target / 0.9995)])
    prof.tone_theta = np.log(np.maximum(inc, 1e-6))


def scores(pred, out, srgb, sel):
    ok_a = sel & ~((out <= LO) | (out >= HI)).any(axis=1)
    ok_s = sel & ~((srgb <= LO) | (srgb >= HI)).any(axis=1)
    ea = delta_e2000(adobe_lab(pred[ok_a]), adobe_lab(out[ok_a]))
    es = delta_e2000(srgb_lab(srgb_from_adobe(pred[ok_s])), srgb_lab(srgb[ok_s]))
    return ea, es


def main(adobe_dir, srgb_dir, out_dir, form, looks):
    os.makedirs(out_dir, exist_ok=True)
    for look in looks:
        rows, cam, out, srgb = load(adobe_dir, srgb_dir, look)
        train = np.array([r["chart"].startswith("train_") for r in rows])
        hue = np.array([float(r["hue"]) for r in rows])
        kind = np.array([r["kind"] for r in rows])
        frac = np.array([float(r["fraction"]) for r in rows])
        held = train & (kind == "grid") & (np.round(hue).astype(int) % 40 == 35) & (frac > 0)
        best = None
        line = f"{look:22s} form {form:3s} held-out hues ({held.sum()}):"
        for w in REG_WEIGHTS:
            prof = Profile(form, w)
            init_tone(prof, cam[train & ~held], out[train & ~held])
            fit(prof, cam[train & ~held], out[train & ~held])
            ea, es = scores(prof.predict(cam), out, srgb, held)
            line += f"  reg {w:g}: {ea.mean():.2f}/{np.percentile(ea, 95):.2f}"
            if best is None or ea.mean() < best[0]:
                best = (ea.mean(), w, ea, es)
        print(line + "  (Adobe RGB mean/p95)", flush=True)
        prof = Profile(form, best[1])
        init_tone(prof, cam[train], out[train])
        fit(prof, cam[train], out[train])
        ea, es = scores(prof.predict(cam), out, srgb, train)
        print(f"{'':22s} form {form:3s} reg {best[1]:g}: held-out hues Adobe RGB mean {best[2].mean():.2f} p95 "
              f"{np.percentile(best[2], 95):.2f}, sRGB mean {best[3].mean():.2f}; all training Adobe RGB mean "
              f"{ea.mean():.2f} median {np.median(ea):.2f} p95 {np.percentile(ea, 95):.2f}, sRGB mean {es.mean():.2f} "
              f"p95 {np.percentile(es, 95):.2f}", flush=True)
        d = prof.to_json()
        d["reg"] = best[1]
        json.dump(d, open(os.path.join(out_dir, f"profile_{form.replace('+', 'plus')}_{look}.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
