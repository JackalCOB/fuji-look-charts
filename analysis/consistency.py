"""The consistency check: does a colour render the same wherever it sits and whatever surrounds it?

Applies the five tests and thresholds set below (the README explains them) to the patch tables
read_colour_charts.py writes, per look. A difference is the largest of the three channel differences,
in 8-bit output levels.

Prints each test's figures against its limit, a pass or fail per look, and diagnostics by chart,
radius, and level for any test that fails. Figures only.

    python analysis/consistency.py PATCHES_DIR [LOOK ...]
"""
import collections
import csv
import os
import sys

import numpy as np

LOOKS = ["provia", "velvia", "astia", "pro_neg_hi", "pro_neg_std", "classic_chrome", "eterna", "classic_neg",
         "eterna_bleach_bypass"]
LIMITS = {
    "anchors_across_charts": {"p95": 1.0, "max": 2.0},
    "anchors_across_positions": {"p95": 1.5},
    "training_duplicates": {"median": 0.6, "p95": 2.0},
    "validation_repeats": {"p95": 1.5},
    "reordered_vs_training": {"median": 0.6, "p95": 2.0},
}


def out(row):
    return np.array([float(row["out_r"]), float(row["out_g"]), float(row["out_b"])])


def stats(dev):
    dev = np.asarray(dev)
    return {"n": len(dev), "median": float(np.median(dev)), "p95": float(np.percentile(dev, 95)),
            "max": float(dev.max())}


def from_median(groups):
    """Each member's largest channel difference from its group's per-channel median."""
    dev, where = [], []
    for key, members in groups.items():
        vals = np.array([v for v, _ in members])
        med = np.median(vals, axis=0)
        for v, info in members:
            dev.append(np.abs(v - med).max())
            where.append((key, info))
    return dev, where


def run(rows):
    res, diag = {}, {}
    # 1. Anchors across charts: the same cell in every chart.
    cells = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "anchor":
            cells[r["name"]].append((out(r), r))
    dev, where = from_median(cells)
    res["anchors_across_charts"] = stats(dev)
    by_chart = collections.defaultdict(list)
    for d, (_, r) in zip(dev, where):
        by_chart[r["chart"]].append(d)
    diag["anchors_across_charts"] = {c: float(np.percentile(v, 95)) for c, v in sorted(by_chart.items())}
    # 2. Anchors across positions: each anchor colour's eight cells, per chart.
    colours = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "anchor":
            colours[(r["chart"], r["anchor"])].append((out(r), r))
    dev, where = from_median(colours)
    res["anchors_across_positions"] = stats(dev)
    by_r = collections.defaultdict(list)
    for d, (_, r) in zip(dev, where):
        by_r[min(int(float(r["r"]) * 10), 9)].append(d)
    diag["anchors_across_positions"] = {f"r {k / 10:.1f}": float(np.median(v)) for k, v in sorted(by_r.items())}
    # 3. Training duplicates: copy 1 against copy 2.
    copies = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "grid" and r["chart"].startswith("train_"):
            copies[r["colour_id"]][r["copy"]] = r
    pairs = [(c["1"], c["2"]) for c in copies.values() if "1" in c and "2" in c]
    dev = [np.abs(out(a) - out(b)).max() for a, b in pairs]
    res["training_duplicates"] = stats(dev)
    by_level = collections.defaultdict(list)
    for d, (a, _) in zip(dev, pairs):
        by_level[round(float(a["stops"]) * 2) / 2].append(d)
    diag["training_duplicates"] = {f"{k:+.1f} stops": float(np.percentile(v, 95)) for k, v in sorted(by_level.items())}
    # 4. Validation repeats: eight cells of each repeated colour.
    reps = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == "repeat":
            reps[r["colour_id"]].append((out(r), r))
    dev, _ = from_median(reps)
    res["validation_repeats"] = stats(dev)
    # 5. Reordered colours against the mean of their training copies.
    train_mean = {k: np.mean([out(r) for r in c.values()], axis=0) for k, c in copies.items()}
    dev = [np.abs(out(r) - train_mean[r["colour_id"]]).max() for r in rows
           if r["kind"] == "reordered" and r["colour_id"] in train_mean]
    res["reordered_vs_training"] = stats(dev)
    return res, diag


def verdict(res):
    fails = []
    for test, lim in LIMITS.items():
        for k, v in lim.items():
            if res[test][k] > v:
                fails.append(f"{test} {k} {res[test][k]:.2f} > {v}")
    return fails


def main(patch_dir, looks):
    looks = looks or LOOKS
    summary = []
    for look in looks:
        path = os.path.join(patch_dir, f"patches_{look}.csv")
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        res, diag = run(rows)
        fails = verdict(res)
        print(f"\n{look}: {len(rows)} patches")
        for test, lim in LIMITS.items():
            s = res[test]
            marks = ", ".join(f"{k} {s[k]:.2f} (limit {v})" for k, v in lim.items())
            extra = "" if "median" in lim else f", median {s['median']:.2f}"
            print(f"  {test:26s} n {s['n']:5d}: {marks}{extra}")
        print(f"  {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
        for test in {f.split()[0] for f in fails}:
            if test in diag:
                print(f"  diagnostics, {test}: " + ", ".join(f"{k} {v:.2f}" for k, v in diag[test].items()))
        summary.append((look, not fails))
    print("\nsummary: " + ", ".join(f"{look} {'pass' if ok else 'FAIL'}" for look, ok in summary))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
