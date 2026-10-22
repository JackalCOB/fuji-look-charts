"""Check every conversion's metadata against what was asked of the camera.

Colour looks must report their own film simulation, DR100, and the fixed neutral settings. `as_shot`
conversions must report the film simulation and dynamic range the raw was shot with, read from the
manifest. Reads named tags only, never location data, and prints counts and failures only.

    python analysis/check_conversion_tags.py CONVERSIONS_DIR MANIFEST.csv
"""
import csv
import io
import os
import re
import subprocess
import sys

FILM_MODE = {"provia": "F0/Standard (Provia)", "velvia": "F2/Fujichrome (Velvia)",
             "astia": "F1b/Studio Portrait Smooth Skin Tone (Astia)", "pro_neg_hi": "Pro Neg. Hi",
             "pro_neg_std": "Pro Neg. Std", "classic_chrome": "Classic Chrome", "eterna": "Eterna",
             "classic_neg": "Classic Negative", "eterna_bleach_bypass": "Bleach Bypass"}
NEUTRAL = {"GrainEffectRoughness": "Off", "ColorChromeEffect": "Off", "ColorChromeFXBlue": "Off",
           "HighlightTone": "0 (normal)", "ShadowTone": "0 (normal)", "Saturation": "0 (normal)",
           "Sharpness": "Normal", "NoiseReduction": "0 (normal)", "Clarity": "0",
           "WhiteBalanceFineTune": "Red +0, Blue +0"}
TAGS = ["FileName", "FilmMode", "DevelopmentDynamicRange", *NEUTRAL]
NAME = re.compile(r"^(?P<stem>.+)__(?P<look>[a-z_]+)__r(?P<r>\d+)\.jpg$")


def main(root, manifest):
    shot = {os.path.splitext(r["file"])[0]: r for r in csv.DictReader(open(manifest, encoding="utf-8"))
            if r.get("role", "raw") == "raw"}
    run = subprocess.run(["exiftool", "-q", "-q", "-csv", "-r", "-ext", "jpg", "-charset", "filename=utf8",
                          *[f"-{t}" for t in TAGS], root], capture_output=True, text=True, encoding="utf-8")
    rows = [r for r in csv.DictReader(io.StringIO(run.stdout)) if "_aggregate" not in r["SourceFile"]]
    failures, checked = [], 0
    for r in rows:
        m = NAME.match(r["FileName"])
        if not m:
            continue
        checked += 1
        look, stem = m["look"], m["stem"]
        problems = []
        if look == "as_shot":
            src = shot.get(stem, {})
            want_mode = src.get("film_mode")
            want_dr = (src.get("auto_dynamic_range") or "").rstrip("%")
            if want_mode and r["FilmMode"] != want_mode:
                problems.append(f"FilmMode {r['FilmMode']!r}, shot {want_mode!r}")
            if want_dr and r["DevelopmentDynamicRange"] != want_dr:
                problems.append(f"DR {r['DevelopmentDynamicRange']}, shot {want_dr}")
        else:
            if r["FilmMode"] != FILM_MODE[look]:
                problems.append(f"FilmMode {r['FilmMode']!r}")
            if r["DevelopmentDynamicRange"] != "100":
                problems.append(f"DR {r['DevelopmentDynamicRange']}")
            problems += [f"{k} {r[k]!r}" for k, v in NEUTRAL.items() if r.get(k) != v]
        if problems:
            failures.append(f"{r['FileName']}: " + "; ".join(problems))
    print(f"checked {checked} conversions under {root}: {checked - len(failures)} as requested, "
          f"{len(failures)} not")
    for line in failures[:30]:
        print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
