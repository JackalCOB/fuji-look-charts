"""Where a sensor position lands in the camera's output, from the geometry fit.

The X-T4 corrects its lens's distortion when it converts a raw, so output positions are not a plain
half-scale copy of sensor positions: the centre of the frame shrinks by about 3% and the edges stretch
by up to 4%. analysis/geometry.py fits the radial model below to a grid of markers, and this
module applies it.

    o = c_o + (s - c_s) * out_scale * (b0 + b1 r^2 + b2 r^4 + ...)

s is a sensor position, c_s the centre of the active area, r the distance from it as a fraction of the
active area's half diagonal, c_o the output centre, and out_scale the output size over the active
area's (0.5 for the S size).
"""
import json

import numpy as np


class Geometry:
    def __init__(self, path, fit=None):
        g = json.load(open(path))
        fits = g["fits"]
        f = fits[fit] if fit else next(iter(fits.values()))
        self.active = g["active_area"]
        self.c_s = np.array(g["c_s"], dtype=np.float64)
        self.half = float(g["half_diagonal"])
        self.c_o = np.array(f["centre_out"], dtype=np.float64)
        self.b = np.array(f["b"], dtype=np.float64)

    def stretch(self, r):
        """Output radius over sensor radius, at sensor radius r (a fraction of the half diagonal)."""
        r2 = np.asarray(r, dtype=np.float64) ** 2
        return sum(c * r2 ** k for k, c in enumerate(self.b))

    def output_radius(self, r):
        """Output radius, as a fraction of the output's half diagonal, of a sensor radius r."""
        return np.asarray(r, dtype=np.float64) * self.stretch(r)

    def to_output(self, x, y, out_scale):
        """Output pixel coordinates of sensor positions x, y."""
        dx, dy = np.asarray(x, dtype=np.float64) - self.c_s[0], np.asarray(y, dtype=np.float64) - self.c_s[1]
        f = self.stretch(np.hypot(dx, dy) / self.half) * out_scale
        return self.c_o[0] + dx * f, self.c_o[1] + dy * f
