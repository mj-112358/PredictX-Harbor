"""Place one SUMO junction of the v9 district onto a real intersection of a Google Maps capture.

The recorded trajectories are unchanged; only the drawing transform differs. Each arm of the SUMO junction is
aligned with one road of the map: distance along the arm keeps the true scale (metres -> capture pixels at
zoom 18, 2x), lateral offsets are stretched so the lanes fill the drawn carriageway, and the short stretch
inside the junction box is stretched so stop lines sit at the edge of the drawn junction. Arms are blended
smoothly by angle, so turning vehicles move continuously.
"""
from pathlib import Path
import json
import math

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MAPS = HERE / "data" / "maps"
M_PER_PX = 156543.03392 * math.cos(math.radians(30.705)) / 2 ** 18 / 2      # zoom 18, device pixel ratio 2

# Junction -> capture and per-arm road geometry, measured on the capture by fitting the road-colour centreline
# of each arm between 160 and 700 px from the junction (scratch measure.py; values in device pixels, y down).
# arm value: (point on the fitted centreline, unit direction away from the junction, half-width of the drawn
#             carriageway in px, distance from the junction centre to the stop line in px)
SPECS = {
    "J21": dict(image="J21_mohali_3b2.png", center=(1888, 992),
                arms={"J22": ((2107.5, 1314.6), 53.8, 34, 66), "J11": ((2237.3, 711.6), -39.1, 23, 78),
                      "J31": ((1505.0, 1243.1), 146.3, 18, 78), "J20": ((1599.6, 607.2), -128.3, 32, 66)}),
    "J11": dict(image="J11_mohali_sarovar.png", center=(1700, 1080),
                arms={"J21": ((1973.2, 1418.8), 54.5, 20, 58), "J01": ((2055.4, 832.3), -34.9, 23, 62),
                      "J10": ((1524.5, 676.1), -112.7, 38, 58)}),
}


class Warp:
    def __init__(self, net, j):
        sp = SPECS[j]
        self.j, self.spec = j, sp
        self.C = np.array(sp["center"], float)
        self.J = net.junctions[j]["xy"]
        self.s = 1 / M_PER_PX                                     # px per metre along roads
        self.arms = []
        for nb, (pt, ang, hw_map, r_map) in sp["arms"].items():
            u = net.junctions[nb]["xy"] - self.J
            u /= np.linalg.norm(u)
            n = np.array([-u[1], u[0]])
            eids = [e for e, r in net.edges.items() if {r["from_node"], r["to_node"]} == {j, nb}]
            hw = sum(int(net.edges[e]["lanes"]) * float(net.edges[e]["lane_width_m"]) for e in eids) / 2
            inc = [e for e in eids if net.edges[e]["to_node"] == j]
            r0 = float(np.dot(net.stopline(inc[0])[0] - self.J, u)) if inc else 10.0
            a = math.radians(ang)
            U = np.array([math.cos(a), math.sin(a)])
            N = np.array([-U[1], U[0]])
            O = np.array(pt, float) + U * np.dot(self.C - np.array(pt, float), U)     # foot of the centre on the arm line
            self.arms.append(dict(nb=nb, u=u, n=n, U=U, N=N, O=O - self.C, phi=math.atan2(u[1], u[0]), r0=r0, s_in=r_map / r0,
                                  k=hw_map / (hw * self.s), edges=eids))
        self.img = Image.open(MAPS / sp["image"]).convert("RGB")

    def _radial(self, A, a):
        """Distance along an arm, metres -> px: stretched inside the junction box, true scale beyond it."""
        r0, si, s = A["r0"], A["s_in"], self.s
        t = 15.0                                                   # metres of smooth transition
        a = np.asarray(a, float)
        inner = a * si
        mid = r0 * si + (a - r0) * si + (s - si) * (a - r0) ** 2 / (2 * t)
        outer = r0 * si + t * (si + s) / 2 + (a - r0 - t) * s
        return np.where(a < r0, inner, np.where(a < r0 + t, mid, outer))

    def _lat_scale(self, A, a):
        return A["k"] * self.s * np.ones_like(np.asarray(a, float))

    def __call__(self, xy):
        q = np.atleast_2d(np.asarray(xy, float)) - self.J
        phi = np.arctan2(q[:, 1], q[:, 0])
        ws, outs = [], []
        for A in self.arms:
            d = (phi - A["phi"] + np.pi) % (2 * np.pi) - np.pi
            ws.append(np.exp(-(d / math.radians(22)) ** 2) + 1e-12)
            a, l = q @ A["u"], q @ A["n"]
            fade = np.clip(np.hypot(*q.T) / max(A["r0"], 1), 0, 1)[:, None]          # arm offset grows from the centre
            outs.append(A["O"] * fade + self._radial(A, a)[:, None] * A["U"] + (l * self._lat_scale(A, a))[:, None] * A["N"])
        W = np.array(ws)
        W /= W.sum(0)
        out = sum(w[:, None] * o for w, o in zip(W, outs))
        return self.C + out

    def heading(self, xy, ang_deg, step=2.0):
        """Screen heading (degrees, same convention as SUMO angle with display y down) after warping."""
        r = np.radians(ang_deg)
        d = np.stack((np.sin(r), np.cos(r)), -1)
        xy = np.atleast_2d(xy)
        a, b = self(xy - d * step), self(xy)
        v = b - a
        return np.degrees(np.arctan2(v[:, 0], v[:, 1]))

    def edges(self):
        return {e for A in self.arms for e in A["edges"]}
