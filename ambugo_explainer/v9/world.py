"""Network geometry, recorded replays and overhead drawing for the AmbuGo v9 explainer.

Every drawn road comes from work/v9/sumo_network/district.net.xml (lane shapes, internal junction
lanes, junction outlines). Every vehicle comes from an instrumented replay of the frozen controller
(data/episodes/*/trajectories.npz). Signal colours come from the logged controller state at each
integer decision tick (data/episodes/*/signals.json).

Display convention: the fixture's y axis points south (boundary nodes BN* have the smallest y,
BS* the largest), so the map is drawn with y increasing downwards (north at the top).
"""
from pathlib import Path
import csv
import json
import math

import numpy as np
import sumolib
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FONTS = HERE.parent / "inputs" / "fonts"
SS = 2  # supersampling

# ------------------------------------------------------------------ palette
BG = (236, 234, 228)
BLOCK = (226, 223, 215)
ROAD = (92, 97, 106)
ROAD_EDGE = (70, 74, 82)
MARK = (232, 232, 228)
INK = (33, 37, 43)
MUTED = (92, 99, 110)
FAINT = (150, 156, 165)
SIG_G = (22, 170, 74)
SIG_A = (246, 180, 0)
SIG_R = (214, 40, 40)
AMB_BLUE = (24, 92, 210)
HOSP = (200, 30, 45)
VCOL = {"car": [(200, 205, 212), (126, 140, 160), (178, 64, 58), (64, 78, 98), (230, 230, 226), (150, 130, 96)],
        "motorcycle": [(44, 48, 56)], "autorickshaw": [(236, 196, 38)], "bus": [(196, 86, 40)], "goods": [(88, 110, 128)]}

_fonts = {}


def F(kind, size):
    key = (kind, size)
    if key not in _fonts:
        name = {"h": "manrope-latin-800-normal", "b": "manrope-latin-700-normal", "s": "manrope-latin-600-normal",
                "m": "manrope-latin-500-normal", "r": "manrope-latin-400-normal",
                "mono": "ibm-plex-mono-latin-500-normal", "monob": "ibm-plex-mono-latin-600-normal"}[kind]
        _fonts[key] = ImageFont.truetype(str(FONTS / f"{name}.ttf"), int(size * SS))
    return _fonts[key]


_cmaps = {}
FALLBACK = {"b": "DejaVuSans-Bold", "h": "DejaVuSans-Bold", "s": "DejaVuSans-Bold", "monob": "DejaVuSansMono-Bold",
            "mono": "DejaVuSansMono"}


def _cmap(font):
    """Code points covered by a loaded font (glyphs missing from Manrope / Plex fall back to DejaVu)."""
    path = font.path
    if path not in _cmaps:
        from fontTools.ttLib import TTFont
        _cmaps[path] = set(TTFont(path).getBestCmap())
    return _cmaps[path]


def _fallback(font):
    kind = next(k for (k, sz), f in _fonts.items() if f is font)
    name = FALLBACK.get(kind, "DejaVuSans")
    key = ("fb:" + name, font.size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}.ttf", int(font.size * .9))
    return _fonts[key]


def runs(s, font):
    """Split s into (text, font) runs; a combining mark stays with its base character in the fallback font."""
    import unicodedata
    if "dejavu" in font.path.lower():
        return [[s, font]]
    have = _cmap(font)
    cl = []
    for ch in s:
        if cl and unicodedata.combining(ch):
            cl[-1] += ch
        else:
            cl.append(ch)
    out = []
    for c in cl:
        f = font if all(ord(x) in have for x in c) else _fallback(font)
        if out and out[-1][1] is f:
            out[-1][0] += c
        else:
            out.append([c, f])
    return out


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], int(max(0, min(1, a)) * 255))


# ------------------------------------------------------------------ network
class Net:
    def __init__(self):
        net = sumolib.net.readNet(str(DATA / "network" / "district.net.xml"), withInternal=True)
        self.lanes, self.internal, self.junctions = [], [], {}
        for e in net.getEdges(withInternal=True):
            for ln in e.getLanes():
                shp = np.array(ln.getShape(), float)
                (self.internal if e.isSpecial() else self.lanes).append(
                    dict(id=ln.getID(), edge=e.getID(), index=ln.getIndex(), shape=shp, width=ln.getWidth(),
                         nlanes=e.getLaneNumber()))
        for n in net.getNodes():
            self.junctions[n.getID()] = dict(xy=np.array(n.getCoord(), float), shape=np.array(n.getShape(), float),
                                             tl=n.getType() == "traffic_light")
        self.edges = {r["edge_id"]: r for r in csv.DictReader(open(DATA / "fixture" / "edges.csv"))}
        self.lane_of_edge = {}
        for ln in self.lanes:
            self.lane_of_edge.setdefault(ln["edge"], []).append(ln)
        self.incoming = {}
        for eid, r in self.edges.items():
            self.incoming.setdefault(r["to_node"], []).append(eid)
        self.bbox = (0, 0, 1729, 1623)

    def stopline(self, eid):
        """Stop-line centre, direction of travel and half-width, from the edge's lanes."""
        lns = self.lane_of_edge[eid]
        ends = np.array([l["shape"][-1] for l in lns])
        d = lns[0]["shape"][-1] - lns[0]["shape"][-2]
        d = d / np.linalg.norm(d)
        return ends.mean(0), d, len(lns) * lns[0]["width"] / 2

    def approach_name(self, eid):
        """Compass side the approach comes FROM, in the north-up display."""
        r = self.edges[eid]
        a, b = self.junctions[r["from_node"]]["xy"], self.junctions[r["to_node"]]["xy"]
        v = a - b                      # from junction towards origin; display y points south
        if abs(v[0]) > abs(v[1]):
            return "east" if v[0] > 0 else "west"
        return "south" if v[1] > 0 else "north"


# ------------------------------------------------------------------ replays
class Run:
    def __init__(self, name):
        d = DATA / "episodes" / name
        z = np.load(d / "trajectories.npz")
        self.t, self.off = z["t"], z["offsets"]
        self.vid, self.x, self.y, self.ang, self.spd, self.lane, self.vt = (z["vehicle"], z["x"], z["y"], z["angle"],
                                                                             z["speed"], z["lane"], z["vtype"])
        ix = json.loads((d / "trajectory_index.json").read_text())
        self.vnames, self.lanes, self.types = ix["vehicles"], ix["lanes"], ix["types"]
        self.lane_edge = np.array([l.rsplit("_", 1)[0] if not l.startswith(":") else ":" for l in self.lanes])
        self.amb = self.vnames.index("a000")
        sig = json.loads((d / "signals.json").read_text())
        self.sig = {}
        for t, j, mode, edge, el, req, amb in sig:
            self.sig[(int(t), j)] = (mode, edge, el, req)
        self.result = json.loads((d / "result.json").read_text())
        self.gates = json.loads((d / "gates.json").read_text())
        self.plans = json.loads((d / "plans.json").read_text())
        self.requests = json.loads((d / "requests.json").read_text())
        z2 = np.load(d / "trace.npz")
        self.counts, self.partition = z2["counts"].astype(int), list(z2["partition"])
        self.cstep, self.cstart = float(z2["step"]), float(z2["start"])
        self.progress = {r["t"]: r["progress"].get("a000") for r in self.requests if r.get("progress")}
        self.colour = {}

    def k(self, t):
        return int(np.clip(np.searchsorted(self.t, t + 1e-9, side="right") - 1, 0, len(self.t) - 1))

    def state(self, t):
        """Vehicles at time t: position/heading interpolated between 0.5 s samples (display only);
        speed and lane use the previous sample."""
        k = self.k(t)
        lo, hi = self.off[k:k + 2]
        vid = self.vid[lo:hi]
        xy = np.stack([self.x[lo:hi], self.y[lo:hi]], 1).astype(float)
        ang = self.ang[lo:hi].astype(float)
        if k + 1 < len(self.t) and t > self.t[k]:
            lo2, hi2 = self.off[k + 1:k + 2 + 1]
            vid2 = self.vid[lo2:hi2]
            _, i, j = np.intersect1d(vid, vid2, return_indices=True)
            u = float(np.clip((t - self.t[k]) / (self.t[k + 1] - self.t[k]), 0, 1))
            xy2 = np.stack([self.x[lo2:hi2], self.y[lo2:hi2]], 1)
            xy[i] = xy[i] * (1 - u) + xy2[j] * u
            da = (self.ang[lo2:hi2][j] - ang[i] + 180) % 360 - 180
            ang[i] = ang[i] + da * u
        return vid, xy, ang, self.spd[lo:hi], self.lane_edge[self.lane[lo:hi]], self.vt[lo:hi]

    def amb_xy(self, t):
        vid, xy, ang, sp, le, vt = self.state(t)
        w = np.flatnonzero(vid == self.amb)
        return (xy[w[0]], ang[w[0]], sp[w[0]], le[w[0]]) if len(w) else None

    def signal(self, j, t):
        """Controller state during [floor(t), floor(t)+1): (mode, served edge, elapsed, request edge)."""
        return self.sig.get((int(math.floor(t)), j), ("green", None, 0, None))

    def halted(self, t, eid):
        vid, xy, ang, sp, le, vt = self.state(t)
        return int(((le == eid) & (sp < .1) & (vid != self.amb)).sum())

    def occ(self, t):
        i = int(np.clip((t - self.cstart) / self.cstep, 0, len(self.counts) - 1))
        return self.counts[i]


# ------------------------------------------------------------------ drawing
class View:
    """A camera onto the world: world (x, y) -> screen, y downwards (north up)."""

    def __init__(self, w, h, cx, cy, scale):
        self.w, self.h, self.cx, self.cy, self.s = w, h, cx, cy, scale
        self.im = Image.new("RGBA", (w * SS, h * SS), BG + (255,))
        self.d = ImageDraw.Draw(self.im)

    def P(self, xy):
        xy = np.asarray(xy, float)
        return np.stack(((xy[..., 0] - self.cx) * self.s + self.w / 2, (xy[..., 1] - self.cy) * self.s + self.h / 2), -1)

    def pts(self, xy):
        return [tuple(p * SS) for p in self.P(xy)]

    def visible(self, xy, pad=60):
        p = self.P(xy)
        return p[..., 0].max() > -pad and p[..., 0].min() < self.w + pad and p[..., 1].max() > -pad and p[..., 1].min() < self.h + pad

    # primitives in screen coordinates. Translucent primitives are drawn on a small tile and
    # alpha-composited (Pillow's ImageDraw overwrites alpha instead of blending).
    def _draw(self, cols, box, fn):
        alphas = [c[3] for c in cols if c is not None and len(c) == 4]
        if alphas and max(alphas) == 0:
            return
        if not alphas or min(alphas) >= 255:
            fn(self.d, 0, 0)
            return
        x0, y0, x1, y1 = box
        X0, Y0 = int(max(0, x0 * SS - 4)), int(max(0, y0 * SS - 4))
        X1, Y1 = int(min(self.w * SS, x1 * SS + 4)), int(min(self.h * SS, y1 * SS + 4))
        if X1 <= X0 or Y1 <= Y0:
            return
        tile = Image.new("RGBA", (X1 - X0, Y1 - Y0), (0, 0, 0, 0))
        fn(ImageDraw.Draw(tile), X0, Y0)
        self.im.alpha_composite(tile, (X0, Y0))

    @staticmethod
    def _bb(pts, pad):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def line(self, pts, col, w):
        if len(pts) < 2:
            return
        pw = max(1, int(round(w * SS)))
        self._draw([col], self._bb(pts, w), lambda d, ox, oy: d.line([(x * SS - ox, y * SS - oy) for x, y in pts],
                                                                    fill=col, width=pw, joint="curve"))

    def poly(self, pts, fill=None, outline=None, w=1):
        self._draw([fill, outline], self._bb(pts, w + 1), lambda d, ox, oy: d.polygon(
            [(x * SS - ox, y * SS - oy) for x, y in pts], fill=fill, outline=outline, width=int(w * SS) if outline else 0))

    def circle(self, p, r, fill=None, outline=None, w=1):
        x, y = p
        self._draw([fill, outline], (x - r - w, y - r - w, x + r + w, y + r + w), lambda d, ox, oy: d.ellipse(
            ((x - r) * SS - ox, (y - r) * SS - oy, (x + r) * SS - ox, (y + r) * SS - oy), fill=fill, outline=outline,
            width=max(1, int(w * SS)) if outline else 0))

    def rect(self, b, fill=None, outline=None, w=1, r=0):
        x0, y0, x1, y1 = b
        x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        self._draw([fill, outline], (x0 - w, y0 - w, x1 + w, y1 + w), lambda d, ox, oy: d.rounded_rectangle(
            (x0 * SS - ox, y0 * SS - oy, x1 * SS - ox, y1 * SS - oy), int(r * SS), fill=fill, outline=outline,
            width=max(1, int(w * SS)) if outline else 0))

    def text(self, p, s, font, col, anchor="la"):
        rs = runs(s, font)
        if len(rs) > 1 or rs and rs[0][1] is not font:
            w = sum(self.d.textlength(t, font=f) for t, f in rs) / SS
            x = p[0] - {"l": 0, "m": w / 2, "r": w}[anchor[0]]
            v = anchor[1]
            # baseline of the primary font for the requested vertical anchor
            y = p[1] + (self.d.textbbox((0, 0), "x", font=font, anchor="l" + v)[1] - self.d.textbbox((0, 0), "x", font=font, anchor="ls")[1]) / SS
            for t, f in rs:
                self.text((x, y), t, f, col, "ls")
                x += self.d.textlength(t, font=f) / SS
            return
        if len(col) == 4 and col[3] < 255:
            bb = self.d.textbbox((p[0] * SS, p[1] * SS), s, font=font, anchor=anchor)
            self._draw([col], (bb[0] / SS, bb[1] / SS, bb[2] / SS, bb[3] / SS),
                       lambda d, ox, oy: d.text((p[0] * SS - ox, p[1] * SS - oy), s, font=font, fill=col, anchor=anchor))
        else:
            self.d.text((p[0] * SS, p[1] * SS), s, font=font, fill=col, anchor=anchor)

    def tw(self, s, font):
        return sum(self.d.textlength(t, font=f) for t, f in runs(s, font)) / SS

    def down(self):
        return self.im.resize((self.w, self.h), Image.Resampling.LANCZOS)

    # ---------------------------------------------------------- network
    def blocks(self, net):
        """City blocks: the faces between junction rows/columns, inset from the road centrelines."""
        J = net.junctions
        for r in range(3):
            for c in range(3):
                q = np.array([J[f"J{r}{c}"]["xy"], J[f"J{r}{c+1}"]["xy"], J[f"J{r+1}{c+1}"]["xy"], J[f"J{r+1}{c}"]["xy"]])
                cen = q.mean(0)
                k = 22.0
                inset = np.array([p + (cen - p) / np.linalg.norm(cen - p) * k * 1.41 for p in q])
                if self.visible(inset):
                    self.poly([tuple(p) for p in self.P(inset)], fill=BLOCK + (255,), outline=(214, 210, 200, 255), w=1)

    def roads(self, net):
        s = self.s
        for j, J in net.junctions.items():
            if len(J["shape"]) > 2 and self.visible(J["shape"]):
                self.poly([tuple(p) for p in self.P(J["shape"])], fill=ROAD)
        for ln in net.lanes + net.internal:
            if not self.visible(ln["shape"]):
                continue
            w = max(4.2 if s < 1 else 1.6, ln["width"] * s)
            self.line([tuple(p) for p in self.P(ln["shape"])], ROAD, w + (0 if ln in net.internal else .6))
        if s < 1.2:
            return
        # lane markings: dashed divider between same-direction lanes; solid line between directions
        for ln in net.lanes:
            if not self.visible(ln["shape"]):
                continue
            shp = ln["shape"]
            d = shp[-1] - shp[0]
            L = np.linalg.norm(d)
            u = d / L
            left = np.array([-u[1], u[0]])              # world left of travel (y-up world)
            if ln["index"] == ln["nlanes"] - 1:         # leftmost lane: direction divider
                a, b = shp[0] + left * ln["width"] / 2, shp[-1] + left * ln["width"] / 2
                self.line([tuple(p) for p in self.P(np.array([a + u * 4, b - u * 4]))], rgba(MARK, .75), max(1, .15 * s))
            if ln["index"] > 0:                         # dashed divider to the lane on its right
                a = shp[0] - left * ln["width"] / 2
                for t0 in np.arange(6, L - 8, 9):
                    p0, p1 = a + u * t0, a + u * min(t0 + 4.5, L - 8)
                    self.line([tuple(p) for p in self.P(np.array([p0, p1]))], rgba(MARK, .85), max(1, .14 * s))

    def signals(self, net, run, t, heads=(), lamp_scale=1.0):
        """Stop-line bars for every signalised approach; optional signal heads for focus junctions."""
        for j, J in net.junctions.items():
            if not J["tl"]:
                continue
            mode, served, el, req = run.signal(j, t)
            for eid in net.incoming.get(j, []):
                c, d, half = net.stopline(eid)
                if not self.visible(c[None], 40):
                    continue
                col = SIG_G if (mode == "green" and served == eid) else SIG_A if (mode == "amber" and served == eid) else SIG_R
                n = np.array([-d[1], d[0]])
                a, b = self.P(c - n * half), self.P(c + n * half)
                self.line([tuple(a), tuple(b)], col, max(2.2, .9 * self.s))
                if j in heads:
                    self.head(c, d, n, half, col, lamp_scale)

    def head(self, c, d, n, half, col, k=1.0):
        """A three-lamp signal head on the near-side kerb, just before the stop line."""
        base = self.P(c - d * 6 - n * (half + 3.5))
        x, y = base
        r = 5.5 * k
        self.rect((x - r - 3, y - 3 * r - 5, x + r + 3, y + 3 * r + 5), fill=(28, 31, 36, 255), outline=(255, 255, 255, 230), w=1.5, r=4)
        for i, lc in enumerate([SIG_R, SIG_A, SIG_G]):
            on = lc == col
            self.circle((x, y + (i - 1) * (2 * r + 1)), r, fill=lc if on else (70, 74, 80))

    # ---------------------------------------------------------- vehicles
    DIMS = {"car": (4.5, 1.8), "motorcycle": (2.1, .8), "autorickshaw": (2.8, 1.3), "bus": (11.5, 2.5), "goods": (8, 2.3),
            "ambulance": (5, 2)}

    def vehicles(self, run, t, min_px=0.0, tint=None, hide_amb=False):
        vid, xy, ang, sp, le, vt = run.state(t)
        P = self.P(xy)
        s = self.s
        amb = None
        for k, p, a, v, e, ty in zip(vid, P, ang, sp, le, vt):
            if not (-30 < p[0] < self.w + 30 and -30 < p[1] < self.h + 30):
                continue
            tname = run.types[ty]
            if k == run.amb:
                amb = (p, a)
                continue
            L, B = self.DIMS[tname]
            L, B = max(L * s, min_px), max(B * s, min_px * .45)
            pal = VCOL[tname]
            col = pal[k % len(pal)]
            if tint is not None:
                tc = tint(k, e, v)
                if tc is not None:
                    col = tc
            self.car(p, a, L, B, col, stopped=v < .1 and s > 1.2)
        if amb is not None and not hide_amb:
            self.ambulance(amb[0], amb[1])
        return amb

    def _frame(self, p, a):
        rad = math.radians(a)
        u = np.array([math.sin(rad), math.cos(rad)])      # SUMO heading, display y down -> (sin, cos)
        n = np.array([-u[1], u[0]])
        return np.asarray(p, float), u, n

    def car(self, p, a, L, B, col, stopped=False):
        p, u, n = self._frame(p, a)
        c = p - u * L / 2                                   # SUMO position is the front bumper
        body = [tuple(c + u * x + n * y) for x, y in [(-L / 2, -B / 2), (L / 2 - B * .25, -B / 2), (L / 2, -B * .3),
                                                     (L / 2, B * .3), (L / 2 - B * .25, B / 2), (-L / 2, B / 2)]]
        self.poly(body, fill=col + (255,), outline=(40, 44, 50, 200), w=.8)
        if L > 9:
            self.poly([tuple(c + u * x + n * y) for x, y in [(L * .12, -B * .38), (L * .3, -B * .32), (L * .3, B * .32), (L * .12, B * .38)]],
                      fill=(40, 52, 66, 150))
        if stopped and L > 9:
            for k in (-1, 1):
                self.circle(tuple(c - u * L * .47 + n * B * .3 * k), max(1.2, B * .13), fill=(220, 30, 30, 255))

    def ambulance(self, p, a, size=1.0):
        p, u, n = self._frame(p, a)
        L = max(5 * self.s, 26) * size
        B = L * .42
        self.circle(tuple(p - u * L / 2), L * .95, fill=rgba(AMB_BLUE, .14), outline=rgba(AMB_BLUE, .85), w=2)
        c = p - u * L / 2
        body = [tuple(c + u * x + n * y) for x, y in [(-L / 2, -B / 2), (L / 2 - B * .2, -B / 2), (L / 2, -B * .3),
                                                     (L / 2, B * .3), (L / 2 - B * .2, B / 2), (-L / 2, B / 2)]]
        self.poly(body, fill=(255, 255, 255, 255), outline=(30, 40, 60, 255), w=1.2)
        self.poly([tuple(c + u * x + n * y) for x, y in [(-L / 2, -B * .5), (L * .3, -B * .5), (L * .3, -B * .32), (-L / 2, -B * .32)]],
                  fill=AMB_BLUE + (255,))
        self.poly([tuple(c + u * x + n * y) for x, y in [(-L / 2, B * .32), (L * .3, B * .32), (L * .3, B * .5), (-L / 2, B * .5)]],
                  fill=AMB_BLUE + (255,))
        cc = c - u * L * .1
        self.line([tuple(cc - u * B * .32), tuple(cc + u * B * .32)], HOSP, max(2, B * .16))
        self.line([tuple(cc - n * B * .32), tuple(cc + n * B * .32)], HOSP, max(2, B * .16))
