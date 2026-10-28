#!/usr/bin/env python3
"""AmbuGo — 20-second cinematic explainer of one selective-priority decision at PCL Chowk.

One intersection, one ambulance (a002), one decision. Everything that moves or is
counted is read from the frozen replays (P3 = AmbuGo B1000, P0 = no priority) and the
controller log; see output/cinematic/verification.md.

    python src/cinematic.py                      # output/cinematic/AmbuGo_20s_Cinematic.mp4
    python src/cinematic.py --stills 0 2 4 ...   # stills (video seconds)
    python src/cinematic.py --contact            # contact sheet of 0,2,...,20 s
    python src/cinematic.py --draft              # half-resolution draft video
"""
from pathlib import Path
import argparse
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapview  # noqa: E402
from data import Episode, ROOT, llxy  # noqa: E402

W, H, FPS, DUR = 1920, 1080, 30, 20.0
SS = 2
OUT = ROOT / "output" / "cinematic"
FONTS = ROOT / "inputs" / "fonts"

# ---------------------------------------------------------------- palette
BG0 = (11, 16, 26)          # deepest navy
BG1 = (36, 46, 64)          # lightest backdrop tone
ROAD = (54, 66, 86)
ROAD_EDGE = (70, 84, 106)
IVORY = (240, 236, 226)
MUTED = (150, 160, 178)
DIM = (96, 108, 128)
CYAN = (0, 140, 230)
EMERALD = (20, 170, 110)
CORAL = (255, 106, 92)
AMBER_SIG = (255, 191, 71)
VIOLET = (167, 139, 250)
AMBER = (245, 166, 35)
TEAL = (45, 212, 191)
SLATE = (74, 86, 106)
APP = {"172380615#1": VIOLET, "172382686#4": AMBER, "172660580#0": TEAL}
APP_NAME = {"172380615#1": "NW", "172382686#4": "SW", "172660580#0": "NE", "335061218": "SE"}

_f = {}


def F(kind, size):
    """Manrope for text, IBM Plex Mono for clock values and numerals."""
    key = (kind, size)
    if key not in _f:
        name = {"h": "manrope-latin-800-normal", "b": "manrope-latin-700-normal", "s": "manrope-latin-600-normal",
                "m": "manrope-latin-500-normal", "r": "manrope-latin-400-normal",
                "mono": "ibm-plex-mono-latin-500-normal", "monob": "ibm-plex-mono-latin-600-normal",
                "monol": "ibm-plex-mono-latin-400-normal"}[kind]
        _f[key] = ImageFont.truetype(str(FONTS / f"{name}.ttf"), int(size * SS))
    return _f[key]


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = float(np.clip(x, 0, 1))
    return 1 - (1 - x) ** 3


def win(v, a, b, fi=.35, fo=.35):
    """Opacity of an element visible in [a, b] with fades."""
    if v < a or v > b:
        return 0.0
    return min(ease((v - a) / fi) if fi else 1, ease((b - v) / fo) if fo else 1)


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], int(np.clip(a, 0, 1) * 255))


def mix(a, b, u):
    u = float(np.clip(u, 0, 1))
    return tuple(int(round(x * (1 - u) + y * u)) for x, y in zip(a[:3], b[:3]))


# ----------------------------------------------------------- time warp
# Video-second -> simulation-second knots. Monotone cubic (PCHIP) keeps the replay
# speed continuous: no jerks, never reverses, and every event keeps its recorded order.
KNOTS = [(0.0, 777.0), (2.5, 785.0), (4.6, 786.6), (4.9, 787.0), (6.6, 791.8), (6.9, 792.0), (8.6, 795.3),
         (9.6, 796.6), (10.4, 797.8), (12.6, 804.5), (16.6, 862.0), (18.2, 880.0), (20.0, 880.0)]


def pchip(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.zeros_like(ys)
    for i in range(1, len(xs) - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = d[0], 0.0 if d[-1] == 0 else d[-1]

    def f(x):
        i = int(np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2))
        t = (x - xs[i]) / h[i]
        t = float(np.clip(t, 0, 1))
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        return h00 * ys[i] + h10 * h[i] * m[i] + h01 * ys[i + 1] + h11 * h[i] * m[i + 1]
    return f


SIM = pchip(*zip(*KNOTS))


def rate(v):
    return (SIM(min(DUR, v + 1 / 60)) - SIM(max(0, v - 1 / 60))) / (min(DUR, v + 1 / 60) - max(0, v - 1 / 60))


# ------------------------------------------------------------ storyboard
STAGES = ["FORECAST", "DECISION", "ACTUATION", "CONSEQUENCES"]
# (start, end, stage index or None, headline, subline)
BEATS = [
    (0.0, 2.5, None, "Ambulance approaching PCL Chowk", "Recorded traffic simulation on the Mohali road network"),
    (2.5, 6.9, 0, "Forecast exceeds the budget", "Expected delay to the other approaches is above 1,000 vehicle-seconds: request withheld"),
    (6.9, 9.6, 1, "Forecast within the budget", "913 vehicle-seconds: request admitted; the pedestrian interval completes first"),
    (9.6, 12.6, 2, "Green at the earliest legal time", "The ambulance is served 72 s earlier than in the no-priority run"),
    (12.6, 16.6, 3, "Effect on adjacent roads", "Conflicting approaches wait 52 s longer; traffic shifts across the junction"),
    (16.6, 20.0, None, "Priority, granted within a forecast budget", "Observed additional stopped time on the conflicting approaches, against the forecast of 913"),
]

# Roads around PCL Chowk, ordered outward from the junction (approaches reversed).
ROADS = [
    ("SE approach", ["335061218", "335061226"], True),
    ("NE approach", ["172660580#0", "129130264#4"], True),
    ("SW approach", ["172382686#4"], True),
    ("NW approach", ["172380615#1"], True),
    ("NE exit", ["172382686#6", "172382686#7"], False),
    ("NW exit", ["172658127#1"], False),
    ("SW exit", ["336665911#0", "336665911#1"], False),
]
MORE, FEWER = CORAL, (120, 160, 255)


def beat_at(v):
    for b in BEATS:
        if b[0] <= v < b[1]:
            return b
    return BEATS[-1]


# ------------------------------------------------------------- canvas
class Cv:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def P(self, p):
        return (p[0] * SS, p[1] * SS)

    def line(self, pts, c, w):
        if len(pts) > 1:
            self.d.line([self.P(p) for p in pts], fill=c, width=max(1, int(round(w * SS))), joint="curve")

    def poly(self, pts, fill=None, outline=None, w=1):
        self.d.polygon([self.P(p) for p in pts], fill=fill, outline=outline, width=int(w * SS) if outline else 0)

    def circle(self, p, r, fill=None, outline=None, w=1):
        x, y = p
        self.d.ellipse(((x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS), fill=fill, outline=outline,
                       width=max(1, int(w * SS)) if outline else 0)

    def rect(self, b, fill=None, outline=None, w=1, r=0):
        x0, y0, x1, y1 = b
        bb = (min(x0, x1) * SS, min(y0, y1) * SS, max(x0, x1) * SS, max(y0, y1) * SS)
        self.d.rounded_rectangle(bb, int(r * SS), fill=fill, outline=outline, width=max(1, int(w * SS)) if outline else 0)

    def arc(self, c, r, a0, a1, col, w):
        x, y = c
        if a1 - a0 < .05:
            return
        self.d.arc(((x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS), a0, a1, fill=col, width=int(w * SS))

    def text(self, p, s, font, col, anchor="la", track=0.0):
        if not track:
            self.d.text(self.P(p), s, font=font, fill=col, anchor=anchor)
            return
        # letter-spaced caps
        widths = [self.d.textlength(ch, font=font) for ch in s]
        total = sum(widths) + track * SS * (len(s) - 1)
        x, y = p[0] * SS, p[1] * SS
        if anchor[0] == "m":
            x -= total / 2
        elif anchor[0] == "r":
            x -= total
        for ch, wch in zip(s, widths):
            self.d.text((x, y), ch, font=font, fill=col, anchor="l" + anchor[1])
            x += wch + track * SS

    def tw(self, s, font, track=0.0):
        return (self.d.textlength(s, font=font) + track * SS * (len(s) - 1)) / SS

    def down(self):
        return self.im.resize((self.w, self.h), Image.Resampling.LANCZOS)


# ------------------------------------------------------------- the film
class Film:
    def __init__(self, scale=1.0):
        self.ep = Episode()
        self.scale_out = scale
        mapview.MW, mapview.MH = W, H
        self.base = mapview.Basemap()
        ep = self.ep
        self.J = ep.J1xy
        self.route = ["335061226", "335061218", "172382686#6", "172382686#7"]
        self.route_pts = ep.route_points(self.route)
        self.trail_t = np.arange(777.0, 880.01, 0.25)
        self.trail = np.array([ep.amb_pos(t) for t in self.trail_t])
        self.road_pts = {e: p for e, p in ep.exy.items()}
        self.conn_pts = [llxy(*np.array(v[0]["shape_lonlat"]).T) for v in ep.connectors.values() if v]
        self.bg_cache = {}
        # recorded green starts at J1 for the mini timeline
        self.tl = {pol: [s for s in ep.timeline[pol, ep.J1] if s[1] > 776 and s[0] < 905] for pol in ("P3", "P0")}

    # ------------------------------------------------------------ camera
    def camera(self, v):
        J = self.J
        s0 = 4.2      # px per Mercator metre (hero framing)
        c_open = np.array([J[0] + 95, J[1] - 70, W / (2 * 3.0)])
        c_hero = np.array([J[0] + 40, J[1] - 18, W / (2 * s0)])
        c_pass = np.array([J[0] + 52, J[1] + 2, W / (2 * s0)])
        c_wide = np.array([J[0] + 30, J[1] - 32, W / (2 * 3.3)])
        c_adj = np.array([J[0] + 10, J[1] - 20, W / (2 * 2.9)])
        keys = [(0, c_open), (2.8, c_hero), (9.6, c_hero), (11.6, c_pass), (12.4, c_pass), (14.0, c_adj), (16.4, c_adj),
                (18.0, c_wide), (20, c_wide)]
        for (a, ca), (b, cb) in zip(keys, keys[1:]):
            if a <= v <= b:
                u = ease((v - a) / (b - a))
                c = ca * (1 - u) + cb * u
                c[2] = math.exp(math.log(ca[2]) * (1 - u) + math.log(cb[2]) * u)
                return c
        return keys[-1][1]

    def S(self, xy, cam):
        xy = np.asarray(xy, float)
        s = W / (2 * cam[2])
        return np.stack(((xy[..., 0] - cam[0]) * s + W / 2, -(xy[..., 1] - cam[1]) * s + H / 2), -1)

    # -------------------------------------------------------- backdrop
    def backdrop(self, cam):
        key = tuple(np.round(cam, 1))
        if key in self.bg_cache:
            return self.bg_cache[key].copy()
        raw = self.base.render(cam)   # Google Maps capture (labels kept), lightly desaturated
        img = Image.blend(raw, ImageOps.grayscale(raw).convert("RGB"), .25)
        img = Image.blend(img, Image.new("RGB", (W, H), (236, 238, 242)), .10)
        if len(self.bg_cache) > 4:
            self.bg_cache.pop(next(iter(self.bg_cache)))
        self.bg_cache[key] = img.copy()
        return img

    # ------------------------------------------------------------ frame
    def frame(self, f):
        v = f / FPS
        t = float(SIM(v))
        cam = self.camera(v)
        img = self.backdrop(cam).convert("RGBA")
        glow = Cv(W, H)
        cv = Cv(W, H)
        self.approaches(cv, glow, cam, v, t)
        self.route_layer(cv, glow, cam, v, t)
        self.adjacent(cv, cam, v, t)
        self.signals(cv, glow, cam, v, t)
        self.vehicles(cv, cam, v, t)
        amb = self.ambulance(cv, glow, cam, v, t)
        self.request_link(cv, glow, cam, v, t, amb)
        self.ped_ring(cv, cam, v, t)
        self.exit_marker(cv, cam, v, t)
        g = glow.down().filter(ImageFilter.GaussianBlur(9))
        img = Image.alpha_composite(img, g)
        img = Image.alpha_composite(img, cv.down())
        ui = Cv(W, H)
        self.chrome(ui, v, t)
        self.card(ui, v, t, img)
        self.headline(ui, v)
        img = Image.alpha_composite(img, self.card_shadow(v))
        img = Image.alpha_composite(img, ui.down())
        img = img.convert("RGB")
        img.paste(self.base.logo.convert("RGB").point(lambda x: int(x * .55)), (W - 300, H - 44))
        cr = self.base.credit.convert("RGB").point(lambda x: int(x * .6))
        img.paste(cr, (W - 172, H - 38))
        if self.scale_out != 1:
            img = img.resize((int(W * self.scale_out), int(H * self.scale_out)), Image.Resampling.LANCZOS)
        return img

    def scrim(self):
        if not hasattr(self, "_scrim"):
            g = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(g)
            d.rectangle((0, 0, 1150, 250), fill=170)
            d.rectangle((0, H - 110, 760, H), fill=150)
            g = g.filter(ImageFilter.GaussianBlur(70))
            self._scrim = Image.merge("RGBA", (Image.new("L", (W, H), BG0[0]), Image.new("L", (W, H), BG0[1]),
                                               Image.new("L", (W, H), BG0[2]), g))
        return self._scrim

    # ------------------------------------------------------------- roads
    def roads(self, cv, cam):
        s = W / (2 * cam[2])
        lw = 3.2 / math.cos(math.radians(30.7))
        for pts in self.conn_pts:
            p = self.S(pts, cam)
            if p[:, 0].max() < -50 or p[:, 0].min() > W + 50 or p[:, 1].max() < -50 or p[:, 1].min() > H + 50:
                continue
            cv.line([tuple(x) for x in p], rgba(ROAD, .9), max(2, lw * s * .9))
        for e, pts in self.road_pts.items():
            p = self.S(pts, cam)
            if p[:, 0].max() < -50 or p[:, 0].min() > W + 50 or p[:, 1].max() < -50 or p[:, 1].min() > H + 50:
                continue
            w = max(2.5, self.ep.edges[e]["lanes"] * lw * s)
            cv.line([tuple(x) for x in p], rgba(ROAD_EDGE, .95), w + 2)
            cv.line([tuple(x) for x in p], rgba(ROAD, 1), w)

    # --------------------------------------------------------- approaches
    def app_strength(self, v):
        return max(win(v, 2.7, 6.8, .7, .7) * .9, win(v, 16.6, 20, .8, 0) * .7)

    def approaches(self, cv, glow, cam, v, t):
        a = self.app_strength(v)
        if a <= 0:
            return
        s = W / (2 * cam[2])
        e785 = self.ep.ev[785]
        for edge, col in APP.items():
            p = self.S(self.ep.exy[edge], cam)
            # extend the glow upstream along predecessors so short links read as approaches
            pts = [tuple(x) for x in p]
            k = .55 if v < 10 else .28
            cv.line(pts, rgba(col, k * a), max(6, 3.7 * s * 1.1))
            glow.line(pts, rgba(col, (.8 if v < 10 else .45) * a), 16)
        # forecast contributions as chips at the approaches (PREDICT / DECIDE only)
        c = win(v, 3.0, 6.8, .6, .6)
        if c > 0:
            for ap in e785["conflicting_approaches"]:
                edge = ap["edge"]
                p = self.S(self.ep.exy[edge], cam)
                q = p[0] * .25 + p[-1] * .75
                off = {"172380615#1": (-70, -46), "172382686#4": (-60, 54), "172660580#0": (66, -40)}[edge]
                x, y = q[0] + off[0], q[1] + off[1]
                label = f"{APP_NAME[edge]}  {ap['burden_proxy_veh_s']:.0f}"
                fnt = F("monob", 20)
                w = cv.tw(label, fnt)
                cv.line([tuple(q), (x, y)], rgba(APP[edge], .7 * c), 1.5)
                cv.rect((x - w / 2 - 14, y - 17, x + w / 2 + 14, y + 17), fill=rgba((14, 20, 32), .88 * c),
                        outline=rgba(APP[edge], .9 * c), w=1.5, r=17)
                cv.text((x, y), label, fnt, rgba(APP[edge], c), "mm")

    # ----------------------------------------------------- adjacent roads
    def road_delta(self, t):
        """Vehicles present, AmbuGo minus no priority, averaged over +-3 s, per road group."""
        ep = self.ep
        k = ep.k(t)
        lo, hi = max(0, k - 6), k + 7
        out = {}
        for name, edges, _ in ROADS:
            idx = [ep.ei[e] for e in edges]
            out[name] = float((ep.c3[lo:hi, idx] - ep.c0[lo:hi, idx]).sum(1).mean())
        return out

    def road_line(self, edges, inbound):
        pts = []
        for e in edges:
            p = self.ep.exy[e]
            pts.extend(p[::-1] if inbound else p)
        return np.array(pts)

    def adjacent(self, cv, cam, v, t):
        a = win(v, 12.7, 16.8, .8, .7)
        if a <= 0:
            return
        s = W / (2 * cam[2])
        d = self.road_delta(t)
        for name, edges, inbound in ROADS:
            val = d[name]
            line = self.road_line(edges, inbound)
            seg = np.cumsum(np.r_[0, np.hypot(*np.diff(line, axis=0).T)])
            keep = line[seg <= 260]
            q = self.S(keep, cam)
            col = MORE if val > 0 else FEWER
            k = min(1.0, abs(val) / 5) * a
            if abs(val) >= .25:
                cv.line([tuple(x) for x in q], rgba(col, .2 + .55 * k), max(10, 3.7 * s * 1.5))
            j = int(np.searchsorted(seg, 120 if inbound else 70))
            j = min(j, len(line) - 1)
            p = self.S(line[j], cam)
            if p[1] < 250 or p[0] > 1300 or abs(val) < .5:
                continue
            lab = f"{val:+.1f}"
            fnt = F("monob", 19)
            w = cv.tw(lab, fnt)
            colt = col if abs(val) >= .5 else MUTED
            cv.rect((p[0] - w / 2 - 12, p[1] - 15, p[0] + w / 2 + 12, p[1] + 15), fill=rgba((14, 20, 32), .9 * a),
                    outline=rgba(colt, .9 * a), w=1.5, r=15)
            cv.text((p[0], p[1]), lab, fnt, rgba(colt, a), "mm")

    # -------------------------------------------------------------- route
    def route_layer(self, cv, glow, cam, v, t):
        p = self.S(self.route_pts, cam)
        cv.line([tuple(x) for x in p], rgba(CYAN, .22), 4)
        k = int(np.searchsorted(self.trail_t, t))
        tr = self.trail[:max(k, 1)]
        tr = tr[np.isfinite(tr).all(1)]
        if len(tr) > 1:
            q = [tuple(x) for x in self.S(tr, cam)]
            cv.line(q, rgba(CYAN, .95), 4.5)
            glow.line(q, rgba(CYAN, .55), 10)

    # ------------------------------------------------------------ signals
    def signals(self, cv, glow, cam, v, t):
        ep = self.ep
        mode, served, req = ep.signal_state("P3", ep.J1, t)
        s = W / (2 * cam[2])
        for edge in ep.signals[ep.J1]["incoming"]:
            p = self.S(ep.exy[edge], cam)
            end, u = p[-1], p[-1] - p[-2]
            u = u / max(np.linalg.norm(u), 1e-6)
            n = np.array([-u[1], u[0]])
            half = max(11, 3.7 * s * .6)
            if mode == 0 and served == edge:
                col = EMERALD
            elif mode == 1 and served == edge:
                col = AMBER_SIG
            else:
                col = CORAL
            a, b = tuple(end - n * half), tuple(end + n * half)
            cv.line([a, b], rgba(col, 1), 6)
            glow.line([a, b], rgba(col, .9 if col == EMERALD else .5), 14)
            if edge == ep.amb_app and 796.5 <= t < 799.5:
                k = (t - 796.5) / 3
                cv.circle(tuple(end), 18 + 60 * k, outline=rgba(EMERALD, 1 - k), w=3)

    # ----------------------------------------------------------- vehicles
    def vehicles(self, cv, cam, v, t):
        ep = self.ep
        vid, xy, ang, spd, edge = ep.P3.positions(t)
        ps = self.S(xy, cam)
        s = W / (2 * cam[2])
        tint = self.app_strength(v)
        app_idx = {ep.ei[e]: c for e, c in APP.items()}
        for k, p, a_, sp, e in zip(vid, ps, ang, spd, edge):
            if k in ep.P3.amb_idx or not (-40 < p[0] < W + 40 and -40 < p[1] < H + 40):
                continue
            L, B = ep.P3.dims[k] * 1.16 * s * 1.6
            L, B = max(L, 14), max(B, 7)
            rad = math.radians(a_)
            u = np.array([math.sin(rad), -math.cos(rad)])
            n = np.array([u[1], -u[0]])
            c0 = p - u * L / 2
            col = SLATE
            if e in app_idx and tint > 0:
                col = mix(SLATE, mix(app_idx[e], IVORY, .25), tint)
            body = [tuple(c0 + u * x + n * y) for x, y in [(-L / 2, -B / 2), (L / 2, -B / 2), (L / 2, B / 2), (-L / 2, B / 2)]]
            cv.poly(body, fill=rgba(col, .97), outline=rgba((12, 16, 26), .9), w=1.2)
            if sp < 0.1:
                for kk in (-1, 1):
                    cv.circle(tuple(c0 - u * L * .46 + n * B * .3 * kk), max(1.4, B * .16), fill=rgba(CORAL, .9))

    # ---------------------------------------------------------- ambulance
    def ambulance(self, cv, glow, cam, v, t):
        ep = self.ep
        p = ep.amb_pos(t)
        if not np.isfinite(p).all():
            return None
        q = self.S(p, cam)
        if not (-60 < q[0] < W + 60 and -60 < q[1] < H + 60):
            return q
        # heading from the recorded trail
        k = int(np.searchsorted(self.trail_t, t))
        a, b = self.trail[max(0, k - 3)], self.trail[min(len(self.trail) - 1, k + 1)]
        d = self.S(b, cam) - self.S(a, cam)
        if np.linalg.norm(d) < .5:
            d = np.array([-0.6, -0.8])
        u = d / np.linalg.norm(d)
        n = np.array([-u[1], u[0]])
        pulse = .5 + .5 * math.sin(v * 2 * math.pi * .9)
        glow.circle(tuple(q), 40 + 4 * pulse, fill=rgba(CYAN, .45))
        cv.circle(tuple(q), 36 + 2 * pulse, fill=rgba((14, 20, 32), .55), outline=rgba(CYAN, .8), w=2.5)
        L, B = 46, 21
        c0 = q - u * 2
        body = [tuple(c0 + u * x + n * y) for x, y in [(-L / 2, -B / 2), (L / 2 - 4, -B / 2), (L / 2, -B / 2 + 4),
                                                        (L / 2, B / 2 - 4), (L / 2 - 4, B / 2), (-L / 2, B / 2)]]
        cv.poly(body, fill=rgba(IVORY), outline=rgba(CYAN), w=2.5)
        cv.line([tuple(c0 - u * 8), tuple(c0 + u * 8)], rgba(CORAL), 4.5)
        cv.line([tuple(c0 - n * 7), tuple(c0 + n * 7)], rgba(CORAL), 4.5)
        tip = q + u * 56
        cv.poly([tuple(tip), tuple(tip - u * 12 + n * 8), tuple(tip - u * 7), tuple(tip - u * 12 - n * 8)], fill=rgba(CYAN, .95))
        return q

    def exit_marker(self, cv, cam, v, t):
        """Closing shot: where the ambulance went (its recorded trail leaves the frame)."""
        a = win(v, 17.0, 20, .8, 0)
        if a <= 0:
            return
        q = self.S(self.trail[np.isfinite(self.trail).all(1)], cam)
        inside = q[(q[:, 0] > 40) & (q[:, 0] < 1250) & (q[:, 1] > 300) & (q[:, 1] < H - 120)]
        if not len(inside):
            return
        p = inside[-1]
        u = q[-1] - p
        u = u / max(np.linalg.norm(u), 1e-6)
        x, y = p[0] + 36, p[1] + 96
        cv.rect((x, y - 22, x + 270, y + 22), fill=rgba((14, 20, 32), .9 * a), outline=rgba(CYAN, a), w=1.5, r=22)
        cv.text((x + 135, y), "a002 CONTINUES TO HOSPITAL", F("monob", 14), rgba(CYAN, a), "mm", track=1.5)
        cv.line([(x + 30, y - 22), tuple(p - u * 6)], rgba(CYAN, .7 * a), 1.5)

    def request_link(self, cv, glow, cam, v, t, amb):
        """Dashed link ambulance -> its stop line: the request and its current status."""
        if amb is None or not (785 <= t < 801.5):
            return
        ep = self.ep
        e = ep.event_at(t)
        if e is None:
            return
        col = CORAL if e["reason"] == "burden_budget_exceeded" else EMERALD
        a = win(v, 2.8, 11.6, .6, .6)
        p = self.S(ep.exy[ep.amb_app], cam)[-1]
        d = p - amb
        L = np.linalg.norm(d)
        if L < 30:
            return
        u = d / L
        ph = (v * 40) % 14
        s0 = 30
        while s0 < L - 14:
            a0, a1 = amb + u * (s0 + ph * 0), amb + u * min(s0 + 7, L - 14)
            cv.line([tuple(a0), tuple(a1)], rgba(col, .9 * a), 3)
            s0 += 14

    def ped_ring(self, cv, cam, v, t):
        ep = self.ep
        if not (771.5 <= t < 796.5):
            return
        a = win(v, 7.4, 9.75, .6, .2) if v > 7 else win(v, 0.3, 2.4, .6, .4) * .7
        if a <= 0:
            return
        c = self.S(ep.J1xy, cam)
        rem = 796.5 - t
        frac = rem / 25.0
        cv.circle(tuple(c), 46, fill=rgba((14, 20, 32), .75 * a))
        cv.arc(tuple(c), 46, -90, -90 + 360, rgba(DIM, .5 * a), 4)
        cv.arc(tuple(c), 46, -90, -90 + 360 * frac, rgba(IVORY, a), 5)
        cv.text((c[0], c[1] - 9), f"{rem:4.1f}s", F("monob", 19), rgba(IVORY, a), "mm")
        cv.text((c[0], c[1] + 13), "PED", F("mono", 12), rgba(MUTED, a), "mm", track=2)

    # ------------------------------------------------------------- chrome
    def chrome(self, ui, v, t):
        b = beat_at(v)
        # headline panel
        hw = max(ui.tw(b[3], F("h", 50)), ui.tw(b[4], F("m", 24)), 760) + 72
        ui.rect((36, 26, 36 + hw, 222), fill=rgba((16, 23, 36), .9), outline=rgba((70, 84, 106), .7), w=1.2, r=20)
        ui.rect((36, H - 92, 700, H - 14), fill=rgba((16, 23, 36), .88), r=16)
        x = 64
        for i, s in enumerate(STAGES):
            active = b[2] == i
            done = b[2] is not None and i < b[2] or v >= 16.6
            col = IVORY if active else (MUTED if done else DIM)
            lab = f"0{i + 1}  {s}"
            fnt = F("mono" if not active else "monob", 17)
            ui.text((x, 58), lab, fnt, rgba(col, 1 if active else .75), "lm", track=2.5)
            w = ui.tw(lab, fnt, 2.5)
            if active:
                ui.rect((x, 74, x + w, 77), fill=rgba(CYAN, 1), r=1)
            x += w + 44
        # clock, bottom left
        el = t - 760
        fnt = F("mono", 18)
        r = rate(v)
        speed = "hold" if r < 0.05 else (f"{r:.1f}x" if r >= 1 else f"{r:.2f}x")
        ui.text((64, H - 52), f"SIM {t:6.1f} s", F("monob", 20), rgba(IVORY, .92), "lm", track=1)
        ui.text((250, H - 52), f"T+{int(el) // 60}:{int(el) % 60:02d}  ·  {speed}", fnt, rgba(MUTED, .9), "lm", track=1)
        ui.text((64, H - 26), "Recorded SUMO replay on mapped Mohali roads  ·  synthetic demand", F("m", 14), rgba(DIM, 1), "lm")

    def headline(self, ui, v):
        b = beat_at(v)
        a = win(v, b[0], b[1], .55 if b[0] > 0 else 0, .45 if b[1] < DUR else 0)
        dy = (1 - ease((v - b[0]) / .6)) * 10
        ui.text((64, 150 + dy), b[3], F("h", 50), rgba(IVORY, a), "ls")
        if b[4]:
            ui.text((66, 190 + dy), b[4], F("m", 24), rgba(MUTED, a), "ls")

    # --------------------------------------------------------------- card
    CARD = (1336, 236, 1856, 836)

    def card_alpha(self, v):
        return win(v, 2.7, 20, .6, 0)

    def card_shadow(self, v):
        a = self.card_alpha(v)
        im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if a <= 0:
            return im
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = self.CARD
        d.rounded_rectangle((x0 + 6, y0 + 18, x1 + 6, y1 + 24), 26, fill=(0, 0, 0, int(150 * a)))
        return im.filter(ImageFilter.GaussianBlur(24))

    def card(self, ui, v, t, img):
        a = self.card_alpha(v)
        if a <= 0:
            return
        x0, y0, x1, y1 = self.CARD
        ui.rect((x0, y0, x1, y1), fill=rgba((16, 23, 36), .97 * a), outline=rgba((70, 84, 106), .8 * a), w=1.2, r=24)
        cx = (x0 + x1) / 2
        if v < 9.6:
            self.gauge_forecast(ui, v, t, a * win(v, 0, 9.6, 0, .5), cx)
        elif v < 12.6:
            self.timeline_card(ui, v, t, a * win(v, 9.6, 12.6, .6, .5))
        elif v < 16.6:
            self.adjacent_card(ui, v, t, a * win(v, 12.6, 16.6, .6, .5))
        else:
            self.gauge_measured(ui, v, t, a * win(v, 16.6, 20, .6, 0))
        if v < 10.5 and v > 10.05:
            pass

    # gauge geometry: 240 deg sweep, 0 at lower left, 1,400 veh*s at lower right
    G_MAX = 1400.0
    G_R = 168

    def g_ang(self, val):
        return 150 + 240 * float(np.clip(val / self.G_MAX, 0, 1))

    def gauge_base(self, ui, a, cx, cy, ticks):
        r = self.G_R
        ui.arc((cx, cy), r, 150, 390, rgba((44, 55, 74), a), 22)
        for val, lab, col, strong in ticks:
            ang = math.radians(self.g_ang(val))
            p0 = (cx + math.cos(ang) * (r - 18), cy + math.sin(ang) * (r - 18))
            p1 = (cx + math.cos(ang) * (r + 22), cy + math.sin(ang) * (r + 22))
            ui.line([p0, p1], rgba(col, a), 4 if strong else 2.5)
            pl = (cx + math.cos(ang) * (r + 44), cy + math.sin(ang) * (r + 44))
            ui.text(pl, lab, F("monob" if strong else "mono", 15), rgba(col, a), "mm", track=1)

    def gauge_forecast(self, ui, v, t, a, cx):
        ep = self.ep
        x0, y0, x1, y1 = self.CARD
        cy = y0 + 262
        ui.text((x0 + 36, y0 + 46), "FORECAST", F("monob", 17), rgba(CYAN, a), "lm", track=3)
        ui.text((x0 + 36, y0 + 76), "additional delay to other approaches", F("m", 20), rgba(MUTED, a), "lm")
        self.gauge_base(ui, a, cx, cy, [(1000, "BUDGET", IVORY, True)])
        e = ep.event_at(t) if t >= 785 else None
        if e is not None and e["t"] > 792:
            e = ep.ev[792]          # the lease runs on the admission-time forecast; later log rows do not re-decide
        grow = ease_out((v - 2.75) / 1.0)
        prev = ep.ev.get(e["t"] - 1) if e else None
        if e is None:
            vals, total = [], 0.0
        else:
            vals = [(x["edge"], x["burden_proxy_veh_s"]) for x in e["conflicting_approaches"]]
            # glide between consecutive logged values over 0.35 video-s, exact once settled
            if prev is not None and prev["reason"] != "" and e["t"] in (787, 792):
                vstart = {787: 4.9, 792: 6.9}[e["t"]]
                u = ease((v - vstart) / .45)
                pv = dict((x["edge"], x["burden_proxy_veh_s"]) for x in prev["conflicting_approaches"])
                vals = [(ed, pv[ed] * (1 - u) + val * u) for ed, val in vals]
            vals = [(ed, val * grow) for ed, val in vals]
            total = sum(val for _, val in vals)
        r = self.G_R
        acc = 0.0
        for ed, val in vals:
            a0, a1 = self.g_ang(acc), self.g_ang(acc + val)
            ui.arc((cx, cy), r, a0, a1 - .6, rgba(APP[ed], a), 22)
            acc += val
        over = total > 1000
        if over and grow >= .999:
            ui.arc((cx, cy), r + 22, self.g_ang(1000), self.g_ang(total), rgba(CORAL, a), 6)
        decided = e is not None and grow > .95
        col = (CORAL if over else EMERALD) if decided else IVORY
        num = f"{math.floor(total + .5):,d}" if e is not None else "-"
        ui.text((cx, cy - 6), num, F("monob", 76), rgba(col, a), "ms")
        ui.text((cx, cy + 32), "/ 1,000 veh·s", F("mono", 22), rgba(MUTED, a), "mm")
        # breakdown row
        if e is not None:
            xs = cx - 150
            for ed, val in [(x["edge"], x["burden_proxy_veh_s"]) for x in e["conflicting_approaches"]]:
                ui.circle((xs, y0 + 428), 6, fill=rgba(APP[ed], a))
                ui.text((xs + 14, y0 + 428), f"{APP_NAME[ed]} {val:.0f}", F("mono", 18), rgba(MUTED, a * grow), "lm")
                xs += 112
        # decision pill
        if decided:
            ok = not over
            pill = "PRIORITY APPROVED" if ok else "PRIORITY WITHHELD"
            pc = EMERALD if ok else CORAL
            ui.rect((x0 + 40, y1 - 92, x1 - 40, y1 - 40), fill=rgba(mix((16, 23, 36), pc, .2), .97 * a), outline=rgba(pc, a), w=2, r=26)
            ui.text((cx, y1 - 66), pill, F("b", 22), rgba(pc, a), "mm", track=2.5)
        else:
            ui.text((cx, y1 - 66), "AWAITING REQUEST", F("mono", 18), rgba(DIM, a), "mm", track=2.5)
        if e is not None:
            ui.text((cx, y0 + 466), f"forecast at t = {e['t']} s  ·  Eq. 20", F("mono", 15), rgba(DIM, a), "mm")

    def mini_rows(self, ui, a, x0, x1, y, highlight, t0=776, t1=905):
        ep = self.ep
        sc = (x1 - x0) / (t1 - t0)
        rows = [("NO PRIORITY", "P0", y), ("AMBUGO", "P3", y + 96)]
        cols = {"335061218": CYAN, **APP}
        for lab, pol, yy in rows:
            ui.text((x0, yy - 20), lab, F("mono", 15), rgba(MUTED, a), "lm", track=2)
            for s0, s1, m, ed in self.tl[pol]:
                xa, xb = x0 + (max(s0, t0) - t0) * sc, x0 + (min(s1, t1) - t0) * sc
                if xb - xa < .5:
                    continue
                if m == 0 and ed is not None:
                    c = cols[ed]
                    al = 1.0 if ed in highlight else .22
                elif m == 3:
                    c, al = (120, 130, 148), .35
                else:
                    c, al = (60, 70, 88), .5
                ui.rect((xa, yy, xb - 1.5, yy + 30), fill=rgba(c, al * a), r=3)
        return sc, rows

    def timeline_card(self, ui, v, t, a):
        ep = self.ep
        x0, y0, x1, y1 = self.CARD
        cx = (x0 + x1) / 2
        L, R = x0 + 36, x1 - 36
        ui.text((L, y0 + 46), "SIGNAL SERVICE", F("monob", 17), rgba(CYAN, a), "lm", track=3)
        ui.text((L, y0 + 76), "green phases at PCL Chowk, both runs", F("m", 20), rgba(MUTED, a), "lm")
        y = y0 + 150
        sc, rows = self.mini_rows(ui, a, L, R, y, {"335061218", *APP})
        X = lambda s: L + (s - 776) * sc
        if 776 <= t <= 905:
            ui.line([(X(t), y - 8), (X(t), y + 136)], rgba(IVORY, .7 * a), 1.5)
        b = ease((v - 10.0) / .6)
        p0, p3 = 868.5, 796.5
        ui.line([(X(p0), y + 34), (X(p0), y + 63), (X(p3), y + 63), (X(p3), y + 88)], rgba(CYAN, a * b), 2.5)
        ui.poly([(X(p3), y + 95), (X(p3) - 6, y + 84), (X(p3) + 6, y + 84)], fill=rgba(CYAN, a * b))
        ui.text((L, y1 - 236), "SE  72 s earlier", F("h", 44), rgba((90, 190, 255), a * b), "ls")
        ui.text((L, y1 - 204), "green for the ambulance", F("m", 20), rgba(MUTED, a * b), "ls")
        c = ease((v - 11.0) / .6)
        for q0, q3, ed in [(795.5, 847.5, "172380615#1"), (812.5, 864.5, "172382686#4"), (839.5, 891.5, "172660580#0")]:
            ui.line([(X(q0), y + 34), (X(q3), y + 88)], rgba(APP[ed], a * c), 2.5)
            ui.poly([(X(q3), y + 95), (X(q3) - 6, y + 84), (X(q3) + 6, y + 84)], fill=rgba(APP[ed], a * c))
        ui.text((L, y1 - 140), "NW, SW, NE  52 s later", F("h", 44), rgba(CORAL, a * c), "ls")
        ui.text((L, y1 - 108), "their next green phases", F("m", 20), rgba(MUTED, a * c), "ls")
        ui.text((L, y1 - 60), "776 s", F("mono", 14), rgba(DIM, a), "lm")
        ui.text((R, y1 - 60), "905 s", F("mono", 14), rgba(DIM, a), "rm")
        ui.text((cx, y1 - 60), "recorded signal states", F("mono", 14), rgba(DIM, a), "mm")

    def adjacent_card(self, ui, v, t, a):
        x0, y0, x1, y1 = self.CARD
        L, R = x0 + 36, x1 - 36
        ui.text((L, y0 + 46), "ADJACENT ROADS", F("monob", 17), rgba(CYAN, a), "lm", track=3)
        ui.text((L, y0 + 76), "vehicles present: AmbuGo minus no priority", F("m", 20), rgba(MUTED, a), "lm")
        d = self.road_delta(t)
        mid = L + 250
        y = y0 + 140
        ui.line([(mid, y - 18), (mid, y + 7 * 56 - 30)], rgba(DIM, .7 * a), 1.5)
        for name, _, _ in ROADS:
            val = d[name]
            ui.text((L, y), name, F("m", 20), rgba(IVORY if abs(val) >= .5 else MUTED, a), "lm")
            wbar = float(np.clip(val, -12, 12)) / 12 * 95
            col = MORE if val > 0 else FEWER
            if abs(wbar) > .5:
                ui.rect((mid, y - 9, mid + wbar, y + 9), fill=rgba(col, .9 * a), r=3)
            ui.text((R, y), f"{val:+.1f}", F("mono", 19), rgba(col if abs(val) >= .5 else MUTED, a), "rm")
            y += 56
        ui.text(((L + R) / 2, y1 - 60), f"averaged over 6 s around t = {t:.0f} s", F("mono", 14), rgba(DIM, a), "mm")

    def gauge_measured(self, ui, v, t, a):
        ep = self.ep
        x0, y0, x1, y1 = self.CARD
        cx = (x0 + x1) / 2
        cy = y0 + 262
        ui.text((x0 + 36, y0 + 46), "OBSERVED", F("monob", 17), rgba(CORAL, a), "lm", track=3)
        ui.text((x0 + 36, y0 + 76), "extra stopped time", F("m", 20), rgba(MUTED, a), "lm")
        ui.text((x0 + 36, y0 + 104), "NW + SW + NE", F("mono", 16), rgba(DIM, a), "lm")
        self.gauge_base(ui, a, cx, cy, [(913.3, "FORECAST 913", CYAN, False), (1000, "BUDGET", IVORY, True)])
        k = ep.k(t)
        val = sum(ep.app_extra[e][k] for e in ep.conf)
        val = max(0.0, val) * ease_out((v - 16.7) / .8)
        ui.arc((cx, cy), self.G_R, self.g_ang(0), self.g_ang(val), rgba(CORAL, a), 22)
        ui.text((cx, cy - 6), f"{math.floor(val + .5):,d}", F("monob", 76), rgba(CORAL, a), "ms")
        ui.text((cx, cy + 32), "vehicle-seconds", F("mono", 22), rgba(MUTED, a), "mm")
        ui.text((cx, y0 + 428), f"785 s  to  {t:.0f} s", F("mono", 18), rgba(MUTED, a), "mm")
        ui.text((cx, y0 + 466), "vs the same vehicles without priority", F("mono", 15), rgba(DIM, a), "mm")
        ui.rect((x0 + 40, y1 - 92, x1 - 40, y1 - 40), outline=rgba(MUTED, .6 * a), w=1.5, r=26)
        ui.text((cx, y1 - 66), "MEASURED IN THE REPLAY", F("b", 19), rgba(MUTED, a), "mm", track=2.5)


# ------------------------------------------------------------------ render
_film = None


def _init(scale):
    global _film
    _film = Film(scale)


def _chunk(args):
    f0, f1, path, scale = args
    w, h = int(W * scale), int(H * scale)
    cmd = [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
           "-tune", "animation", "-threads", "1", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(_film.frame(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return f1 - f0


def render(jobs, scale=1.0, name="AmbuGo_20s_Cinematic.mp4"):
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(DUR * FPS)
    tmp = OUT / "_parts"
    tmp.mkdir(exist_ok=True)
    parts = [(f0, min(n, f0 + 50), tmp / f"p{i:03d}.mp4", scale) for i, f0 in enumerate(range(0, n, 50))]
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(scale,)) as pool:
        done = 0
        for k in pool.imap_unordered(_chunk, parts):
            done += k
            print(f"{done}/{n} frames  {time.time() - t0:.0f}s", flush=True)
    (tmp / "list.txt").write_text("".join(f"file '{p[2].name}'\n" for p in parts))
    final = OUT / name
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    shutil.rmtree(tmp)
    print(final)


def contact(film, times, path, cols=4):
    ims = [film.frame(int(min(t, DUR - 1 / FPS) * FPS)).resize((480, 270), Image.Resampling.LANCZOS) for t in times]
    rows = math.ceil(len(ims) / cols)
    sheet = Image.new("RGB", (cols * 490 + 10, rows * 310 + 10), (8, 10, 16))
    d = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(zip(times, ims)):
        x, y = 10 + (i % cols) * 490, 10 + (i // cols) * 310
        sheet.paste(im, (x, y))
        d.text((x, y + 276), f"{t:5.2f} s  ·  sim {SIM(min(t, DUR)):.1f} s", fill=(200, 205, 215))
    sheet.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stills", type=float, nargs="*")
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stills is not None and a.stills:
        film = Film()
        d = OUT / "stills"
        d.mkdir(exist_ok=True)
        for v in a.stills:
            film.frame(int(min(v, DUR - 1 / FPS) * FPS)).save(d / f"v{v:05.2f}.jpg", quality=92)
        return
    if a.contact:
        film = Film()
        contact(film, [0, 2, 4, 6, 7.5, 9, 10.5, 12, 14, 16, 18, 20], OUT / "contact_sheet.jpg")
        contact(film, [2.45, 2.6, 3.9, 4.1, 5.45, 5.6, 6.85, 7.1, 7.95, 8.1, 9.65, 9.8, 10.45, 10.6, 12.35, 12.5,
                       12.55, 12.7, 16.55, 16.7, 18.0, 19.0], OUT / "contact_transitions.jpg")
        print(OUT / "contact_sheet.jpg")
        return
    if a.draft:
        render(a.jobs, .5, "draft_960.mp4")
        return
    render(a.jobs)


if __name__ == "__main__":
    main()
