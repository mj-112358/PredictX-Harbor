#!/usr/bin/env python3
"""AmbuGo v9 explainer film: one archived JSEL episode (seed 40, load 0.95, one mission) replayed on the
synthetic district, plus one labelled single-decision counterfactual (J21 request withheld).

    python film.py                  # output/AmbuGo_v9_Explainer.mp4
    python film.py --stills 5 40    # stills at video seconds
    python film.py --contact        # contact sheet
    python film.py --draft          # 960x540 draft
"""
from pathlib import Path
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world import (Net, Run, View, F, rgba, BG, INK, MUTED, FAINT, SIG_G, SIG_A, SIG_R, AMB_BLUE, HOSP, ROAD)  # noqa: E402

W, H, FPS = 1920, 1080, 30
OUT = Path(__file__).resolve().parent / "output"
ROUTE_COL = {"A": (70, 110, 180), "B": (40, 140, 140), "C": (130, 80, 170), "D": (150, 125, 60)}
MORE = (140, 70, 200)        # occupancy higher than the comparison replay (not a signal colour)
LESS = (40, 150, 190)
PANEL = (255, 255, 255)


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def win(u, a, b, fi=.06, fo=.06):
    if u < a or u > b:
        return 0.0
    return min(ease((u - a) / fi) if fi else 1, ease((b - u) / fo) if fo else 1)


def interp(knots, u):
    us = [k[0] for k in knots]
    i = int(np.clip(np.searchsorted(us, u, side="right") - 1, 0, len(knots) - 2))
    a, b = knots[i], knots[i + 1]
    if b[0] == a[0]:
        return b[1]
    w = (u - a[0]) / (b[0] - a[0])
    return a[1] + (b[1] - a[1]) * w


_eq = {}


def equation(tex, size=26, color=INK):
    key = (tex, size, color)
    if key not in _eq:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        fig = plt.figure(figsize=(.01, .01), dpi=144)
        fig.text(0, 0, tex, fontsize=size, color=tuple(c / 255 for c in color))
        buf = io.BytesIO()
        fig.savefig(buf, dpi=144, transparent=True, bbox_inches="tight", pad_inches=.03)
        plt.close(fig)
        _eq[key] = Image.open(io.BytesIO(buf.getvalue())).convert("RGBA")
    return _eq[key]


# ---------------------------------------------------------------- the film
class Film:
    def __init__(self, scale=1.0):
        self.net = Net()
        self.P3, self.WH, self.P0 = Run("P3"), Run("P3_withholdJ21"), Run("P0")
        self.scale_out = scale
        p0 = self.P3.plans[0]
        uniq, self.routes = [], {}
        names = {}
        for c in p0["candidates"]:
            key = tuple(c["route"])
            if key not in uniq:
                uniq.append(key)
        # fixed naming: A = incumbent at dispatch, then the remaining in log order
        order = sorted(uniq, key=lambda r: (0 if [c for c in p0["candidates"] if tuple(c["route"]) == r][0]["incumbent"] else 1, uniq.index(r)))
        for i, r in enumerate(order):
            n = "ABCD"[i]
            cs = [c for c in p0["candidates"] if tuple(c["route"]) == r]
            self.routes[n] = dict(edges=list(r), none=[c for c in cs if c["action"][1] == 0][0],
                                  req=[c for c in cs if c["action"][1] > 0][0], chosen=list(r) == p0["chosen_route"])
            names[r] = n
        self.route_name = names
        self.g = {(g["t"], g["junction"]): g for g in self.P3.gates}
        self.build()
        self.starts = np.cumsum([0] + [s["dur"] for s in self.scenes])
        self.total = float(self.starts[-1])

    # ---------------------------------------------------------- scenes
    def build(self):
        J = self.net.junctions
        j21 = J["J21"]["xy"]; j11 = J["J11"]["xy"]
        S = []
        S.append(dict(key="problem", chap="The problem", dur=11, run="P3", sim=[(0, 3992), (1, 4066)],
                      cam=[(0, (865, 812, .62)), (1, (860, 770, .66))], ov=self.ov_problem))
        S.append(dict(key="routes", chap="1 · Choosing a route and a priority action together", dur=14, run="P3",
                      sim=[(0, 4065.5), (1, 4065.5)], cam=[(0, (860, 770, .66)), (.15, (880, 760, .98)), (1, (880, 760, .98))],
                      ov=self.ov_routes))
        S.append(dict(key="travel", chap="2 · Every road link: check the route, then the request", dur=7, run="P3",
                      sim=[(0, 4065.5), (1, 4091.0)], cam=[(0, (880, 760, .98)), (.35, ("track", 2.0)), (1, (520, 600, 2.4))],
                      ov=self.ov_travel))
        S.append(dict(key="recheck", chap="2 · Every road link: check the route, then the request", dur=5.5, run="P3",
                      sim=[(0, 4091.0), (1, 4091.0)], cam=[(0, (520, 600, 2.4)), (.25, (1000, 740, 1.0)), (1, (1000, 740, 1.0))],
                      ov=self.ov_recheck))
        S.append(dict(key="refuse", chap="3 · A priority request that is withheld", dur=9, run="P3",
                      sim=[(0, 4091.0), (1, 4091.0)], cam=[(0, (1000, 740, 1.0)), (.2, (j11[0] + 40, j11[1] - 10, 3.4)), (1, (j11[0] + 40, j11[1] - 10, 3.4))],
                      ov=self.ov_refuse))
        S.append(dict(key="cycle", chap="4 · How a normal signal cycle runs", dur=13, run="P3",
                      sim=[(0, 4091.0), (1, 4130.5)], cam=[(0, (j11[0] + 40, j11[1] - 10, 3.4)), (1, (j11[0] + 40, j11[1] - 10, 3.4))],
                      ov=self.ov_cycle, cap_y=866))
        S.append(dict(key="j21", chap="5 · The decision at J21", dur=5, run="P3", sim=[(0, 4130.5), (.6, 4132.0), (1, 4132.0)],
                      cam=[(0, (j11[0] + 40, j11[1] - 10, 3.4)), (.6, (j21[0] + 130, j21[1] + 20, 4.6)), (1, (j21[0] + 130, j21[1] + 20, 4.6))],
                      ov=self.ov_j21))
        S.append(dict(key="s1", chap="Scenario 1 — give the ambulance priority (what AmbuGo did)", dur=12, run="P3",
                      sim=[(0, 4132.0), (1, 4180.0)], cam=[(0, (j21[0] + 130, j21[1] + 20, 4.6)), (1, (j21[0] + 130, j21[1] + 20, 4.6))],
                      ov=self.ov_s1, cap_y=866))
        S.append(dict(key="burden", chap="6 · How AmbuGo forecast that cost (Eq. 20)", dur=19, run="P3",
                      sim=[(0, 4132.0), (1, 4132.0)], cam=[(0, (j21[0] + 130, j21[1] + 20, 4.6)), (1, (j21[0] + 130, j21[1] + 20, 4.6))],
                      ov=self.ov_burden, cap_y=866))
        S.append(dict(key="compare", chap="Scenario 1 vs Scenario 2 — same moment, same traffic", dur=17, run="split",
                      sim=[(0, 4132.0), (.06, 4132.0), (.94, 4236.0), (1, 4236.0)],
                      cam=[(0, (j21[0] + 20, j21[1] - 40, 2.7)), (1, (j21[0] + 20, j21[1] - 40, 2.7))], ov=self.ov_compare))
        S.append(dict(key="network", chap="7 · One junction, several roads — and the rest of the trip", dur=21, run="P3",
                      sim=[(0, 4171.0), (.5, 4255.0), (.56, 4256.0), (.66, 4256.0), (.92, 4316.5), (1, 4320.0)],
                      cam=[(0, (j21[0] + 130, j21[1] + 20, 4.6)), (.12, (1130, 760, 1.12)), (1, (1130, 760, 1.12))],
                      ov=self.ov_network))
        S.append(dict(key="result", chap="Results", dur=12, run="P3", sim=[(0, 4320.0), (1, 4320.0)],
                      cam=[(0, (1130, 760, 1.12)), (.25, (865, 812, .62)), (1, (865, 812, .62))], ov=self.ov_result))
        self.scenes = S

    def locate(self, v):
        i = int(np.clip(np.searchsorted(self.starts, v, side="right") - 1, 0, len(self.scenes) - 1))
        sc = self.scenes[i]
        return i, sc, float(np.clip((v - self.starts[i]) / sc["dur"], 0, 1))

    def sim(self, sc, u):
        return float(interp(sc["sim"], u))

    def rate(self, sc, u):
        du = 1 / (FPS * sc["dur"])
        return (self.sim(sc, min(1, u + du)) - self.sim(sc, max(0, u - du))) / ((min(1, u + du) - max(0, u - du)) * sc["dur"])

    def cam(self, sc, u, run, t):
        ks = sc["cam"]

        def spec(c):
            if isinstance(c[0], str):
                a = run.amb_xy(t)
                xy = a[0] if a else self.net.junctions["J11"]["xy"]
                return np.array([xy[0], xy[1], c[1]], float)
            return np.array(c, float)
        us = [k[0] for k in ks]
        i = int(np.clip(np.searchsorted(us, u, side="right") - 1, 0, len(ks) - 2)) if len(ks) > 1 else 0
        if len(ks) == 1:
            return spec(ks[0][1])
        a, b = ks[i], ks[i + 1]
        w = ease((u - a[0]) / max(b[0] - a[0], 1e-9))
        A, B = spec(a[1]), spec(b[1])
        c = A * (1 - w) + B * w
        c[2] = math.exp(math.log(A[2]) * (1 - w) + math.log(B[2]) * w)
        return c

    # ---------------------------------------------------------- frame
    def frame(self, f):
        v = f / FPS
        i, sc, u = self.locate(v)
        t = self.sim(sc, u)
        ctx = dict(v=v, u=u, t=t, sc=sc, film=self)
        self._cap_y = sc.get("cap_y")
        if sc["run"] == "split":
            img = Image.new("RGB", (W, H), (30, 33, 38))
            for k, run in enumerate((self.P3, self.WH)):
                cx, cy, s = self.cam(sc, u, run, t)
                view = View(956, H, cx, cy, s)
                self.world(view, run, t, heads=("J21",))
                sc["ov"](view, dict(ctx, side=k, run=run), "map")
                img.paste(view.down().convert("RGB"), (k * 964, 0))
            ui = View(W, H, 0, 0, 1)
            ui.im = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
            from PIL import ImageDraw
            ui.d = ImageDraw.Draw(ui.im)
            sc["ov"](ui, dict(ctx, run=self.P3), "ui")
            self.chrome(ui, ctx)
            img = Image.alpha_composite(img.convert("RGBA"), ui.down()).convert("RGB")
        else:
            run = self.P3
            cx, cy, s = self.cam(sc, u, run, t)
            view = View(W, H, cx, cy, s)
            heads = sc.get("heads", ("J10", "J11", "J21", "J22", "J12", "J13") if s > 2.5 else ())
            self.world(view, run, t, heads=heads, tint=sc["key"] == "network")
            sc["ov"](view, dict(ctx, run=run), "map")
            ui = View(W, H, 0, 0, 1)
            from PIL import ImageDraw
            ui.im = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0)); ui.d = ImageDraw.Draw(ui.im)
            sc["ov"](ui, dict(ctx, run=run), "ui")
            self.chrome(ui, ctx)
            img = Image.alpha_composite(view.down(), ui.down()).convert("RGB")
        if self.scale_out != 1:
            img = img.resize((int(W * self.scale_out), int(H * self.scale_out)), Image.Resampling.LANCZOS)
        return img

    def world(self, view, run, t, heads=(), tint=False):
        view.blocks(self.net)
        view.roads(self.net)
        if tint:
            self.occupancy_tint(view, t)
        view.signals(self.net, run, t, heads=heads, lamp_scale=min(1.2, max(.8, view.s / 4)))
        self.markers(view)
        view.vehicles(run, t, min_px=10 if view.s < 1.3 else 0)

    def markers(self, view):
        n = self.net.junctions
        for name, lab, col in [("BE1", "Hospital (BE1)", HOSP), ("BW1", "Ambulance base (BW1)", AMB_BLUE)]:
            p = view.P(n[name]["xy"])
            if not (-200 < p[0] < view.w + 200 and -100 < p[1] < view.h + 100):
                continue
            r = 16
            view.rect((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=(255, 255, 255, 255), outline=col + (255,), w=2.5, r=5)
            if name == "BE1":
                view.line([(p[0] - 8, p[1]), (p[0] + 8, p[1])], col, 4.5); view.line([(p[0], p[1] - 8), (p[0], p[1] + 8)], col, 4.5)
            else:
                view.text(p, "A", F("b", 16), col + (255,), "mm")
            dx = -24 if name == "BE1" else 24
            view.text((p[0] + dx, p[1] - 30), lab, F("b", 19), col + (255,), "rm" if name == "BE1" else "lm")
        if view.s < 1.4:
            for j, J in n.items():
                if J["tl"]:
                    p = view.P(J["xy"])
                    view.text((p[0] + 12, p[1] + 12), j, F("monob", 14 if view.s < .8 else 16), (70, 76, 86, 255), "la")

    # ---------------------------------------------------------- chrome
    def chrome(self, ui, ctx):
        sc, u, t = ctx["sc"], ctx["u"], ctx["t"]
        title = sc["chap"]
        fw = ui.tw(title, F("b", 22))
        ui.rect((24, 20, 24 + fw + 36, 64), fill=(255, 255, 255, 236), outline=(210, 212, 216, 255), w=1, r=10)
        ui.text((42, 42), title, F("b", 22), INK + (255,), "lm")
        el = t - 4064
        r = self.rate(sc, u)
        sp = "paused" if abs(r) < .05 else (f"{r:.1f}× real time" if r >= 1 else f"{r:.2f}× real time")
        s1 = f"simulation t = {t:,.1f} s"
        s2 = f"{'dispatch + ' if el >= 0 else 'dispatch − '}{int(abs(el)) // 60}:{int(abs(el)) % 60:02d}   ·   {sp}"
        w1 = max(ui.tw(s1, F("monob", 19)), ui.tw(s2, F("mono", 15))) + 36
        ui.rect((W - 24 - w1, 20, W - 24, 82), fill=(255, 255, 255, 236), outline=(210, 212, 216, 255), w=1, r=10)
        ui.text((W - 42, 40), s1, F("monob", 19), INK + (255,), "rm")
        ui.text((W - 42, 66), s2, F("mono", 15), MUTED + (255,), "rm")
        ui.text((W - 28, 98), "synthetic SUMO district (AmbuGo v9 fixture) · seed 40, load 0.95 · not a calibrated city",
                F("m", 13), (110, 116, 126, 255), "rm")

    def caption(self, ui, lines, a=1.0, y=None):
        """Lower-left caption; lines is a list of strings (each <= ~95 chars)."""
        if a <= 0:
            return
        fnt = F("m", 25)
        lh = 36
        w = max(ui.tw(s, fnt) for s in lines) + 48
        y1 = (y or H - 48)
        y0 = y1 - len(lines) * lh - 26
        ui.rect((24, y0, 24 + w, y1), fill=(255, 255, 255, int(238 * a)), outline=(205, 208, 214, int(255 * a)), w=1, r=12)
        for k, s in enumerate(lines):
            ui.text((48, y0 + 13 + k * lh), s, fnt, INK + (int(255 * a),), "la")

    def captions(self, ui, u, items):
        for a, b, lines in items:
            al = win(u, a, b, .04 if a > 0 else 0, .04 if b < 1 else 0)
            if al > 0:
                self.caption(ui, lines, al, self._cap_y)

    def panel(self, ui, box, title, a=1.0):
        x0, y0, x1, y1 = box
        ui.rect((x0 + 4, y0 + 8, x1 + 4, y1 + 10), fill=(0, 0, 0, int(28 * a)), r=14)
        ui.rect(box, fill=(255, 255, 255, int(246 * a)), outline=(205, 208, 214, int(255 * a)), w=1, r=14)
        if title:
            ui.text((x0 + 26, y0 + 30), title, F("b", 22), INK + (int(255 * a),), "lm")

    def tag(self, view, xy, lines, col, a=1.0, off=(40, -60), world=True):
        p = view.P(xy) if world else np.array(xy, float)
        fnt = [F("b", 18)] + [F("m", 17)] * (len(lines) - 1)
        w = max(view.tw(s, f) for s, f in zip(lines, fnt)) + 26
        h = 26 * len(lines) + 12
        x0, y0 = p[0] + off[0], p[1] + off[1] - h / 2
        x0 = float(np.clip(x0, 10, view.w - w - 10)); y0 = float(np.clip(y0, 90, view.h - h - 10))
        cx, cy = float(np.clip(p[0], x0, x0 + w)), float(np.clip(p[1], y0, y0 + h))
        view.line([tuple(p), (cx, cy)], rgba(col, a), 2)
        view.circle(tuple(p), 4, fill=rgba(col, a))
        view.rect((x0, y0, x0 + w, y0 + h), fill=(255, 255, 255, int(242 * a)), outline=rgba(col, a), w=2, r=8)
        for k, (s, f) in enumerate(zip(lines, fnt)):
            view.text((x0 + 13, y0 + 8 + 26 * k), s, f, rgba(col if k == 0 else INK, a), "la")

    def route_line(self, view, edges, col, width, a=1.0, offset=0.0):
        pts = []
        for e in edges:
            ls = self.net.lane_of_edge[e]
            shp = np.mean([l["shape"] for l in ls], 0) if len({len(l["shape"]) for l in ls}) == 1 else ls[0]["shape"]
            pts.extend(list(shp))
        P = view.P(np.array(pts))
        if offset:
            d = np.gradient(P, axis=0)
            n = np.stack((-d[:, 1], d[:, 0]), 1)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            P = P + n * offset
        view.line([tuple(p) for p in P], (255, 255, 255, int(220 * a)), width + 4)
        view.line([tuple(p) for p in P], rgba(col, a), width)

    def strip(self, ui, run, j, t0, t1, box, t, label=None, names=None, a=1.0, highlight=None):
        """Recorded signal stages of junction j between t0 and t1."""
        x0, y0, x1, y1 = box
        names = names or {}
        sc = (x1 - x0) / (t1 - t0)
        if label:
            ui.text((x0, y0 - 16), label, F("s", 17), INK + (int(255 * a),), "lm")
        segs, cur = [], None
        for tt in range(int(t0), int(t1)):
            m, e, el, req = run.signal(j, tt)
            key = (m, e)
            if cur and cur[2] == key:
                cur[1] = tt + 1
            else:
                cur = [tt, tt + 1, key]; segs.append(cur)
        for s0, s1, (m, e) in segs:
            xa, xb = x0 + (s0 - t0) * sc, x0 + (s1 - t0) * sc
            if m == "green":
                col, txt, tc = SIG_G, f"{names.get(e, '?')} green", (255, 255, 255)
            elif m == "amber":
                col, txt, tc = SIG_A, "amber", INK
            elif m == "allred":
                col, txt, tc = (150, 40, 40), "all-red", (255, 255, 255)
            else:
                col, txt, tc = (120, 60, 60), "pedestrian interval (all red)", (255, 255, 255)
            ui.rect((xa, y0, xb - 1.5, y1), fill=rgba(col, a), r=3)
            if xb - xa > ui.tw(txt, F("s", 15)) + 10:
                ui.text(((xa + xb) / 2, (y0 + y1) / 2), txt, F("s", 15), tc + (int(255 * a),), "mm")
        if highlight:
            ha, hb, hl = highlight
            xa, xb = x0 + (ha - t0) * sc, x0 + (hb - t0) * sc
            ui.rect((xa, y0 - 6, xb, y1 + 6), outline=(30, 30, 30, int(255 * a)), w=2.5, r=4)
            ui.text(((xa + xb) / 2, y1 + 20), hl, F("b", 15), INK + (int(255 * a),), "mm")
        if t0 <= t <= t1:
            xx = x0 + (t - t0) * sc
            ui.line([(xx, y0 - 8), (xx, y1 + 8)], (20, 20, 20, int(255 * a)), 2.5)
        for tt in range(int(math.ceil(t0 / 20) * 20), int(t1) + 1, 20):
            xx = x0 + (tt - t0) * sc
            ui.text((xx, y1 + (36 if highlight else 14)), f"{tt}", F("mono", 12), MUTED + (int(255 * a),), "mm")

    def approach_labels(self, view, j, names, a=1.0, extra=None):
        for e, nm in names.items():
            c, d, half = self.net.stopline(e)
            p = view.P(c - d * 70)
            txt = nm + (extra.get(e, "") if extra else "")
            w = view.tw(txt, F("b", 17)) + 22
            view.rect((p[0] - w / 2, p[1] - 15, p[0] + w / 2, p[1] + 15), fill=(255, 255, 255, int(235 * a)),
                      outline=(AMB_BLUE if "ambulance" in nm else (90, 96, 106)) + (int(255 * a),), w=1.5, r=8)
            view.text((p[0], p[1]), txt, F("b", 17), (AMB_BLUE if "ambulance" in nm else INK) + (int(255 * a),), "mm")

    def occupancy_tint(self, view, t):
        """Links where occupancy (vehicles present) exceeds the normal-cycle replay at the same time (±10 s)."""
        P3, WH = self.P3, self.WH
        k = int((t - P3.cstart) / P3.cstep)
        lo, hi = max(0, k - 20), k + 21
        d = (P3.counts[lo:hi] - WH.counts[lo:hi]).mean(0)
        for i, e in enumerate(P3.partition):
            if e not in self.net.lane_of_edge or abs(d[i]) < .6:
                continue
            col = MORE if d[i] > 0 else LESS
            al = min(.75, .25 + abs(d[i]) * .07)
            for ln in self.net.lane_of_edge[e]:
                view.line([tuple(p) for p in view.P(ln["shape"])], rgba(col, al), max(7, ln["width"] * view.s * 1.6))

    # ---------------------------------------------------------- overlays
    def ov_problem(self, view, ctx, layer):
        u = ctx["u"]
        if layer == "map":
            a = ctx["run"].amb_xy(ctx["t"])
            if a and u > .6:
                self.tag(view, a[0], ["Ambulance dispatched", "from BW1 at t = 4,064 s"], AMB_BLUE, win(u, .62, 1, .08, 0), off=(40, 70))
            return
        self.captions(view, u, [
            (0, .38, ["A synthetic district: 16 signal-controlled junctions, 59 one-way road links,",
                      "cars, motorcycles, autorickshaws, buses and goods vehicles. Every vehicle is a recorded SUMO trajectory."]),
            (.38, .7, ["Each junction runs its own signal cycle. Queues form on red and clear on green."]),
            (.7, 1, ["An ambulance must reach the hospital at BE1. How can it get there quickly",
                     "without needlessly disrupting everyone else?"])])

    def ov_routes(self, view, ctx, layer):
        u = ctx["u"]
        rts = self.routes
        if layer == "map":
            for k, n in enumerate("ABCD"):
                a = win(u, .1 + k * .07, 1, .05, 0)
                if a > 0:
                    r = rts[n]
                    self.route_line(view, r["edges"], ROUTE_COL[n], 6 if r["chosen"] else 4.5, a, offset=(k - 1.5) * 7)
            for k, n in enumerate("ABCD"):
                a = win(u, .14 + k * .07, 1, .05, 0)
                if a > 0:
                    e = rts[n]["edges"][2 if n != "D" else 3]
                    mid = np.mean(self.net.lane_of_edge[e][0]["shape"], 0)
                    p = view.P(mid)
                    off = {"A": (0, -26), "B": (-30, 0), "C": (0, 26), "D": (0, 28)}[n]
                    view.circle((p[0] + off[0], p[1] + off[1]), 15, fill=rgba(ROUTE_COL[n], a), outline=(255, 255, 255, int(255 * a)), w=2)
                    view.text((p[0] + off[0], p[1] + off[1]), n, F("b", 17), (255, 255, 255, int(255 * a)), "mm")
            return
        a = win(u, .36, 1, .05, 0)
        if a > 0:
            box = (1140, 640, 1890, 1010)
            self.panel(view, box, "Forecast at dispatch (t = 4,065 s)", a)
            x0, y0 = box[0] + 26, box[1] + 74
            cols = [x0, x0 + 60, x0 + 255, x0 + 455, x0 + 640]
            hd = ["", "normal signals", "with request", "burden Ĉ", "eligible"]
            for x, h in zip(cols, hd):
                view.text((x, y0), h, F("s", 15), MUTED + (int(255 * a),), "la")
            tmin = min(r["req"]["eta"] for r in rts.values())
            for k, n in enumerate("ABCD"):
                r = rts[n]
                y = y0 + 38 + k * 46
                hl = r["chosen"]
                if hl:
                    view.rect((box[0] + 12, y - 9, box[2] - 12, y + 33), fill=rgba(ROUTE_COL[n], .12 * a), r=8)
                view.circle((cols[0] + 14, y + 12), 13, fill=rgba(ROUTE_COL[n], a))
                view.text((cols[0] + 14, y + 12), n, F("b", 15), (255, 255, 255, int(255 * a)), "mm")
                view.text((cols[1], y), f"{r['none']['eta']:.1f} s", F("mono", 19), INK + (int(255 * a),), "la")
                view.text((cols[2], y), f"{r['req']['eta']:.1f} s", F("monob" if hl else "mono", 19), INK + (int(255 * a),), "la")
                view.text((cols[3], y), f"{r['req']['burden']:,.0f} veh·s", F("mono", 19), INK + (int(255 * a),), "la")
                ok = r["req"]["eta"] <= tmin + 1e-9
                view.text((cols[4], y), "yes" if ok else "no", F("b", 18), ((20, 130, 60) if ok else (150, 60, 60)) + (int(255 * a),), "la")
            view.text((x0, box[3] - 28), "forecast ETA to the hospital boundary; request = 30 s look-ahead, 60 s hold",
                      F("m", 14), MUTED + (int(255 * a),), "lm")
        self.captions(view, u, [
            (0, .36, ["At dispatch AmbuGo scores every pairing of a candidate route with an action:",
                      "no signal request, or a request (look ahead 30 s, hold up to 60 s)."]),
            (.36, .7, ["Selection rule (ε = 0 s): only pairs whose forecast equals the fastest are eligible.",
                       "The forecast burden would only break a tie. Route C with a request is chosen."]),
            (.7, 1, ["Route C is the slowest under normal signals (616.9 s) but the fastest with a request (248.0 s):",
                     "the route cannot be chosen properly without considering the signal action."])])

    def ov_travel(self, view, ctx, layer):
        u, t = ctx["u"], ctx["t"]
        if layer == "map":
            rid = "C" if t < 4091 else "A"
            self.route_line(view, self.routes[rid]["edges"], ROUTE_COL[rid], 4.5, .55)
            g = self.g[(4065, "J10")]
            a = win(u, .25, 1, .08, 0)
            if a > 0:
                self.tag(view, self.net.junctions["J10"]["xy"], ["J10 · request admitted",
                         f"forecast gain {g['benefit']:.0f} s · burden {g['burden']:,.0f} veh·s"], (20, 130, 60), a, off=(-40, -110))
            return
        self.captions(view, u, [(0, 1, ["Each time the ambulance enters a road link, the route is re-checked and the next",
                                        "junction's request is tested. At J10 it is admitted and the ambulance gets green."])])

    def ov_recheck(self, view, ctx, layer):
        u = ctx["u"]
        p = self.P3.plans[1]
        cand = {}
        for c in p["candidates"]:
            if c["action"][1] > 0:
                cand[tuple(c["route"])] = c
        inc = [c for c in cand.values() if c["incumbent"]][0]
        best = min(cand.values(), key=lambda c: c["eta"])
        if layer == "map":
            for c, n, w in [(inc, "C", 4.5), (best, "A", 6)]:
                self.route_line(view, c["route"], ROUTE_COL[n], w, win(u, .15, 1, .06, 0))
            return
        a = win(u, .2, 1, .06, 0)
        if a > 0:
            box = (1180, 700, 1890, 960)
            self.panel(view, box, "Re-check on entering J10 → J11 (t = 4,091 s)", a)
            y = box[1] + 80
            for c, n, lab in [(best, "A", "route A"), (inc, "C", "route C (current)")]:
                view.circle((box[0] + 40, y + 12), 13, fill=rgba(ROUTE_COL[n], a))
                view.text((box[0] + 40, y + 12), n, F("b", 15), (255, 255, 255, int(255 * a)), "mm")
                view.text((box[0] + 66, y), f"{lab}: forecast {c['eta']:.1f} s", F("mono", 20), INK + (int(255 * a),), "la")
                y += 48
            view.text((box[0] + 26, y + 12), f"gap {inc['eta'] - best['eta']:.1f} s > 10 s hysteresis → switch to A",
                      F("b", 19), INK + (int(255 * a),), "la")
        self.captions(view, u, [(0, 1, ["With newer observations, route A is now forecast 12.6 s faster. The current route is kept",
                                        "only if it is within 10 s (hysteresis), so AmbuGo switches back to route A."])])

    def ov_refuse(self, view, ctx, layer):
        u = ctx["u"]
        g = self.g[(4091, "J11")]
        names = {"E_J10_J11": "west · ambulance", "E_J01_J11": "north", "E_J21_J11": "south"}
        if layer == "map":
            self.approach_labels(view, "J11", names, win(u, .2, 1, .06, 0))
            return
        a = win(u, .2, 1, .06, 0)
        if a > 0:
            box = (1160, 120, 1890, 500)
            self.panel(view, box, "Request at J11 (t = 4,091 s)", a)
            rows = [(False, "Forecast gain for the ambulance", f"{g['benefit']:.1f} s", "needs ≥ 5 s"),
                    (True, "Feasible within the 60 s hold (Eq. 19)", f"{g['feasible_lhs']:.1f} s", "≤ 60 s"),
                    (True, "Space on the next link (J11 → J21)", f"{g['headroom_obs']} veh", f"< {g['headroom_cap']:.1f}"),
                    (True, "Forecast burden on others (Eq. 20)", f"{g['burden']:,.0f} veh·s", "≤ 10,000")]
            y = box[1] + 78
            for ok, lab, val, lim in rows:
                c = (20, 130, 60) if ok else (190, 40, 40)
                view.circle((box[0] + 40, y + 12), 12, fill=rgba(c, .14 * a), outline=rgba(c, a), w=2)
                view.text((box[0] + 40, y + 12), "✓" if ok else "✗", F("b", 16), rgba(c, a), "mm")
                view.text((box[0] + 66, y), lab, F("m", 18), INK + (int(255 * a),), "la")
                view.text((box[2] - 26, y), f"{val}  {lim}", F("mono", 17), (c if not ok else INK) + (int(255 * a),), "ra")
                y += 46
            view.text((box[0] + 26, y + 6), f"Signal now: pedestrian interval, {25 - g['signal_elapsed']:.0f} s left",
                      F("m", 17), MUTED + (int(255 * a),), "la")
            b = win(u, .45, 1, .06, 0)
            view.rect((box[0] + 22, box[3] - 64, box[2] - 22, box[3] - 18), fill=(190, 40, 40, int(30 * b)), outline=(190, 40, 40, int(255 * b)), w=2, r=23)
            view.text(((box[0] + box[2]) / 2, box[3] - 41), "PRIORITY WITHHELD: too little to gain", F("b", 19), (190, 40, 40, int(255 * b)), "mm")
        self.captions(view, u, [
            (0, .45, ["At J11 the ambulance is about 27 s from the stop line. The forecast says the normal cycle",
                      "would reach its approach only about 3.6 s after it arrives."]),
            (.45, 1, ["A priority pulse would interrupt the others for a forecast gain below 5 s, so the request is",
                      "withheld. In all 432 archived JSEL runs, every refusal was of this kind (too little gain",
                      "or no feasible pulse); none failed on the burden budget or downstream space."])])

    def ov_cycle(self, view, ctx, layer):
        u, t = ctx["u"], ctx["t"]
        names = {"E_J10_J11": "west", "E_J01_J11": "north", "E_J21_J11": "south"}
        if layer == "map":
            self.approach_labels(view, "J11", {"E_J10_J11": "west · ambulance", "E_J01_J11": "north", "E_J21_J11": "south"}, 1)
            return
        box = (420, 1000 - 70, 1880, 1000 - 30)
        view.rect((400, 880, 1900, 1062), fill=(255, 255, 255, 236), outline=(205, 208, 214, 255), w=1, r=12)
        self.strip(view, self.P3, "J11", 4070, 4150, (430, 930, 1880, 968), t, label="J11 signal stages (recorded)", names=names)
        box = (1300, 110, 1890, 470)
        self.panel(view, box, "Fixed rules at every junction", 1)
        for k, s in enumerate(["one approach has green at a time", "minimum green 8 s", "amber 3 s, then all-red 2 s",
                               "a 25 s all-red pedestrian interval each cycle;", "   once started it is never cut short",
                               "an occupied approach unserved for 150 s is", "   served next (a service-age guard, not a cap on", "   any one driver's wait)"]):
            view.text((box[0] + 26 + (18 if s.startswith("   ") else 0), box[1] + 68 + k * 34), ("•  " if not s.startswith("   ") else "") + s.strip(), F("m", 18), INK + (255,), "la")
        self.captions(view, u, [(0, .55, ["With the request withheld, J11 keeps its normal cycle: the pedestrian interval ends,",
                                          "north gets green, then amber and all-red, then the ambulance's west approach."]),
                                (.55, 1, ["Signals change only through these legal stages. No light jumps from red to green",
                                          "for anyone, ambulance or not."])], )

    def ov_j21(self, view, ctx, layer):
        u = ctx["u"]
        names = {"E_J11_J21": "north · ambulance", "E_J22_J21": "east", "E_J31_J21": "south", "E_J20_J21": "west"}
        if layer == "map":
            self.approach_labels(view, "J21", names, win(u, .55, 1, .08, 0))
            return
        self.captions(view, u, [(0, 1, ["At J21 the ambulance asks again (t = 4,132 s). East has had green for 3 s;",
                                        "south and west are waiting. What if J21 serves the ambulance now?"])])

    J21N = {"E_J11_J21": "north", "E_J22_J21": "east", "E_J31_J21": "south", "E_J20_J21": "west"}

    def stopped_badges(self, view, run, t, a=1.0):
        for e, nm in self.J21N.items():
            c, d, half = self.net.stopline(e)
            p = view.P(c - d * 70)
            n = run.halted(t, e)
            txt = f"{nm} · {n} stopped" if "north" not in nm else f"north (ambulance) · {n} stopped"
            w = view.tw(txt, F("b", 16)) + 22
            col = AMB_BLUE if "north" in nm else INK
            view.rect((p[0] - w / 2, p[1] - 15, p[0] + w / 2, p[1] + 15), fill=(255, 255, 255, int(238 * a)), outline=col + (int(255 * a),), w=1.5, r=8)
            view.text((p[0], p[1]), txt, F("b", 16), col + (int(255 * a),), "mm")

    def ov_s1(self, view, ctx, layer):
        u, t = ctx["u"], ctx["t"]
        if layer == "map":
            self.stopped_badges(view, self.P3, t)
            return
        view.rect((400, 880, 1900, 1062), fill=(255, 255, 255, 236), outline=(205, 208, 214, 255), w=1, r=12)
        self.strip(view, self.P3, "J21", 4100, 4260, (430, 930, 1880, 968), t, label="J21 signal stages (recorded, priority granted)",
                   names=self.J21N)
        self.captions(view, u, [
            (0, .3, ["East must still finish its 8 s minimum green, then amber (3 s) and all-red (2 s)."]),
            (.3, .66, ["From 4,142 s the ambulance's north approach is green. East and south now wait longer,",
                       "and arriving vehicles join their queues."]),
            (.66, 1, ["The ambulance clears J21 at about 4,171 s. East and south get their next green later",
                      "than they otherwise would have."])], )

    def ov_burden(self, view, ctx, layer):
        u = ctx["u"]
        g = self.g[(4132, "J21")]
        arms = {a["edge"]: a for a in g["arms"]}
        if layer == "map":
            hl = [(.07, .33, "E_J22_J21"), (.33, .62, "E_J22_J21")]
            for a0, a1, e in hl:
                a = win(u, a0, a1, .03, .03)
                if a > 0:
                    for ln in self.net.lane_of_edge[e]:
                        view.line([tuple(p) for p in view.P(ln["shape"])], rgba(MORE, .35 * a), ln["width"] * view.s * 1.5)
            a = win(u, .07, .33, .03, .03)
            if a > 0:
                E = arms["E_J22_J21"]
                c, d, half = self.net.stopline("E_J22_J21")
                self.tag(view, c - d * 40, ["East: vehicles already waiting (estimated)",
                                            f"q̂ = {E['q_hat']:.1f} vehicles",
                                            f"from a count of {E['n_obs']} on the link, sampled every 5 s",
                                            "and received 2 s late; moving vehicles removed using"
                                            , "the expected flow. Not the exact stopped-car count."], MORE, a, off=(40, 200))
            a = win(u, .33, .62, .03, .03)
            if a > 0:
                E = arms["E_J22_J21"]
                c, d, half = self.net.stopline("E_J22_J21")
                self.tag(view, c - d * 40, ["East: new arrivals while it waits",
                                            f"expected flow λ = {E['lam']:.3f} veh/s, design loading X = {E['X']:.2f}",
                                            f"λD² / (2 max(1 − X, 0.05)) = {E['term_arrival']:,.0f} veh·s",
                                            "the closer X is to 1, the slower a queue recovers"], MORE, a, off=(40, 200))
            return
        # rewind label
        a = win(u, 0, .12, 0, .04)
        if a > 0:
            view.rect((760, 100, 1160, 146), fill=(30, 33, 40, int(230 * a)), r=10)
            view.text((960, 123), "↺  back to t = 4,132 s, before the decision", F("b", 19), (255, 255, 255, int(255 * a)), "mm")
        # signal strip with displaced green
        view.rect((400, 880, 1900, 1062), fill=(255, 255, 255, 236), outline=(205, 208, 214, 255), w=1, r=12)
        hl = (4132, 4132 + g["D"], f"green displaced from others: D = {g['D']:.1f} s") if u > .2 else None
        self.strip(view, self.WH, "J21", 4100, 4260, (430, 930, 1880, 968), 4132,
                   label="J21 if the normal cycle had continued (counterfactual replay)", names=self.J21N, highlight=hl)
        # formula + breakdown
        a = win(u, .2, .82, .04, .03)
        if a > 0:
            box = (1200, 120, 1890, 420 if u < .62 else 700)
            self.panel(view, box, "Forecast burden of a priority pulse (Eq. 20)", a)
            img = equation(r"$\hat{C}_e=\sum_{j\neq e}\left[\hat q_j D_e+\frac{\lambda_j D_e^{2}}{2\max(1-X_j,\,0.05)}\right]$", 20)
            view.im.alpha_composite(img, (int((box[0] + 24) * 2), int((box[1] + 60) * 2)))
            y = box[1] + 60 + img.size[1] / 2 + 18
            img2 = equation(r"$D_e=\min\!\left(60,\ \max[5,\ 30+\hat d_e]\right)$", 17)
            view.im.alpha_composite(img2, (int((box[0] + 24) * 2), int(y * 2)))
            y += img2.size[1] / 2 + 14
            view.text((box[0] + 26, y), f"d̂ₑ = {g['d_hat']:.1f} s to clear the ambulance approach's own", F("m", 17), INK + (int(255 * a),), "la")
            view.text((box[0] + 26, y + 26), f"queue estimate ({g['q_hat_own']:.1f} vehicles)  →  D = {g['D']:.1f} s", F("m", 17), INK + (int(255 * a),), "la")
            if u > .62:
                b = win(u, .62, 1, .04, 0)
                y += 74
                view.text((box[0] + 26, y), "approach", F("s", 15), MUTED + (int(255 * b),), "la")
                for x, h in [(box[0] + 200, "waiting q̂D"), (box[0] + 345, "arrivals"), (box[0] + 500, "total")]:
                    view.text((x, y), h, F("s", 15), MUTED + (int(255 * b),), "la")
                y += 30
                tot = 0
                for e in ["E_J22_J21", "E_J31_J21", "E_J20_J21"]:
                    A = arms[e]; s = A["term_queue"] + A["term_arrival"]; tot += s
                    view.text((box[0] + 26, y), self.J21N[e], F("b", 19), INK + (int(255 * b),), "la")
                    view.text((box[0] + 200, y), f"{A['term_queue']:,.0f}", F("mono", 19), INK + (int(255 * b),), "la")
                    view.text((box[0] + 345, y), f"{A['term_arrival']:,.0f}", F("mono", 19), INK + (int(255 * b),), "la")
                    view.text((box[0] + 500, y), f"{s:,.0f}", F("monob", 19), INK + (int(255 * b),), "la")
                    y += 32
                view.line([(box[0] + 26, y + 4), (box[2] - 26, y + 4)], (200, 203, 208, int(255 * b)), 1.5)
                view.text((box[0] + 26, y + 14), "Ĉ =", F("b", 21), INK + (int(255 * b),), "la")
                view.text((box[0] + 500, y + 14), f"{g['burden']:,.0f} veh·s", F("monob", 21), INK + (int(255 * b),), "la")
        a = win(u, .82, 1, .04, 0)
        if a > 0:
            box = (1150, 120, 1890, 480)
            self.panel(view, box, "Admission test at J21 (t = 4,132 s)", a)
            rows = [("Forecast gain for the ambulance", f"{g['benefit']:.1f} s ≥ 5 s"),
                    ("Feasible within the 60 s hold (Eq. 19)", f"{g['feasible_lhs']:.1f} ≤ 60 s"),
                    ("Space on the next link (J21 → J22)", f"{g['headroom_obs']} < {g['headroom_cap']:.1f}"),
                    ("Forecast burden (Eq. 20)", f"{g['burden']:,.0f} ≤ 10,000"),
                    ("Legal transition", f"east min. green, amber, all-red: {g['switch_delay']:.0f} s")]
            y = box[1] + 74
            for lab, val in rows:
                view.circle((box[0] + 40, y + 12), 12, fill=(20, 130, 60, int(36 * a)), outline=(20, 130, 60, int(255 * a)), w=2)
                view.text((box[0] + 40, y + 12), "✓", F("b", 16), (20, 130, 60, int(255 * a)), "mm")
                view.text((box[0] + 64, y), lab, F("m", 17), INK + (int(255 * a),), "la")
                view.text((box[2] - 24, y), val, F("mono", 16), INK + (int(255 * a),), "ra")
                y += 44
            view.rect((box[0] + 22, box[3] - 62, box[2] - 22, box[3] - 18), fill=(20, 130, 60, int(30 * a)), outline=(20, 130, 60, int(255 * a)), w=2, r=22)
            view.text(((box[0] + box[2]) / 2, box[3] - 40), "PRIORITY ADMITTED", F("b", 20), (20, 130, 60, int(255 * a)), "mm")
        self.captions(view, u, [
            (.0, .33, ["Before deciding, AmbuGo estimates the cost of a priority pulse to the other approaches.",
                       "First: how many vehicles are already waiting there (an estimate from delayed counts)."]),
            (.33, .62, ["Second: how much green the others lose (D), and how many more vehicles arrive meanwhile.",
                        "Heavily loaded approaches recover slowly, so their arrivals weigh more."]),
            (.62, .82, ["The three approaches add up to a forecast burden of 3,326 vehicle-seconds.",
                        "This is a ranking estimate from observations, not a simulation of the future."]),
            (.82, 1, ["Every gate passes, including the burden budget of 10,000 veh·s, so the request is admitted.",
                      "Next: what actually happened, compared with keeping the normal cycle."])])

    def ov_compare(self, view, ctx, layer):
        u, t = ctx["u"], ctx["t"]
        if layer == "map":
            run = ctx["run"]
            self.stopped_badges(view, run, t)
            side = ctx["side"]
            lab = ["Scenario 1 — priority granted (AmbuGo's decision)", "Scenario 2 — normal cycle kept (counterfactual replay)"][side]
            col = (20, 130, 60) if side == 0 else (90, 96, 106)
            w = view.tw(lab, F("b", 21)) + 30
            view.rect((view.w / 2 - w / 2, 96, view.w / 2 + w / 2, 140), fill=(255, 255, 255, 240), outline=col + (255,), w=2, r=10)
            view.text((view.w / 2, 118), lab, F("b", 21), col + (255,), "mm")
            am = run.amb_xy(t)
            ns = "ambulance waiting at red" if (am and am[2] < .1 and am[3] == "E_J11_J21") else ("ambulance through J21" if am and am[3] != "E_J11_J21" else "ambulance approaching")
            green = 4142 if side == 0 else 4228
            view.text((view.w / 2, 168), f"{ns} · ambulance green at {green:,} s", F("m", 18), INK + (255,), "mm")
            return
        if u > .9:
            a = win(u, .9, 1, .03, 0)
            box = (420, 760, 1500, 1050)
            self.panel(view, box, "Stopped vehicle-seconds at J21, 4,125–4,300 s (sampled every 5 s)", a)
            rows = [("east", 1775, 670), ("south", 1080, 530), ("west", 110, 615), ("north (ambulance approach)", 90, 605)]
            view.text((box[0] + 520, box[1] + 70), "priority granted", F("s", 16), MUTED + (int(255 * a),), "ra")
            view.text((box[0] + 760, box[1] + 70), "normal cycle", F("s", 16), MUTED + (int(255 * a),), "ra")
            for k, (n, g, w) in enumerate(rows):
                y = box[1] + 100 + k * 36
                view.text((box[0] + 26, y), n, F("b", 19), INK + (int(255 * a),), "la")
                view.text((box[0] + 520, y), f"{g:,}", F("mono", 19), INK + (int(255 * a),), "ra")
                view.text((box[0] + 760, y), f"{w:,}", F("mono", 19), INK + (int(255 * a),), "ra")
            view.text((box[0] + 26, box[3] - 30), "Measured in the two replays. At most 19 vehicles on any J21 approach (storage 53–132 each): no spillback.",
                      F("m", 15), MUTED + (int(255 * a),), "la")

    def ov_network(self, view, ctx, layer):
        u, t = ctx["u"], ctx["t"]
        if layer == "map":
            run = self.P3
            for tt, j, lab, off in [(4175, "J22", "J22 · admitted", (-30, -90)), (4224, "J12", "J12 · admitted", (40, -80))]:
                a = 1.0 if t >= tt else 0
                if a and t < tt + 60:
                    g = self.g[(tt, j)]
                    self.tag(view, self.net.junctions[j]["xy"], [lab, f"gain {g['benefit']:.0f} s · burden {g['burden']:,.0f} veh·s"], (20, 130, 60), 1, off=off)
            if t >= 4256:
                a = 1.0 if t < 4300 else 0
                if a:
                    am = run.amb_xy(t)
                    if am:
                        self.tag(view, am[0], ["Stronger priority mode (JFAST)", "for the rest of this mission"], (200, 110, 0), 1, off=(-60, 110))
            # link labels for the redistribution
            if u < .5:
                for e, lab in [("E_J22_J23", "J22 → J23"), ("E_J23_J22", "J23 → J22")]:
                    mid = np.mean(self.net.lane_of_edge[e][0]["shape"], 0)
                    p = view.P(mid)
            return
        # progress deficit gauge
        pr = self.P3.progress
        k = int(math.floor(t))
        rec = pr.get(k) or pr.get(k - 1)
        a = win(u, .25, .92, .05, .05)
        if a > 0 and rec:
            box = (24, 650, 720, 830)
            self.panel(view, box, "Progress check (Eq. 21)", a)
            view.text((box[0] + 26, box[1] + 58), "elapsed time beyond the free-flow time of the roads already driven",
                      F("m", 16), MUTED + (int(255 * a),), "la")
            D = rec["D"] if t < 4256 or k >= 4256 else rec["D"]
            x0, x1, y = box[0] + 26, box[2] - 120, box[1] + 120
            sc = (x1 - x0) / 30
            view.rect((x0, y - 10, x1, y + 10), fill=(225, 227, 231, int(255 * a)), r=5)
            col = (200, 110, 0) if D > 20 else AMB_BLUE
            view.rect((x0, y - 10, x0 + min(30, D) * sc, y + 10), fill=col + (int(255 * a),), r=5)
            tx = x0 + 20 * sc
            view.line([(tx, y - 18), (tx, y + 18)], INK + (int(255 * a),), 3)
            view.text((tx, y + 30), "20 s threshold", F("s", 14), INK + (int(255 * a),), "mm")
            view.text((box[2] - 26, y), f"{D:.1f} s", F("monob", 22), col + (int(255 * a),), "rm")
        a = win(u, .0, .45, .02, .05)
        if a > 0:
            box = (1300, 640, 1890, 840)
            self.panel(view, box, "Tinted links (priority vs normal cycle)", a)
            view.rect((box[0] + 26, box[1] + 64, box[0] + 70, box[1] + 80), fill=rgba(MORE, .6 * a), r=4)
            view.text((box[0] + 84, box[1] + 72), "more vehicles present than in the normal-cycle replay", F("m", 16), INK + (int(255 * a),), "lm")
            view.rect((box[0] + 26, box[1] + 100, box[0] + 70, box[1] + 116), fill=rgba(LESS, .6 * a), r=4)
            view.text((box[0] + 84, box[1] + 108), "fewer vehicles present", F("m", 16), INK + (int(255 * a),), "lm")
            view.text((box[0] + 26, box[1] + 150), "same time, ±10 s average · not a signal colour", F("m", 15), MUTED + (int(255 * a),), "lm")
        self.captions(view, u, [
            (0, .26, ["The change at J21 spreads: platoons leave at different times, so the links beyond J22, between J22 and J23,",
                      "also carry more or fewer vehicles than with the normal cycle."]),
            (.26, .5, ["Meanwhile the ambulance's requests are admitted at J22 and J12. Its progress is compared",
                       "with the free-flow time of the roads it has actually driven."]),
            (.5, .66, ["At 4,256 s the shortfall reaches 20.5 s, over the 20 s threshold: for the rest of this trip",
                       "it requests stronger priority without the burden test. Legal signal stages still apply."]),
            (.66, 1, ["The ambulance reaches the hospital boundary BE1 at 4,316.5 s, 252.5 s after dispatch."])])

    def ov_result(self, view, ctx, layer):
        u = ctx["u"]
        if layer == "map":
            return
        a = win(u, .2, 1, .05, 0)
        if a <= 0:
            return
        box = (430, 150, 1490, 840)
        self.panel(view, box, "", a)
        x, y = box[0] + 40, box[1] + 50
        A = int(255 * a)
        view.text((x, y), "This episode (seed 40, load 0.95; identical demand in every run)", F("b", 24), INK + (A,), "la")
        rows = [("AmbuGo (JSEL), as decided", "252.5 s"), ("same run, J21 request withheld (one-decision counterfactual)", "323.5 s"),
                ("no signal priority (S0)", "572.0 s")]
        y += 50
        for lab, val in rows:
            view.text((x + 10, y), lab, F("m", 21), INK + (A,), "la"); view.text((box[2] - 50, y), val, F("monob", 22), INK + (A,), "ra"); y += 38
        view.text((x + 10, y + 6), "Around J21, 4,132–4,400 s, priority vs normal cycle: positive spatial exposure A = 5,992 veh·s,",
                  F("m", 18), MUTED + (A,), "la")
        view.text((x + 10, y + 32), "but signed residence change −74 veh·s: vehicles were moved between links more than delayed overall.",
                  F("m", 18), MUTED + (A,), "la")
        y += 100
        view.text((x, y), "Across the paper's 1,728 microscopic runs (balanced means)", F("b", 24), INK + (A,), "la")
        y += 50
        rows = [("route-only, no priority (S0)", "401.57 s"), ("aggressive joint control (JFAST)", "211.19 s"),
                ("selective joint control (JSEL)", "222.29 s")]
        for lab, val in rows:
            view.text((x + 10, y), lab, F("m", 21), INK + (A,), "la"); view.text((box[2] - 50, y), val, F("monob", 22), INK + (A,), "ra"); y += 38
        y += 8
        for s in ["JSEL vs JFAST: +11.10 s ambulance time [95% CI 9.47, 12.94],",
                  "positive spatial exposure −3.84 veh·h; positive individual residence (vs S0) −1.64 veh·h.",
                  "JSEL's mean is 44.6% below S0's in this synthetic grid, and 51.6% of JSEL missions",
                  "escalated to the aggressive (JFAST) mode, so that time is not due to selectivity alone."]:
            view.text((x + 10, y), s, F("m", 19), INK + (A,), "la"); y += 32
        y += 18
        view.text((x, y), "Synthetic, uncalibrated network and demand: a reproducible method, not a field or clinical result.",
                  F("s", 18), (150, 60, 60) + (A,), "la")


# ---------------------------------------------------------------- render
_film = None


def _init(scale):
    global _film
    _film = Film(scale)


def _chunk(args):
    f0, f1, path, scale = args
    w, h = int(W * scale), int(H * scale)
    cmd = [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-threads", "1", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(_film.frame(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return f1 - f0


def render(jobs, scale=1.0, name="AmbuGo_v9_Explainer.mp4"):
    OUT.mkdir(parents=True, exist_ok=True)
    film = Film(scale)
    n = int(round(film.total * FPS))
    tmp = OUT / "_parts"; tmp.mkdir(exist_ok=True)
    parts = [(f0, min(n, f0 + 90), tmp / f"p{i:04d}.mp4", scale) for i, f0 in enumerate(range(0, n, 90))]
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs, initializer=_init, initargs=(scale,)) as pool:
        done = 0
        for k in pool.imap_unordered(_chunk, parts):
            done += k
            print(f"{done}/{n} frames {time.time() - t0:.0f}s", flush=True)
    (tmp / "list.txt").write_text("".join(f"file '{p[2].name}'\n" for p in parts))
    final = OUT / name
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    shutil.rmtree(tmp)
    print(final, film.total)


def contact(film, times, path, cols=4):
    ims = [film.frame(int(min(t, film.total - 1 / FPS) * FPS)).resize((480, 270), Image.Resampling.LANCZOS) for t in times]
    rows = math.ceil(len(ims) / cols)
    from PIL import ImageDraw
    sheet = Image.new("RGB", (cols * 490 + 10, rows * 300 + 10), (20, 22, 26))
    d = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(zip(times, ims)):
        x, y = 10 + (i % cols) * 490, 10 + (i // cols) * 300
        sheet.paste(im, (x, y))
        _, sc, u = film.locate(t)
        d.text((x, y + 276), f"{t:6.1f} s · {sc['key']} · sim {film.sim(sc, u):.1f}", fill=(220, 220, 220))
    sheet.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stills", type=float, nargs="*")
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stills:
        film = Film()
        d = OUT / "stills"; d.mkdir(exist_ok=True)
        for v in a.stills:
            film.frame(int(v * FPS)).save(d / f"v{v:06.2f}.jpg", quality=90)
        print("total", film.total)
        return
    if a.contact:
        film = Film()
        times = [float(film.starts[i] + film.scenes[i]["dur"] * f) for i in range(len(film.scenes)) for f in (.3, .85)]
        contact(film, times, OUT / "contact_sheet.jpg")
        print(OUT / "contact_sheet.jpg", film.total)
        return
    render(a.jobs, .5 if a.draft else 1.0, "draft.mp4" if a.draft else "AmbuGo_v9_Explainer.mp4")


if __name__ == "__main__":
    main()
