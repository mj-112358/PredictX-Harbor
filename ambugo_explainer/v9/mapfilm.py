#!/usr/bin/env python3
"""AmbuGo v9 short explainer on Google Maps captures of Mohali: two junctions, two decisions.

    python mapfilm.py                  # output/AmbuGo_v9_Short.mp4
    python mapfilm.py --stills 5 40    # stills at video seconds
    python mapfilm.py --contact        # contact sheet
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world  # noqa: E402
from world import Net, Run, View, rgba, SS, VCOL, AMB_BLUE, HOSP, SIG_G, SIG_A, SIG_R  # noqa: E402
from mapfit import Warp, M_PER_PX  # noqa: E402

W, H, FPS = 1920, 1080, 30
OUT = Path(__file__).resolve().parent / "output"
FONTS = Path(__file__).resolve().parent.parent / "inputs" / "fonts"
INK = (32, 33, 36)
MUTED = (95, 99, 104)
OK = (24, 128, 56)
NO = (197, 34, 31)
VEH_SCALE = 1.6          # vehicles drawn 1.6x their SUMO size so they read on the map


def G(w, size, italic=False):
    """EB Garamond, registered with world's font cache so missing glyphs fall back to DejaVu."""
    kind = f"g{w}{'i' if italic else ''}"
    key = (kind, size)
    if key not in world._fonts:
        name = "eb-garamond-italic-400" if italic else f"eb-garamond-{w}"
        world._fonts[key] = ImageFont.truetype(str(FONTS / f"{name}.ttf"), int(size * SS))
    return world._fonts[key]


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def win(u, a, b, fi=.05, fo=.05):
    if u < a or u > b:
        return 0.0
    return min(ease((u - a) / fi) if fi else 1, ease((b - u) / fo) if fo else 1)


def interp(knots, u):
    us = [k[0] for k in knots]
    i = int(np.clip(np.searchsorted(us, u, side="right") - 1, 0, len(knots) - 2))
    a, b = knots[i], knots[i + 1]
    return b[1] if b[0] == a[0] else a[1] + (b[1] - a[1]) * (u - a[0]) / (b[0] - a[0])


class Film:
    def __init__(self):
        self.net = Net()
        self.runs = {n: Run(n) for n in ("P3", "P2", "P3_withholdJ21")}
        self.warp = {j: Warp(self.net, j) for j in ("J11", "J21")}
        self.g = {(g["t"], g["junction"]): g for g in self.runs["P3"].gates}
        self.lane_ids = {n: np.array(r.lanes) for n, r in self.runs.items()}
        self.build()
        self.starts = np.cumsum([0] + [s["dur"] for s in self.scenes])
        self.total = float(self.starts[-1])

    # ------------------------------------------------------------------ scenes
    def build(self):
        S = []
        S.append(dict(key="intro", j="J11", runs=("P3",), dur=7, sim=[(0, 4078), (1, 4091)], zoom=[(0, .95), (1, 1.25)],
                      ov=self.ov_intro))
        S.append(dict(key="j11", j="J11", runs=("P3",), dur=12, sim=[(0, 4091), (1, 4091)], zoom=[(0, 1.25), (1, 1.25)],
                      ov=self.ov_j11))
        S.append(dict(key="j11cmp", j="J11", runs=("P3", "P2"), dur=15, sim=[(0, 4091), (.05, 4091), (.82, 4170), (1, 4170)],
                      zoom=[(0, 1.05), (1, 1.05)], ov=self.ov_j11cmp))
        S.append(dict(key="j21in", j="J21", runs=("P3",), dur=6, sim=[(0, 4118), (1, 4132)], zoom=[(0, .95), (1, 1.2)],
                      ov=self.ov_j21in))
        S.append(dict(key="j21", j="J21", runs=("P3",), dur=15, sim=[(0, 4132), (1, 4132)], zoom=[(0, 1.2), (1, 1.2)],
                      ov=self.ov_j21))
        S.append(dict(key="j21cmp", j="J21", runs=("P3", "P3_withholdJ21"), dur=17,
                      sim=[(0, 4132), (.04, 4132), (.86, 4236), (1, 4236)], zoom=[(0, 1.0), (1, 1.0)], ov=self.ov_j21cmp))
        S.append(dict(key="close", j="J21", runs=("P3",), dur=12, sim=[(0, 4236), (1, 4236)], zoom=[(0, 1.0), (1, .9)],
                      ov=self.ov_close))
        self.scenes = S

    def locate(self, v):
        i = int(np.clip(np.searchsorted(self.starts, v, side="right") - 1, 0, len(self.scenes) - 1))
        sc = self.scenes[i]
        return sc, float(np.clip((v - self.starts[i]) / sc["dur"], 0, 1))

    def rate(self, sc, u):
        du = 1 / (FPS * sc["dur"])
        a, b = max(0, u - du), min(1, u + du)
        return (interp(sc["sim"], b) - interp(sc["sim"], a)) / ((b - a) * sc["dur"])

    # ------------------------------------------------------------------ drawing
    def pane(self, j, run_name, t, w, h, zoom, label=None, lab_col=None):
        wp = self.warp[j]
        cx, cy = wp.C
        sw, sh = w / zoom, h / zoom
        box = (int(cx - sw / 2), int(cy - sh / 2), int(cx - sw / 2) + int(sw), int(cy - sh / 2) + int(sh))
        bg = wp.img.crop(box).resize((w * SS, h * SS), Image.Resampling.BICUBIC).convert("RGBA")
        v = View(w, h, 0, 0, 1)
        v.im, v.d = bg, ImageDraw.Draw(bg)
        to = lambda P: (np.asarray(P, float) - np.array(box[:2], float)) * zoom      # capture px -> pane px
        v.s = zoom / M_PER_PX * VEH_SCALE / 1.6                                       # px per metre for the ambulance glyph
        run = self.runs[run_name]
        self.draw_signals(v, wp, run, j, t, to, zoom)
        self.draw_vehicles(v, wp, run, run_name, j, t, to, zoom)
        return v, to

    def draw_signals(self, v, wp, run, j, t, to, zoom):
        mode, served, el, req = run.signal(j, t)
        for e in self.net.incoming[j]:
            c, d, half = self.net.stopline(e)
            n = np.array([-d[1], d[0]])
            col = SIG_G if (mode == "green" and served == e) else SIG_A if (mode == "amber" and served == e) else SIG_R
            a, b = to(wp(c - n * half)[0]), to(wp(c + n * half)[0])
            v.line([tuple(a), tuple(b)], (255, 255, 255, 230), 9 * zoom + 4)
            v.line([tuple(a), tuple(b)], col, 9 * zoom)
            # signal head on the kerb side, just before the stop line
            p = to(wp(c - d * 3 - n * (half + 4))[0])
            r = 9 * zoom + 2
            v.rect((p[0] - r - 4, p[1] - 3 * r - 6, p[0] + r + 4, p[1] + 3 * r + 6), fill=(30, 32, 36, 255),
                   outline=(255, 255, 255, 235), w=2, r=5)
            for i, lc in enumerate([SIG_R, SIG_A, SIG_G]):
                v.circle((p[0], p[1] + (i - 1) * (2 * r + 2)), r, fill=lc if lc == col else (74, 78, 84))

    def draw_vehicles(self, v, wp, run, run_name, j, t, to, zoom):
        vid, xy, ang, sp, le, vt = run.state(t)
        k = run.k(t)
        lanes = self.lane_ids[run_name][run.lane[run.off[k]:run.off[k + 1]]]
        E = wp.edges()
        keep = np.array([(l.rsplit("_", 1)[0] in E) or l.startswith(f":{j}_") for l in lanes])
        if not keep.any():
            return
        P = to(wp(xy[keep]))
        A = wp.heading(xy[keep], ang[keep])
        amb = None
        for idx, p, a, ty in zip(vid[keep], P, A, vt[keep]):
            if not (-60 < p[0] < v.w + 60 and -60 < p[1] < v.h + 60):
                continue
            if idx == run.amb:
                amb = (p, a)
                continue
            tn = run.types[ty]
            L, B = View.DIMS[tn]
            s = zoom / M_PER_PX * VEH_SCALE
            pal = VCOL[tn]
            v.car(p, a, L * s, B * s, pal[idx % len(pal)])
        if amb is not None:
            v.ambulance(amb[0], amb[1], size=1.0)

    # ------------------------------------------------------------------ chrome
    def caption(self, ui, lines, a=1.0, y0=26):
        if a <= 0:
            return
        f = G(500, 36)
        lh = 46
        w = max(ui.tw(s, f) for s in lines) + 80
        x0 = W / 2 - w / 2
        h = len(lines) * lh + 34
        ui.rect((x0 + 3, y0 + 6, x0 + w + 3, y0 + h + 6), fill=(0, 0, 0, int(30 * a)), r=16)
        ui.rect((x0, y0, x0 + w, y0 + h), fill=(255, 255, 255, int(246 * a)), outline=(218, 220, 224, int(255 * a)), w=1, r=16)
        for i, s in enumerate(lines):
            ui.text((W / 2, y0 + 17 + lh / 2 + i * lh), s, f, INK + (int(255 * a),), "mm")

    def captions(self, ui, u, items):
        for a, b, lines in items:
            al = win(u, a, b, .035 if a > 0 else 0, .035 if b < 1 else 0)
            if al > 0:
                self.caption(ui, lines, al)

    def chrome(self, ui, sc, u, t, label):
        # junction label (bottom left) and simulation clock (bottom right)
        if label:
            f = G(600, 26)
            w = ui.tw(label, f) + 40
            ui.rect((24, H - 74, 24 + w, H - 28), fill=(255, 255, 255, 240), outline=(218, 220, 224, 255), w=1, r=10)
            ui.text((44, H - 51), label, f, INK + (255,), "lm")
        r = self.rate(sc, u)
        s = f"t = {t:,.1f} s   ·   " + ("paused" if abs(r) < .05 else f"{r:.1f}× real time")
        f = G(500, 22)
        w = ui.tw(s, f) + 36
        ui.rect((W - 24 - w, H - 70, W - 24, H - 30), fill=(255, 255, 255, 235), outline=(218, 220, 224, 255), w=1, r=10)
        ui.text((W - 42, H - 50), s, f, MUTED + (255,), "rm")
        # Google attribution
        s = "Google Maps  ·  Map data ©2026 Google"
        f = world.F("s", 15)
        w = ui.tw(s, f) + 24
        ui.rect((W / 2 - w / 2, H - 40, W / 2 + w / 2, H - 14), fill=(255, 255, 255, 215), r=6)
        ui.text((W / 2, H - 27), s, f, (60, 62, 66, 255), "mm")

    def side_label(self, ui, x, text, col):
        f = G(600, 27)
        w = ui.tw(text, f) + 36
        ui.rect((x - w / 2, 178, x + w / 2, 222), fill=(255, 255, 255, 245), outline=col + (255,), w=2.5, r=12)
        ui.text((x, 200), text, f, col + (255,), "mm")

    def tick(self, ui, x, y, ok, a=1.0, r=14):
        c = OK if ok else NO
        ui.circle((x, y), r, fill=rgba(c, .12 * a), outline=rgba(c, a), w=2.2)
        if ok:
            ui.line([(x - r * .45, y + r * .02), (x - r * .1, y + r * .38), (x + r * .5, y - r * .38)], rgba(c, a), 3)
        else:
            ui.line([(x - r * .4, y - r * .4), (x + r * .4, y + r * .4)], rgba(c, a), 3)
            ui.line([(x - r * .4, y + r * .4), (x + r * .4, y - r * .4)], rgba(c, a), 3)

    def gate_panel(self, ui, rows, verdict, ok, a, box, reveal):
        """rows: (ok, label, value, limit); reveal in [0,1] uncovers rows one by one, then the verdict."""
        x0, y0, x1, y1 = box
        A = int(255 * a)
        ui.rect((x0 + 4, y0 + 8, x1 + 4, y1 + 8), fill=(0, 0, 0, int(30 * a)), r=18)
        ui.rect(box, fill=(255, 255, 255, int(248 * a)), outline=(218, 220, 224, A), w=1, r=18)
        ui.text((x0 + 32, y0 + 44), "Should the light change for the ambulance?", G(600, 31), INK + (A,), "lm")
        n = len(rows)
        for i, (g, lab, val, lim) in enumerate(rows):
            b = a * ease((reveal * (n + 1) - i) / .6)
            if b <= 0:
                continue
            y = y0 + 110 + i * 74
            self.tick(ui, x0 + 48, y + 2, g, b)
            ui.text((x0 + 80, y - 12), lab, G(500, 27), INK + (int(255 * b),), "lm")
            ui.text((x0 + 80, y + 18), lim, G(400, 22, True), MUTED + (int(255 * b),), "lm")
            ui.text((x1 - 32, y + 2), val, G(600, 30), (OK if g else NO) + (int(255 * b),), "rm")
        b = a * ease((reveal * (n + 1) - n) / .6)
        if b > 0:
            c = OK if ok else NO
            yy = y1 - 70
            ui.rect((x0 + 24, yy - 30, x1 - 24, yy + 30), fill=rgba(c, .1 * b), outline=rgba(c, b), w=2.5, r=30)
            ui.text(((x0 + x1) / 2, yy), verdict, G(700, 30), rgba(c, b), "mm")

    # ------------------------------------------------------------------ overlays
    def ov_intro(self, ui, u, t, panes):
        self.captions(ui, u, [(0, 1, ["An ambulance is on its way to hospital. At each signal it may ask for the light",
                                      "to be changed. AmbuGo decides, junction by junction, whether that is worth it."])])
        v, to = panes[0]
        self.amb_tag(ui, v, to, "J11", "P3", t, "Ambulance", win(u, .3, 1, .1, 0))

    def amb_tag(self, ui, v, to, j, run_name, t, text, a):
        if a <= 0:
            return
        run = self.runs[run_name]
        am = run.amb_xy(t)
        if not am:
            return
        p = to(self.warp[j](am[0])[0])
        f = G(600, 24)
        w = ui.tw(text, f) + 28
        x, y = p[0] - w / 2, p[1] - 92
        ui.line([(p[0], p[1] - 36), (p[0], y + 40)], rgba(AMB_BLUE, a), 2.5)
        ui.rect((x, y, x + w, y + 40), fill=(255, 255, 255, int(245 * a)), outline=rgba(AMB_BLUE, a), w=2, r=10)
        ui.text((p[0], y + 20), text, f, rgba(AMB_BLUE, a), "mm")

    def ov_j11(self, ui, u, t, panes):
        g = self.g[(4091, "J11")]
        self.captions(ui, u, [
            (0, .5, ["At this junction the light would turn green for the ambulance only",
                     f"about {g['nominal_wait']:.1f} seconds after it arrives, even without any change."]),
            (.5, 1, ["Interrupting the cross traffic would save the ambulance less than the 5-second minimum,",
                     "so AmbuGo leaves the signal on its normal cycle."])])
        a = win(u, .08, 1, .05, 0)
        rows = [(False, "Time saved for the ambulance", f"{g['benefit']:.1f} s", "must be at least 5 s"),
                (True, "Extra waiting caused for other traffic", f"{g['burden']:,.0f} veh·s", "limit 10,000 vehicle-seconds"),
                (True, "Room on the road ahead", f"{g['headroom_obs']} of {g['headroom_cap']:.0f}", "vehicles, must stay below 85% of space"),
                (True, "Possible within one 60 s priority hold", f"{g['feasible_lhs']:.1f} s", "must be at most 60 s")]
        self.gate_panel(ui, rows, "No change: normal signal cycle", False, a, (1170, 250, 1890, 830), np.clip((u - .1) / .5, 0, 1))
        v, to = panes[0]
        self.amb_tag(ui, v, to, "J11", "P3", t, "Ambulance", 1)

    def ov_j11cmp(self, ui, u, t, panes):
        self.side_label(ui, W / 4, "AmbuGo: signal not changed", OK)
        self.side_label(ui, 3 * W / 4, "Always-priority controller (JFAST)", (150, 70, 20))
        for k, (name, (v, to)) in enumerate(zip(("P3", "P2"), panes)):
            self.road_badges(ui, v, to, "J11", name, t, k * (W // 2 + 4))
            self.amb_tag(ui, v, to, "J11", name, t, "Ambulance", 1, ) if False else None
        self.captions(ui, u, [
            (0, .55, ["The same moment, the same traffic. On the right, a controller that always changes the light",
                      "for the ambulance, as in the paper's aggressive comparison (JFAST)."]),
            (.55, 1, ["Changing the light saves the ambulance 11 s here, but the cross road waits 64 s longer",
                      "for its green: 463 instead of 79 vehicle-seconds of stopped time."])])

    def road_badges(self, ui, v, to, j, run_name, t, xoff):
        run = self.runs[run_name]
        names = {"J11": {"E_J01_J11": "cross road", "E_J10_J11": "ambulance's road"},
                 "J21": {"E_J22_J21": "east road", "E_J31_J21": "south road", "E_J20_J21": "west road",
                         "E_J11_J21": "ambulance's road"}}[j]
        wp = self.warp[j]
        for e, nm in names.items():
            c, d, half = self.net.stopline(e)
            p = to(wp(c - d * 45)[0]) + np.array([xoff, 0])
            n = run.halted(t, e)
            txt = f"{nm}: {n} stopped"
            f = G(600, 22)
            w = ui.tw(txt, f) + 24
            col = AMB_BLUE if "ambulance" in nm else INK
            ui.rect((p[0] - w / 2, p[1] - 17, p[0] + w / 2, p[1] + 17), fill=(255, 255, 255, 240), outline=col + (255,), w=1.8, r=9)
            ui.text((p[0], p[1]), txt, f, col + (255,), "mm")

    def ov_j21in(self, ui, u, t, panes):
        g = self.g[(4132, "J21")]
        self.captions(ui, u, [(0, 1, ["Forty seconds later, the next junction. Here the ambulance would otherwise",
                                      f"wait about {g['nominal_wait']:.0f} seconds for its green."])])
        v, to = panes[0]
        self.amb_tag(ui, v, to, "J21", "P3", t, "Ambulance", win(u, .2, 1, .1, 0))

    def ov_j21(self, ui, u, t, panes):
        g = self.g[(4132, "J21")]
        arms = {a["edge"]: a for a in g["arms"]}
        self.captions(ui, u, [
            (0, .45, ["Before changing the light, AmbuGo estimates the extra waiting it would cause on the",
                      "other three roads, from delayed and noisy vehicle counts."]),
            (.45, 1, [f"About {g['burden']:,.0f} vehicle-seconds in total, well within its limit of 10,000,",
                      f"and the ambulance saves about {g['benefit']:.0f} s. So the light is changed."])])
        v, to = panes[0]
        wp = self.warp["J21"]
        b = win(u, .12, 1, .06, 0)
        if b > 0:
            for e, nm in [("E_J22_J21", "east road"), ("E_J31_J21", "south road"), ("E_J20_J21", "west road")]:
                A = arms[e]
                c, d, half = self.net.stopline(e)
                p = to(wp(c - d * 55)[0])
                txt1, txt2 = nm, f"+{A['term_queue'] + A['term_arrival']:,.0f} veh·s"
                f1, f2 = G(500, 22), G(700, 28)
                w = max(ui.tw(txt1, f1), ui.tw(txt2, f2)) + 28
                ui.rect((p[0] - w / 2, p[1] - 36, p[0] + w / 2, p[1] + 36), fill=(255, 255, 255, int(245 * b)),
                        outline=(150, 70, 20, int(255 * b)), w=2, r=10)
                ui.text((p[0], p[1] - 14), txt1, f1, MUTED + (int(255 * b),), "mm")
                ui.text((p[0], p[1] + 14), txt2, f2, (150, 70, 20, int(255 * b)), "mm")
        a = win(u, .08, 1, .05, 0)
        rows = [(True, "Time saved for the ambulance", f"{g['benefit']:.1f} s", "must be at least 5 s"),
                (True, "Extra waiting caused for other traffic", f"{g['burden']:,.0f} veh·s", "limit 10,000 vehicle-seconds"),
                (True, "Room on the road ahead", f"{g['headroom_obs']} of {g['headroom_cap']:.0f}", "vehicles, must stay below 85% of space"),
                (True, "Possible within one 60 s priority hold", f"{g['feasible_lhs']:.1f} s", "must be at most 60 s")]
        self.gate_panel(ui, rows, "Change the light: priority given", True, a, (1170, 250, 1890, 830), np.clip((u - .1) / .55, 0, 1))
        self.amb_tag(ui, v, to, "J21", "P3", t, "Ambulance", 1)

    def ov_j21cmp(self, ui, u, t, panes):
        self.side_label(ui, W / 4, "AmbuGo: priority given", OK)
        self.side_label(ui, 3 * W / 4, "Same moment, light not changed (replay)", (90, 94, 100))
        for k, (name, (v, to)) in enumerate(zip(("P3", "P3_withholdJ21"), panes)):
            self.road_badges(ui, v, to, "J21", name, t, k * (W // 2 + 4))
        self.captions(ui, u, [
            (0, .4, ["The change is made safely: the current green first runs its 8-second minimum,",
                     "then amber and all-red, before the ambulance's road turns green."]),
            (.4, .75, ["The ambulance gets its green at 4,142 s instead of 4,228 s. The extra waiting",
                       "falls on the east and south roads, as the estimate predicted."]),
            (.75, 1, ["No queue comes close to filling its road (at most 19 vehicles), and the",
                      "normal cycle resumes once the ambulance has passed."])])

    def ov_close(self, ui, u, t, panes):
        a = win(u, .05, 1, .06, 0)
        A = int(255 * a)
        box = (300, 150, 1620, 960)
        ui.rect((box[0] + 4, box[1] + 8, box[2] + 4, box[3] + 8), fill=(0, 0, 0, int(34 * a)), r=22)
        ui.rect(box, fill=(255, 255, 255, int(250 * a)), outline=(218, 220, 224, A), w=1, r=22)
        x = box[0] + 70
        ui.text((x, box[1] + 70), "How AmbuGo keeps the after-effects small", G(700, 44), INK + (A,), "lm")
        items = [("Change a light only when it saves the ambulance at least 5 s.", "Junction 1: 3.6 s, so no change."),
                 ("Change it only when the estimated extra waiting is within budget.", "Junction 2: 3,326 of 10,000 vehicle-seconds."),
                 ("Always through legal stages.", "Minimum green, amber, all-red; then the normal cycle resumes.")]
        for i, (s1, s2) in enumerate(items):
            b = a * ease((u - .1 - i * .1) / .08)
            y = box[1] + 165 + i * 100
            ui.text((x, y), f"{i + 1}.", G(700, 34), (150, 70, 20, int(255 * b)), "lm")
            ui.text((x + 46, y), s1, G(500, 32), INK + (int(255 * b),), "lm")
            ui.text((x + 46, y + 38), s2, G(400, 26, True), MUTED + (int(255 * b),), "lm")
        b = a * ease((u - .5) / .08)
        y = box[1] + 500
        ui.line([(x, y - 20), (box[2] - 70, y - 20)], (218, 220, 224, int(255 * b)), 1.5)
        ui.text((x, y + 20), "Across the study's 1,728 simulation runs (averages)", G(600, 30), INK + (int(255 * b),), "lm")
        for i, s in enumerate(["Trips 44.6% faster than with no signal priority.",
                               "11.1 s slower per trip than changing every light (JFAST),",
                               "but 3.84 vehicle-hours less extra road occupancy per run."]):
            ui.text((x + 10, y + 74 + i * 42), s, G(500, 29), INK + (int(255 * b),), "lm")

    # ------------------------------------------------------------------ frame
    def frame(self, f):
        v = f / FPS
        sc, u = self.locate(v)
        t = float(interp(sc["sim"], u))
        zoom = float(interp(sc["zoom"], u))
        j = sc["j"]
        img = Image.new("RGBA", (W, H), (240, 240, 240, 255))
        panes = []
        if len(sc["runs"]) == 1:
            pv, to = self.pane(j, sc["runs"][0], t, W, H, zoom)
            img = pv.down()
            panes.append((pv, to))
        else:
            for k, rn in enumerate(sc["runs"]):
                pv, to = self.pane(j, rn, t, W // 2 - 2, H, zoom)
                img.paste(pv.down(), (k * (W // 2 + 2), 0))
                panes.append((pv, to))
        if sc["key"] == "close":
            img = img.filter(ImageFilter.GaussianBlur(6 * ease(u / .1)))
        ui = View(W, H, 0, 0, 1)
        ui.im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        ui.d = ImageDraw.Draw(ui.im)
        sc["ov"](ui, u, t, panes)
        label = {"intro": "Junction 1", "j11": "Junction 1", "j11cmp": "Junction 1", "j21in": "Junction 2",
                 "j21": "Junction 2", "j21cmp": "Junction 2", "close": None}[sc["key"]]
        self.chrome(ui, sc, u, t, label)
        return Image.alpha_composite(img.convert("RGBA"), ui.down()).convert("RGB")


# ---------------------------------------------------------------------- render
_film = None


def _init():
    global _film
    _film = Film()


def _chunk(args):
    f0, f1, path = args
    cmd = [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-threads", "1", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(_film.frame(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return f1 - f0


def render(jobs, name="AmbuGo_v9_Short.mp4"):
    film = Film()
    n = int(round(film.total * FPS))
    tmp = OUT / "_parts_short"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    parts = [(f0, min(n, f0 + 90), tmp / f"p{i:04d}.mp4") for i, f0 in enumerate(range(0, n, 90))]
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs, initializer=_init) as pool:
        done = 0
        for k in pool.imap_unordered(_chunk, parts):
            done += k
            print(f"{done}/{n} frames {time.time() - t0:.0f}s", flush=True)
    (tmp / "list.txt").write_text("".join(f"file '{p[2].name}'\n" for p in parts))
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c", "copy", "-movflags", "+faststart", str(OUT / name)], check=True)
    shutil.rmtree(tmp)
    print(OUT / name, film.total)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stills", type=float, nargs="*")
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stills:
        film = Film()
        d = OUT / "stills"
        d.mkdir(exist_ok=True)
        for v in a.stills:
            film.frame(int(v * FPS)).save(d / f"short_{v:06.2f}.jpg", quality=90)
        print("total", film.total)
        return
    if a.contact:
        film = Film()
        times = [float(film.starts[i] + film.scenes[i]["dur"] * f) for i in range(len(film.scenes)) for f in (.3, .85)]
        ims = [film.frame(int(min(t, film.total - .04) * FPS)).resize((480, 270), Image.Resampling.LANCZOS) for t in times]
        cols = 4
        rows = math.ceil(len(ims) / cols)
        sheet = Image.new("RGB", (cols * 490 + 10, rows * 300 + 10), (20, 22, 26))
        d = ImageDraw.Draw(sheet)
        for i, (t, im) in enumerate(zip(times, ims)):
            x, y = 10 + (i % cols) * 490, 10 + (i // cols) * 300
            sheet.paste(im, (x, y))
            sc, u = film.locate(t)
            d.text((x, y + 276), f"{t:5.1f} s  {sc['key']}  sim {interp(sc['sim'], u):.1f}", fill=(220, 220, 220))
        sheet.save(OUT / "contact_sheet_short.jpg", quality=90)
        print(OUT / "contact_sheet_short.jpg", film.total)
        return
    render(a.jobs)


if __name__ == "__main__":
    main()
