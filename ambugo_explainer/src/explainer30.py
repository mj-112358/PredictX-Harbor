#!/usr/bin/env python3
"""AmbuGo — 30-second walkthrough of how the controller decides, step by step.

1 ROUTES        candidate routes scored with and without priority (decision log, t = 761 s)
2 CONDITIONS    what the controller measures at PCL Chowk (queue estimates, arrival rates, signal)
3 OPTIONS       each option forecast: priority now vs keep the normal cycle; cost drawn on each road
4 RE-EVALUATION the same test every second until the forecast falls within the budget (785-792 s)
5 ACTUATION     legal transition, green at 796.5 s, ambulance passes
6 EFFECTS       measured change on the surrounding roads, and the observed cost vs the forecast

Reuses the drawing code of cinematic.py; only timing, beats and the step-specific panels differ.

    python src/explainer30.py                 # output/cinematic/AmbuGo_30s_Explainer.mp4
    python src/explainer30.py --stills 2 10   # stills
    python src/explainer30.py --contact       # contact sheet
"""
from pathlib import Path
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cinematic as C  # noqa: E402
from cinematic import (F, rgba, mix, ease, ease_out, win, W, H, CYAN, CORAL, EMERALD, IVORY, MUTED, DIM,  # noqa: E402
                       APP, APP_NAME, ROADS, MORE, FEWER)

DUR = 30.0
KNOTS = [(0.0, 761.0), (4.0, 766.0), (7.0, 780.0), (8.0, 785.0), (12.8, 786.4), (13.4, 787.0), (15.6, 791.8),
         (15.9, 792.0), (17.0, 793.2), (19.2, 796.6), (20.0, 797.8), (22.0, 804.5), (26.0, 862.0), (27.6, 880.0),
         (30.0, 880.0)]
STAGES = ["ROUTES", "CONDITIONS", "OPTIONS", "RE-EVALUATION", "ACTUATION", "EFFECTS"]
BEATS = [
    (0.0, 4.5, 0, "Candidate routes are scored",
     "Each route to the hospital is forecast with and without signal priority"),
    (4.5, 8.0, 1, "Conditions at the junction are measured",
     "Queue estimates from delayed sensor counts, historical arrival rates and the signal state"),
    (8.0, 12.8, 2, "Each option is forecast before it is used",
     "Priority now would save the ambulance about 73 s, but add 1,045 vehicle-seconds of delay elsewhere"),
    (12.8, 17.6, 3, "The decision is re-evaluated every second",
     "Priority is admitted once the forecast cost falls within the budget of 1,000"),
    (17.6, 22.0, 4, "The signal changes at the earliest legal time",
     "The pedestrian interval completes first; the ambulance is served 72 s earlier than without priority"),
    (22.0, 26.5, 5, "Measured effect on the surrounding roads",
     "Vehicles present on each road, relative to the same traffic without priority"),
    (26.5, 30.0, None, "Priority is granted only within a forecast budget",
     "Observed extra stopped time on the conflicting approaches, against the forecast of 913"),
]

# Install the 30 s timing into the shared renderer.
C.DUR, C.KNOTS, C.STAGES, C.BEATS = DUR, KNOTS, STAGES, BEATS
C.SIM = C.pchip(*zip(*KNOTS))
SIM = C.SIM

ROUTE_COL = {"A": (99, 102, 241), "B": CYAN, "C": (219, 39, 119), "D": (101, 163, 13)}
PANEL = (16, 23, 36)


class Film30(C.Film):
    def __init__(self, scale=1.0):
        super().__init__(scale)
        fd = self.ep.decisions["first_decision"]["unique_routes"]
        self.routes = [(r["route_id"], self.ep.route_points(r["route"]), r) for r in fd]

    # ------------------------------------------------------------ camera
    def camera(self, v):
        J = self.J
        c_route = np.array([8541300.0, 3594330.0, W / (2 * 0.40)])
        c_hero = np.array([J[0] + 40, J[1] - 18, W / (2 * 4.2)])
        c_pass = np.array([J[0] + 52, J[1] + 2, W / (2 * 4.2)])
        c_adj = np.array([J[0] + 10, J[1] - 20, W / (2 * 2.9)])
        c_wide = np.array([J[0] + 30, J[1] - 32, W / (2 * 3.3)])
        keys = [(0, c_route), (4.3, c_route), (7.6, c_hero), (19.2, c_hero), (21.0, c_pass), (21.8, c_pass),
                (23.2, c_adj), (26.2, c_adj), (27.6, c_wide), (30, c_wide)]
        for (a, ca), (b, cb) in zip(keys, keys[1:]):
            if a <= v <= b:
                u = ease((v - a) / (b - a))
                c = ca * (1 - u) + cb * u
                c[2] = math.exp(math.log(ca[2]) * (1 - u) + math.log(cb[2]) * u)
                return c
        return keys[-1][1]

    # ------------------------------------------------------------ frame
    def frame(self, f):
        v = f / C.FPS
        t = float(SIM(v))
        cam = self.camera(v)
        img = self.backdrop(cam).convert("RGBA")
        glow, cv = C.Cv(W, H), C.Cv(W, H)
        self.candidate_routes(cv, cam, v)
        self.approaches(cv, glow, cam, v, t)
        self.forecast_bands(cv, cam, v, t)
        self.route_layer(cv, glow, cam, v, t)
        self.adjacent(cv, cam, v, t)
        self.signals(cv, glow, cam, v, t)
        self.vehicles(cv, cam, v, t)
        amb = self.ambulance(cv, glow, cam, v, t)
        self.request_link(cv, glow, cam, v, t, amb)
        self.ped_ring(cv, cam, v, t)
        self.exit_marker(cv, cam, v, t)
        from PIL import ImageFilter
        img = Image.alpha_composite(img, glow.down().filter(ImageFilter.GaussianBlur(9)))
        img = Image.alpha_composite(img, cv.down())
        ui = C.Cv(W, H)
        self.chrome(ui, v, t)
        self.card(ui, v, t, img)
        self.headline(ui, v)
        img = Image.alpha_composite(img, self.card_shadow(v))
        img = Image.alpha_composite(img, ui.down()).convert("RGB")
        img.paste(self.base.logo.convert("RGB").point(lambda x: int(x * .55)), (W - 300, H - 44))
        img.paste(self.base.credit.convert("RGB").point(lambda x: int(x * .6)), (W - 172, H - 38))
        return img

    # ------------------------------------------------------- step 1 map
    def candidate_routes(self, cv, cam, v):
        a = win(v, 0.2, 5.6, .6, .9)
        if a <= 0:
            return
        for i, (rid, pts, r) in enumerate(self.routes):
            ai = a * ease((v - .4 - i * .45) / .6)
            if ai <= 0:
                continue
            d = np.gradient(pts, axis=0)
            n = np.stack((-d[:, 1], d[:, 0]), 1)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            s = W / (2 * cam[2])
            p = self.S(pts + n * ((i - 1.5) * 7 / s), cam)
            w = 7 if r["chosen"] else 4.5
            cv.line([tuple(x) for x in p], rgba((255, 255, 255), .85 * ai), w + 4)
            cv.line([tuple(x) for x in p], rgba(ROUTE_COL[rid], ai), w)
        hp = self.S(self.ep.hosp, cam)
        cv.circle(tuple(hp), 15, fill=rgba(PANEL, .95 * a), outline=rgba(EMERALD, a), w=2.5)
        cv.text(tuple(hp), "H", F("b", 16), rgba(EMERALD, a), "mm")
        cv.text((hp[0] + 24, hp[1]), "Fortis hospital", F("s", 18), rgba(PANEL, a), "lm")

    # ------------------------------------------------- approach lanes/chips
    def app_strength(self, v):
        return max(win(v, 4.8, 12.8, .7, .6) * .9, win(v, 26.5, 30, .8, 0) * .7)

    def approaches(self, cv, glow, cam, v, t):
        a = self.app_strength(v)
        if a <= 0:
            return
        s = W / (2 * cam[2])
        for edge, col in APP.items():
            pts = [tuple(x) for x in self.S(self.ep.exy[edge], cam)]
            cv.line(pts, rgba(col, .5 * a), max(6, 3.7 * s * 1.1))
            glow.line(pts, rgba(col, .6 * a), 16)
        # step 2: measured queue estimate on each approach
        c = win(v, 5.2, 8.1, .6, .4)
        if c > 0:
            e = self.ep.ev[785]
            for ap in e["conflicting_approaches"] + [{"edge": self.ep.amb_app, "queue_proxy": e["queue_proxy"]}]:
                self.chip(cv, cam, ap["edge"], f"{APP_NAME[ap['edge']]}  queue {ap['queue_proxy']:.1f}",
                          APP.get(ap["edge"], CYAN), c)

    def chip(self, cv, cam, edge, label, col, a):
        p = self.S(self.ep.exy[edge], cam)
        q = p[0] * .25 + p[-1] * .75
        off = {"172380615#1": (-96, -46), "172382686#4": (-70, 58), "172660580#0": (96, -40), "335061218": (150, 40)}[edge]
        x, y = q[0] + off[0], q[1] + off[1]
        fnt = F("monob", 19)
        w = cv.tw(label, fnt)
        cv.line([tuple(q), (x, y)], rgba(col, .8 * a), 1.5)
        cv.rect((x - w / 2 - 14, y - 17, x + w / 2 + 14, y + 17), fill=rgba(PANEL, .92 * a), outline=rgba(col, a), w=1.5, r=17)
        cv.text((x, y), label, fnt, rgba(col if col != CYAN else (90, 190, 255), a), "mm")

    # -------------------------------------------- steps 3-4: forecast bands
    def decision_event(self, t):
        e = self.ep.event_at(t) if t >= 785 else None
        if e is not None and e["t"] > 792:
            e = self.ep.ev[792]
        return e

    def forecast_bands(self, cv, cam, v, t):
        """Forecast cost of 'priority now' drawn on each conflicting road, length proportional to C_j."""
        a = win(v, 8.4, 17.8, .7, .5)
        if a <= 0:
            return
        e = self.decision_event(max(t, 785))
        grow = ease_out((v - 8.4) / 1.2)
        s = W / (2 * cam[2])
        lines = {edges[0]: self.road_line(edges, True) for _, edges, inbound in ROADS if inbound}
        ok = e["raw_burden_proxy_veh_s"] <= 1000
        for ap in e["conflicting_approaches"]:
            line = lines[ap["edge"]]
            seg = np.cumsum(np.r_[0, np.hypot(*np.diff(line, axis=0).T)])
            L = ap["burden_proxy_veh_s"] / 501.0 * 150 * grow
            keep = line[seg <= max(L, 1)]
            if len(keep) < 2:
                j = min(1, len(line) - 1)
                keep = np.array([line[0], line[0] + (line[j] - line[0]) * min(1, L / max(seg[j], 1e-6))])
            elif seg[len(keep) - 1] < L and len(keep) < len(line):
                j = len(keep)
                f = (L - seg[j - 1]) / max(seg[j] - seg[j - 1], 1e-6)
                keep = np.vstack([keep, line[j - 1] + (line[j] - line[j - 1]) * f])
            q = self.S(keep, cam)
            col = EMERALD if ok else CORAL
            cv.line([tuple(x) for x in q], rgba(col, .45 * a), max(16, 3.7 * s * 2.2))
            self.chip(cv, cam, ap["edge"], f"{APP_NAME[ap['edge']]}  +{ap['burden_proxy_veh_s']:.0f}", APP[ap["edge"]], a * grow)

    # ------------------------------------------------ shifted time windows
    def adjacent(self, cv, cam, v, t):
        if win(v, 22.1, 26.6) <= 0:
            return
        saved = C.win
        C.win = lambda vv, a_, b_, fi=.35, fo=.35: saved(vv - 9.4, a_, b_, fi, fo) if (a_, b_) == (12.7, 16.8) else saved(vv, a_, b_, fi, fo)
        try:
            super().adjacent(cv, cam, v, t)
        finally:
            C.win = saved

    def request_link(self, cv, glow, cam, v, t, amb):
        if amb is None or not (785 <= t < 801.5):
            return
        e = self.ep.event_at(t)
        if e is None:
            return
        col = CORAL if e["reason"] == "burden_budget_exceeded" else EMERALD
        a = win(v, 8.1, 21.2, .6, .6)
        p = self.S(self.ep.exy[self.ep.amb_app], cam)[-1]
        d = p - amb
        L = np.linalg.norm(d)
        if L < 30:
            return
        u = d / L
        s0 = 30
        while s0 < L - 14:
            cv.line([tuple(amb + u * s0), tuple(amb + u * min(s0 + 7, L - 14))], rgba(col, .9 * a), 3)
            s0 += 14

    def ped_ring(self, cv, cam, v, t):
        if not (771.5 <= t < 796.5):
            return
        a = max(win(v, 5.4, 8.1, .6, .4) * .85, win(v, 17.4, 19.3, .5, .2))
        if a <= 0:
            return
        c = self.S(self.ep.J1xy, cam)
        rem = 796.5 - t
        cv.circle(tuple(c), 46, fill=rgba((14, 20, 32), .8 * a))
        cv.arc(tuple(c), 46, -90, 270, rgba(DIM, .5 * a), 4)
        cv.arc(tuple(c), 46, -90, -90 + 360 * rem / 25.0, rgba(IVORY, a), 5)
        cv.text((c[0], c[1] - 9), f"{rem:4.1f}s", F("monob", 19), rgba(IVORY, a), "mm")
        cv.text((c[0], c[1] + 13), "PED", F("mono", 12), rgba(MUTED, a), "mm", track=2)

    def exit_marker(self, cv, cam, v, t):
        if v < 27.0:
            return
        saved = C.win
        C.win = lambda vv, a_, b_, fi=.35, fo=.35: saved(vv - 10.0, a_, b_, fi, fo) if (a_, b_) == (17.0, 20) else saved(vv, a_, b_, fi, fo)
        try:
            super().exit_marker(cv, cam, v, t)
        finally:
            C.win = saved

    # ---------------------------------------------------------- chrome
    def chrome(self, ui, v, t):
        b = C.beat_at(v)
        hw = max(ui.tw(b[3], F("h", 50)), ui.tw(b[4], F("m", 24)), 1130) + 72
        ui.rect((36, 26, 36 + hw, 222), fill=rgba(PANEL, .92), outline=rgba((70, 84, 106), .7), w=1.2, r=20)
        ui.rect((36, H - 92, 700, H - 14), fill=rgba(PANEL, .88), r=16)
        x = 64
        for i, s in enumerate(STAGES):
            active = b[2] == i
            done = (b[2] is not None and i < b[2]) or v >= 26.5
            col = IVORY if active else (MUTED if done else DIM)
            lab = f"{i + 1}  {s}"
            fnt = F("monob" if active else "mono", 17)
            ui.text((x, 58), lab, fnt, rgba(col, 1 if active else .8), "lm", track=2)
            w = ui.tw(lab, fnt, 2)
            if active:
                ui.rect((x, 74, x + w, 77), fill=rgba(CYAN, 1), r=1)
            x += w + 38
        el = t - 760
        r = C.rate(v)
        speed = "hold" if r < 0.05 else (f"{r:.1f}x" if r >= 1 else f"{r:.2f}x")
        ui.text((64, H - 52), f"SIM {t:6.1f} s", F("monob", 20), rgba(IVORY, .92), "lm", track=1)
        ui.text((250, H - 52), f"T+{int(el) // 60}:{int(el) % 60:02d}  ·  {speed}", F("mono", 18), rgba(MUTED, .9), "lm", track=1)
        ui.text((64, H - 26), "Recorded SUMO replay on mapped Mohali roads  ·  synthetic demand", F("m", 14), rgba(DIM, 1), "lm")

    # ------------------------------------------------------------ cards
    def card_alpha(self, v):
        return win(v, 0.6, DUR, .6, 0)

    def card(self, ui, v, t, img):
        a = self.card_alpha(v)
        if a <= 0:
            return
        x0, y0, x1, y1 = self.CARD
        ui.rect((x0, y0, x1, y1), fill=rgba(PANEL, .97 * a), outline=rgba((70, 84, 106), .8 * a), w=1.2, r=24)
        steps = [(0, 4.5, self.routes_card), (4.5, 8.0, self.inputs_card), (8.0, 12.8, self.options_card),
                 (12.8, 17.6, self.reeval_card), (17.6, 22.0, self.timeline30), (22.0, 26.5, self.adjacent_card),
                 (26.5, DUR, self.measured30)]
        for s0, s1, fn in steps:
            if s0 <= v < s1 or (s1 == DUR and v >= s0):
                fn(ui, v, t, a * win(v, s0, s1, .6 if s0 > 0 else 0, .5 if s1 < DUR else 0))

    def head(self, ui, a, over, sub, col=CYAN):
        x0, y0, x1, y1 = self.CARD
        ui.text((x0 + 36, y0 + 46), over, F("monob", 17), rgba(col, a), "lm", track=3)
        ui.text((x0 + 36, y0 + 76), sub, F("m", 20), rgba(MUTED, a), "lm")

    def routes_card(self, ui, v, t, a):
        x0, y0, x1, y1 = self.CARD
        L, R = x0 + 36, x1 - 36
        self.head(ui, a, "ROUTE OPTIONS", "forecast travel time to the hospital, t = 761 s")
        ui.text((L + 60, y0 + 128), "with priority", F("mono", 15), rgba(DIM, a), "lm")
        ui.text((L + 250, y0 + 128), "without", F("mono", 15), rgba(DIM, a), "lm")
        y = y0 + 176
        for i, (rid, _, r) in enumerate(self.routes):
            ai = a * ease((v - .6 - i * .45) / .6)
            col = ROUTE_COL[rid]
            if r["chosen"]:
                ui.rect((L - 14, y - 30, R + 14, y + 30), fill=rgba(mix(PANEL, CYAN, .14), ai), outline=rgba(CYAN, ai), w=1.5, r=12)
            ui.circle((L + 14, y), 13, fill=rgba(col, ai))
            ui.text((L + 14, y), rid, F("b", 16), rgba((255, 255, 255), ai), "mm")
            ui.text((L + 60, y), f"{r['forecast_with_priority_s']:.0f} s", F("monob", 26), rgba(IVORY, ai), "lm")
            ui.text((L + 250, y), f"{r['forecast_without_priority_s']:.0f} s", F("mono", 22), rgba(MUTED, ai), "lm")
            if r["chosen"]:
                ui.text((R, y), "SELECTED", F("b", 15), rgba((90, 190, 255), ai), "rm", track=2)
            y += 74
        b = a * ease((v - 2.8) / .6)
        ui.text((L, y1 - 104), "The fastest route-and-priority pair is chosen;", F("m", 19), rgba(MUTED, b), "lm")
        ui.text((L, y1 - 76), "its cost to other traffic is checked junction by junction.", F("m", 19), rgba(MUTED, b), "lm")

    def inputs_card(self, ui, v, t, a):
        x0, y0, x1, y1 = self.CARD
        L, R = x0 + 36, x1 - 36
        e = self.ep.ev[785]
        self.head(ui, a, "JUNCTION CONDITIONS", "PCL Chowk, as received by the controller")
        cols = [L, L + 120, L + 260, R]
        for x, h, anc in zip(cols, ["road", "queue est.", "arrivals/h", "loading"], ["lm", "lm", "lm", "rm"]):
            ui.text((x, y0 + 130), h, F("mono", 15), rgba(DIM, a), anc)
        y = y0 + 176
        for ap in e["conflicting_approaches"]:
            col = APP[ap["edge"]]
            ui.circle((L + 7, y), 7, fill=rgba(col, a))
            ui.text((L + 24, y), APP_NAME[ap["edge"]], F("b", 20), rgba(col, a), "lm")
            ui.text((cols[1], y), f"{ap['queue_proxy']:.1f} veh", F("mono", 20), rgba(IVORY, a), "lm")
            ui.text((cols[2], y), f"{ap['arrival_rate'] * 3600:.0f}", F("mono", 20), rgba(IVORY, a), "lm")
            ui.text((cols[3], y), f"{ap['nominal_design_X']:.2f}", F("mono", 20), rgba(IVORY, a), "rm")
            y += 50
        ui.circle((L + 7, y), 7, fill=rgba(CYAN, a))
        ui.text((L + 24, y), "SE", F("b", 20), rgba((90, 190, 255), a), "lm")
        ui.text((cols[1], y), f"{e['queue_proxy']:.1f} veh", F("mono", 20), rgba(IVORY, a), "lm")
        ui.text((cols[2], y), "ambulance approach", F("m", 17), rgba(MUTED, a), "lm")
        y += 64
        ui.line([(L, y - 22), (R, y - 22)], rgba((60, 72, 92), a), 1)
        ui.text((L, y + 4), "Signal", F("mono", 15), rgba(DIM, a), "lm")
        ui.text((cols[1], y + 4), f"pedestrian interval, {e['switch_delay_s']:.0f} s left", F("m", 20), rgba(IVORY, a), "lm")
        ui.text((L, y + 44), "Ambulance", F("mono", 15), rgba(DIM, a), "lm")
        ui.text((cols[1], y + 44), f"{e['free_flow_eta_to_stopline_s']:.1f} s from the stop line", F("m", 20), rgba(IVORY, a), "lm")
        ui.text((L, y1 - 76), "Counts every 5 s, received 2 s late; arrivals and loading", F("m", 16), rgba(DIM, a), "lm")
        ui.text((L, y1 - 52), "from historical demand and signal capacity.", F("m", 16), rgba(DIM, a), "lm")

    def options_card(self, ui, v, t, a):
        x0, y0, x1, y1 = self.CARD
        L, R = x0 + 36, x1 - 36
        e = self.ep.ev[785]
        self.head(ui, a, "OPTIONS AT 785 s", "forecast for each choice, before acting")
        bud = 1000
        sc = (R - L - 20) / 1400

        def option(y, title, saving, cost, verdict, vcol, delay):
            b = a * ease((v - delay) / .6)
            ui.rect((L - 12, y, R + 12, y + 230), fill=rgba(mix(PANEL, (255, 255, 255), .04), b), outline=rgba((70, 84, 106), b), w=1.2, r=14)
            ui.text((L + 6, y + 30), title, F("b", 22), rgba(IVORY, b), "lm")
            ui.text((L + 6, y + 70), "ambulance", F("mono", 15), rgba(DIM, b), "lm")
            ui.text((L + 130, y + 70), saving, F("m", 19), rgba(IVORY, b), "lm")
            ui.text((L + 6, y + 108), "other traffic", F("mono", 15), rgba(DIM, b), "lm")
            ui.text((L + 130, y + 108), f"+{cost:,.0f} vehicle-seconds", F("m", 19), rgba(IVORY, b), "lm")
            by = y + 146
            ui.rect((L + 6, by, R - 6, by + 16), fill=rgba((44, 55, 74), b), r=4)
            if cost > 0:
                acc = L + 6
                for ap in e["conflicting_approaches"]:
                    w = ap["burden_proxy_veh_s"] * sc * ease_out((v - delay - .2) / .9)
                    ui.rect((acc, by, acc + w, by + 16), fill=rgba(APP[ap["edge"]], b))
                    acc += w
            bx = L + 6 + bud * sc
            ui.line([(bx, by - 8), (bx, by + 24)], rgba(IVORY, b), 3)
            ui.text((bx, by + 34), "budget 1,000", F("mono", 13), rgba(MUTED, b), "mm")
            c = a * ease((v - delay - 1.2) / .5)
            ui.text((R - 6, y + 30), verdict, F("b", 16), rgba(vcol, c), "rm", track=1.5)

        option(y0 + 112, "Priority green now", f"saves about {e['predicted_benefit_s']:.0f} s",
               e["raw_burden_proxy_veh_s"], "OVER BUDGET", CORAL, 8.3)
        option(y0 + 358, "Keep the normal cycle", f"waits about {e['nominal_signal_wait_s']:.0f} s for its green",
               0, "WITHIN BUDGET", EMERALD, 9.4)
        c = a * ease((v - 11.0) / .5)
        ui.rect((L, y1 - 70, R, y1 - 26), fill=rgba(mix(PANEL, CORAL, .18), c), outline=rgba(CORAL, c), w=2, r=22)
        ui.text(((L + R) / 2, y1 - 48), "PRIORITY WITHHELD FOR NOW", F("b", 19), rgba(CORAL, c), "mm", track=2)

    def reeval_card(self, ui, v, t, a):
        x0, y0, x1, y1 = self.CARD
        L, R = x0 + 36, x1 - 36
        ev = self.ep.ev
        self.head(ui, a, "RE-EVALUATED EVERY SECOND", "forecast cost of priority now, vehicle-seconds")
        base, top = y0 + 360, y0 + 150
        sc = (base - top) / 1300
        ui.line([(L, base), (R, base)], rgba((70, 84, 106), a), 1.5)
        by = base - 1000 * sc
        for xx in np.arange(L, R, 16):
            ui.line([(xx, by), (min(xx + 8, R), by)], rgba(IVORY, .8 * a), 2)
        ui.text((L, top - 24), "dashed line: budget 1,000", F("mono", 14), rgba(MUTED, a), "lm")
        bw = (R - L) / 8
        for i, s in enumerate(range(785, 793)):
            if t < s:
                continue
            val = ev[s]["raw_burden_proxy_veh_s"]
            k = ease((t - s) / .35) if s > 786 else 1.0
            col = CORAL if val > 1000 else EMERALD
            xa = L + i * bw + 8
            ui.rect((xa, base - val * sc * k, xa + bw - 16, base), fill=rgba(col, a), r=3)
            ly = base - val * sc * k - 14 if val > 1000 else base - val * sc * k + 16
            ui.text((xa + (bw - 16) / 2, ly), f"{val:,.0f}", F("monob" if val <= 1000 else "mono", 13), rgba(IVORY, a * k), "mm")
            ui.text((xa + (bw - 16) / 2, base + 18), str(s), F("mono", 13), rgba(DIM, a), "mm")
        if t >= 792:
            c = a * ease((v - 15.95) / .5)
            yy = y0 + 420
            ui.text((L, yy), "New count sample at 792 s; physical queues unchanged", F("s", 18), rgba(IVORY, c), "lm")
            for j, (lab, o, n) in enumerate([("SE queue", ev[791]["queue_proxy"], ev[792]["queue_proxy"]),
                                             ("SW queue", 9.37, 8.37), ("NW queue", 1.94, 0.94)]):
                ui.text((L, yy + 34 + j * 28), f"{lab} estimate  {o:.1f}  to  {n:.1f} veh", F("mono", 16), rgba(MUTED, c), "lm")
            ui.rect((L - 4, y1 - 64, R + 4, y1 - 22), fill=rgba(mix(PANEL, EMERALD, .2), c), outline=rgba(EMERALD, c), w=2, r=21)
            ui.text(((L + R) / 2, y1 - 43), "WITHIN BUDGET: PRIORITY ADMITTED", F("b", 18), rgba(EMERALD, c), "mm", track=2)
        else:
            ui.rect((L - 4, y1 - 64, R + 4, y1 - 22), fill=rgba(mix(PANEL, CORAL, .18), a), outline=rgba(CORAL, a), w=2, r=21)
            ui.text(((L + R) / 2, y1 - 43), "OVER BUDGET: PRIORITY WITHHELD", F("b", 18), rgba(CORAL, a), "mm", track=2)

    def timeline30(self, ui, v, t, a):
        x0, y0, x1, y1 = self.CARD
        cx = (x0 + x1) / 2
        L, R = x0 + 36, x1 - 36
        self.head(ui, a, "SIGNAL SERVICE", "green phases at PCL Chowk, both runs")
        y = y0 + 150
        sc, rows = self.mini_rows(ui, a, L, R, y, {"335061218", *APP})
        X = lambda s: L + (s - 776) * sc
        if 776 <= t <= 905:
            ui.line([(X(t), y - 8), (X(t), y + 136)], rgba(IVORY, .7 * a), 1.5)
        w0 = a * (1 - ease((v - 19.2) / .5))
        if w0 > 0 and t < 796.6:
            ui.text((L, y1 - 236), "Waiting for the pedestrian interval", F("s", 22), rgba(IVORY, w0), "ls")
            ui.text((L, y1 - 150), f"{max(0.0, 796.5 - t):4.1f} s", F("monob", 64), rgba(IVORY, w0), "ls")
            ui.text((L, y1 - 112), "until the earliest legal switch to SE green", F("m", 20), rgba(MUTED, w0), "ls")
        b = ease((v - 19.6) / .6)
        ui.line([(X(868.5), y + 34), (X(868.5), y + 63), (X(796.5), y + 63), (X(796.5), y + 88)], rgba(CYAN, a * b), 2.5)
        ui.poly([(X(796.5), y + 95), (X(796.5) - 6, y + 84), (X(796.5) + 6, y + 84)], fill=rgba(CYAN, a * b))
        ui.text((L, y1 - 236), "SE  72 s earlier", F("h", 44), rgba((90, 190, 255), a * b), "ls")
        ui.text((L, y1 - 204), "green for the ambulance", F("m", 20), rgba(MUTED, a * b), "ls")
        c = ease((v - 20.8) / .6)
        for q0, q3, ed in [(795.5, 847.5, "172380615#1"), (812.5, 864.5, "172382686#4"), (839.5, 891.5, "172660580#0")]:
            ui.line([(X(q0), y + 34), (X(q3), y + 88)], rgba(APP[ed], a * c), 2.5)
            ui.poly([(X(q3), y + 95), (X(q3) - 6, y + 84), (X(q3) + 6, y + 84)], fill=rgba(APP[ed], a * c))
        ui.text((L, y1 - 140), "NW, SW, NE  52 s later", F("h", 44), rgba(CORAL, a * c), "ls")
        ui.text((L, y1 - 108), "their next green phases", F("m", 20), rgba(MUTED, a * c), "ls")
        ui.text((L, y1 - 60), "776 s", F("mono", 14), rgba(DIM, a), "lm")
        ui.text((R, y1 - 60), "905 s", F("mono", 14), rgba(DIM, a), "rm")
        ui.text((cx, y1 - 60), "recorded signal states", F("mono", 14), rgba(DIM, a), "mm")

    def measured30(self, ui, v, t, a):
        ep = self.ep
        x0, y0, x1, y1 = self.CARD
        cx, cy = (x0 + x1) / 2, y0 + 262
        self.head(ui, a, "OBSERVED", "extra stopped time, NW + SW + NE", CORAL)
        self.gauge_base(ui, a, cx, cy, [(913.3, "FORECAST 913", CYAN, False), (1000, "BUDGET", IVORY, True)])
        k = ep.k(t)
        val = max(0.0, sum(ep.app_extra[e][k] for e in ep.conf)) * ease_out((v - 26.6) / .8)
        ui.arc((cx, cy), self.G_R, self.g_ang(0), self.g_ang(val), rgba(CORAL, a), 22)
        ui.text((cx, cy - 6), f"{math.floor(val + .5):,d}", F("monob", 76), rgba(CORAL, a), "ms")
        ui.text((cx, cy + 32), "vehicle-seconds", F("mono", 22), rgba(MUTED, a), "mm")
        ui.text((cx, y0 + 428), f"785 s  to  {t:.0f} s", F("mono", 18), rgba(MUTED, a), "mm")
        ui.text((cx, y0 + 466), "vs the same vehicles without priority", F("mono", 15), rgba(DIM, a), "mm")
        ui.rect((x0 + 40, y1 - 92, x1 - 40, y1 - 40), outline=rgba(MUTED, .6 * a), w=1.5, r=26)
        ui.text((cx, y1 - 66), "MEASURED IN THE REPLAY", F("b", 19), rgba(MUTED, a), "mm", track=2.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stills", type=float, nargs="*")
    ap.add_argument("--contact", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    C.Film = Film30
    C.OUT.mkdir(parents=True, exist_ok=True)
    if a.stills:
        film = Film30()
        d = C.OUT / "stills30"
        d.mkdir(exist_ok=True)
        for v in a.stills:
            film.frame(int(min(v, DUR - 1 / 30) * 30)).save(d / f"v{v:05.2f}.jpg", quality=92)
        return
    if a.contact:
        film = Film30()
        C.contact(film, [1, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17, 19, 20.5, 23, 25, 27.5, 29.9], C.OUT / "contact_sheet_30s.jpg", cols=5)
        print(C.OUT / "contact_sheet_30s.jpg")
        return
    C.render(a.jobs, 1.0, "AmbuGo_30s_Explainer.mp4")


if __name__ == "__main__":
    main()
