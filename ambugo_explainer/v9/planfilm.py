#!/usr/bin/env python3
"""AmbuGo v9 explainer: the plan (route + signal changes chosen together at dispatch) and the delay forecast.

One caption and at most one highlighted thing per beat.

    python planfilm.py                 # output/AmbuGo_v9_Plan.mp4
    python planfilm.py --stills 5 20   # stills at video seconds
    python planfilm.py --contact       # one frame per beat
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
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import world  # noqa: E402
from world import View, rgba, SS, AMB_BLUE, HOSP  # noqa: E402
from mapfilm import G, ease, interp, W, H, FPS, INK  # noqa: E402
from film import equation  # noqa: E402
import simplefilm as SF  # noqa: E402
from simplefilm import Simple, COST, GAIN, AMB, J1_AMB, J1_CROSS, J2_AMB, J2_EAST, J2_SOUTH, J2_WEST  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
MUTED = (95, 99, 104)
RCOL = {"A": (52, 101, 164), "B": (31, 140, 130), "C": (128, 64, 160), "D": (170, 120, 40)}


def N(dur, cap, routes=(), focus=None, tag=None, verdict=None, markers=True, plan=0, amb=None):
    """A beat on the route-planning diagram."""
    return dict(view="net", dur=dur, cap=cap, routes=list(routes), focus=focus, tag=tag, verdict=verdict, markers=markers,
                plan=plan, amb=amb, sim=[(0, 4065), (1, 4065)], card=None, hl=[])


def M(dur, j, run, sim, cap, hl=(), zoom=1.15, shift=(0, 60), verdict=None, card=None, eq=False):
    b = SF.B(dur, j, run, sim, cap, hl, zoom, shift, verdict, card)
    b["view"], b["eq"] = "map", eq
    return b


class Plan(Simple):
    def build(self):
        P = self.runs["P3"].plans
        self.p0, self.p1 = P[0], P[1]
        self.cand = {}
        names = {}
        for c in self.p0["candidates"]:
            r = tuple(c["route"])
            if r not in names:
                names[r] = None
        order = sorted(names, key=lambda r: (0 if [c for c in self.p0["candidates"] if tuple(c["route"]) == r][0]["incumbent"] else 1,
                                             list(names).index(r)))
        for i, r in enumerate(order):
            cs = [c for c in self.p0["candidates"] if tuple(c["route"]) == r]
            self.cand["ABCD"[i]] = dict(route=list(r), none=[c for c in cs if c["action"][1] == 0][0], req=[c for c in cs if c["action"][1] > 0][0])
        rq1 = {tuple(c["route"]): c for c in self.p1["candidates"] if c["action"][1] > 0}
        self.cand1 = {n: rq1[tuple(self.cand[n]["route"][1:])] for n in ("A", "C")}
        g1, g2 = self.g[(4091, "J11")], self.g[(4132, "J21")]
        arms = {a["edge"]: a for a in g2["arms"]}
        E = arms[J2_EAST]

        def rtag(n):
            c = self.cand[n]["req"]
            return [f"Route {n}", f"{c['eta']:.0f} s  ·  {c['admitted_junctions']} planned signal changes",
                    f"others wait +{c['burden']:,.0f} veh·s"]

        cap_routes = ["For each route it forecasts which signals to change, how long", "the trip takes, and how much extra waiting it causes."]
        S = [
            # ------------------------------------------------ the plan, made at dispatch
            N(5, ["When the ambulance is dispatched, AmbuGo plans", "the trip before it sets off."]),
            N(5, ["It compares several routes to the hospital."], routes="ABCD"),
            N(6, cap_routes, routes="ABCD", focus="A", tag=rtag("A")),
            N(4.5, cap_routes, routes="ABCD", focus="B", tag=rtag("B")),
            N(4.5, cap_routes, routes="ABCD", focus="C", tag=rtag("C")),
            N(4.5, cap_routes, routes="ABCD", focus="D", tag=rtag("D")),
            N(7, ["A signal change is planned only where its forecast delay", "for other traffic stays within AmbuGo's limit."], routes="ABCD"),
            N(7, [f"It takes the fastest plan: route C, {self.cand['C']['req']['eta']:.0f} s.",
                  "Forecast delay decides only between equally fast plans."], focus="C",
              verdict=(f"Route C  ·  {self.cand['C']['req']['eta']:.0f} s", RCOL["C"])),
            N(8, ["The plan is redone on every road. Half a minute later, route A", "had become faster and caused less waiting, so it switched."],
              focus="A", plan=1, amb=4091,
              tag=[f"Route A, now  {self.cand1['A']['eta']:.0f} s  ·  +{self.cand1['A']['burden']:,.0f} veh·s",
                   f"route C, now  {self.cand1['C']['eta']:.0f} s  ·  +{self.cand1['C']['burden']:,.0f} veh·s"]),
            # ------------------------------------------------ junction 1: a planned change that is not made
            M(5, "J11", "P3", [(0, 4079), (1, 4091)], ["Along the route, each signal is asked about", "30 seconds before the ambulance gets there."],
              [("amb", "Ambulance")], zoom=1.05),
            M(6, "J11", "P3", [(0, 4091), (1, 4091)], [f"Here a change would save only {g1['benefit']:.1f} seconds,", "under the 5-second minimum."],
              [("road", J1_AMB, AMB, f"saves {g1['benefit']:.1f} s")], verdict=("No change", COST)),
            M(8, "J11", "P2", [(0, 4091), (.8, 4170), (1, 4170)], ["In a recorded run where it was changed anyway, the cross road",
                                                                   "waited 64 s longer so the ambulance passed 11 s sooner."],
              [("road", J1_CROSS, COST, "+64 s wait")], zoom=1.05),
            # ------------------------------------------------ junction 2: what the change is, what it costs, then watch it
            M(6, "J21", "P3", [(0, 4118), (1, 4132)], ["At the next junction the ambulance is still about",
                                                       "50 seconds away, on this road. Its light is red."],
              [("road", J2_AMB, AMB, "Ambulance's road"), ("amb", "Ambulance", "Ambulance, about 50 s away")], zoom=.95),
            M(5, "J21", "P3", [(0, 4132), (1, 4132)], ["Right now the east road has the green light."],
              [("road", J2_EAST, GAIN, "East road: green now")]),
            M(6, "J21", "P3", [(0, 4132), (1, 4132)], ["Without any change, the ambulance would wait",
                                                       f"about {g2['nominal_wait']:.0f} seconds for its turn."],
              [("road", J2_AMB, AMB, f"would wait {g2['nominal_wait']:.0f} s")]),
            M(7, "J21", "P3", [(0, 4132), (1, 4132)], ["The change AmbuGo considers: end the east road's green",
                                                       "early and give the green to the ambulance."],
              verdict=("East green ends early  →  ambulance gets green", (70, 74, 80))),
            M(6, "J21", "P3", [(0, 4132), (1, 4132)], [f"That would take about {g2['D']:.0f} seconds of green from the",
                                                       "other three roads, so their cars would wait longer."],
              [("road", J2_EAST, COST, None), ("road", J2_SOUTH, COST, None), ("road", J2_WEST, COST, None)]),
            M(9, "J21", "P3", [(0, 4132), (1, 4132)], [], card=True, eq=True),
            M(7, "J21", "P3", [(0, 4132), (1, 4132)], [f"East road: about {E['q_hat']:.0f} cars waiting, and more arriving.",
                                                       f"Extra waiting, added up over all its cars: {E['term_queue'] + E['term_arrival']:,.0f} seconds."],
              [("road", J2_EAST, COST, f"+{E['term_queue'] + E['term_arrival']:,.0f} s waiting")]),
            M(6, "J21", "P3", [(0, 4132), (1, 4132)],
              [f"South road: {arms[J2_SOUTH]['term_queue'] + arms[J2_SOUTH]['term_arrival']:,.0f} seconds.  "
               f"West road: {arms[J2_WEST]['term_queue'] + arms[J2_WEST]['term_arrival']:,.0f} seconds."],
              [("road", J2_SOUTH, COST, f"+{arms[J2_SOUTH]['term_queue'] + arms[J2_SOUTH]['term_arrival']:,.0f} s"),
               ("road", J2_WEST, COST, f"+{arms[J2_WEST]['term_queue'] + arms[J2_WEST]['term_arrival']:,.0f} s")]),
            M(7, "J21", "P3", [(0, 4132), (1, 4132)], [f"In total {g2['burden']:,.0f} seconds of extra waiting, under the limit",
                                                       f"of 10,000, to save the ambulance {g2['benefit']:.0f} seconds."],
              verdict=("Make the change", GAIN)),
            M(8, "J21", "P3", [(0, 4132), (1, 4141.5)], ["Now watch the east road's light:", "green, then amber, then red."],
              [("road", J2_EAST, lambda t: self.light(J2_EAST, t)[1], lambda t: f"east light: {self.light(J2_EAST, t)[0]}")]),
            M(6, "J21", "P3", [(0, 4141.5), (1, 4146)], ["After two seconds of red in every direction,", "the ambulance's light turns green."],
              [("road", J2_AMB, lambda t: self.light(J2_AMB, t)[1], lambda t: f"ambulance's light: {self.light(J2_AMB, t)[0]}")]),
            M(7, "J21", "P3", [(0, 4146), (1, 4180)], ["The ambulance drives through, 86 seconds sooner", "than it would have without the change."],
              [("amb", "Ambulance")], zoom=.85),
            # ------------------------------------------------ close
            M(6, "J21", "P3", [(0, 4180), (1, 4180)], [], card=["AmbuGo plans the route and the signal changes together,",
                                                                "forecasting the delay every change would cause."]),
            M(6, "J21", "P3", [(0, 4180), (1, 4180)], [], card=["Across 1,728 simulations: trips 44.6% faster than with",
                                                                "no priority, with less extra congestion than changing every light."]),
        ]
        for s in S:
            s.setdefault("j", "J21")
            s.setdefault("run", "P3")
            s.setdefault("zoom", 1.0)
            s.setdefault("shift", (0, 60))
            s.setdefault("eq", False)
            s["ov"], s["key"] = None, "beat"
        self.scenes = S
        self._netbg = None

    def light(self, e, t, j="J21", run="P3"):
        mode, served, el, rq = self.runs[run].signal(j, t)
        if served == e and mode == "green":
            return "GREEN", (24, 128, 56)
        if served == e and mode == "amber":
            return "AMBER", (196, 132, 0)
        return "RED", (197, 34, 31)

    # ------------------------------------------------------------------ the planning diagram
    def net_xy(self, xy):
        xy = np.asarray(xy, float)
        s = 0.66
        return np.stack((960 - 1712 * s / 2 + xy[..., 0] * s, 262 - 267 * s + xy[..., 1] * s), -1)

    def netbg(self):
        if self._netbg is None:
            im = Image.open(Path(__file__).resolve().parent / "data" / "maps" / "wide_mohali_z15.png").convert("RGB")
            im = im.crop((200, 200, 3400, 2000)).resize((W * SS, H * SS), Image.Resampling.BICUBIC)
            im = Image.blend(im, Image.new("RGB", im.size, (255, 255, 255)), .55).filter(ImageFilter.GaussianBlur(2))
            self._netbg = im.convert("RGBA")
        return self._netbg.copy()

    def route_nodes(self, route):
        E = self.net.edges
        return [E[route[0]]["from_node"]] + [E[e]["to_node"] for e in route]

    def draw_net(self, sc, u):
        a_in = ease(min(1.0, u * sc["dur"] / .45))
        v = View(W, H, 0, 0, 1)
        v.im = self.netbg()
        v.d = ImageDraw.Draw(v.im)
        J = self.net.junctions
        nodes = {k for k in J if k.startswith("J")} | {"BW1", "BE1"}
        for e, r in self.net.edges.items():
            a, b = r["from_node"], r["to_node"]
            if a in nodes and b in nodes and a < b:
                v.line([tuple(self.net_xy(J[a]["xy"])), tuple(self.net_xy(J[b]["xy"]))], (150, 156, 166, 255), 7)
        for k in nodes:
            if k.startswith("J"):
                v.circle(tuple(self.net_xy(J[k]["xy"])), 7, fill=(120, 126, 136, 255))
        offs = {"A": -9, "B": -3, "C": 3, "D": 9}
        for n in sc["routes"]:
            if n == sc["focus"]:
                continue
            P = self.net_xy([J[k]["xy"] for k in self.route_nodes(self.cand[n]["route"])]) + offs[n]
            v.line([tuple(p) for p in P], rgba(RCOL[n], .55 * (1 if sc["focus"] is None else .45)), 6)
        if sc["focus"]:
            n = sc["focus"]
            route = self.cand[n]["route"] if sc["plan"] == 0 else self.cand[n]["route"][1:]
            P = self.net_xy([J[k]["xy"] for k in self.route_nodes(route)])
            v.line([tuple(p) for p in P], (255, 255, 255, 255), 22)
            v.line([tuple(p) for p in P], rgba(RCOL[n], a_in), 14)
        img = v.down()
        ui = View(W, H, 0, 0, 1)
        ui.im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        ui.d = ImageDraw.Draw(ui.im)
        if sc["markers"]:
            for k, lab, col, dx in [("BW1", "Ambulance base", AMB_BLUE, 0), ("BE1", "Hospital", HOSP, 0)]:
                p = self.net_xy(J[k]["xy"])
                ui.rect((p[0] - 20, p[1] - 20, p[0] + 20, p[1] + 20), fill=(255, 255, 255, 255), outline=col + (255,), w=3, r=7)
                if k == "BE1":
                    ui.line([(p[0] - 10, p[1]), (p[0] + 10, p[1])], col, 5)
                    ui.line([(p[0], p[1] - 10), (p[0], p[1] + 10)], col, 5)
                else:
                    ui.circle(tuple(p), 8, fill=col + (255,))
                ui.text((p[0], p[1] + 44), lab, G(600, 28), col + (255,), "mm")
        if sc["amb"]:
            am = self.runs["P3"].amb_xy(sc["amb"])
            p = self.net_xy(am[0])
            ui.circle(tuple(p), 16, fill=(255, 255, 255, 255), outline=AMB_BLUE + (255,), w=4)
            ui.circle(tuple(p), 7, fill=AMB_BLUE + (255,))
        if sc["tag"] and sc["focus"]:
            n = sc["focus"]
            f1, f2 = G(700, 34), G(500, 28)
            lines = sc["tag"]
            w = max(ui.tw(lines[0], f1), *[ui.tw(s, f2) for s in lines[1:]]) + 50
            h = 58 + 40 * (len(lines) - 1)
            x0, y0 = 1920 - 60 - w, 1080 - 70 - h
            ui.rect((x0, y0, x0 + w, y0 + h), fill=(255, 255, 255, int(250 * a_in)), outline=rgba(RCOL[n], a_in), w=3, r=14)
            ui.text((x0 + 25, y0 + 32), lines[0], f1, rgba(RCOL[n], a_in), "lm")
            for i, s in enumerate(lines[1:]):
                ui.text((x0 + 25, y0 + 74 + 40 * i), s, f2, INK + (int(255 * a_in),), "lm")
        return img, ui, a_in

    # ------------------------------------------------------------------ the forecast-model card
    def eq_card(self, ui, a):
        A = int(255 * a)
        g2 = self.g[(4132, "J21")]
        box = (140, 170, 1780, 820)
        ui.rect((box[0] + 4, box[1] + 8, box[2] + 4, box[3] + 8), fill=(0, 0, 0, int(30 * a)), r=22)
        ui.rect(box, fill=(255, 255, 255, int(252 * a)), r=22)
        ui.text((960, box[1] + 70), "How AmbuGo forecasts the extra waiting on each other road", G(700, 40), INK + (A,), "mm")
        img = equation(r"$\hat{C}=\sum_{\mathrm{other\ roads}}\left[\ \hat q\,D\ +\ \frac{\lambda\,D^{2}}{2\,\max(1-X,\ 0.05)}\ \right]$", 44)
        if a < 1:
            img = img.copy()
            img.putalpha(img.getchannel("A").point(lambda x: int(x * a)))
        ui.im.alpha_composite(img, (int(960 * SS - img.width / 2), int((box[1] + 130) * SS)))
        y = box[1] + 130 + img.height / SS + 70
        items = [(r"$D$", f"green time the other roads lose  (here ≈ {g2['D']:.0f} s)"),
                 (r"$\hat q\,D$", "cars already queued × the green they lose"),
                 (r"$\lambda D^2/2\max(1-X,0.05)$", "cars expected to arrive meanwhile; grows as the road nears capacity X")]
        for sym, txt in items:
            im2 = equation(sym, 30)
            if a < 1:
                im2 = im2.copy()
                im2.putalpha(im2.getchannel("A").point(lambda x: int(x * a)))
            ui.im.alpha_composite(im2, (int((box[0] + 80) * SS), int(y * SS - im2.height / 2)))
            ui.text((box[0] + 470, y), txt, G(500, 34), INK + (A,), "lm")
            y += 96

    # ------------------------------------------------------------------ frame
    def frame(self, f):
        v_ = f / FPS
        sc, u = self.locate(v_)
        if sc["view"] == "net":
            img, ui, a_in = self.draw_net(sc, u)
            prev = self.scenes[max(0, self.scenes.index(sc) - 1)]
            cap_a = 1.0 if (prev is not sc and prev.get("cap") == sc["cap"]) else a_in
            self.caption(ui, sc["cap"], cap_a)
            if sc["verdict"]:
                self.verdict(ui, sc, u)
            self.attrib(ui)
            return Image.alpha_composite(img.convert("RGBA"), ui.down()).convert("RGB")
        if sc.get("eq"):
            sc2 = dict(sc, card=None)
            base = Simple.frame(self, f) if False else None
            img = self.map_only(sc, f)
            ui = View(W, H, 0, 0, 1)
            ui.im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
            ui.d = ImageDraw.Draw(ui.im)
            self.eq_card(ui, ease(min(1.0, u * sc["dur"] / .45)))
            self.attrib(ui)
            return Image.alpha_composite(img.convert("RGBA"), ui.down()).convert("RGB")
        return Simple.frame(self, f)

    def map_only(self, sc, f):
        sc_, u = self.locate(f / FPS)
        t = float(interp(sc["sim"], u))
        v, to, wp = self.mpane(sc, t, sc["zoom"])
        run = self.runs[sc["run"]]
        self.bars(v, wp, run, sc["j"], t, to, sc["zoom"])
        self.draw_vehicles(v, wp, run, sc["run"], sc["j"], t, to, sc["zoom"])
        return v.down().filter(ImageFilter.GaussianBlur(8))

    def verdict(self, ui, sc, u):
        txt, col = sc["verdict"]
        b = ease(min(1.0, max(0.0, (u * sc["dur"] - 1.6) / .5)))
        f2 = G(700, 50)
        w = ui.tw(txt, f2) + 90
        y = H - 190
        ui.rect((W / 2 - w / 2 + 4, y + 6, W / 2 + w / 2 + 4, y + 96), fill=(0, 0, 0, int(30 * b)), r=46)
        ui.rect((W / 2 - w / 2, y, W / 2 + w / 2, y + 90), fill=(255, 255, 255, int(252 * b)), outline=rgba(col, b), w=4, r=45)
        ui.text((W / 2, y + 45), txt, f2, rgba(col, b), "mm")

    def attrib(self, ui):
        s = "Google Maps  ·  Map data ©2026 Google"
        fa = world.F("s", 15)
        w = ui.tw(s, fa) + 24
        ui.rect((W / 2 - w / 2, H - 40, W / 2 + w / 2, H - 14), fill=(255, 255, 255, 215), r=6)
        ui.text((W / 2, H - 27), s, fa, (60, 62, 66, 255), "mm")


_film = None


def _init():
    global _film
    _film = Plan()


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


def render(jobs, name="AmbuGo_v9_Plan.mp4"):
    film = Plan()
    n = int(round(film.total * FPS))
    tmp = OUT / "_parts_plan"
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
    film = Plan() if (a.stills or a.contact) else None
    if a.stills:
        d = OUT / "stills"
        d.mkdir(exist_ok=True)
        for v in a.stills:
            film.frame(int(v * FPS)).save(d / f"plan_{v:06.2f}.jpg", quality=90)
        print("total", film.total)
        return
    if a.contact:
        times = [float(film.starts[i] + film.scenes[i]["dur"] * .8) for i in range(len(film.scenes))]
        ims = [film.frame(int(t * FPS)).resize((480, 270), Image.Resampling.LANCZOS) for t in times]
        cols = 4
        rows = math.ceil(len(ims) / cols)
        sheet = Image.new("RGB", (cols * 490 + 10, rows * 300 + 10), (20, 22, 26))
        d = ImageDraw.Draw(sheet)
        for i, (t, im) in enumerate(zip(times, ims)):
            x, y = 10 + (i % cols) * 490, 10 + (i // cols) * 300
            sheet.paste(im, (x, y))
            d.text((x, y + 276), f"beat {i + 1}   {t:5.1f} s", fill=(220, 220, 220))
        sheet.save(OUT / "contact_sheet_plan.jpg", quality=90)
        print(OUT / "contact_sheet_plan.jpg", film.total)
        return
    render(a.jobs)


if __name__ == "__main__":
    main()
