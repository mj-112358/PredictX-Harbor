#!/usr/bin/env python3
"""20-second cut of the AmbuGo explainer, for a human voiceover.

Same frozen episode, overlays and maths panels as the long film, but: no narration
panel (the map spans the full width), nine fast beats, short holds on the two
decisions. Panel animations that were paced for the long film run on a virtual
clock scaled to each beat.

    python src/short.py                 # output/AmbuGo_20s.mp4 (+ voiceover cue sheet)
    python src/short.py --stills 3 6 9  # stills at given video seconds
"""
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapview  # noqa: E402

FULL_W = 1920
mapview.MW = FULL_W                      # map spans the frame; no narration column

import story as ST  # noqa: E402
import render as R  # noqa: E402
import draw as D  # noqa: E402

ST.MW = FULL_W
R.MW = FULL_W
FPS = 30
K = FULL_W / 1250   # keeps the long film's map scale with the wider viewport
OUT = R.OUT


def cam(c):
    return (c[0], c[1], c[2] * K) if isinstance(c, tuple) and len(c) == 3 else ("track", c[1] * K)


def ov_focus(ml, c, colour="approach"):
    """Single-junction view: coloured approach lanes, cars, signal heads, ambulance. No tags."""
    if colour == "delta":
        focus, *_ = ST.junction_scene(ml, c, heads=True, colour="delta", labels=True, bands=.5)
    else:
        focus, *_ = ST.junction_scene(ml, c, heads=True)
    p = ST.draw_focus(ml, c, focus, .85)
    if colour != "delta":
        ST.request_badge(ml, c, p)


def build(ep):
    """One junction (PCL Chowk), about 70 s of simulated time, one camera."""
    J1 = ep.J1xy
    c_close = cam((J1[0] - 30, J1[1] - 30, 170))
    c_mid = cam((J1[0] - 40, J1[1] - 30, 215))
    S = ST.Shot
    beats = [
        S("approach", "Ambulance approaches PCL Chowk", 3.5, (776.0, 785.0), [(0, c_close)], ov_focus, "gauge", [], ramp=.0),
        S("decide", "Priority withheld: cost to other drivers is over budget", 4.0, (785.0, 791.9), [(0, c_close)], ov_focus, "gauge", [], ramp=.0),
        S("admit", "New sensor data: cost falls under budget, request admitted", 1.6, 792.0, [(0, c_close)], ov_focus, "gauge", []),
        S("green", "Ambulance gets green at the earliest legal moment", 4.4, (792.0, 803.0), [(0, c_close)], ov_focus, "timeline", [], ramp=.0),
        S("shift", "The other three approaches now wait 52 s longer", 2.0, 803.0, [(0, c_close)], ov_focus, "timeline_shift", []),
        S("pay", "Who pays: extra waiting per car vs no priority", 4.5, (803.0, 850.0), [(0, c_close), (.3, c_mid)],
          lambda ml, c: ov_focus(ml, c, "delta"), "cost", [], ramp=.0),
    ]
    for b in beats:
        b.virtual = None
    return beats


def big_gauge(cv, c):
    ep, t = c.ep, c.t
    e = ep.event_at(t) if t >= 785 else None
    cv.text((40, 30), "Forecast cost to other drivers if the ambulance gets priority now", 24, D.INK, "bold")
    cv.text((40, 66), "waiting added to the NW, SW and NE approaches (vehicle-seconds) · budget B = 1000", 17, D.MUTED)
    x0, x1, y0, y1 = 40, 1500, 120, 190
    sc = (x1 - x0) / 1300
    cv.rect((x0, y0, x1, y1), fill=(240, 242, 245), r=8)
    total = 0.0
    if e is not None:
        x = x0
        for a in e["conflicting_approaches"]:
            w = a["burden_proxy_veh_s"] * sc
            cv.rect((x, y0, x + w, y1), fill=D.APP_COL[a["edge"]])
            cv.text((x + w / 2, (y0 + y1) / 2), ep.app_name[a["edge"]], 20, (255, 255, 255), "bold", anchor="mm")
            x += w
        total = e["raw_burden_proxy_veh_s"]
    bx = x0 + 1000 * sc
    cv.line([(bx, y0 - 16), (bx, y1 + 16)], D.INK, 5)
    cv.text((bx, y1 + 22), "budget B = 1000", 18, D.INK, "bold", anchor="ma")
    if e is None:
        cv.text((x0 + 20, (y0 + y1) / 2), "not evaluated yet: ambulance outside its request horizon", 20, D.MUTED, "semibold", anchor="lm")
        msg, col = "WAITING", D.MUTED
    elif total > 1000:
        msg, col = "WITHHELD", D.BAD
    else:
        msg, col = "ADMITTED", D.AMB
    if e is not None:
        cv.text((1880, 120), f"{total:,.0f}", 64, col, "bold", anchor="ra")
        cv.text((1880, 196), "vehicle-seconds", 18, D.MUTED, anchor="ra")
    cv.rect((1580, 240, 1880, 300), fill=D.alpha(col, .12), outline=col, width=3, r=10)
    cv.text((1730, 270), msg, 28, col, "bold", anchor="mm")


def simple_timeline(cv, c, shift=False):
    import panels as PN
    ep, t = c.ep, c.t
    cv.text((40, 24), "Which approach has green at PCL Chowk", 24, D.INK, "bold")
    x = 900
    for col, s in [(D.AMB, "SE (ambulance)"), (D.NW_C, "NW"), (D.SW_C, "SW"), (D.NE_C, "NE"), ((196, 199, 204), "pedestrian")]:
        PN.chip(cv, x, 30, col, 18, 18)
        cv.text((x + 26, 28), s, 18, D.INK)
        x += 60 + cv.tsize(s, 18)[0]
    y0, y1 = 80, 300
    ch = PN.gantt(cv, c, ep.J1, 770, 905, (40, y0, 1880, y1), reveal_t=None if shift else t, preview=False)
    if shift:
        gap = 40
        rowh = (y1 - y0 - 30 - gap) / 2
        g0, g1 = y0 + rowh, y0 + rowh + gap
        a = D.fade(c.ls, .2, .8)
        for p0, p3, col in [(795.5, 847.5, D.NW_C), (812.5, 864.5, D.SW_C), (839.5, 891.5, D.NE_C)]:
            xa, xb = float(ch.X(p0)), float(ch.X(p3))
            cv.arrow((xa, g1 - 1), (xb, g0 + 1), D.alpha(col, a), 4, 14)
            cv.text((xb + 6, g0 + 4), "+52 s", 17, D.alpha(col, a), "bold")


def cost_panel(cv, c):
    ep, t = c.ep, c.t
    k = ep.k(t)
    extra = sum(ep.app_extra[e][k] for e in ep.conf)
    cols = [(40, "Ambulance at this light", "served 72 s earlier", "than in the no-priority run", D.AMB),
            (680, "Other three approaches", f"+{max(extra, 0):,.0f} veh·s", "extra waiting so far (measured)", D.BAD),
            (1320, "Controller's forecast", "913 veh·s", "at admission (Eq. 20)", D.INK)]
    for x, h, big, sub, col in cols:
        cv.text((x, 40), h, 22, D.MUTED, "semibold")
        cv.text((x, 90), big, 54, col, "bold")
        cv.text((x, 170), sub, 19, D.MUTED)
    cv.text((40, 250), "Car colour = its extra stopped time vs the same car without priority:", 18, D.INK, "semibold")
    x0 = 700
    for i in range(-40, 41, 2):
        cv.rect((x0 + (i + 40) * 4, 248, x0 + (i + 42) * 4, 270), fill=D.diverging(i))
    cv.text((x0, 278), "less", 15, D.MUTED)
    cv.text((x0 + 164, 278), "same", 15, D.MUTED, anchor="ma")
    cv.text((x0 + 328, 278), "more", 15, D.MUTED, anchor="ra")


class ShortFilm(R.Film):
    def __init__(self):
        from data import Episode
        self.ep = Episode()
        self.base = mapview.Basemap()
        self.shots = build(self.ep)
        for s in self.shots:
            s.appear = []
        self.starts = np.cumsum([0] + [s.dur for s in self.shots])
        self.total = float(self.starts[-1])
        self.nframes = int(round(self.total * FPS))
        ep = self.ep
        k760 = ep.k(760)
        d = ep.c3 - ep.c0
        self.first_div = {}
        for i, e in enumerate(ep.edge_ids):
            if d[k760, i] == 0:
                nz = np.flatnonzero(d[k760:, i] != 0)
                if len(nz):
                    self.first_div[e] = float(ep.t[k760 + nz[0]])
        self._cameras()

    def ctx(self, f):
        c = super().ctx(f)
        if c.shot.virtual:
            c.ls = c.u * c.shot.virtual
        return c

    def sidebar(self, img, c):
        pass

    def bottom(self, img, c):
        cv = D.Canvas(R.W, R.H - R.BOTTOM_Y, (255, 255, 255, 255))
        cv.line([(0, 0), (R.W, 0)], D.RULE, 2)
        p = c.shot.panel
        if p == "gauge":
            big_gauge(cv, c)
        elif p == "timeline":
            simple_timeline(cv, c)
        elif p == "timeline_shift":
            simple_timeline(cv, c, shift=True)
        elif p == "cost":
            cost_panel(cv, c)
        im = cv.down()
        img.paste(im, (0, R.BOTTOM_Y), im)

    def header(self, img, c):
        cv = D.Canvas(R.W, mapview.MY, (255, 255, 255, 255))
        cv.text((24, 28), c.shot.chapter, 26, D.INK, "bold", anchor="lm")
        t = c.t
        el = t - 760
        cv.text((R.W - 330, 28), f"t = {t:.1f} s  ·  dispatch + {D.fmt_mmss(abs(el))}", 19, D.INK, "semibold", anchor="rm")
        if c.rate <= 1e-6:
            pill, col = "HOLD", (150, 100, 0)
        else:
            pill, col = f"▶ {c.rate:.1f}× real time" if c.rate >= 1 else f"▶ {c.rate:.2f}× real time", D.AMB
        w, _ = cv.tsize(pill, 16, "bold")
        cv.rect((R.W - 24 - w - 28, 12, R.W - 24, 44), fill=D.alpha(col, .12), outline=col, width=2, r=16)
        cv.text((R.W - 24 - w / 2 - 14, 28), pill, 16, col, "bold", anchor="mm")
        x = 0
        for k, s in enumerate(self.shots):
            w = s.dur / self.total * R.W
            if k < c.index:
                cv.rect((x, mapview.MY - 4, x + w - 2, mapview.MY), fill=D.AMB)
            elif k == c.index:
                cv.rect((x, mapview.MY - 4, x + max(1, w * c.u), mapview.MY), fill=D.AMB)
            else:
                cv.rect((x, mapview.MY - 4, x + w - 2, mapview.MY), fill=(225, 229, 234))
            x += w
        im = cv.down()
        img.paste(im, (0, 0), im)


_film = None


def _init():
    global _film
    _film = ShortFilm()


def _chunk(args):
    f0, f1, path = args
    cmd = [shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-threads", "1", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(_film.frame(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return f1 - f0


def render(out, jobs):
    film = ShortFilm()
    n = film.nframes
    tmp = out / "_short_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = [(f0, min(n, f0 + 60), tmp / f"p{i:03d}.mp4") for i, f0 in enumerate(range(0, n, 60))]
    t0 = time.time()
    with mp.get_context("fork").Pool(jobs, initializer=_init) as pool:
        done = 0
        for k in pool.imap_unordered(_chunk, parts):
            done += k
            print(f"{done}/{n} frames ({time.time() - t0:.0f}s)", flush=True)
    (tmp / "list.txt").write_text("".join(f"file '{p[2].name}'\n" for p in parts))
    final = out / "AmbuGo_20s.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    shutil.rmtree(tmp)
    cues = []
    for i, s in enumerate(film.shots):
        sim = s.sim if isinstance(s.sim, tuple) else (s.sim, s.sim)
        cues.append(f"{film.starts[i]:5.1f}–{film.starts[i + 1]:5.1f} s   sim {sim[0]:.1f}–{sim[1]:.1f} s   {s.chapter}")
    (out / "AmbuGo_20s_cues.txt").write_text("\n".join(cues) + "\n")
    print(final, f"{film.total:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stills", type=float, nargs="*")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.stills:
        film = ShortFilm()
        d = OUT / "preview_short"
        d.mkdir(exist_ok=True)
        for vt in a.stills:
            film.frame(int(vt * FPS)).save(d / f"s_{vt:05.2f}.jpg", quality=90)
        print("length", film.total)
        return
    render(OUT, a.jobs)


if __name__ == "__main__":
    main()
