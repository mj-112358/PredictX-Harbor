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


def build(ep):
    J1 = ep.J1xy
    c_close = cam((J1[0] - 38, J1[1] - 30, 178))
    c_mid = cam((J1[0] - 25, J1[1] - 20, 260))
    c_route = cam((8540650.0, 3594190.0, 2300.0))
    c_netd = cam((8540650.0, 3594200.0, 2500.0))
    c_hosp = cam((8541700.0, 3593275.0, 250.0))
    c_j2 = cam((ep.J2xy[0], ep.J2xy[1], 260.0))
    S = ST.Shot
    # (key, header label, video s, sim range or frozen time, cameras, overlay, panel, virtual-clock seconds)
    beats = [
        S("routes", "Joint route + priority choice", 2.0, 761.0, [(0, c_route)], ST.ov_routes, "routes", []),
        S("approach", "Approaching PCL Chowk", 1.8, (761.5, 785.0), [(0, c_route), (.6, c_close)], ST.ov_approach, "speed", [], ramp=.15),
        S("decide", "Priority withheld: forecast burden > budget", 2.6, (785.0, 791.6), [(0, c_close)],
          lambda ml, c: ST.ov_decision(ml, c, True), "burden_live", [], ramp=.0),
        S("admit", "Admitted: new estimate brings burden under B", 1.4, 792.0, [(0, c_close)],
          lambda ml, c: ST.ov_decision(ml, c, True), "burden_prev", []),
        S("transition", "Legal transition → ambulance green", 2.4, (792.0, 804.0), [(0, c_close)], ST.ov_transition, "gantt", [], ramp=.0),
        S("shift", "One switched light shifts the whole cycle (+52 s)", 2.0, 804.0, [(0, c_close), (.5, c_mid)], ST.ov_shift, "gantt_preview", []),
        S("consequence", "Who pays: extra waiting per car vs no priority", 2.0, (804.0, 851.0), [(0, c_close)], ST.ov_consequence, "approaches", [], ramp=.0),
        S("escalate", "Progress deficit > 20 s → escalation; J2", 1.8, (851.0, 1001.0),
          [(0, ("track", 650 * K)), (.55, c_j2)], ST.ov_j2, "speed", [], ramp=.0),
        S("hospital", "Hospital re-route and arrival: 323.5 s vs 516.5 s", 2.2, (1045.0, 1100.0),
          [(0, ("track", 500 * K)), (.4, c_hosp)], ST.ov_hospital, "hospital", [], ramp=.25),
        S("netdelta", "Replay, whole network: where the cost went", 1.8, (760.0, 1125.0), [(0, c_netd)], ST.ov_network_delta, "network", [], ramp=.0),
    ]
    # virtual clock: the long film's timed reveals run to completion within each beat
    virt = {"routes": 7.5, "shift": 12.0, "hospital": 2.0, "netdelta": 25.0}
    for b in beats:
        b.virtual = virt.get(b.key)
        b.chapter = b.chapter
    return beats


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
