#!/usr/bin/env python3
"""Live data panel for AmbuGo_v9_Plan.mp4: the film on the left, two graphs of the same recorded runs on the right.

    python panel_data.py               # once: data/panel_series.json
    python panelfilm.py                # output/AmbuGo_v9_Plan_with_graphs.mp4 (2880x1080)
    python panelfilm.py --stills 60    # composite stills

Every frame of the panel follows the film's own clock: the run shown on the map (AmbuGo, or the always-change
JFAST run during that comparison), the simulation time on the map, and whether the film is paused.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from world import View, rgba, SS  # noqa: E402
from mapfilm import G, interp, FPS  # noqa: E402
from planfilm import Plan  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PW, PH, MW = 960, 1080, 1920
INK, MUTED, GRID = (32, 33, 36), (100, 104, 110), (226, 228, 232)
COL = {"P3": (40, 92, 160), "P2": (196, 96, 24), "P0": (150, 154, 160)}
NAME = {"P3": "AmbuGo run", "P2": "Comparison run: light always changed (JFAST)"}
DISPATCH = 4064.0
XMAX = 300.0


class Panel:
    def __init__(self):
        self.d = json.loads((HERE / "data" / "panel_series.json").read_text())
        for k in ("P3", "P0", "P2"):
            s = self.d[k]
            s["t"] = np.array(s["t"])
            s["amb_kmh"] = np.array([np.nan if x is None else x for x in s["amb_kmh"]])
            s["stopped"] = np.array(s["stopped"], float)
        self.film = Plan()

    def state(self, f):
        sc, u = self.film.locate(f / FPS)
        t = float(interp(sc["sim"], u))
        r = self.film.rate(sc, u)
        view = sc["view"]
        run = sc.get("run", "P3")
        if view == "net":
            return dict(run="P3", el=0.0, mode="plan", rate=0.0)
        mode = "card" if (sc.get("card") or sc.get("eq")) else "map"
        return dict(run=run, el=max(0.0, t - DISPATCH), mode=mode, rate=r)

    # ------------------------------------------------------------------ drawing helpers
    def plot(self, v, box, ymax, yticks, series, el, ylab):
        x0, y0, x1, y1 = box
        sx = lambda x: x0 + (x1 - x0) * x / XMAX
        sy = lambda y: y1 - (y1 - y0) * y / ymax
        for yt in yticks:
            v.line([(x0, sy(yt)), (x1, sy(yt))], GRID + (255,), 1.2)
            v.text((x0 - 12, sy(yt)), f"{yt:g}", G(400, 19), MUTED + (255,), "rm")
        for xt in range(0, int(XMAX) + 1, 60):
            v.line([(sx(xt), y1), (sx(xt), y1 + 6)], MUTED + (255,), 1.2)
            v.text((sx(xt), y1 + 22), f"{xt}", G(400, 19), MUTED + (255,), "mm")
        v.line([(x0, y1), (x1, y1)], (180, 183, 188, 255), 1.5)
        v.text((x0 - 52, (y0 + y1) / 2), ylab, G(400, 18, True), MUTED + (255,), "mm")
        for t, y, col, w in series:
            m = (t <= el + 1e-9) & ~np.isnan(y)
            if m.sum() < 2:
                if m.sum() == 1:
                    v.circle((sx(t[m][0]), sy(y[m][0])), 4, fill=col + (255,))
                continue
            # break the line where data is missing (before insertion / after arrival)
            idx = np.flatnonzero(m)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for rr in runs:
                if len(rr) > 1:
                    v.line([(sx(t[i]), sy(y[i])) for i in rr], col + (255,), w)
            v.circle((sx(t[idx[-1]]), sy(y[idx[-1]])), 5.5, fill=col + (255,))
        cx = sx(min(el, XMAX))
        v.line([(cx, y0 - 6), (cx, y1)], (120, 124, 130, 200), 1.5)

    def value_at(self, run, key, el):
        s = self.d[run]
        i = int(np.clip(np.searchsorted(s["t"], el + 1e-9) - 1, 0, len(s["t"]) - 1))
        return s[key][i]

    def render(self, f):
        st = self.state(f)
        v = View(PW, PH, 0, 0, 1)
        v.im = Image.new("RGBA", (PW * SS, PH * SS), (250, 250, 248, 255))
        v.d = ImageDraw.Draw(v.im)
        run, el = st["run"], st["el"]
        col = COL[run]
        v.line([(0, 0), (0, PH)], (210, 212, 216, 255), 3)
        v.text((48, 58), "Live data from this simulation", G(700, 36), INK + (255,), "lm")
        v.circle((58, 112), 9, fill=col + (255,))
        v.text((78, 112), NAME[run], G(600, 25), col + (255,), "lm")
        if st["mode"] == "plan":
            status = "Planning at dispatch: the ambulance has not moved yet"
        else:
            m, s = divmod(int(el), 60)
            r = st["rate"]
            status = f"{m}:{s:02d} after dispatch  ·  " + ("paused" if abs(r) < .05 else f"playing at {r:.1f}× real time")
        v.text((48, 150), status, G(400, 23, True), MUTED + (255,), "lm")
        for i, s in enumerate(["The same recorded traffic as the map. The graphs grow as the",
                               "film plays and stop when it pauses; the grey line marks now."]):
            v.text((48, 198 + i * 30), s, G(500, 22), INK + (255,), "lm")

        # ---- graph 1: ambulance speed
        y = 292
        v.text((48, y), "Ambulance speed", G(700, 29), INK + (255,), "lm")
        sp = self.value_at(run, "amb_kmh", el)
        trav = self.d[run]["travel_s"]
        if st["mode"] != "plan" and el >= trav:
            val = "arrived"
        elif np.isnan(sp) or st["mode"] == "plan":
            val = "—"
        else:
            val = f"{sp:.0f} km/h"
        v.text((PW - 48, y), val, G(700, 32), col + (255,), "rm")
        v.text((48, y + 34), "Drops to 0 while the ambulance waits at a red light.", G(400, 21, True), MUTED + (255,), "lm")
        self.plot(v, (110, y + 70, PW - 50, y + 250), 60, (0, 25, 50),
                  [(self.d[run]["t"], self.d[run]["amb_kmh"], col, 3.2)], el, "km/h")

        # ---- graph 2: stopped vehicles, this run vs no priority
        y = 640
        v.text((48, y), "Cars standing still in the whole district", G(700, 29), INK + (255,), "lm")
        a, b = self.value_at(run, "stopped", el), self.value_at("P0", "stopped", el)
        v.text((48, y + 34), "Every ordinary vehicle at a standstill right now, in this run", G(400, 21, True), MUTED + (255,), "lm")
        v.text((48, y + 60), "and in the same traffic with no signal priority.", G(400, 21, True), MUTED + (255,), "lm")
        lx = 48
        for lab, c, n in [(NAME[run].split(":")[0] if run == "P3" else "JFAST run", col, a), ("No signal priority", COL["P0"], b)]:
            v.line([(lx, y + 100), (lx + 34, y + 100)], c + (255,), 4)
            txt = f"{lab}: {int(n)}" if st["mode"] != "plan" else lab
            v.text((lx + 44, y + 100), txt, G(600, 22), c + (255,), "lm")
            lx += 44 + v.tw(txt, G(600, 22)) + 46
        self.plot(v, (110, y + 130, PW - 50, y + 330), 130, (0, 40, 80, 120),
                  [(self.d["P0"]["t"], self.d["P0"]["stopped"], COL["P0"], 2.4),
                   (self.d[run]["t"], self.d[run]["stopped"], col, 3.0)], el, "cars")
        v.text(((110 + PW - 50) / 2, PH - 50), "seconds since the ambulance was dispatched", G(500, 22), MUTED + (255,), "mm")
        v.text(((110 + PW - 50) / 2, PH - 20), "sampled every 0.5 s  ·  same seed and traffic demand in every run",
               G(400, 18, True), MUTED + (255,), "mm")
        return v.down().convert("RGB")


def composite(panel, frame_rgb, f):
    im = Image.new("RGB", (MW + PW, PH), (255, 255, 255))
    im.paste(frame_rgb, (0, 0))
    im.paste(panel.render(f), (MW, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", type=float, nargs="*")
    a = ap.parse_args()
    src = OUT / "AmbuGo_v9_Plan.mp4"
    panel = Panel()
    if a.stills:
        d = OUT / "stills"
        d.mkdir(exist_ok=True)
        for s in a.stills:
            raw = subprocess.run([shutil.which("ffmpeg"), "-loglevel", "error", "-ss", f"{s}", "-i", str(src), "-frames:v", "1",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            fr = Image.frombytes("RGB", (MW, PH), raw)
            composite(panel, fr, int(round(s * FPS))).save(d / f"graphs_{s:06.2f}.jpg", quality=90)
        return
    n = int(round(panel.film.total * FPS))
    dec = subprocess.Popen([shutil.which("ffmpeg"), "-loglevel", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    out = OUT / "AmbuGo_v9_Plan_with_graphs.mp4"
    enc = subprocess.Popen([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{MW + PW}x{PH}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    size = MW * PH * 3
    for f in range(n):
        raw = dec.stdout.read(size)
        if len(raw) < size:
            break
        enc.stdin.write(composite(panel, Image.frombytes("RGB", (MW, PH), raw), f).tobytes())
        if f % 300 == 0:
            print(f"{f}/{n}", flush=True)
    enc.stdin.close()
    enc.wait()
    dec.wait()
    print(out)


if __name__ == "__main__":
    main()
