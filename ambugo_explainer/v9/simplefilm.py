#!/usr/bin/env python3
"""AmbuGo v9 simple explainer: one idea per beat, one caption, at most one highlighted thing on the map.

    python simplefilm.py                 # output/AmbuGo_v9_Simple.mp4
    python simplefilm.py --stills 5 20   # stills at video seconds
    python simplefilm.py --contact       # contact sheet (one frame per beat)
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
from world import View, rgba, SS, AMB_BLUE, SIG_G, SIG_A, SIG_R  # noqa: E402
from mapfilm import Film as MapFilm, G, ease, interp, W, H, FPS, INK, M_PER_PX, VEH_SCALE  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
COST = (214, 96, 24)          # roads that would pay for a change
GAIN = (24, 128, 56)          # roads that gain
AMB = AMB_BLUE

J1_AMB, J1_CROSS, J1_SOUTH = "E_J10_J11", "E_J01_J11", "E_J21_J11"
J2_AMB, J2_EAST, J2_SOUTH, J2_WEST = "E_J11_J21", "E_J22_J21", "E_J31_J21", "E_J20_J21"


def B(dur, j, run, sim, cap, hl=(), zoom=1.15, shift=(0, 60), verdict=None, card=None):
    """One beat. hl: list of ("road", edge, colour, label) | ("amb", label) | ("junction", label)."""
    return dict(dur=dur, j=j, run=run, sim=sim, cap=cap, hl=list(hl), zoom=zoom, shift=shift, verdict=verdict, card=card)


class Simple(MapFilm):
    def build(self):
        g1, g2 = self.g[(4091, "J11")], self.g[(4132, "J21")]
        S = [
            # ---------------- junction 1: the request is withheld
            B(5, "J11", "P3", [(0, 4079), (1, 4091)], ["An ambulance is on its way to hospital."],
              [("amb", "Ambulance")], zoom=1.05),
            B(4, "J11", "P3", [(0, 4091), (1, 4091)], ["At this junction it asks for a green light."],
              [("road", J1_AMB, AMB, "Ambulance's road")]),
            B(5, "J11", "P3", [(0, 4091), (1, 4091)],
              ["Without any change, its light turns green", f"only {g1['nominal_wait']:.1f} seconds after it arrives."],
              [("road", J1_AMB, AMB, f"wait: {g1['nominal_wait']:.1f} s")]),
            B(4, "J11", "P3", [(0, 4091), (1, 4091)], ["Changing the light would make the cross road wait."],
              [("road", J1_CROSS, COST, "Cross road")]),
            B(6, "J11", "P3", [(0, 4091), (1, 4091)],
              ["AmbuGo changes a light only if it saves", "the ambulance at least 5 seconds."],
              verdict=(f"{g1['benefit']:.1f} s saved  →  no change", COST)),
            # what a change would have done: the archived always-priority run, one road at a time
            B(4, "J11", "P2", [(0, 4091), (1, 4091)], ["What if the light had been changed anyway?"],
              [("junction", None)]),
            B(11, "J11", "P2", [(0, 4091), (1, 4170)], ["Recorded run of a controller that always changes the light."],
              [("road", J1_CROSS, COST, "Cross road")], zoom=1.05),
            B(4, "J11", "P2", [(0, 4170), (1, 4170)], ["The cross road waited 64 seconds longer for green."],
              [("road", J1_CROSS, COST, "+64 s wait")], zoom=1.05),
            B(4, "J11", "P2", [(0, 4170), (1, 4170)], ["The south road got its green 22 seconds sooner."],
              [("road", J1_SOUTH, GAIN, "22 s sooner")], zoom=1.05),
            B(4, "J11", "P2", [(0, 4170), (1, 4170)], ["The ambulance got through just 11 seconds sooner."],
              [("road", J1_AMB, AMB, "11 s sooner")], zoom=1.05),
            B(6, "J11", "P2", [(0, 4170), (1, 4170)], ["In total, 296 more vehicle-seconds of waiting", "to save 11 seconds."],
              verdict=("Not worth it  →  no change", COST), zoom=1.05),
            # ---------------- junction 2: the request is granted
            B(5, "J21", "P3", [(0, 4120), (1, 4132)], ["At the next junction, the ambulance would wait",
                                                       f"{g2['nominal_wait']:.0f} seconds for green."],
              [("road", J2_AMB, AMB, f"wait: {g2['nominal_wait']:.0f} s")], zoom=1.05),
            B(6, "J21", "P3", [(0, 4132), (1, 4132)], ["A change would make the other three roads wait:",
                                                       f"about {round(g2['burden'], -2):,.0f} vehicle-seconds in total."],
              [("road", J2_EAST, COST, None), ("road", J2_SOUTH, COST, None), ("road", J2_WEST, COST, None)]),
            B(6, "J21", "P3", [(0, 4132), (1, 4132)], ["That is within AmbuGo's limit of 10,000,", "and it saves the ambulance 43 seconds."],
              verdict=("43 s saved  →  change the light", GAIN)),
            B(8, "J21", "P3", [(0, 4132), (1, 4146)], ["The change is made safely: minimum green,", "then amber, then all-red."],
              [("junction", None)]),
            B(7, "J21", "P3", [(0, 4146), (1, 4180)], ["The ambulance gets green 86 seconds sooner", "than without the change."],
              [("amb", "Ambulance")], zoom=1.05),
            B(6, "J21", "P3", [(0, 4180), (1, 4205)], ["The east and south roads wait a little longer;", "no queue comes close to filling its road."],
              [("road", J2_EAST, COST, None), ("road", J2_SOUTH, COST, None)], zoom=1.05),
            # ---------------- close
            B(5, "J21", "P3", [(0, 4205), (1, 4205)], [], card=["AmbuGo changes a light only when it clearly helps",
                                                                "the ambulance and the cost to others stays small."]),
            B(6, "J21", "P3", [(0, 4205), (1, 4205)], [], card=["Across 1,728 simulations: trips 44.6% faster than with",
                                                                "no priority, with less extra congestion than changing every light."]),
        ]
        for s in S:
            s["ov"] = None
            s["key"] = "beat"
        self.scenes = S

    # ------------------------------------------------------------------ drawing
    def mpane(self, sc, t, zoom):
        j = sc["j"]
        wp = self.warp[j]
        cx, cy = wp.C + np.array(sc["shift"], float) / zoom
        sw, sh = W / zoom, H / zoom
        box = (int(cx - sw / 2), int(cy - sh / 2), int(cx - sw / 2) + int(sw), int(cy - sh / 2) + int(sh))
        bg = wp.img.crop(box).resize((W * SS, H * SS), Image.Resampling.BICUBIC)
        bg = Image.blend(bg, Image.new("RGB", bg.size, (255, 255, 255)), .38).convert("RGBA")     # quiet the map
        v = View(W, H, 0, 0, 1)
        v.im, v.d = bg, ImageDraw.Draw(bg)
        to = lambda P: (np.asarray(P, float) - np.array(box[:2], float)) * zoom
        v.s = zoom / M_PER_PX * VEH_SCALE / 1.6
        return v, to, wp

    def road_glow(self, v, wp, to, e, col, a):
        for ln in self.net.lane_of_edge[e]:
            shp = ln["shape"]
            L = np.linalg.norm(np.diff(shp, axis=0), axis=1).sum()
            keep = np.linspace(0, 1, 40)
            pts = np.array([np.interp(k * L, np.r_[0, np.cumsum(np.linalg.norm(np.diff(shp, axis=0), axis=1))], shp[:, i])
                            for k in keep for i in (0, 1)]).reshape(-1, 2)
            P = to(wp(pts))
            v.line([tuple(p) for p in P], rgba(col, .30 * a), 34 * v.s / 3.9 + 18)

    def road_label(self, ui, wp, to, e, col, text, a):
        c, d, half = self.net.stopline(e)
        p = to(wp(c - d * (30 if e == J1_AMB else 48))[0])
        f = G(700, 32)
        w = ui.tw(text, f) + 36
        ui.rect((p[0] - w / 2, p[1] - 26, p[0] + w / 2, p[1] + 26), fill=(255, 255, 255, int(250 * a)), outline=rgba(col, a), w=3, r=12)
        ui.text((p[0], p[1]), text, f, rgba(col, a), "mm")

    def bars(self, v, wp, run, j, t, to, zoom):
        mode, served, el, req = run.signal(j, t)
        for e in self.net.incoming[j]:
            c, d, half = self.net.stopline(e)
            n = np.array([-d[1], d[0]])
            col = SIG_G if (mode == "green" and served == e) else SIG_A if (mode == "amber" and served == e) else SIG_R
            a, b = to(wp(c - n * (half + 1))[0]), to(wp(c + n * (half + 1))[0])
            v.line([tuple(a), tuple(b)], (255, 255, 255, 240), 12 * zoom + 5)
            v.line([tuple(a), tuple(b)], col, 12 * zoom)

    def caption(self, ui, lines, a=1.0, y0=34):
        if a <= 0 or not lines:
            return
        f = G(500, 42)
        lh = 54
        w = max(ui.tw(s, f) for s in lines) + 96
        x0 = W / 2 - w / 2
        h = len(lines) * lh + 40
        ui.rect((x0 + 3, y0 + 6, x0 + w + 3, y0 + h + 6), fill=(0, 0, 0, int(26 * a)), r=18)
        ui.rect((x0, y0, x0 + w, y0 + h), fill=(255, 255, 255, int(250 * a)), r=18)
        for i, s in enumerate(lines):
            ui.text((W / 2, y0 + 20 + lh / 2 + i * lh), s, f, INK + (int(255 * a),), "mm")

    def frame(self, f):
        v_ = f / FPS
        sc, u = self.locate(v_)
        t = float(interp(sc["sim"], u))
        zoom = sc["zoom"]
        run = self.runs[sc["run"]]
        a_in = ease(min(1.0, u * sc["dur"] / .45))                 # every beat fades its own overlay in
        v, to, wp = self.mpane(sc, t, zoom)
        hls = [tuple(x(t) if callable(x) else x for x in h) for h in sc["hl"]]     # labels/colours may follow the clock
        for h in hls:
            if h[0] == "road":
                self.road_glow(v, wp, to, h[1], h[2], a_in)
        self.bars(v, wp, run, sc["j"], t, to, zoom)
        self.draw_vehicles(v, wp, run, sc["run"], sc["j"], t, to, zoom)
        img = v.down()
        if sc["card"]:
            img = img.filter(ImageFilter.GaussianBlur(8))
        ui = View(W, H, 0, 0, 1)
        ui.im = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        ui.d = ImageDraw.Draw(ui.im)
        for h in hls:
            if h[0] == "road" and h[3]:
                self.road_label(ui, wp, to, h[1], h[2], h[3], a_in)
            elif h[0] == "amb":
                self.amb_tag(ui, v, to, sc["j"], sc["run"], t, h[1], a_in,
                             off_text=h[2] if len(h) > 2 else "Ambulance (approaching)")
            elif h[0] == "junction":
                p = to(wp.C)
                r = 190 * zoom
                ui.circle(tuple(p), r, outline=rgba((40, 40, 40), .85 * a_in), w=4)
        self.caption(ui, sc["cap"], a_in)
        if sc["verdict"]:
            txt, col = sc["verdict"]
            b = ease(min(1.0, max(0.0, (u * sc["dur"] - 1.6) / .5)))
            f2 = G(700, 50)
            w = ui.tw(txt, f2) + 90
            y = H - 190
            ui.rect((W / 2 - w / 2 + 4, y + 6, W / 2 + w / 2 + 4, y + 96), fill=(0, 0, 0, int(30 * b)), r=46)
            ui.rect((W / 2 - w / 2, y, W / 2 + w / 2, y + 90), fill=(255, 255, 255, int(252 * b)), outline=rgba(col, b), w=4, r=45)
            ui.text((W / 2, y + 45), txt, f2, rgba(col, b), "mm")
        if sc["card"]:
            f3 = G(500, 50)
            for i, s in enumerate(sc["card"]):
                ui.text((W / 2, H / 2 - 40 + i * 70), s, f3, INK + (int(255 * a_in),), "mm")
        s = "Google Maps  ·  Map data ©2026 Google"
        fa = world.F("s", 15)
        w = ui.tw(s, fa) + 24
        ui.rect((W / 2 - w / 2, H - 40, W / 2 + w / 2, H - 14), fill=(255, 255, 255, 215), r=6)
        ui.text((W / 2, H - 27), s, fa, (60, 62, 66, 255), "mm")
        return Image.alpha_composite(img.convert("RGBA"), ui.down()).convert("RGB")


_film = None


def _init():
    global _film
    _film = Simple()


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


def render(jobs, name="AmbuGo_v9_Simple.mp4"):
    film = Simple()
    n = int(round(film.total * FPS))
    tmp = OUT / "_parts_simple"
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
    film = Simple() if (a.stills or a.contact) else None
    if a.stills:
        d = OUT / "stills"
        d.mkdir(exist_ok=True)
        for v in a.stills:
            film.frame(int(v * FPS)).save(d / f"simple_{v:06.2f}.jpg", quality=90)
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
        sheet.save(OUT / "contact_sheet_simple.jpg", quality=90)
        print(OUT / "contact_sheet_simple.jpg", film.total)
        return
    render(a.jobs)


if __name__ == "__main__":
    main()
