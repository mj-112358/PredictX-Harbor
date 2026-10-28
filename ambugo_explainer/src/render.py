#!/usr/bin/env python3
"""Render the AmbuGo explainer film (1920x1080, 30 fps) from the frozen B1000 episode.

    python src/render.py                 # full film -> output/AmbuGo_Explainer.mp4
    python src/render.py --preview       # one still per chapter -> output/preview/
    python src/render.py --stills 12.5 80 # stills at given video seconds
    python src/render.py --jobs 4        # parallel workers (default: CPU count)
"""
from pathlib import Path
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import draw as D  # noqa: E402
import panels as PN  # noqa: E402
import story as ST  # noqa: E402
from data import Episode, ROOT  # noqa: E402
from draw import Canvas, alpha, fade  # noqa: E402
from mapview import Basemap, MapLayer, MX, MY, MW, MH  # noqa: E402

W, H, FPS = 1920, 1080, 30
SIDE = (MW, MY, W, 744)
BOTTOM_Y = 744
OUT = ROOT / "output"
READ_WPS = 3.2  # words per second used to pace narration


# ------------------------------------------------------------------ timeline
class Film:
    def __init__(self):
        self.ep = Episode()
        self.base = Basemap()
        self.shots = ST.build(self.ep)
        for s in self.shots:
            words = sum(len(txt.split()) for _, txt in s.narration)
            s.dur = max(s.dur, 2.5 + words / READ_WPS + 3.0)
            # appearance time of each narration item, paced by reading speed
            acc, s.appear = 1.0, []
            for kind, txt in s.narration:
                s.appear.append(acc)
                acc += len(txt.split()) / READ_WPS + (0.6 if kind == "h" else 0.4)
        self.starts = np.cumsum([0] + [s.dur for s in self.shots])
        self.total = float(self.starts[-1])
        self.nframes = int(round(self.total * FPS))
        ep = self.ep
        k760 = ep.k(760)
        self.first_div = {}
        d = ep.c3 - ep.c0
        for i, e in enumerate(ep.edge_ids):
            if d[k760, i] != 0:
                continue
            nz = np.flatnonzero(d[k760:, i] != 0)
            if len(nz):
                self.first_div[e] = float(ep.t[k760 + nz[0]])
        self._cameras()

    def locate(self, vt):
        i = int(np.clip(np.searchsorted(self.starts, vt, side="right") - 1, 0, len(self.shots) - 1))
        s = self.shots[i]
        ls = vt - self.starts[i]
        return i, s, float(np.clip(ls / s.dur, 0, 1)), ls

    def _spec(self, spec, t, last):
        if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "track":
            p = self.ep.amb_pos(t)
            if not np.isfinite(p).all():
                p = last[:2]
            return np.array([p[0], p[1], spec[1]], float)
        return np.array(spec, float)

    def _cameras(self):
        cams = np.zeros((self.nframes + 1, 3))
        last = np.array(self.shots[0].cams[0][1], float)
        prev_end = last.copy()
        cur_shot = 0
        for f in range(self.nframes + 1):
            vt = f / FPS
            i, s, u, ls = self.locate(vt)
            if i != cur_shot:
                prev_end = cams[f - 1].copy()
                cur_shot = i
            t = s.sim_time(u)
            ks = s.cams
            j = max(n for n in range(len(ks)) if ks[n][0] <= u + 1e-12)
            a = self._spec(ks[j][1], t, last)
            if j + 1 < len(ks):
                b = self._spec(ks[j + 1][1], t, last)
                w = D.smooth((u - ks[j][0]) / (ks[j + 1][0] - ks[j][0]))
                cam = blend(a, b, w)
            else:
                cam = a
            if i > 0:
                T = min(3.5, .3 * s.dur)
                w = D.smooth(ls / T)
                if w < 1:
                    cam = blend(prev_end, cam, w, arc=True)
            cams[f] = cam
            last = cam
        # light smoothing removes 0.5 s position-sample tremor from tracking shots
        ker = np.exp(-0.5 * (np.arange(-9, 10) / 3.5) ** 2)
        ker /= ker.sum()
        for c in range(3):
            v = np.log(cams[:, c]) if c == 2 else cams[:, c]
            pad = np.pad(v, (9, 9), mode="edge")
            v = np.convolve(pad, ker, mode="valid")
            cams[:, c] = np.exp(v) if c == 2 else v
        self.cams = cams

    # -------------------------------------------------------------- frame
    def ctx(self, f):
        vt = f / FPS
        i, s, u, ls = self.locate(vt)
        c = ST.Ctx()
        c.ep, c.shot, c.u, c.ls, c.vt, c.index = self.ep, s, u, ls, vt, i
        c.t = s.sim_time(u)
        c.cam = self.cams[min(f, self.nframes)]
        c.first_div = self.first_div
        if isinstance(s.sim, tuple):
            r = s.ramp
            c.rate = (s.sim[1] - s.sim[0]) / (s.dur * (1 - r) if r > 0 else s.dur)
        else:
            c.rate = 0.0
        return c

    def frame(self, f):
        c = self.ctx(f)
        img = Image.new("RGB", (W, H), D.PAPER)
        bg = self.base.render(c.cam)
        img.paste(bg, (MX, MY))
        ml = MapLayer(self.ep, c.cam)
        c.shot.overlay(ml, c)
        ov = ml.done()
        img.paste(ov, (MX, MY), ov)
        self.header(img, c)
        self.sidebar(img, c)
        self.bottom(img, c)
        self.attribution(img)
        return img

    # ------------------------------------------------------------- chrome
    def header(self, img, c):
        cv = Canvas(W, MY, (255, 255, 255, 255))
        cv.text((24, 28), c.shot.chapter, 24, D.INK, "bold", anchor="lm")
        t = c.t
        el = t - 760
        clock = f"t = {t:.1f} s"
        sub = f"dispatch {'+' if el >= 0 else '−'} {D.fmt_mmss(abs(el))}"
        cv.text((W - 330, 28), sub, 15, D.MUTED, anchor="rm")
        sw, _ = cv.tsize(sub, 15)
        cv.text((W - 330 - sw - 14, 28), clock, 21, D.INK, "semibold", anchor="rm")
        if c.rate <= 1e-6:
            pill, col = "PAUSED · explanation", D.AMBERC
        else:
            r = c.rate
            pill, col = (f"▶  {r:.2f}× real time" if r < 1 else f"▶  {r:.1f}× real time"), D.AMB
        w, _ = cv.tsize(pill, 16, "bold")
        cv.rect((W - 24 - w - 28, 12, W - 24, 44), fill=alpha(col, .13), outline=col, width=2, r=16)
        cv.text((W - 24 - w / 2 - 14, 28), pill, 16, col if col != D.AMBERC else (150, 100, 0), "bold", anchor="mm")
        # chapter progress bar
        x = 0
        for k, s in enumerate(self.shots):
            w = s.dur / self.total * W
            col = D.AMB if k < c.index else (D.AMB if k == c.index else (225, 229, 234))
            if k == c.index:
                cv.rect((x, MY - 4, x + w * c.u, MY), fill=D.AMB)
                if w * c.u < w - 1:
                    cv.rect((x + w * c.u, MY - 4, x + w - 1, MY), fill=(190, 205, 230))
            else:
                cv.rect((x, MY - 4, x + w - 1, MY), fill=col)
            x += w
        im = cv.down()
        img.paste(im, (0, 0), im)

    def sidebar(self, img, c):
        x0, y0, x1, y1 = SIDE
        wv, hv = x1 - x0, y1 - y0
        cv = Canvas(wv, hv, (*D.PAPER, 255))
        cv.line([(0, 0), (0, hv)], D.RULE, 2)
        s = c.shot
        size = 18
        for _ in range(6):
            if self._layout(cv, s, c, size, dry=True) <= hv - 14:
                break
            size -= 1
        self._layout(cv, s, c, size, dry=False)
        im = cv.down()
        img.paste(im, (x0, y0), im)

    def _layout(self, cv, s, c, size, dry):
        x, y, wmax = 26, 20, SIDE[2] - SIDE[0] - 52
        for (kind, txt), at in zip(s.narration, s.appear):
            a = fade(c.ls, at, at + .7)
            if dry:
                a = 1.0
            if a <= 0:
                continue
            new = (not dry) and (c.ls - at) < 3.5
            if kind == "h":
                y += 6
                if not dry:
                    cv.text((x, y), txt, size + 3, alpha(D.INK, a), "bold")
                y += (size + 3) * 1.45
            elif kind in ("k", "w"):
                col = D.AMB if kind == "k" else (190, 120, 0)
                bgc = (232, 240, 252) if kind == "k" else (255, 244, 214)
                if dry:
                    yy = self._wrap_h(cv, txt, wmax - 22, size)
                else:
                    hh = self._wrap_h(cv, txt, wmax - 22, size)
                    cv.rect((x - 8, y - 6, x + wmax + 6, y + hh + 2), fill=alpha(bgc, a), r=6)
                    cv.rect((x - 8, y - 6, x - 3, y + hh + 2), fill=alpha(col, a), r=2)
                    yy = cv.wrap((x + 8, y), txt, wmax - 22, size, alpha(D.INK, a), "medium") - y
                y += yy + 14
            else:
                if dry:
                    yy = self._wrap_h(cv, txt, wmax, size)
                else:
                    if new:
                        cv.rect((x - 12, y + 2, x - 8, y + size * 1.2), fill=alpha(D.AMB, a))
                    yy = cv.wrap((x, y), txt, wmax, size, alpha(D.INK, a)) - y
                y += yy + 10
        return y

    def _wrap_h(self, cv, txt, width, size):
        words, lines, cur = txt.split(" "), 0, ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if cv.d.textlength(trial, font=D.font(size)) / D.SS <= width or not cur:
                cur = trial
            else:
                lines += 1
                cur = wd
        return (lines + 1) * size * 1.38

    def bottom(self, img, c):
        cv = Canvas(W, H - BOTTOM_Y, (255, 255, 255, 255))
        cv.line([(0, 0), (W, 0)], D.RULE, 2)
        p = c.shot.panel
        t = c.t
        if p == "legend":
            PN.legend(cv, c)
        elif p == "routes":
            PN.route_table(cv, c)
        elif p == "speed":
            PN.speed_deficit(cv, c)
        elif p == "burden":
            PN.burden(cv, c)
        elif p == "burden_live":
            PN.burden(cv, c, compare_prev=True)
        elif p == "burden_prev":
            PN.burden(cv, c, compare_prev=True)
        elif p == "gantt":
            PN.gantt_panel(cv, c)
        elif p == "gantt_preview":
            PN.gantt_panel(cv, c, preview=True)
        elif p in ("approaches", "approaches_end"):
            PN.approaches(cv, c)
        elif p == "j2":
            PN.j2_panel(cv, c)
        elif p == "hospital":
            if t < 1050:
                PN.speed_deficit(cv, c)
            elif t < 1083.5:
                PN.reroute(cv, c)
            else:
                PN.hospital(cv, c)
        elif p == "network":
            PN.network(cv, c)
        elif p == "summary":
            summary_panel(cv, c)
        elif p == "none":
            title_panel(cv, c, self)
        im = cv.down()
        img.paste(im, (0, BOTTOM_Y), im)

    def attribution(self, img):
        y = MY + MH
        cv = Canvas(MW, BOTTOM_Y - y, (255, 255, 255, 255))
        cv.text((268, 14), "Map imagery: fixed Google Maps captures.  Vehicles, signals and routes: recorded SUMO replay drawn on top.", 13, D.MUTED, anchor="lm")
        im = cv.down()
        img.paste(im, (0, y), im)
        img.paste(self.base.logo, (12, y + 2))
        img.paste(self.base.credit, (134, y + 6))


def blend(a, b, w, arc=False):
    """Interpolate cameras; zoom is geometric. With arc=True a long move also zooms out
    mid-way, so the apparent on-screen pan speed stays moderate."""
    out = np.array(a, float) * (1 - w) + np.array(b, float) * w
    out[2] = math.exp(math.log(a[2]) * (1 - w) + math.log(b[2]) * w)
    if arc:
        dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        k = float(np.clip(dist / (1.5 * max(a[2], b[2])) - 0.3, 0, 1.5))
        out[2] *= 1 + k * 4 * w * (1 - w)
    return out


def title_panel(cv, c, film):
    PN.title(cv, 30, 14, "Chapters")
    x, y = 30, 50
    for k, s in enumerate(film.shots[1:], 1):
        col = k // 6
        row = (k - 1) % 6 if k < 7 else (k - 7) % 6
        xx = 30 + ((k - 1) // 6) * 640
        yy = 50 + ((k - 1) % 6) * 40
        start = film.starts[k]
        cv.text((xx, yy), f"{D.fmt_mmss(start):>5}", 16, D.MUTED, "semibold")
        cv.text((xx + 60, yy), s.chapter, 16, D.INK)
    cv.text((30, 305), f"Total length {D.fmt_mmss(film.total)}.  Source: AmbuGo_20s_Reproducible package (P3/P0 replays, controller log, decisions), paper draft v9.1.", 13, D.MUTED)


def summary_panel(cv, c):
    PN.title(cv, 30, 12, "What this film does not establish")
    lim = [
        "A single synthetic episode (seed 19, case B1000) on real geometry with assumed demand and signal plans — not a field measurement.",
        "Ĉ (Eq. 20) is a ranking surrogate; its closeness to the realised +843 veh·s here is illustrative, not a validation.",
        "Differences against the no-priority run include all three ambulance missions; they are not attributable to a002 alone.",
        "No claim of global route optimality, guaranteed response time, minimum aftereffects or complete recovery.",
    ]
    y = 46
    for s in lim:
        cv.circle((40, y + 10), 4, fill=D.MUTED)
        cv.wrap((54, y), s, 860, 15, D.INK)
        y += 44 if len(s) > 110 else 26
    x = 1000
    PN.title(cv, x, 12, "Reproduce this film")
    cmds = ["pip install -r requirements.txt", "python src/verify.py        # checks every on-screen number", "python src/render.py --preview", "python src/render.py         # output/AmbuGo_Explainer.mp4"]
    y = 50
    for s in cmds:
        cv.rect((x, y - 4, 1880, y + 24), fill=(244, 246, 248), r=4)
        cv.text((x + 10, y), s, 15, D.INK, "mono")
        y += 36
    cv.text((x, y + 6), "Inputs are the frozen replays and logs of the AmbuGo_20s_Reproducible package; nothing is re-simulated.", 13, D.MUTED)


# ------------------------------------------------------------------ workers
_film = None


def _init():
    global _film
    _film = Film()


def _render_chunk(args):
    f0, f1, path = args
    ff = shutil.which("ffmpeg")
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
           "-g", "60", "-threads", "1", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(f0, f1):
        p.stdin.write(_film.frame(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed on " + str(path))
    return f1 - f0


def render(jobs, out):
    out.mkdir(parents=True, exist_ok=True)
    film = Film()
    n = film.nframes
    print(f"{len(film.shots)} chapters, {film.total:.1f} s, {n} frames", flush=True)
    chunk = 300
    parts = []
    tmp = out / "_parts"
    tmp.mkdir(exist_ok=True)
    for i, f0 in enumerate(range(0, n, chunk)):
        parts.append((f0, min(n, f0 + chunk), tmp / f"part_{i:04d}.mp4"))
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(jobs, initializer=_init) as pool:
        for k in pool.imap_unordered(_render_chunk, parts):
            done += k
            el = time.time() - t0
            print(f"  {done}/{n} frames  ({el / 60:.1f} min, eta {(n - done) * el / max(done, 1) / 60:.1f} min)", flush=True)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p[2].name}'\n" for p in parts))
    final = out / "AmbuGo_Explainer.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    shutil.rmtree(tmp)
    chapters = [{"start_s": round(float(film.starts[i]), 2), "end_s": round(float(film.starts[i + 1]), 2),
                 "chapter": s.chapter, "simulation_s": s.sim if isinstance(s.sim, tuple) else [s.sim, s.sim]}
                for i, s in enumerate(film.shots)]
    (out / "chapters.json").write_text(json.dumps({"duration_s": film.total, "fps": FPS, "frames": n,
                                                   "chapters": chapters}, indent=2))
    lines = [f"{D.fmt_mmss(c['start_s']):>6}  {c['chapter']}" for c in chapters]
    (out / "chapters.txt").write_text("\n".join(lines) + "\n")
    film.frame(int(film.starts[4] * FPS + 12 * FPS)).save(out / "poster.jpg", quality=92)
    print(final)
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="one still per chapter")
    ap.add_argument("--stills", type=float, nargs="*", help="stills at these video seconds")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--output", type=Path, default=OUT)
    a = ap.parse_args()
    if a.preview or a.stills:
        film = Film()
        d = a.output / "preview"
        d.mkdir(parents=True, exist_ok=True)
        times = a.stills or [film.starts[i] + film.shots[i].dur * .8 for i in range(len(film.shots))]
        for vt in times:
            t0 = time.time()
            im = film.frame(int(vt * FPS))
            p = d / f"still_{vt:07.2f}.jpg"
            im.save(p, quality=92)
            print(p, f"{time.time() - t0:.2f}s")
        print(f"film length {film.total:.1f} s")
        return
    render(a.jobs, a.output)


if __name__ == "__main__":
    main()
