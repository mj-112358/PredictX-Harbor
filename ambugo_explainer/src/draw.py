"""Drawing primitives: a 2x supersampled RGBA canvas, fonts, LaTeX-style equations, charts."""
from pathlib import Path
import io
import math
import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ambugo-mpl"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

SS = 2  # supersampling factor for overlays

# ----------------------------------------------------------------- palette
INK = (28, 34, 42)
MUTED = (96, 106, 118)
FAINT = (160, 168, 178)
RULE = (222, 226, 231)
PAPER = (250, 250, 248)
PANEL = (255, 255, 255)
AMB = (25, 98, 214)          # ambulance / its approach / AmbuGo series
BASE = (120, 128, 140)       # no-priority baseline series
NW_C = (142, 68, 173)        # NW approach (violet)
SW_C = (230, 126, 34)        # SW approach (orange)
NE_C = (22, 160, 133)        # NE approach (teal)
RED = (214, 39, 40)
GREEN = (24, 150, 72)
AMBERC = (240, 165, 0)
PED = (120, 120, 120)
ESC = (232, 93, 4)           # escalated (aggressive) mode
OK = (24, 150, 72)
BAD = (214, 39, 40)
APP_COL = {"335061218": AMB, "172380615#1": NW_C, "172382686#4": SW_C, "172660580#0": NE_C}

FONT_DIR = Path("/usr/share/fonts/opentype/inter")
_fonts = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _fonts:
        names = {"regular": "Inter-Regular.otf", "medium": "Inter-Medium.otf",
                 "semibold": "Inter-SemiBold.otf", "bold": "Inter-Bold.otf",
                 "italic": "Inter-Italic.otf", "mono": None}
        path = None
        if weight == "mono":
            for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
                if Path(p).exists():
                    path = p
        else:
            p = FONT_DIR / names[weight]
            if p.exists():
                path = str(p)
        if path is None:
            from matplotlib import font_manager
            path = font_manager.findfont(FontProperties(family="DejaVu Sans",
                                                        weight="bold" if weight in ("bold", "semibold") else "normal"))
        _fonts[key] = ImageFont.truetype(path, int(size * SS))
    return _fonts[key]


def mix(c1, c2, u):
    u = float(np.clip(u, 0, 1))
    return tuple(int(round(a * (1 - u) + b * u)) for a, b in zip(c1[:3], c2[:3]))


def alpha(c, a):
    return (c[0], c[1], c[2], int(np.clip(a, 0, 1) * 255))


def diverging(v, vmax=40.0):
    """Blue (less waiting than baseline) - light grey - red (more waiting)."""
    u = float(np.clip(v / vmax, -1, 1))
    mid = (205, 208, 213)
    if u >= 0:
        return mix(mid, (200, 20, 30), u ** 0.75)
    return mix(mid, (20, 90, 200), (-u) ** 0.75)


# ----------------------------------------------------------- equations
_eq_cache = {}


def equation(tex, size=22, color=INK):
    key = (tex, size, color)
    if key in _eq_cache:
        return _eq_cache[key]
    import matplotlib.pyplot as plt
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(0.01, 0.01), dpi=72 * SS)
    fig.text(0, 0, tex, fontsize=size, color=tuple(c / 255 for c in color))
    buf = io.BytesIO()
    fig.savefig(buf, dpi=72 * SS, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    _eq_cache[key] = im
    return im


# --------------------------------------------------------------- canvas
class Canvas:
    """Draw in 1x coordinates on a 2x RGBA layer; downsample for antialiasing."""

    def __init__(self, w, h, bg=(0, 0, 0, 0)):
        self.w, self.h = w, h
        self.im = Image.new("RGBA", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.im)

    def P(self, p):
        return (p[0] * SS, p[1] * SS)

    def line(self, pts, fill, width=2, joint="curve"):
        pts = [self.P(p) for p in pts]
        if len(pts) >= 2:
            self.d.line(pts, fill=fill, width=max(1, int(round(width * SS))), joint=joint)

    def poly(self, pts, fill=None, outline=None, width=1):
        self.d.polygon([self.P(p) for p in pts], fill=fill, outline=outline,
                       width=max(1, int(round(width * SS))) if outline else 0)

    def circle(self, p, r, fill=None, outline=None, width=1):
        x, y = p
        self.d.ellipse(((x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS), fill=fill,
                       outline=outline, width=max(1, int(round(width * SS))) if outline else 0)

    def rect(self, box, fill=None, outline=None, width=1, r=0):
        x0, y0, x1, y1 = box
        b = (min(x0, x1) * SS, min(y0, y1) * SS, max(x0, x1) * SS, max(y0, y1) * SS)
        if r:
            self.d.rounded_rectangle(b, int(r * SS), fill=fill, outline=outline,
                                     width=max(1, int(round(width * SS))) if outline else 0)
        else:
            self.d.rectangle(b, fill=fill, outline=outline,
                             width=max(1, int(round(width * SS))) if outline else 0)

    def text(self, p, s, size=18, fill=INK, weight="regular", anchor="la"):
        self.d.text(self.P(p), s, font=font(size, weight), fill=fill, anchor=anchor)

    def tsize(self, s, size=18, weight="regular"):
        b = self.d.textbbox((0, 0), s, font=font(size, weight))
        return (b[2] - b[0]) / SS, (b[3] - b[1]) / SS

    def rich(self, p, parts, size=18, anchor_y="top"):
        """Draw a sequence of (text, color, weight) on one line. Returns end x."""
        x, y = p
        for s, c, w in parts:
            self.text((x, y), s, size, c, w)
            x += self.d.textlength(s, font=font(size, w)) / SS
        return x

    def wrap(self, p, s, width, size=18, fill=INK, weight="regular", lh=1.38):
        words = s.split(" ")
        lines, cur = [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if self.d.textlength(trial, font=font(size, weight)) / SS <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        x, y = p
        for ln in lines:
            self.text((x, y), ln, size, fill, weight)
            y += size * lh
        return y

    def paste(self, im, p, anchor="la", opacity=1.0):
        w, h = im.size
        x, y = p[0] * SS, p[1] * SS
        if anchor[0] == "m":
            x -= w / 2
        elif anchor[0] == "r":
            x -= w
        if anchor[1] == "m":
            y -= h / 2
        elif anchor[1] == "b":
            y -= h
        if opacity < 1:
            im = im.copy()
            im.putalpha(im.getchannel("A").point(lambda a: int(a * opacity)))
        self.im.alpha_composite(im, (int(round(x)), int(round(y))))
        return w / SS, h / SS

    def eq(self, p, tex, size=22, color=INK, anchor="la", opacity=1.0):
        return self.paste(equation(tex, size, color), p, anchor, opacity)

    def arrow(self, a, b, fill, width=3, head=12):
        a, b = np.array(a, float), np.array(b, float)
        v = b - a
        L = np.linalg.norm(v)
        if L < 1:
            return
        u = v / L
        n = np.array([-u[1], u[0]])
        self.line([tuple(a), tuple(b - u * head * .6)], fill, width)
        self.poly([tuple(b), tuple(b - u * head + n * head * .55), tuple(b - u * head - n * head * .55)], fill=fill)

    def down(self):
        return self.im.resize((self.w, self.h), Image.Resampling.LANCZOS)


# ------------------------------------------------------------------ charts
class Chart:
    def __init__(self, cv, box, xlim, ylim):
        self.cv, self.box, self.xlim, self.ylim = cv, box, xlim, ylim

    def X(self, x):
        x0, y0, x1, y1 = self.box
        return x0 + (np.asarray(x, float) - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * (x1 - x0)

    def Y(self, y):
        x0, y0, x1, y1 = self.box
        return y1 - (np.asarray(y, float) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * (y1 - y0)

    def axes(self, xticks=(), yticks=(), xlabel=None, ylabel=None, xfmt=str, yfmt=str, grid=True, size=13):
        cv = self.cv
        x0, y0, x1, y1 = self.box
        for v in yticks:
            y = float(self.Y(v))
            if grid:
                cv.line([(x0, y), (x1, y)], RULE, 1)
            cv.text((x0 - 6, y), yfmt(v), size, MUTED, anchor="rm")
        for v in xticks:
            x = float(self.X(v))
            cv.line([(x, y1), (x, y1 + 4)], FAINT, 1)
            cv.text((x, y1 + 6), xfmt(v), size, MUTED, anchor="ma")
        cv.line([(x0, y1), (x1, y1)], FAINT, 1.2)
        cv.line([(x0, y0), (x0, y1)], FAINT, 1.2)
        if xlabel:
            cv.text((x1, y1 + 22), xlabel, size, MUTED, anchor="ra")
        if ylabel:
            cv.text((x0, y0 - 18), ylabel, size, MUTED, anchor="la")

    def series(self, xs, ys, color, width=2.5, step=False):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(ys)
        if step and len(xs) > 1:
            xx = np.repeat(xs, 2)[1:]
            yy = np.repeat(ys, 2)[:-1]
            xs, ys, ok = xx, yy, np.isfinite(yy)
        # split on NaN gaps
        idx = np.flatnonzero(ok)
        if not len(idx):
            return
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, len(idx) - 1]
        for s, e in zip(starts, ends):
            sel = idx[s:e + 1]
            pts = list(zip(self.X(xs[sel]).tolist(), self.Y(np.clip(ys[sel], *self.ylim)).tolist()))
            if len(pts) == 1:
                self.cv.circle(pts[0], width, fill=color)
            else:
                self.cv.line(pts, color, width)

    def fill_between(self, xs, y_a, y_b, pos_color, neg_color=None):
        xs = np.asarray(xs, float)
        ya, yb = np.asarray(y_a, float), np.asarray(y_b, float)
        for i in range(len(xs) - 1):
            d = ya[i] - yb[i]
            if d == 0:
                continue
            c = pos_color if d > 0 else (neg_color or pos_color)
            if c is None:
                continue
            x0, x1 = float(self.X(xs[i])), float(self.X(xs[i + 1]))
            self.cv.poly([(x0, float(self.Y(ya[i]))), (x1, float(self.Y(ya[i]))),
                          (x1, float(self.Y(yb[i]))), (x0, float(self.Y(yb[i])))], fill=c)

    def hline(self, y, color, width=1.5, label=None, size=13):
        x0, _, x1, _ = self.box
        yy = float(self.Y(y))
        self.cv.line([(x0, yy), (x1, yy)], color, width)
        if label:
            self.cv.text((x1 - 4, yy - 4), label, size, color, "semibold", anchor="rb")

    def cursor(self, x, color=MUTED):
        _, y0, _, y1 = self.box
        xx = float(self.X(x))
        self.cv.line([(xx, y0), (xx, y1)], alpha(color, .7), 1.5)

    def dot(self, x, y, color, r=5):
        if np.isfinite(y):
            self.cv.circle((float(self.X(x)), float(self.Y(np.clip(y, *self.ylim)))), r, fill=color, outline=(255, 255, 255), width=1.5)


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t * t * t * (10 + t * (-15 + 6 * t))


def fade(u, a, b):
    """0 before a, 1 after b, smooth in between."""
    if b <= a:
        return 1.0 if u >= a else 0.0
    return smooth((u - a) / (b - a))


def fmt_mmss(s):
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"
