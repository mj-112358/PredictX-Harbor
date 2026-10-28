"""Map panel: Google Maps captures under a shared Web Mercator camera, plus replay overlays."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from data import INP, llxy
import draw as D
from draw import Canvas

MX, MY, MW, MH = 0, 56, 1250, 660   # map viewport on the 1920x1080 frame
CAPTURES = [("far", 76.7226514, 30.7042347, 14), ("wide", 76.7226514, 30.7042347, 15),
            ("middle", 76.7245, 30.7065, 16), ("junction_mid", 76.7129, 30.71042, 17),
            ("hospital", 76.73105, 30.6951, 17), ("junction", 76.7129, 30.71042, 18)]
LANE_W = 3.2 / math.cos(math.radians(30.7))   # ground lane width expressed in Mercator metres


class Basemap:
    def __init__(self):
        self.patches = []
        for name, lon, lat, z in CAPTURES:
            im = Image.open(INP / "maps" / f"{name}.png").convert("RGBA")
            self.patches.append({"name": name, "image": im, "center": llxy(lon, lat),
                                 "res": 156543.03392804097 / 2 ** z, "z": z})
        src = Image.open(INP / "maps" / "junction.png").convert("RGB")
        # Native Google Maps logo and copyright wording, shown at fixed size beside the map.
        self.logo = src.crop((620, 677, 735, 701)).resize((115, 24), Image.Resampling.LANCZOS)
        self.credit = src.crop((754, 703, 878, 720)).resize((124, 17), Image.Resampling.LANCZOS)
        self.cache = {}

    def render(self, cam):
        key = tuple(np.round(cam, 1))
        if key in self.cache:
            return self.cache[key]
        cx, cy, hw = cam
        out = Image.new("RGBA", (MW, MH), (238, 238, 236, 255))
        worldres = 2 * hw / MW
        for p in self.patches:
            r = p["res"]
            a = worldres / r
            left = cx - hw
            top = cy + hw * MH / MW
            x = (left - p["center"][0]) / r + 640
            y = (p["center"][1] - top) / r + 360
            x2, y2 = x + MW * a, y + MH * a
            margin = min(x - 82, 1268 - x2, y - 74, 672 - y2)
            for rx1, ry1, rx2, ry2 in [(72, 610, 188, 720), (1103, 669, 1280, 720), (1212, 565, 1280, 720)]:
                margin = min(margin, max(rx1 - x2, x - rx2, ry1 - y2, y - ry2))
            opacity = 1.0 if p["z"] == 14 else float(D.smooth(margin / 18))
            if opacity < 0.001:
                continue
            patch = p["image"].transform((MW, MH), Image.Transform.AFFINE, (a, 0, x, 0, a, y),
                                         Image.Resampling.BICUBIC)
            if opacity < 1:
                patch.putalpha(patch.getchannel("A").point(lambda v: int(v * opacity)))
            out = Image.alpha_composite(out, patch)
        rgb = out.convert("RGB")
        # Quiet the basemap so the colour-coded overlays carry the meaning.
        grey = ImageOps.grayscale(rgb).convert("RGB")
        rgb = Image.blend(rgb, grey, 0.55)
        rgb = Image.blend(rgb, Image.new("RGB", rgb.size, (255, 255, 255)), 0.30)
        if len(self.cache) > 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = rgb
        return rgb


class MapLayer:
    """Overlay drawing in screen coordinates for one camera."""

    def __init__(self, ep, cam):
        self.ep = ep
        self.cam = np.asarray(cam, float)
        self.cv = Canvas(MW, MH)
        self.scale = MW / (2 * self.cam[2])  # screen px per Mercator metre

    def S(self, xy):
        xy = np.asarray(xy, float)
        cx, cy, hw = self.cam
        return np.stack(((xy[..., 0] - cx) * self.scale + MW / 2,
                         -(xy[..., 1] - cy) * self.scale + MH / 2), axis=-1)

    def visible(self, pts, pad=40):
        p = self.S(pts)
        return (p[:, 0].max() > -pad and p[:, 0].min() < MW + pad and
                p[:, 1].max() > -pad and p[:, 1].min() < MH + pad)

    # ---------------------------------------------------------- network
    def roads(self, color=(150, 156, 166), alpha_=150):
        """Simulated road links drawn at their modelled lane widths."""
        cv = self.cv
        if self.scale < 0.5:   # network-wide views: darker, thicker so the simulated network reads
            color, alpha_ = (84, 94, 110), 215
        for eid, pts in self.ep.exy.items():
            if eid.startswith(":") or not self.visible(pts):
                continue
            lanes = self.ep.edges[eid]["lanes"]
            w = float(np.clip(lanes * LANE_W * self.scale, 2.2 if self.scale < 0.5 else 1.4, 60))
            cv.line([tuple(p) for p in self.S(pts)], D.alpha(color, alpha_ / 255), w)

    def edge_band(self, eid, color, a=0.55, extra=6, width=None):
        pts = self.ep.exy[eid]
        lanes = self.ep.edges[eid]["lanes"]
        w = width if width is not None else float(np.clip(lanes * LANE_W * self.scale + extra, 4, 90))
        self.cv.line([tuple(p) for p in self.S(pts)], D.alpha(color, a), w)

    def route(self, pts_world, color, width=7, casing=True, arrows=False, a=1.0):
        pts = self.S(pts_world)
        ps = [tuple(p) for p in pts]
        if casing:
            self.cv.line(ps, (255, 255, 255, int(220 * a)), width + 5)
        self.cv.line(ps, D.alpha(color, a), width)
        if arrows:
            delta = np.diff(pts, axis=0)
            length = np.linalg.norm(delta, axis=1)
            cum = np.r_[0, np.cumsum(length)]
            for s in np.arange(60, cum[-1], 120):
                j = min(len(delta) - 1, max(0, np.searchsorted(cum, s) - 1))
                if length[j] < .01:
                    continue
                u = delta[j] / length[j]
                n = np.array([-u[1], u[0]])
                p = pts[j] + u * (s - cum[j])
                self.cv.line([tuple(p - u * 5 + n * 4), tuple(p), tuple(p - u * 5 - n * 4)], (255, 255, 255, int(255 * a)), 2)

    # --------------------------------------------------------- vehicles
    def vehicle(self, p, ang, L, B, fill, outline=(40, 46, 56), stopped=False, ghost=False):
        rad = math.radians(ang)
        u = np.array([math.sin(rad), -math.cos(rad)])
        n = np.array([u[1], -u[0]])
        p = np.asarray(p) - u * L / 2  # SUMO position is the vehicle front
        shape = [(-L / 2, -B / 2), (-L / 2, B / 2), (L * .38, B / 2), (L / 2, B * .3), (L / 2, -B * .3), (L * .38, -B / 2)]
        pts = [tuple(p + u * x + n * y) for x, y in shape]
        if ghost:
            self.cv.poly(pts, fill=None, outline=(70, 76, 86, 210), width=1.6)
            return
        self.cv.poly(pts, fill=fill, outline=outline, width=1)
        if L >= 9:
            self.cv.poly([tuple(p + u * x + n * y) for x, y in [(L * .12, -B * .38), (L * .12, B * .38), (L * .3, B * .3), (L * .3, -B * .3)]],
                         fill=(255, 255, 255, 120))
        if stopped:
            for k in (-1, 1):
                self.cv.circle(tuple(p - u * L * .47 + n * B * .32 * k), max(1.6, B * .17), fill=D.RED)

    def vehicles(self, t, colour_fn, ghosts=False, label_fn=None, min_len=7):
        ep = self.ep
        vid, xy, ang, spd, edge = ep.P3.positions(t)
        ps = self.S(xy)
        sc = self.scale
        focus_xy = None
        inside = (ps[:, 0] > -30) & (ps[:, 0] < MW + 30) & (ps[:, 1] > -30) & (ps[:, 1] < MH + 30)
        if ghosts:
            gv, gxy, gang, gspd, _ = ep.P0.positions(t)
            gps = self.S(gxy)
            for k, p, a_ in zip(gv, gps, gang):
                if k in ep.P0.amb_idx or not (-30 < p[0] < MW + 30 and -30 < p[1] < MH + 30):
                    continue
                le, wi = ep.P0.dims[k]
                self.vehicle(p, a_, max(le * sc * 1.45, min_len), max(wi * sc * 1.45, min_len * .45), None, ghost=True)
        labels = []
        for k, p, a_, v, e, ok in zip(vid, ps, ang, spd, edge, inside):
            if not ok:
                continue
            if k in ep.P3.amb_idx:
                if k == ep.focus:
                    focus_xy = (p, a_)
                else:
                    self.other_ambulance(p, a_)
                continue
            le, wi = ep.P3.dims[k]
            L = max(le * sc * 1.45, min_len)
            B = max(wi * sc * 1.45, min_len * .45)
            fill = colour_fn(k, e, v)
            if fill is None:
                continue
            self.vehicle(p, a_, L, B, fill, stopped=v < 0.1)
            if label_fn:
                s = label_fn(k, e, v)
                if s:
                    labels.append((p, s, fill))
        placed = []
        for p, s, c in labels:
            w, h = self.cv.tsize(s, 13, "bold")
            box = (p[0] + 8, p[1] - 26, p[0] + 16 + w, p[1] - 6)
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
                continue
            placed.append(box)
            self.cv.rect(box, fill=(255, 255, 255, 225), outline=c, width=1.5, r=4)
            self.cv.text((box[0] + 4, box[1] + 10), s, 13, D.INK, "bold", anchor="lm")
        return focus_xy

    def other_ambulance(self, p, ang):
        self.cv.circle(tuple(p), 11, fill=(255, 255, 255, 230), outline=(110, 120, 135), width=2)
        self.cv.line([(p[0] - 5, p[1]), (p[0] + 5, p[1])], D.RED, 3)
        self.cv.line([(p[0], p[1] - 5), (p[0], p[1] + 5)], D.RED, 3)

    def ambulance(self, p, ang, t, escalated=False, size=1.0):
        p = np.asarray(p, float)
        ring = D.ESC if escalated else D.AMB
        r = 30 * size
        self.cv.circle(tuple(p), r + 6, fill=D.alpha(ring, .18))
        self.cv.circle(tuple(p), r, fill=(255, 255, 255, 240), outline=ring, width=3.5)
        rad = math.radians(ang)
        u = np.array([math.sin(rad), -math.cos(rad)])
        n = np.array([u[1], -u[0]])
        L, B = 34 * size, 16 * size
        q = p - u * 0
        shape = [(-L / 2, -B / 2), (-L / 2, B / 2), (L * .38, B / 2), (L / 2, B * .3), (L / 2, -B * .3), (L * .38, -B / 2)]
        self.cv.poly([tuple(q + u * x + n * y) for x, y in shape], fill=(255, 255, 255), outline=D.INK, width=2)
        c = q - u * 3
        self.cv.line([tuple(c - u * 6), tuple(c + u * 6)], D.RED, 4)
        self.cv.line([tuple(c - n * 6), tuple(c + n * 6)], D.RED, 4)
        blink = int(t * 4) % 2
        self.cv.circle(tuple(q + u * L * .3 + n * (B / 2 + 2)), 3, fill=D.RED if blink else D.AMB)
        self.cv.circle(tuple(q + u * L * .3 - n * (B / 2 + 2)), 3, fill=D.AMB if blink else D.RED)

    # ---------------------------------------------------------- signals
    def stopline(self, eid):
        pts = self.S(self.ep.exy[eid])
        p = pts[-1]
        u = pts[-1] - pts[-2]
        u = u / max(np.linalg.norm(u), 1e-6)
        n = np.array([-u[1], u[0]])
        return p, u, n

    def signal_colour(self, mode, served, eid):
        if mode == 0 and served == eid:
            return D.GREEN
        if mode == 1 and served == eid:
            return D.AMBERC
        return D.RED

    def junction_signals(self, j, t, pol="P3", heads=True, head_scale=1.0, outward=58):
        mode, served, req = self.ep.signal_state(pol, j, t)
        for eid in self.ep.signals[j]["incoming"]:
            p, u, n = self.stopline(eid)
            if not (0 < p[0] < MW and 0 < p[1] < MH):
                continue
            c = self.signal_colour(mode, served, eid)
            lanes = self.ep.edges[eid]["lanes"]
            half = max(9, lanes * LANE_W * self.scale * .62)
            self.cv.line([tuple(p - n * half), tuple(p + n * half)], (255, 255, 255), 9)
            self.cv.line([tuple(p - n * half), tuple(p + n * half)], c, 6)
            if heads:
                s = head_scale
                q = p - u * 20 - n * (half + outward * s)
                self.cv.line([tuple(p - n * half), tuple(q)], (255, 255, 255, 200), 3)
                self.cv.line([tuple(p - n * half), tuple(q)], (90, 98, 108), 1.2)
                x, y = q
                self.cv.rect((x - 12 * s, y - 31 * s, x + 12 * s, y + 31 * s), fill=(36, 42, 50), outline=(255, 255, 255), width=2, r=5 * s)
                for k, col in enumerate([D.RED, D.AMBERC, D.GREEN]):
                    on = col == c
                    self.cv.circle((x, y + (k - 1) * 19 * s), 7.5 * s, fill=col if on else (75, 82, 90))
        return mode, served, req

    def small_signals(self, j, t, pol="P3"):
        mode, served, req = self.ep.signal_state(pol, j, t)
        for eid in self.ep.signals[j]["incoming"]:
            p, u, n = self.stopline(eid)
            if not (-10 < p[0] < MW + 10 and -10 < p[1] < MH + 10):
                continue
            c = self.signal_colour(mode, served, eid)
            self.cv.circle(tuple(p), 5, fill=c, outline=(255, 255, 255), width=1.5)

    # ------------------------------------------------------------ labels
    def tag(self, anchor, offset, lines, color, size=14, w=None, fill=(255, 255, 255, 238), line=True):
        """A boxed annotation with a leader line to `anchor` (screen coords)."""
        cv = self.cv
        ax, ay = anchor
        bx, by = ax + offset[0], ay + offset[1]
        lh = size * 1.35
        widths = [cv.tsize(s if isinstance(s, str) else s[0], size, "semibold" if i == 0 else "regular")[0] for i, s in enumerate(lines)]
        bw = (w or max(widths)) + 18
        bh = lh * len(lines) + 10
        x0 = bx - bw / 2
        y0 = by - bh / 2
        x0 = float(np.clip(x0, 6, MW - bw - 6))
        y0 = float(np.clip(y0, 6, MH - bh - 6))
        if line:
            cx = float(np.clip(ax, x0, x0 + bw))
            cy = float(np.clip(ay, y0, y0 + bh))
            cv.line([(ax, ay), (cx, cy)], (255, 255, 255, 220), 4)
            cv.line([(ax, ay), (cx, cy)], color, 2)
            cv.circle((ax, ay), 4, fill=color, outline=(255, 255, 255), width=1.5)
        cv.rect((x0, y0, x0 + bw, y0 + bh), fill=fill, outline=color, width=2, r=7)
        y = y0 + 6
        for i, s in enumerate(lines):
            if isinstance(s, str):
                cv.text((x0 + 9, y), s, size, color if i == 0 else D.INK, "semibold" if i == 0 else "regular")
            else:
                cv.text((x0 + 9, y), s[0], size, s[1], s[2] if len(s) > 2 else "regular")
            y += lh
        return (x0, y0, x0 + bw, y0 + bh)

    def pin(self, xy, text, color, size=15, dy=-26):
        p = self.S(xy)
        self.cv.circle(tuple(p), 7, fill=color, outline=(255, 255, 255), width=2.5)
        w, h = self.cv.tsize(text, size, "semibold")
        x, y = p[0], p[1] + dy
        self.cv.rect((x - w / 2 - 8, y - h / 2 - 6, x + w / 2 + 8, y + h / 2 + 6), fill=(255, 255, 255, 235), outline=color, width=2, r=6)
        self.cv.text((x, y), text, size, color, "semibold", anchor="mm")

    def done(self):
        return self.cv.down()
