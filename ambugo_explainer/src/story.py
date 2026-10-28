"""Storyboard: chapters, timing, camera, map overlays, narration and bottom panels."""
import math

import numpy as np

import draw as D
import panels as PN
from draw import alpha, fade
from mapview import MW, MH

FPS = 30


class Shot:
    def __init__(self, key, chapter, dur, sim, cams, overlay, panel, narration, ramp=0.12):
        self.key, self.chapter, self.dur = key, chapter, dur
        self.sim = sim            # (t0, t1) or a single frozen time
        self.cams = cams          # list of (u, cam) ; cam = (cx,cy,hw) | ("track", hw) | "prev"
        self.overlay, self.panel, self.narration = overlay, panel, narration
        self.ramp = ramp

    def sim_time(self, u):
        if not isinstance(self.sim, tuple):
            return float(self.sim)
        t0, t1 = self.sim
        r = self.ramp
        # trapezoidal velocity profile: smooth start and stop, constant rate in between
        if r <= 0:
            s = u
        else:
            vmax = 1 / (1 - r)
            if u < r:
                s = vmax * u * u / (2 * r)
            elif u > 1 - r:
                s = 1 - vmax * (1 - u) ** 2 / (2 * r)
            else:
                s = vmax * (r / 2 + (u - r))
        return t0 + (t1 - t0) * s


class Ctx:
    pass


# ----------------------------------------------------------------- colours
def approach_colour(ep):
    apps = {ep.ei[e]: D.APP_COL[e] for e in [ep.amb_app] + ep.conf}

    def f(k, e, v):
        if e in apps:
            return apps[e]
        return (96, 104, 116) if v < 0.1 else (150, 158, 170)
    return f


def delta_colour(ep, t):
    dw = ep.extra_wait(t)

    def f(k, e, v):
        return D.diverging(float(dw[k]))
    return f


def delta_label(ep, t, thr=5):
    dw = ep.extra_wait(t)

    def f(k, e, v):
        x = float(dw[k])
        return f"{x:+.0f} s" if abs(x) >= thr else None
    return f


def plain_colour(k, e, v):
    return (90, 98, 110) if v < 0.1 else (140, 148, 160)


# ----------------------------------------------------------------- overlays
def ov_title(ml, ctx):
    ml.roads(alpha_=120)
    ml.cv.rect((0, 0, MW, MH), fill=(255, 255, 255, 228))
    cv = ml.cv
    cv.text((70, 150), "AmbuGo", 64, D.AMB, "bold")
    cv.text((70, 235), "How one selective signal-priority decision", 34, D.INK, "semibold")
    cv.text((70, 280), "propagates through an urban traffic network", 34, D.INK, "semibold")
    cv.text((70, 350), "A slow, step-by-step replay of one recorded SUMO episode on mapped Mohali roads,", 21, D.MUTED)
    cv.text((70, 380), "with the controller's own numbers shown beside every visual change.", 21, D.MUTED)
    cv.text((70, 450), "Case B1000 · ambulance a002 · seed 19 · paired against a no-priority run", 18, D.MUTED, "semibold")
    cv.text((70, 520), "Synthetic traffic and assumed signal plans on real road geometry — an explanation of the", 17, D.MUTED)
    cv.text((70, 545), "mechanism, not a field measurement.", 17, D.MUTED)


def ov_network(ml, ctx):
    ep, t = ctx.ep, ctx.t
    ml.roads(alpha_=170)
    a = fade(ctx.ls, 1, 2.5)
    for j, s in ep.signals.items():
        nm = {ep.J1: "J1 · PCL Chowk", ep.J2: "J2", ep.J3: "J3"}.get(j)
        xy = np.array([s["lon"], s["lat"]])
        from data import llxy
        p = ml.S(llxy(*xy))
        ml.cv.circle(tuple(p), 9, fill=(255, 255, 255), outline=D.INK, width=2.5)
        ml.cv.circle(tuple(p), 4, fill=D.INK)
        if nm and a > 0:
            ml.cv.text((p[0] + 14, p[1] - 4), nm, 16, alpha(D.INK, a), "bold", anchor="lb")
    b = fade(ctx.ls, 3, 4.5)
    if b > 0:
        ml.pin(ep.hosp, "Fortis hospital road approach", D.GREEN)
        ml.pin(ep.amb_pos(760.5), "a002 dispatched, t = 760 s", D.AMB, dy=30)
    c = fade(ctx.ls, 6, 7.5)
    if c > 0:
        for col in (0, 1):
            p = ep.amb_pos(t, col)
            if np.isfinite(p).all():
                q = ml.S(p)
                ml.other_ambulance(q, 0)
                ml.cv.text((q[0] + 14, q[1] + 4), ["a000 (en route)", "a001 (en route)"][col], 14, alpha(D.MUTED, c), "semibold")
    ml.vehicles(t, plain_colour, min_len=3)


ROUTE_COLS = [(155, 89, 182), D.AMB, (39, 174, 96), (230, 126, 34)]


def ov_routes(ml, ctx):
    ep = ctx.ep
    ml.roads(alpha_=140)
    routes = ep.decisions["first_decision"]["unique_routes"]
    for i, r in enumerate(routes):
        a = fade(ctx.ls, 1 + i * 1.2, 2 + i * 1.2)
        if a <= 0:
            continue
        pts = ep.route_points(r["route"])
        off = (i - 1.5) * 9 / ml.scale
        # lateral offset so overlapping corridors stay distinguishable
        d = np.gradient(pts, axis=0)
        n = np.stack((-d[:, 1], d[:, 0]), 1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        ml.route(pts + n * off, ROUTE_COLS[i], 9 if r["chosen"] else 5, casing=r["chosen"], a=a)
    for i, r in enumerate(routes):
        a = fade(ctx.ls, 1 + i * 1.2, 2 + i * 1.2)
        if a <= 0:
            continue
        sec = r["forecast_with_priority_s"]
        lab = f"{r['route_id']}  {sec:.1f} s" + ("  ✓ chosen" if r["chosen"] else "")
        p = (60, 40 + i * 34)
        ml.cv.rect((p[0] - 12, p[1] - 4, p[0] + 260, p[1] + 26), fill=(255, 255, 255, int(235 * a)), outline=alpha(ROUTE_COLS[i], a), width=2, r=6)
        ml.cv.text((p[0], p[1]), lab, 18, alpha(ROUTE_COLS[i], a), "bold")
    ml.pin(ep.hosp, "H", D.GREEN, dy=-24)
    p = ml.S(ep.amb_pos(ctx.t))
    ml.ambulance(p, 330, ctx.t, size=.75)


def junction_scene(ml, ctx, heads=True, colour="approach", ghosts=False, labels=False, tags=None, bands=1.0):
    ep, t = ctx.ep, ctx.t
    ml.roads(alpha_=150)
    for e in [ep.amb_app] + ep.conf:
        ml.edge_band(e, D.APP_COL[e], a=.42 * bands, extra=10)
    rt = ep.amb_route(t)
    if rt:
        ml.route(ep.route_points(rt), D.AMB, 4, casing=False, arrows=False, a=.55)
    for j in ep.signals:
        if j != ep.J1:
            ml.small_signals(j, t)
    mode, served, req = ml.junction_signals(ep.J1, t, heads=heads)
    if colour == "approach":
        cf = approach_colour(ep)
        lf = None
    elif colour == "delta":
        cf = delta_colour(ep, t)
        lf = delta_label(ep, t) if labels else None
    else:
        cf, lf = plain_colour, None
    focus = ml.vehicles(t, cf, ghosts=ghosts, label_fn=lf)
    signal_card(ml, ctx, mode, served)
    return focus, mode, served, req


def signal_card(ml, ctx, mode, served):
    """Fixed card: what PCL Chowk is doing now and for how long."""
    ep, t = ctx.ep, ctx.t
    seg = next(((a, b, m, e) for a, b, m, e in ep.timeline["P3", ep.J1] if a <= t < b), None)
    names = {0: "green", 1: "amber", 2: "all-red clearance", 3: "pedestrian all-red reservation"}
    if seg is None:
        return
    a, b, m, e = seg
    s1 = "PCL Chowk now:"
    if m in (0, 1):
        s2 = f"{ep.app_name.get(e, '?')} {names[m]}"
        col = D.APP_COL.get(e, D.INK) if m == 0 else (170, 110, 0)
    else:
        s2 = names[m]
        col = D.INK
    s3 = f"since {a:.1f} s · ends {b:.1f} s ({max(0, b - t):.1f} s left)"
    ml.cv.rect((14, 14, 400, 92), fill=(255, 255, 255, 240), outline=(200, 205, 212), width=1.5, r=8)
    ml.cv.text((28, 24), s1, 14, D.MUTED, "semibold")
    ml.cv.text((28, 42), s2, 19, col, "bold")
    ml.cv.text((28, 68), s3, 13, D.MUTED)


def draw_focus(ml, ctx, focus, size=.8):
    ep, t = ctx.ep, ctx.t
    esc = bool(ep.P3.a["fallback"][ep.k(t), ep.focus_col])
    if focus is None:
        p = ep.amb_pos(t)
        if not np.isfinite(p).all():
            return None
        focus = (ml.S(p), 330)
    ml.ambulance(focus[0], focus[1], t, escalated=esc, size=size)
    return focus[0]


def request_badge(ml, ctx, p):
    ep, t = ctx.ep, ctx.t
    e = ep.event_at(t)
    if e is None or p is None or e["approach"] != ep.amb_app:
        return
    if e["reason"] == "burden_budget_exceeded":
        s, col = "REQUEST WITHHELD  ✗   ΣĈ > B", D.BAD
    elif e["reason"] == "selective_gate_passed":
        s, col = "REQUEST ADMITTED  ✓   lease starts", D.AMB
    else:
        s, col = "LEASE ACTIVE  ·  request forwarded", D.AMB
    w, _ = ml.cv.tsize(s, 15, "bold")
    x, y = p[0] - w - 70, p[1] - 15
    if x < 10:
        x = p[0] + 44
    ml.cv.rect((x, y, x + w + 24, y + 30), fill=(255, 255, 255, 240), outline=col, width=2.5, r=7)
    ml.cv.text((x + 12, y + 15), s, 15, col, "bold", anchor="lm")


def app_tags(ml, ctx, show=True, which=None, highlight_changes=False):
    ep, t = ctx.ep, ctx.t
    e = ep.event_at(t)
    if e is None:
        return
    prev = ep.ev.get(e["t"] - 1) if highlight_changes else None
    if highlight_changes and e["t"] == 792:
        prev = ep.ev[791]
    offs = {"172380615#1": (-230, -60), "172382686#4": (-60, 92), "172660580#0": (200, -40), "335061218": (240, 30)}
    for a in e["conflicting_approaches"]:
        nm = ep.app_name[a["edge"]]
        pts = ml.S(ep.exy[a["edge"]])
        anchor = pts[len(pts) // 2] if len(pts) > 2 else (pts[0] * .5 + pts[-1] * .5)
        q = a["queue_proxy"]
        lines = [f"{nm} approach (conflicting)", f"q̂ = {q:.2f} veh   λ = {a['arrival_rate']:.4f}/s",
                 f"X = {a['nominal_design_X']:.3f}   Ĉ = {a['burden_proxy_veh_s']:.0f} veh·s"]
        if prev is not None:
            pa = next(x for x in prev["conflicting_approaches"] if x["edge"] == a["edge"])
            if abs(pa["queue_proxy"] - q) > 1e-9:
                lines[1] = (f"q̂ = {pa['queue_proxy']:.2f} → {q:.2f} veh  (new sample)", D.BAD, "semibold")
        ml.tag(tuple(anchor), offs[a["edge"]], lines, D.APP_COL[a["edge"]], 14)
    pts = ml.S(ep.exy[ep.amb_app])
    anchor = pts[0] * .4 + pts[-1] * .6
    lines = ["SE approach (ambulance)", f"q̂ = {e['queue_proxy']:.2f} veh → d̂ = {e['queue_clearance_s']:.2f} s",
             f"switch delay δ = {e['switch_delay_s']:.0f} s   benefit b = {e['predicted_benefit_s']:.1f} s"]
    if prev is not None and abs(prev["queue_proxy"] - e["queue_proxy"]) > 1e-9:
        lines[1] = (f"q̂ = {prev['queue_proxy']:.2f} → {e['queue_proxy']:.2f};  d̂ → {e['queue_clearance_s']:.2f} s", D.BAD, "semibold")
    ml.tag(tuple(anchor), offs["335061218"], lines, D.AMB, 14)


def ov_approach(ml, ctx):
    focus, *_ = junction_scene(ml, ctx, heads=ctx.cam[2] < 400)
    draw_focus(ml, ctx, focus, .8)


def ov_decision(ml, ctx, highlight=False):
    focus, mode, served, req = junction_scene(ml, ctx)
    p = draw_focus(ml, ctx, focus, .8)
    app_tags(ml, ctx, highlight_changes=highlight)
    request_badge(ml, ctx, p)


def ov_transition(ml, ctx):
    focus, mode, served, req = junction_scene(ml, ctx)
    p = draw_focus(ml, ctx, focus, .8)
    request_badge(ml, ctx, p)
    t = ctx.t
    if 796.5 <= t < 806:
        a = 1 - fade(t, 803, 806)
        ml.cv.rect((14, 102, 530, 144), fill=(255, 255, 255, int(240 * a)), outline=alpha(D.GREEN, a), width=3, r=8)
        ml.cv.text((30, 123), "796.5 s: SE approach green — earliest legal moment", 18, alpha(D.GREEN, a), "bold", anchor="lm")


def ov_shift(ml, ctx):
    focus, *_ = junction_scene(ml, ctx, heads=True)
    draw_focus(ml, ctx, focus, .8)


def ov_consequence(ml, ctx):
    focus, *_ = junction_scene(ml, ctx, heads=True, colour="delta", ghosts=True, labels=True, bands=.6)
    draw_focus(ml, ctx, focus, .7)


def ov_track(ml, ctx):
    ep, t = ctx.ep, ctx.t
    ml.roads(alpha_=150)
    rt = ep.amb_route(t)
    if rt:
        ml.route(ep.route_points(rt), D.ESC if ep.P3.a["fallback"][ep.k(t), ep.focus_col] else D.AMB, 7, arrows=True)
    for j in ep.signals:
        ml.small_signals(j, t)
    focus = ml.vehicles(t, plain_colour, min_len=6)
    p = draw_focus(ml, ctx, focus, .7)
    if ctx.shot.key == "escalate" and p is not None:
        esc = t >= 851.5
        s = "ESCALATED: aggressive priority for the rest of the mission" if esc else f"progress deficit D_a = {ep.D_a[ep.k(t)]:.2f} s  (γ = 20 s)"
        col = D.ESC if esc else D.INK
        w, _ = ml.cv.tsize(s, 15, "bold")
        x, y = min(p[0] + 40, MW - w - 40), p[1] + 36
        ml.cv.rect((x, y, x + w + 24, y + 30), fill=(255, 255, 255, 240), outline=col, width=2.5, r=7)
        ml.cv.text((x + 12, y + 15), s, 15, col, "bold", anchor="lm")


J2_COLS = {"129126240#7": D.AMB, "1215822559#0": (90, 150, 230), "383450897#1": D.NE_C, "383450898#0": D.SW_C}


def ov_j2(ml, ctx):
    ep, t = ctx.ep, ctx.t
    ml.roads(alpha_=150)
    for e, c in J2_COLS.items():
        ml.edge_band(e, c, a=.4, extra=10)
    rt = ep.amb_route(t)
    if rt:
        ml.route(ep.route_points(rt), D.ESC, 4, casing=False, a=.5)
    for j in ep.signals:
        if j not in (ep.J2, ep.J3):
            ml.small_signals(j, t)
    ml.junction_signals(ep.J2, t, heads=True, head_scale=.8, outward=40)
    ml.junction_signals(ep.J3, t, heads=False)
    apps = {ep.ei[e]: c for e, c in J2_COLS.items()}
    focus = ml.vehicles(t, lambda k, e, v: apps.get(e, plain_colour(k, e, v)))
    p = draw_focus(ml, ctx, focus, .7)
    e = ep.event_at(t)
    if e is not None and p is not None and e["t"] >= 971:
        s = f"ESCALATED REQUEST  ·  δ = {e['switch_delay_s']:.0f} s  ·  B not applied"
        w, _ = ml.cv.tsize(s, 15, "bold")
        x, y = min(p[0] + 40, MW - w - 40), p[1] + 34
        ml.cv.rect((x, y, x + w + 24, y + 30), fill=(255, 255, 255, 240), outline=D.ESC, width=2.5, r=7)
        ml.cv.text((x + 12, y + 15), s, 15, D.ESC, "bold", anchor="lm")
    # who else is here
    for col in (0, 1):
        q = ep.amb_pos(t, col)
        if np.isfinite(q).all():
            s = ml.S(q)
            if 0 < s[0] < MW and 0 < s[1] < MH:
                ml.cv.text((s[0] + 14, s[1] - 14), ["a000", "a001 (escalated)"][col], 14, D.MUTED, "bold")


def ov_hospital(ml, ctx):
    ep, t = ctx.ep, ctx.t
    ml.roads(alpha_=150)
    u = ep.decisions["route_updates"][-1]
    if t >= 1055:
        ml.route(ep.route_points(u["incumbent_route"][-4:]), D.BASE, 7, a=.9)
        ml.route(ep.route_points(u["chosen_route"][-4:]), D.ESC, 9, arrows=True)
    else:
        rt = ep.amb_route(t)
        if rt:
            ml.route(ep.route_points(rt), D.ESC, 7, arrows=True)
    for j in ep.signals:
        ml.small_signals(j, t)
    focus = ml.vehicles(t, plain_colour, min_len=6)
    ml.pin(ep.hosp, "H  Fortis hospital road approach", D.GREEN, dy=-30)
    if t >= 1083.5:
        p = ml.S(ep.hosp)
        ml.cv.circle((p[0] + 120, p[1] - 30), 13, fill=(255, 255, 255), outline=D.GREEN, width=2.5)
        PN.tick(ml.cv, (p[0] + 120, p[1] - 30), True, 10)
    else:
        draw_focus(ml, ctx, focus, .75)
    if 1055 <= t:
        a = fade(t, 1055, 1056)
        ml.cv.rect((20, 20, 470, 92), fill=(255, 255, 255, int(240 * a)), outline=alpha(D.ESC, a), width=2.5, r=8)
        ml.cv.text((36, 34), "A (new):  T̂ = 25.25 s   chosen", 18, alpha(D.ESC, a), "bold")
        ml.cv.text((36, 62), "B (incumbent):  T̂ = 35.30 s", 18, alpha(D.BASE, a), "bold")


def ov_network_delta(ml, ctx):
    ep, t = ctx.ep, ctx.t
    ml.roads(alpha_=120)
    k = ep.k(t)
    lo, hi = max(0, k - 10), min(len(ep.t), k + 11)
    d = (ep.c3[lo:hi] - ep.c0[lo:hi]).mean(0)
    for i, eid in enumerate(ep.edge_ids):
        if abs(d[i]) < 0.25 or eid not in ep.exy:
            continue
        col = (205, 30, 40) if d[i] > 0 else (25, 95, 205)
        w = float(np.clip(3 + abs(d[i]) * 2.2, 3, 16))
        ml.cv.line([tuple(p) for p in ml.S(ep.exy[eid])], (255, 255, 255, 200), w + 4)
        ml.cv.line([tuple(p) for p in ml.S(ep.exy[eid])], alpha(col, min(1, .45 + abs(d[i]) * .12)), w)
    # propagation front: occupancy first diverges from baseline (after a002 dispatch)
    for eid, tf in ctx.first_div.items():
        age = t - tf
        if 0 <= age < 25:
            pts = ml.S(ep.exy[eid])
            p = pts[len(pts) // 2]
            r = 8 + age * 1.6
            ml.cv.circle(tuple(p), r, outline=alpha((142, 68, 173), 1 - age / 25), width=3)
    for j in ep.signals:
        ml.small_signals(j, t)
    for col in (0, 1, 2):
        p = ep.amb_pos(t, col)
        if np.isfinite(p).all():
            q = ml.S(p)
            if col == 2:
                ml.ambulance(q, 0, t, escalated=bool(ep.P3.a["fallback"][k, 2]), size=.5)
            else:
                ml.other_ambulance(q, 0)
                ml.cv.text((q[0] + 13, q[1] + 3), ["a000", "a001"][col], 13, D.MUTED, "bold")
    ml.pin(ep.hosp, "H", D.GREEN, dy=-22)
    from data import llxy
    s = ml.S(ep.J1xy)
    ml.cv.text((s[0] - 14, s[1] - 10), "J1 PCL Chowk", 14, D.INK, "bold", anchor="rb")
    s = ml.S(ep.J2xy)
    ml.cv.text((s[0] + 14, s[1] - 10), "J2", 14, D.INK, "bold", anchor="lb")


def ov_recovery(ml, ctx):
    focus, *_ = junction_scene(ml, ctx, heads=True, colour="approach")


def ov_summary(ml, ctx):
    ml.roads(alpha_=90)
    ml.cv.rect((0, 0, MW, MH), fill=(255, 255, 255, 248))
    cv = ml.cv
    cv.text((40, 26), "The causal chain shown in this film", 26, D.INK, "bold")
    steps = [
        ("Sensors", "link counts every 5 s, received 2 s late, ±2 noise", D.MUTED),
        ("Estimator", "q̂ = [n̂ − λL/v]₊ ,  d̂ = q̂ / (s/p)", D.MUTED),
        ("Forecast", "ETA per route–action;  burden Ĉ per junction (Eq. 20)", D.AMB),
        ("Gate", "b ≥ 5 s,  Eq. 19 feasible,  headroom,  Ĉ ≤ B", D.AMB),
        ("Lease", "bounded request; legal min-green / amber / all-red / pedestrian", D.AMB),
        ("Signal", "phase inserted → cycle of J1 shifted by 52 s", D.SW_C),
        ("Queues", "NW, SW, NE wait longer; SE served earlier", D.SW_C),
        ("Network", "platoons leave J1 at new times → downstream links change", (142, 68, 173)),
        ("Escalation", "deficit D_a > γ → aggressive priority for the rest of the trip", D.ESC),
    ]
    y = 78
    a_ = ctx.ls
    for i, (h, s, c) in enumerate(steps):
        a = fade(a_, .4 * i, .4 * i + .8)
        cv.rect((40, y, 210, y + 44), fill=D.mix((255, 255, 255), c, .14 * a) + (255,), outline=alpha(c, a), width=2, r=8)
        cv.text((125, y + 22), h, 17, alpha(c, a), "bold", anchor="mm")
        cv.text((230, y + 22), s, 17, alpha(D.INK, a), anchor="lm")
        if i < len(steps) - 1:
            cv.arrow((125, y + 46), (125, y + 62), alpha(D.FAINT, a), 2, 8)
        y += 64


# --------------------------------------------------------------- narration
def N(*items):
    return list(items)


def build(ep):
    J1 = ep.J1xy
    c_close = (J1[0] - 38, J1[1] - 30, 178)
    c_mid = (J1[0] - 25, J1[1] - 20, 300)
    c_net = (8540250.0, 3593950.0, 4000.0)
    c_route = (8540650.0, 3594190.0, 2150.0)
    c_netd = (8540650.0, 3594200.0, 2500.0)
    c_hosp = (8541700.0, 3593275.0, 270.0)
    c_j2 = (ep.J2xy[0] - 20, ep.J2xy[1] + 25, 230.0)

    shots = [
        Shot("title", "AmbuGo — what this film shows", 9, 760.0, [(0, c_net)], ov_title, "none", N(
            ("h", "How to watch"),
            ("p", "The map (top left) replays recorded vehicles and signals. The panel on the right explains what to look at. The panel at the bottom shows the mathematics the controller evaluated at that instant, with its logged numbers."),
            ("p", "The header shows the simulation clock and the playback speed. Decisions are shown slowed down or paused; uneventful driving is sped up, and the speed is always stated."),
            ("k", "Every number on screen is read from the recorded replay or the controller's decision log. Nothing is re-simulated or invented."),
        )),
        Shot("network", "1 · The network and the paired experiment", 26, 760.0, [(0, c_net)], ov_network, "legend", N(
            ("h", "What is real, what is simulated"),
            ("p", "Real: Mohali road geometry (OpenStreetMap, compiled by SUMO) and the Google Maps backdrop."),
            ("p", "Simulated: 414 ordinary vehicles from synthetic Poisson demand (seed 19), fixed-time signal plans at 7 junctions, and three ambulance missions dispatched at 600, 660 and 760 s."),
            ("h", "Two paired runs"),
            ("k", "AmbuGo: joint routing + selective signal priority, burden budget B = 1000 veh·s."),
            ("k", "No priority: identical roads, demand, signal plans and routing logic, but no signal priority."),
            ("p", "Because everything else is identical, any difference between the two runs is caused by priority decisions (of all three ambulances)."),
            ("p", "We follow ambulance a002 from dispatch (t = 760 s) to the hospital road approach."),
        )),
        Shot("routes", "2 · Choosing a route jointly with a priority action", 24, 761.0,
             [(0, c_net), (.25, c_route)], ov_routes, "routes", N(
            ("h", "Joint route–action choice"),
            ("p", "At dispatch the planner lists up to four loop-free routes to the hospital. Each is paired with two actions: no priority (0, 0) or strong priority (h, l) = (30 s look-ahead, 60 s maximum hold)."),
            ("p", "For each of the 8 pairs it forecasts the ETA T̂ and a burden score Ĉ (the junction burdens of Eq. 20, summed along the route)."),
            ("k", "Only pairs within ε of the fastest ETA are eligible. Here ε = 0 s, so the fastest pair wins: route B with strong priority, 306.97 s, versus 500.24 s without priority."),
            ("p", "The four routes share the corridor. Two of them (C, D) use a longer middle section, and all four differ around the hospital roundabout."),
        )),
        Shot("approach", "3 · Approaching PCL Chowk (junction J1)", 18, (761.5, 784.5),
             [(0, c_route), (.55, c_mid), (1, c_close)], ov_approach, "speed", N(
            ("h", "What to watch"),
            ("p", "Each approach lane of PCL Chowk has its own colour: SE (blue, the ambulance's approach), NW (violet), SW (orange), NE (teal). Cars take the colour of the lane they are on. Red rear dots mark stopped cars."),
            ("p", "Since 771.5 s the junction has been in a 25 s pedestrian all-red reservation. Once started it cannot be interrupted, so no approach can turn green before 796.5 s."),
            ("p", "Bottom left: recorded ambulance speed. Bottom right: the progress deficit D_a(t), which decides escalation later (scene 10)."),
        ), ramp=.18),
        Shot("decide", "4 · The request is evaluated at t = 785 s", 24, 785.0, [(0, c_close)],
             ov_decision, "burden", N(
            ("h", "Step 1 — Is a request due?"),
            ("p", "Free-flow time to the stop line is 3.0 s, within h + d̂ = 30 + 8.3 s, so the ambulance is inside its request horizon."),
            ("h", "Step 2 — What would priority cost the others?"),
            ("p", "Serving SE now would take green away from NW, SW and NE for D_e = 38.3 s. Each loses the waiting of its current queue (q̂·D) plus new arrivals amplified by its loading X (Eq. 20). The tags on the map show each approach's inputs."),
            ("h", "Step 3 — Gate"),
            ("k", "Benefit b = 72.6 s ≥ 5 s, the pulse fits the hold (Eq. 19) and there is receiving headroom. But ΣĈ = 1044.5 veh·s > B = 1000, so the request is WITHHELD."),
        )),
        Shot("withheld", "5 · Withheld, re-checked every second", 21, (785.0, 791.99),
             [(0, c_close)], lambda ml, ctx: ov_decision(ml, ctx, True), "burden_live", N(
            ("h", "Re-evaluated each second"),
            ("p", "The controller re-runs the same test once per second with the newest sensor data it has received (5 s sampling, 2 s delay)."),
            ("p", "At 787 s a new sample raises the SW queue proxy from 6.37 to 9.37 vehicles, and ΣĈ rises to 1184 veh·s, still over budget."),
            ("k", "The refusal does not delay the ambulance here: the pedestrian reservation blocks every approach until 796.5 s anyway (δ counts down 11 → 5 s)."),
        ), ramp=0),
        Shot("admit", "6 · Admitted at t = 792 s", 22, 792.0, [(0, c_close)],
             lambda ml, ctx: ov_decision(ml, ctx, True), "burden_prev", N(
            ("h", "Why the gate passes now"),
            ("p", "A sample taken at 790 s arrives at 792 s. It lowers the ambulance's own queue estimate (4.78 → 2.78 veh), so d̂ falls 10.5 → 6.1 s and the displaced green D_e falls 40.5 → 36.1 s. It also lowers the conflicting estimates: NW 1.94 → 0.94 and SW 9.37 → 8.37 vehicles."),
            ("k", "ΣĈ = 913.3 veh·s ≤ B. All gates pass, so the request is ADMITTED and a bounded lease starts."),
            ("w", "Note: the true vehicle counts on all four approaches are identical at 785 s and 792 s. What changed is the delayed, noisy estimate, not the physical queues."),
            ("p", "Admission is not green. The light can change only when the pedestrian reservation ends."),
        )),
        Shot("transition", "7 · Legal transition and the ambulance's green", 36, (792.0, 804.0),
             [(0, c_close)], ov_transition, "gantt", N(
            ("h", "Signal legality is never bypassed"),
            ("p", "The pedestrian reservation completes at 796.5 s. The SE approach then gets green directly, skipping the normal order NW → SW → NE → SE once."),
            ("p", "Rules kept throughout: minimum green 8 s, amber 3 s, all-red 2 s, pedestrian reservation 25 s uninterrupted."),
            ("k", "The ambulance crosses at ≈801.5 s. NW, SW and NE stay red, so their queues keep growing (watch the orange SW lane fill up)."),
            ("p", "Bottom: the recorded signal timeline of both runs, revealed as time advances."),
        ), ramp=0.05),
        Shot("shift", "8 · One switched light shifts the whole cycle", 22, 804.0,
             [(0, c_close), (.5, c_mid)], ov_shift, "gantt_preview", N(
            ("h", "From one decision to every later cycle"),
            ("p", "The base controller is fixed-time with a cyclic pointer. After the priority green the pointer is synchronised to the served approach (SE), and the normal cycle continues from there."),
            ("k", "So 51 s of service (SE green + clearance + pedestrian) were inserted in front of NW, SW and NE. Each of their greens now comes 52 s later than in the no-priority run, in this cycle and every following one."),
            ("p", "SE is served 72 s earlier, which is the ambulance's gain. This is how one signal change re-times the service of every other approach at the junction."),
        )),
        Shot("consequence", "9 · What the other approaches pay", 30, (804.0, 851.0),
             [(0, c_close)], ov_consequence, "approaches", N(
            ("h", "New colour code: individual cars"),
            ("p", "Each car is now coloured by its extra stopped time since dispatch, compared with the same car in the no-priority run: red = waited longer, blue = waited less, grey = same. Labels give the difference in seconds."),
            ("p", "Hollow outlines show where those same cars are at this moment in the no-priority run."),
            ("k", "The NW, SW and NE queues build while they wait for their delayed greens. On SE the opposite happens: its chart below shows fewer stopped vehicle-seconds than the baseline (blue area)."),
            ("p", "Bottom: stopped vehicles per approach in both runs; the shaded area between them is extra (red) or saved (blue) vehicle-seconds."),
        ), ramp=0.1),
        Shot("escalate", "10 · Progress check and escalation  (t ≈ 851 s)", 20, (846.0, 853.0),
             [(0, ("track", 650))], ov_track, "speed", N(
            ("h", "A safeguard against an untrustworthy forecast"),
            ("p", "P_a(t) credits only the links the ambulance has actually driven, at free-flow speed. Junction connectors earn no credit."),
            ("k", "At 851 s: elapsed 91.0 s − progress 70.74 s = 20.26 s > γ = 20 s. The ambulance escalates irreversibly."),
            ("p", "Most of the deficit was the stop at PCL Chowk (785–797 s), which the pedestrian reservation imposed."),
            ("p", "From now on a002 uses the aggressive comparator (JFAST): strong (30, 60) requests, admitted without the burden gate. Later priority is no longer 'selective'."),
        ), ramp=0.1),
        Shot("corridor", "11 · Along the corridor", 18, (853.0, 966.0),
             [(0, ("track", 650)), (.15, ("track", 800)), (.85, ("track", 800)), (1, ("track", 600))], ov_track, "speed", N(
            ("h", "Sped up: no decisions on this stretch"),
            ("p", "The corridor between J1 and J2 has no controlled junction, so the controller issues no requests. Playback is sped up; the header shows the exact rate."),
            ("p", "Ambulance a001 (escalated since 704 s) reaches J2 first and is served there from 955.5 s, shortly before a002 arrives."),
            ("p", "Orange ring = escalated mode. The bottom charts keep growing from the same records."),
        ), ramp=0.1),
        Shot("j2", "12 · Junction J2 — escalated requests", 30, (966.0, 1001.0),
             [(0, ("track", 600)), (.18, c_j2), (.85, c_j2), (1, ("track", 500))], ov_j2, "j2", N(
            ("h", "Two stop lines in series"),
            ("p", "J2 is a joined junction: the route crosses stop line NW-1, then a 6.7 m link, then NW-2, and then the single-approach signal J3."),
            ("p", "a002 requests NW-1 at 971 s. NW-2 is green for a001 at that moment, so the switch must wait for amber + all-red (δ = 10 s). NW-1 turns green at 981.5 s, and NW-2 at 994.5 s."),
            ("k", "Escalated mode: Ĉ is still computed and logged (36.5–168.5 veh·s) but B is not applied."),
            ("p", "The deficit D_a rises from 24 to 48 s here. Legal clearances can cost an escalated ambulance time as well."),
        ), ramp=0.1),
        Shot("hospital", "13 · Hospital approach, re-route and arrival", 34, (1001.0, 1090.0),
             [(0, ("track", 500)), (.25, c_hosp)], ov_hospital, "hospital", N(
            ("h", "Last re-plan"),
            ("p", "Entering roundabout arc 5 at 1055 s, the planner re-scores the remaining continuations: A 25.255 s versus incumbent B 35.295 s."),
            ("p", "Hysteresis keeps the incumbent only if it is within 10 s of the best. The gap is 10.04 s > 10 s, so the planner switches to A."),
            ("k", "Arrival at the modelled hospital road approach: t = 1083.5 s, 323.5 s after dispatch. The same mission in the no-priority run took 516.5 s, 193 s longer."),
            ("p", "The endpoint is a modelled road approach, not a surveyed emergency entrance."),
        ), ramp=0.08),
        Shot("netdelta", "14 · The whole network — where did the cost go?", 40, (760.0, 1125.0),
             [(0, c_hosp), (.08, c_netd)], ov_network_delta, "network", N(
            ("h", "Replay of the same 6 minutes, whole network"),
            ("p", "Link colour shows the vehicles present on each link in AmbuGo minus the no-priority run at the same moment (±5 s average): red = more, blue = fewer. Width grows with the difference. Summing the red part over links and time gives A, shown below."),
            ("p", "Violet rings mark the moment a link's vehicle count first departs from the baseline. The disturbance travels with platoons that leave J1 at new times."),
            ("k", "Signals are fixed-time and uncoordinated. Junctions affect each other only through vehicles (platoon timing, queues, spill-back) and through other ambulances' requests."),
            ("p", "Bottom: network-wide stopped vehicles in both runs and the accumulated difference. All three missions contribute."),
        ), ramp=0.04),
        Shot("recovery", "15 · Back at PCL Chowk after arrival", 24, (1083.5, 1125.0),
             [(0, c_net), (.18, c_mid)], ov_recovery, "approaches_end", N(
            ("h", "Ordinary service continues"),
            ("p", "No requests are active after arrival (none recorded after 1083.5 s). NW turns green at 1095.5 s and its three stopped cars clear by 1100 s. SW turns green at 1112.5 s and clears by 1120 s."),
            ("k", "The 52 s cycle shift is still in place. It does not decay in a fixed-time plan without a resynchronisation rule."),
            ("p", "This is real local queue discharge. It does not establish complete network recovery."),
        ), ramp=0.1),
        Shot("summary", "16 · Summary: mechanism, numbers and real-world deployment", 44, 1125.0,
             [(0, c_mid), (.1, c_net)], ov_summary, "summary", N(
            ("h", "This episode (one synthetic run)"),
            ("k", "a002 travel: 323.5 s with AmbuGo vs 516.5 s without priority."),
            ("k", "PCL Chowk: forecast burden 913 veh·s at admission. Realised extra stopped time on the three conflicting approaches: +843 veh·s by 880 s (first shifted cycle)."),
            ("k", "Network, 760–1125 s: stopped vehicle-seconds +307 (+5.2 %), all three missions together."),
            ("h", "Paper evidence (1,728 runs, Sec. 5)"),
            ("p", "Selective vs aggressive joint control: +11.10 s ambulance time [9.47, 12.94], −3.84 veh·h positive spatial exposure [−4.73, −2.95]. These are 95 % seed-block bootstrap intervals."),
            ("h", "Deployment would need"),
            ("p", "Link counts with known delay and noise, signal controllers that accept bounded requests with legal clearances, calibrated λ, X and saturation flows, and a field check that Ĉ ranks real harm correctly. None of these is validated here."),
        )),
    ]
    return shots
