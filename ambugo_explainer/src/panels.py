"""Bottom-panel mathematics and charts. Every number is read from the episode data."""
import numpy as np

import draw as D
from draw import Canvas, Chart, alpha

BW, BH = 1920, 336  # bottom panel size

APPS = ["335061218", "172380615#1", "172382686#4", "172660580#0"]


def title(cv, x, y, s, sub=None):
    cv.text((x, y), s, 17, D.INK, "bold")
    if sub:
        w, _ = cv.tsize(s, 17, "bold")
        cv.text((x + w + 12, y + 2), sub, 14, D.MUTED)


def chip(cv, x, y, color, w=14, h=14):
    cv.rect((x, y, x + w, y + h), fill=color, r=3)


def tick(cv, p, ok, size=11):
    x, y = p
    if ok:
        cv.circle((x, y), size, fill=alpha(D.OK, .12), outline=D.OK, width=2)
        cv.line([(x - size * .45, y), (x - size * .1, y + size * .4), (x + size * .5, y - size * .4)], D.OK, 3)
    else:
        cv.circle((x, y), size, fill=alpha(D.BAD, .12), outline=D.BAD, width=2)
        cv.line([(x - size * .4, y - size * .4), (x + size * .4, y + size * .4)], D.BAD, 3)
        cv.line([(x - size * .4, y + size * .4), (x + size * .4, y - size * .4)], D.BAD, 3)


# ------------------------------------------------------------------ legend
def legend(cv, ctx):
    ep = ctx.ep
    title(cv, 30, 16, "How to read the colours", "(used consistently in every scene)")
    rows = [
        (D.AMB, "Ambulance a002 · its route · its approach at PCL Chowk (SE)"),
        (D.NW_C, "NW approach of PCL Chowk (conflicting)"),
        (D.SW_C, "SW approach of PCL Chowk (conflicting)"),
        (D.NE_C, "NE approach of PCL Chowk (conflicting)"),
        (D.ESC, "Escalated (aggressive) priority mode, after 851 s"),
        (D.BASE, "No-priority baseline run (same demand, same signals)"),
    ]
    y = 52
    for c, s in rows:
        chip(cv, 30, y + 2, c, 26, 14)
        cv.text((68, y), s, 15, D.INK)
        y += 27
    cv.text((30, y + 6), "Cars:", 15, D.INK, "semibold")
    cv.text((80, y + 6), "red rear dots = stopped (speed < 0.1 m/s);  hollow outline = same car in the no-priority run", 15, D.INK)
    y += 30
    x = 30
    cv.text((x, y + 4), "Extra waiting vs. baseline:", 15, D.INK, "semibold")
    x0 = 250
    for i in range(-40, 41, 2):
        cv.rect((x0 + (i + 40) * 3, y + 4, x0 + (i + 42) * 3, y + 20), fill=D.diverging(i))
    cv.text((x0, y + 24), "−40 s (less)", 13, D.MUTED)
    cv.text((x0 + 120, y + 24), "0", 13, D.MUTED, anchor="ma")
    cv.text((x0 + 243, y + 24), "+40 s (more)", 13, D.MUTED, anchor="ra")

    x = 1010
    title(cv, x, 16, "Notation", "(symbols follow the paper, Sections 2 and 5)")
    notes = [
        (r"$t$", "simulation clock (s); ambulance dispatched at $d_a=760$ s"),
        (r"$\hat q_j,\ \hat d_j$", "queue proxy (veh) and its discharge time (s) from delayed counts"),
        (r"$\lambda_j,\ X_j$", "historical arrival rate (veh/s) and nominal loading of approach $j$"),
        (r"$(h,\,l)$", "priority action: 30 s look-ahead, at most 60 s requested hold"),
        (r"$D_e,\ \hat C_e$", "displaced green (s) and forecast burden (veh·s), Eq. 20"),
        (r"$B,\ b_e,\ \delta_e$", "burden budget 1000 veh·s, time benefit (s), legal switch delay (s)"),
        (r"$P_a,\ D_a,\ \gamma$", "free-flow progress, progress deficit, escalation threshold 20 s (Eq. 21)"),
    ]
    y = 50
    for sym, s in notes:
        cv.eq((x, y - 3), sym, 17)
        mixed(cv, (x + 120, y), s, 15)
        y += 38


def mixed(cv, p, s, size=15, color=D.INK):
    """Text with inline $...$ math segments."""
    x, y = p
    parts = s.split("$")
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2:
            w, h = cv.eq((x, y - 2), "$" + part + "$", size + 1, color)
            x += w + 2
        else:
            cv.text((x, y), part, size, color)
            x += cv.d.textlength(part, font=D.font(size)) / D.SS
    return x


# ------------------------------------------------------------- route table
def route_table(cv, ctx):
    ep = ctx.ep
    u = ep.decisions["route_updates"][0]
    title(cv, 30, 14, "Route–action candidates at t = 761 s", "(controller log, a002 at dispatch)")
    cols = [30, 120, 220, 380, 540, 700, 850]
    heads = ["Pair", "Route", "Action (h, l)", "Forecast ETA T̂", "Burden Ĉ (veh·s)", "Eligible", "Chosen"]
    y = 48
    for x, h in zip(cols, heads):
        cv.text((x, y), h, 14, D.MUTED, "semibold")
    cv.line([(30, y + 22), (930, y + 22)], D.RULE, 1)
    y += 30
    names = "AABBCCDD"
    reveal = ctx.ls
    for i, c in enumerate(u["candidates"]):
        a = D.fade(reveal, 1.0 + i * 0.35, 1.5 + i * 0.35)
        if a <= 0:
            continue
        hi = c["chosen"]
        col = D.AMB if hi else D.INK
        w = "bold" if hi else "regular"
        if hi:
            cv.rect((24, y - 4, 936, y + 22), fill=alpha(D.AMB, .10), r=4)
        vals = [str(c["candidate_index"]), names[c["candidate_index"]],
                "(0, 0) none" if c["action"][0] == 0 else "(30, 60) strong",
                f"{c['forecast_remaining_time_s']:.2f} s", f"{c['unvalidated_burden_proxy_vehicle_s']:.1f}"]
        for x, v in zip(cols, vals):
            cv.text((x, y), v, 15, D.alpha(col, a), w)
        tick(cv, (cols[5] + 22, y + 9), c["eligible"], 9)
        if c["chosen"]:
            tick(cv, (cols[6] + 22, y + 9), True, 9)
        y += 27
    x = 1000
    title(cv, x, 14, "Selection rule (selective mode)")
    cv.eq((x, 50), r"$\mathcal{E}=\{\,c:\ \hat T_c\leq \hat T_{\min}+\varepsilon\,\},\qquad \varepsilon=0\ \mathrm{s}$", 21)
    cv.eq((x, 92), r"$c^{*}=\arg\min_{c\in\mathcal{E}}\ (\hat C_c,\ \hat T_c,\ \mathrm{route\ id},\ \mathrm{action\ id})$", 21)
    cv.eq((x, 136), r"$\hat T_{\min}=306.97\ \mathrm{s}\ \Rightarrow\ \mathcal{E}=\{3\}\ \Rightarrow\ c^{*}=$ route B, strong action", 19)
    mixed(cv, (x, 186), "With $\\varepsilon = 0$ the burden term only breaks ties: the fastest pair wins.", 15)
    mixed(cv, (x, 212), "Without priority the same route is forecast at 500.24 s, so priority is", 15)
    mixed(cv, (x, 236), "expected to save ≈193 s; the burden of that saving is priced junction by junction (Eq. 20).", 15)
    mixed(cv, (x, 270), "Route B's Ĉ = 838.5 veh·s is the sum of admitted junction burdens along the whole route.", 15, D.MUTED)


# --------------------------------------------------------- speed / deficit
EVENTS = [(785, "withheld", D.BAD), (792, "admitted", D.AMB), (796.5, "green", D.GREEN), (851, "escalated", D.ESC),
          (971, "J2 request", D.ESC), (1055, "re-route", D.AMB), (1083.5, "arrived", D.GREEN)]


def speed_deficit(cv, ctx, x_end=1090):
    ep, t = ctx.ep, ctx.t
    k = ep.k(t)
    ts = ep.t[ep.k(760):k + 1]
    sl = slice(ep.k(760), k + 1)
    c1 = Chart(cv, (90, 50, 900, 252), (760, x_end), (0, 60))
    title(cv, 30, 12, "Ambulance a002 speed", "recorded SUMO speed, km/h")
    c1.axes(xticks=range(780, x_end + 1, 40), yticks=[0, 25, 50], xlabel="simulation time t (s)")
    for i, (te, lab, col) in enumerate(EVENTS):
        if te <= t:
            xx = float(c1.X(te))
            cv.line([(xx, 50), (xx, 252)], alpha(col, .45), 1.5)
            cv.text((xx + 3, 52 + (i % 3) * 15), lab, 12, col, "semibold")
    c1.series(ts, ep.amb_speed[sl], D.AMB, 2.6)
    c1.dot(t, ep.interp(ep.amb_speed, t), D.AMB)
    v = ep.amb_speed[k]
    cv.text((900, 12), f"{v:.0f} km/h" if np.isfinite(v) else "arrived", 22, D.AMB, "bold", anchor="ra")

    c2 = Chart(cv, (1060, 50, 1880, 252), (760, x_end), (0, 70))
    title(cv, 990, 12, "Progress deficit", "Eq. 21, recomputed from the trajectory")
    c2.axes(xticks=range(780, x_end + 1, 40), yticks=[0, 20, 40, 60], xlabel="simulation time t (s)")
    c2.hline(20, D.ESC, 1.6, "γ = 20 s escalation threshold")
    c2.series(ts, ep.D_a[sl], D.INK, 2.4)
    if t >= 851:
        c2.dot(851, 20.26, D.ESC, 6)
        cv.text((float(c2.X(851)) + 8, float(c2.Y(20.26)) - 24), "851 s: 20.26 > 20", 13, D.ESC, "semibold")
    da = ep.D_a[k]
    c2.dot(t, da if np.isfinite(da) else np.nan, D.INK)
    if np.isfinite(da):
        cv.text((1880, 12), f"D_a = {da:.2f} s", 20, D.INK, "bold", anchor="ra")
    cv.eq((1000, 286), r"$D_a(t)=\left[\,t-d_a-P_a(t)\,\right]_+,\quad P_a(t)=\sum_{f\ \mathrm{completed}}\frac{L_f}{v_f}+\frac{x_a(t)}{v_e}$", 17, D.MUTED)
    cv.text((90, 300), "Event lines are from the controller log; the curve itself is the recorded speed.", 13, D.MUTED)
    cv.text((1620, 300), "only driven links earn credit", 13, D.MUTED)


# ----------------------------------------------------------- burden panel
def burden(cv, ctx, compare_prev=False):
    ep, t = ctx.ep, ctx.t
    e = ep.event_at(t)
    if e is None:
        return
    prev = ep.ev.get(e["t"] - 1) if compare_prev else None
    if compare_prev and e["t"] == 792:
        prev = ep.ev[791]
    h, l = e["action"]
    d_hat = e["queue_clearance_s"]
    De = min(l, max(5, h + d_hat))
    title(cv, 30, 10, f"Burden forecast for the requested pulse at t = {e['t']} s", "paper Eq. 20 — evaluated with the controller's own inputs")
    cv.eq((30, 40), r"$\hat C_e=\sum_{j\neq e}\left[\ \hat q_j\,D_e\ +\ \frac{\lambda_j\,D_e^{2}}{2\,\max(1-X_j,\ 0.05)}\ \right],\qquad D_e=\min\!\left(l,\ \max[5,\ h+\hat d_e]\right)$", 21)
    hl_d = prev is not None and abs(prev["queue_clearance_s"] - d_hat) > 1e-9
    s = f"$D_e=\\min(60,\\ \\max[5,\\ 30+{d_hat:.2f}])={De:.2f}\\ \\mathrm{{s}}$"
    if hl_d:
        pD = min(l, max(5, h + prev["queue_clearance_s"]))
        cv.rect((26, 92, 640, 122), fill=(255, 240, 160), r=5)
        s = f"$D_e=\\min(60,\\ \\max[5,\\ 30+{d_hat:.2f}])={De:.2f}\\ \\mathrm{{s}}\\quad(\\mathrm{{was}}\\ {pD:.2f})$"
    cv.eq((30, 94), s, 19)
    cv.text((660, 98), f"own approach: q̂ = {e['queue_proxy']:.2f} veh → d̂ = q̂ / 0.4545 veh/s = {d_hat:.2f} s", 14, D.MUTED)
    cols = [30, 130, 250, 370, 470, 600, 760, 900]
    heads = ["approach j", "q̂_j (veh)", "λ_j (veh/s)", "X_j", "q̂_j·D_e", "λD²/2(1−X)", "Ĉ_j (veh·s)"]
    y = 130
    for x, hh in zip(cols, heads):
        cv.text((x, y), hh, 13, D.MUTED, "semibold")
    y += 22
    total = 0
    for a in e["conflicting_approaches"]:
        name = ep.app_name[a["edge"]]
        col = D.APP_COL[a["edge"]]
        t1 = a["queue_proxy"] * De
        t2 = a["arrival_rate"] * De ** 2 / (2 * max(1 - a["nominal_design_X"], .05))
        total += a["burden_proxy_veh_s"]
        pa = None
        if prev is not None:
            pa = next(x for x in prev["conflicting_approaches"] if x["edge"] == a["edge"])
        chip(cv, 30, y + 3, col)
        cv.text((52, y), name, 15, col, "bold")
        vals = [(f"{a['queue_proxy']:.2f}", pa is not None and abs(pa["queue_proxy"] - a["queue_proxy"]) > 1e-9),
                (f"{a['arrival_rate']:.4f}", False), (f"{a['nominal_design_X']:.3f}", False),
                (f"{t1:.1f}", False), (f"{t2:.1f}", False), (f"{a['burden_proxy_veh_s']:.1f}", False)]
        for x, (v, changed) in zip(cols[1:], vals):
            if changed:
                cv.rect((x - 5, y - 3, x + 92, y + 21), fill=(255, 240, 160), r=4)
                v = f"{pa['queue_proxy']:.2f}→{v}"
            cv.text((x, y), v, 15, D.INK, "semibold" if x == cols[6] else "regular")
        y += 26
    cv.line([(30, y + 2), (860, y + 2)], D.RULE, 1)
    y += 8
    over = total > 1000
    cv.text((30, y), "Σ over conflicting approaches", 15, D.INK, "semibold")
    cv.text((cols[6], y), f"{total:.1f}", 17, D.BAD if over else D.OK, "bold")
    cv.text((cols[6] + 70, y + 1), "veh·s", 14, D.MUTED)
    cv.text((30, y + 30), "A ranking surrogate in vehicle-seconds, not a bound on network harm (paper, Sec. 5).", 13, D.MUTED)

    # stacked gauge
    gx0, gx1, gy = 1000, 1880, 26
    title(cv, gx0, 6, "Gate:  Σ Ĉ  ≤  B = 1000 veh·s ?")
    scale = (gx1 - gx0) / 1400
    x = gx0
    cv.rect((gx0, gy + 22, gx1, gy + 52), fill=(240, 242, 245), r=4)
    for a in e["conflicting_approaches"]:
        w = a["burden_proxy_veh_s"] * scale
        cv.rect((x, gy + 22, x + w, gy + 52), fill=D.APP_COL[a["edge"]])
        if w > 40:
            cv.text((x + w / 2, gy + 37), f"{a['burden_proxy_veh_s']:.0f}", 13, (255, 255, 255), "bold", anchor="mm")
        x += w
    bx = gx0 + 1000 * scale
    cv.line([(bx, gy + 14), (bx, gy + 60)], D.INK, 3)
    cv.text((bx, gy + 62), "B = 1000", 13, D.INK, "bold", anchor="ma")
    for v in (0, 500, 1400):
        cv.text((gx0 + v * scale, gy + 62), str(v), 12, D.MUTED, anchor="ma")
    cv.text((min(x + 8, gx1 - 4), gy + 37), f"{total:.0f}", 16, D.BAD if over else D.OK, "bold", anchor="lm" if x + 60 < gx1 else "rm")

    # checklist
    y = 118
    feas_lhs = max(h + d_hat - e["switch_delay_s"], 0) + d_hat + 2
    rows = [
        (e["free_flow_eta_to_stopline_s"] <= h + d_hat, f"Trigger: free-flow time to stop line {e['free_flow_eta_to_stopline_s']:.2f} s ≤ h + d̂ = {h + d_hat:.1f} s"),
        (e["pulse_feasible"], f"Eq. 19 pulse fits hold: [h+d̂−δ]₊ + d̂ + 2 = {feas_lhs:.1f} ≤ l = 60 s   (δ = {e['switch_delay_s']:.0f} s)"),
        (e["predicted_benefit_s"] >= 5, f"Benefit b_e = {e['predicted_benefit_s']:.1f} s ≥ 5 s   (nominal red wait avoided)"),
        (e["headroom"], "Receiving headroom: downstream count < 0.85 × storage"),
        (not over, f"Burden test Σ Ĉ ≤ B:  {total:.1f} {'>' if over else '≤'} 1000 veh·s"),
    ]
    for ok, s in rows:
        tick(cv, (1012, y + 9), ok, 10)
        cv.text((1032, y), s, 14, D.INK)
        y += 27
    gate = e["reason"]
    if gate == "burden_budget_exceeded":
        txt, col = "Decision: REQUEST WITHHELD — re-evaluated next second", D.BAD
    elif gate == "selective_gate_passed":
        txt, col = "Decision: REQUEST ADMITTED — bounded lease starts", D.AMB
    else:
        txt, col = "Lease active: request forwarded until link departure", D.AMB
    cv.rect((1000, y + 6, 1880, y + 40), fill=alpha(col, .1), outline=col, width=2, r=6)
    cv.text((1440, y + 23), txt, 16, col, "bold", anchor="mm")


# -------------------------------------------------------------- gantt (J1)
def mode_colour(mode, edge):
    if mode == 0:
        return D.APP_COL.get(edge, (90, 160, 90))
    if mode == 1:
        return D.AMBERC
    if mode == 2:
        return (70, 74, 82)
    return (196, 199, 204)


def gantt(cv, ctx, j, x0t, x1t, box, reveal_t=None, preview=False, apps=None, names=None):
    ep = ctx.ep
    x0, y0, x1, y1 = box
    ch = Chart(cv, (x0 + 150, y0, x1, y1), (x0t, x1t), (0, 1))
    rows = [("P3", "AmbuGo", D.AMB), ("P0", "No priority", D.BASE)]
    gap = 40
    rh = (y1 - y0 - 30 - gap) / 2
    for r, (pol, lab, lc) in enumerate(rows):
        ry = y0 + r * (rh + gap)
        cv.text((x0, ry + rh / 2), lab, 16, lc, "bold", anchor="lm")
        for a, b, mode, edge in ep.timeline[pol, j]:
            if b < x0t or a > x1t:
                continue
            a2, b2 = max(a, x0t), min(b, x1t)
            vis = 1.0
            if reveal_t is not None and a2 > reveal_t:
                if not preview:
                    continue
                vis = .28
            if reveal_t is not None and a2 <= reveal_t < b2 and not preview:
                b2 = reveal_t
            col = mode_colour(mode, edge)
            xa, xb = float(ch.X(a2)), float(ch.X(b2))
            cv.rect((xa, ry, xb, ry + rh), fill=alpha(col, vis))
            label = None
            if mode == 0 and edge is not None:
                label = (names or ep.app_name).get(edge, "")
            elif mode == 3:
                label = "ped"
            if label and xb - xa > 26:
                cv.text(((xa + xb) / 2, ry + rh / 2), label, 13, (255, 255, 255) if mode == 0 else D.INK, "bold", anchor="mm")
    ticks = [v for v in range(int(np.ceil(x0t / 20) * 20), int(x1t) + 1, 40)]
    for v in ticks:
        xx = float(ch.X(v))
        cv.line([(xx, y1 - 14), (xx, y1 - 10)], D.FAINT, 1)
        cv.text((xx, y1 - 8), str(v), 12, D.MUTED, anchor="ma")
    if reveal_t is not None and x0t <= reveal_t <= x1t:
        xx = float(ch.X(reveal_t))
        cv.line([(xx, y0 - 6), (xx, y1 - 14)], D.INK, 2)
    return ch


def gantt_panel(cv, ctx, preview=False):
    ep, t = ctx.ep, ctx.t
    title(cv, 30, 10, "PCL Chowk signal timeline — which approach is served", "recorded signal state, both runs, same clock")
    leg = [(D.AMB, "SE (ambulance)"), (D.NW_C, "NW"), (D.SW_C, "SW"), (D.NE_C, "NE"), (D.AMBERC, "amber 3 s"),
           ((70, 74, 82), "all-red 2 s"), ((196, 199, 204), "pedestrian 25 s")]
    x = 1010
    for c, s in leg:
        chip(cv, x, 14, c)
        cv.text((x + 20, 12), s, 14, D.INK)
        x += 30 + cv.tsize(s, 14)[0] + 10
    y0, y1 = 64, 268
    ch = gantt(cv, ctx, ep.J1, 765, 1125, (30, y0, 1890, y1), reveal_t=t, preview=preview)
    if preview:
        gap = 40
        rowh = (y1 - y0 - 30 - gap) / 2
        g0 = y0 + rowh          # top of the gap between the rows
        g1 = g0 + gap           # top of the no-priority row
        pairs = [(795.5, 847.5, D.NW_C), (812.5, 864.5, D.SW_C), (839.5, 891.5, D.NE_C)]
        a = D.fade(ctx.ls, 2, 3.5)
        for p0, p3, col in pairs:
            xa, xb = float(ch.X(p0)), float(ch.X(p3))
            cv.arrow((xa, g1 - 1), (xb, g0 + 1), alpha(col, a), 3, 11)
            cv.text((xb + 4, g0 + 2), "+52 s", 13, alpha(col, a), "bold")
        b = D.fade(ctx.ls, 6, 7.5)
        xa, xb = float(ch.X(868.5)), float(ch.X(796.5))
        cv.arrow((xa, g1 - 1), (xb, g0 + 1), alpha(D.AMB, b), 3, 11)
        cv.text((xa + 6, g1 - 17), "SE served 72 s earlier", 13, alpha(D.AMB, b), "bold")
        c = D.fade(ctx.ls, 9, 10.5)
        cv.rect((float(ch.X(796.5)), y0 - 4, float(ch.X(847.5)), y0 + rowh + 2), outline=alpha(D.INK, c), width=2.5, r=3)
        cv.text((float(ch.X(796.5)), y0 - 8), "inserted before NW: SE green 21 s + amber/all-red 5 s + pedestrian 25 s = 51 s", 13, alpha(D.INK, c), "bold", anchor="lb")
    cv.text((30, 296), "Fixed-time base controller: 124 s cycle (ped 25 · NW 12 · SW 22 · NE 24 · SE 21 s, each followed by amber 3 s + all-red 2 s).", 14, D.MUTED)
    cv.text((30, 316), "After the priority green the cycle pointer is synchronised to the served approach, so the whole cycle stays shifted afterwards.", 14, D.MUTED)


# --------------------------------------------------------- approach multiples
def approaches(cv, ctx, x_end=1125):
    ep, t = ctx.ep, ctx.t
    k = ep.k(t)
    s = ep.k(780)
    ts = ep.t[s:k + 1]
    title(cv, 30, 8, "Stopped vehicles on each PCL Chowk approach", "AmbuGo (colour) vs. no priority (grey); red fill = extra queue, blue fill = shorter queue")
    w = 335
    order = ["172380615#1", "172382686#4", "172660580#0", "335061218"]
    for i, e in enumerate(order):
        x0 = 40 + i * (w + 22)
        col = D.APP_COL[e]
        idx = ep.ei[e]
        ch = Chart(cv, (x0 + 26, 64, x0 + w, 262), (780, x_end), (0, 14))
        ch.axes(xticks=range(800, x_end + 1, 100), yticks=[0, 5, 10], size=12)
        y3 = ep.h3[s:k + 1, idx]
        y0 = ep.h0[s:k + 1, idx]
        ch.fill_between(ts, y3, y0, alpha(D.RED, .28), alpha(D.AMB, .22))
        ch.series(ts, y0, D.BASE, 1.8, step=True)
        ch.series(ts, y3, col, 2.6, step=True)
        chip(cv, x0, 40, col)
        nm = ep.app_name[e] + (" (ambulance)" if e == ep.amb_app else " (conflicting)")
        cv.text((x0 + 20, 38), nm, 15, col, "bold")
        ex = ep.app_extra[e][k]
        cv.text((x0 + w, 38), f"{ex:+.0f} veh·s", 15, D.BAD if ex > 0 else D.AMB, "bold", anchor="ra")
    conf = sum(ep.app_extra[e][k] for e in ep.conf)
    cv.text((40, 290), f"Realised extra stopped vehicle-seconds since 785 s on the three conflicting approaches:  {conf:+.0f} veh·s", 16, D.INK, "semibold")
    cv.text((40, 312), "Forecast before admission (Eq. 20, t = 792 s): 913.3 veh·s.  Measured as ∫(stopped_AmbuGo − stopped_baseline) dt on the same links; includes later cycles of the 52 s shift.", 13, D.MUTED)


# --------------------------------------------------------------- network
def network(cv, ctx):
    ep, t = ctx.ep, ctx.t
    k = ep.k(t)
    s = ep.k(760)
    ts = ep.t[s:k + 1]
    title(cv, 30, 8, "Whole network: ordinary vehicles stopped", "all ordinary vehicles with speed < 0.1 m/s, incl. junction connectors")
    ch = Chart(cv, (80, 54, 900, 262), (760, 1125), (0, 40))
    ch.axes(xticks=range(780, 1126, 40), yticks=[0, 10, 20, 30, 40], xlabel="simulation time t (s)")
    ch.fill_between(ts, ep.net_stop3[s:k + 1], ep.net_stop0[s:k + 1], alpha(D.RED, .22), alpha(D.AMB, .2))
    ch.series(ts, ep.net_stop0[s:k + 1], D.BASE, 2, step=True)
    ch.series(ts, ep.net_stop3[s:k + 1], D.AMB, 2.6, step=True)
    cv.text((896, 58), f"now: AmbuGo {ep.net_stop3[k]:d}  ·  no priority {ep.net_stop0[k]:d}", 15, D.INK, "semibold", anchor="ra")
    for te, lab, col in EVENTS:
        if te <= t:
            xx = float(ch.X(te))
            cv.line([(xx, 54), (xx, 262)], alpha(col, .35), 1.2)
    x = 980
    title(cv, x, 8, "Accumulated since dispatch (t ≥ 760 s)")
    s3, s0 = ep.stop_cum3[k], ep.stop_cum0[k]
    rows = [("Stopped vehicle-seconds, AmbuGo", f"{s3:,.0f}", D.AMB),
            ("Stopped vehicle-seconds, no priority", f"{s0:,.0f}", D.BASE),
            ("Difference", f"{s3 - s0:+,.0f}  ({(s3 - s0) / max(s0, 1) * 100:+.1f} %)", D.BAD if s3 > s0 else D.AMB),
            ("Positive occupancy difference A (links)", f"{ep.A_cum[k]:,.0f} veh·s", D.INK),
            ("Signed occupancy difference ΔJ (links)", f"{ep.sJ_cum[k]:+,.0f} veh·s", D.INK)]
    y = 46
    for lab, v, c in rows:
        cv.text((x, y), lab, 15, D.INK)
        cv.text((1880, y), v, 16, c, "bold", anchor="ra")
        y += 30
    cv.eq((x, y + 4), r"$A=\Delta t\sum_{t}\sum_{g}\left[N^{\pi}_g(t)-N^{0}_g(t)\right]_+,\qquad \Delta J=\Delta t\sum_t\sum_g\left(N^{\pi}_g-N^0_g\right)$", 17)
    cv.text((x, y + 48), "Paper Eq. 23 form evaluated on the recorded road links (internal connectors and source queues not split out).", 13, D.MUTED)
    cv.text((x, y + 66), "A counts every displaced vehicle, moving or not, so A ≥ max(ΔJ, 0). All three missions contribute.", 13, D.MUTED)


# --------------------------------------------------------------------- J2
def j2_panel(cv, ctx):
    ep, t = ctx.ep, ctx.t
    names = {"129126240#7": "NW-1", "1215822559#0": "NW-2", "383450897#1": "NE", "383450898#0": "SW"}
    title(cv, 30, 8, "Junction J2 signal timeline", "two stop lines in series on the ambulance route (NW-1 then NW-2)")
    cols = {"129126240#7": D.AMB, "1215822559#0": (90, 150, 230), "383450897#1": D.NE_C, "383450898#0": D.SW_C}
    old = dict(D.APP_COL)
    D.APP_COL.update(cols)
    gantt(cv, ctx, ep.J2, 920, 1010, (30, 50, 1100, 230), reveal_t=t, names=names)
    D.APP_COL.clear()
    D.APP_COL.update(old)
    cv.text((30, 262), "a001 (escalated) is served at 955.5–968.5 s; a002 arrives at 971 s and must wait for the legal", 14, D.MUTED)
    cv.text((30, 282), "amber + all-red of NW-2 before its own green at 981.5 s, then again at NW-2 until 994.5 s.", 14, D.MUTED)
    x = 1150
    title(cv, x, 8, "Controller log, escalated mode")
    heads = ["t (s)", "approach", "δ (s)", "logged Ĉ", "gate"]
    xs = [x, x + 80, x + 200, x + 280, x + 400]
    for xx, hh in zip(xs, heads):
        cv.text((xx, 40), hh, 13, D.MUTED, "semibold")
    y = 62
    for tt in [971, 975, 981, 988, 994, 995]:
        e = ep.ev[tt]
        if tt > t + 1e-9:
            break
        nm = names.get(e["approach"], "J3")
        vals = [str(tt), nm, f"{e['switch_delay_s']:.0f}", f"{e['raw_burden_proxy_veh_s']:.1f}", "not applied"]
        for xx, v in zip(xs, vals):
            cv.text((xx, y), v, 15, D.ESC if xx == xs[4] else D.INK)
        y += 26
    cv.text((x, 232), "After escalation, B is not used: requests are admitted", 14, D.INK)
    cv.text((x, 252), "as in the aggressive comparator (JFAST). Ĉ is still logged.", 14, D.INK)
    cv.text((x, 280), "Signal legality (min green 8 s, amber, all-red) still applies.", 14, D.MUTED)


# --------------------------------------------------------------- hospital
def hospital(cv, ctx):
    ep, t = ctx.ep, ctx.t
    speed_like = Chart(cv, (90, 50, 900, 280), (760, 1090), (0, 60))
    k = ep.k(t)
    sl = slice(ep.k(760), k + 1)
    title(cv, 30, 12, "Ambulance a002 speed", "recorded, km/h")
    speed_like.axes(xticks=range(780, 1091, 40), yticks=[0, 25, 50], xlabel="simulation time t (s)")
    for te, lab, col in EVENTS:
        if te <= t:
            xx = float(speed_like.X(te))
            cv.line([(xx, 50), (xx, 280)], alpha(col, .45), 1.5)
            cv.text((xx + 3, 52), lab, 12, col, "semibold")
    speed_like.series(ep.t[sl], ep.amb_speed[sl], D.AMB, 2.6)
    speed_like.dot(t, ep.interp(ep.amb_speed, t), D.AMB)
    x = 990
    title(cv, x, 12, "Mission travel time (dispatch → hospital road approach)")
    m3 = {m["id"]: m for m in ep.P3.meta["missions"]}
    m0 = {m["id"]: m for m in ep.P0.meta["missions"]}
    a = D.fade(ctx.ls, 0, 1) if t >= 1083.5 else .35
    y = 56
    for i, mid in enumerate(["a000", "a001", "a002"]):
        p3, p0 = m3[mid]["travel_s"], m0[mid]["travel_s"]
        foc = mid == "a002"
        cv.text((x, y + 12), mid + (" (focus)" if foc else ""), 15, D.INK, "bold" if foc else "regular")
        sc = 1.0
        cv.rect((x + 110, y, x + 110 + p3 * sc, y + 16), fill=alpha(D.AMB, a if foc else .55))
        cv.rect((x + 110, y + 19, x + 110 + p0 * sc, y + 35), fill=alpha(D.BASE, a if foc else .55))
        cv.text((x + 116 + p3 * sc, y - 1), f"{p3:.1f} s", 13, D.AMB, "bold")
        cv.text((x + 116 + p0 * sc, y + 19), f"{p0:.1f} s  (−{p0 - p3:.0f} s with AmbuGo)", 13, D.MUTED, "bold")
        y += 62
    chip(cv, x, 250, D.AMB)
    cv.text((x + 20, 248), "AmbuGo (B = 1000)", 14, D.INK)
    chip(cv, x + 200, 250, D.BASE)
    cv.text((x + 220, 248), "no priority (routing only)", 14, D.INK)
    cv.text((x, 280), "One synthetic episode (seed 19). Not a field measurement; see summary for the 1,728-run study.", 13, D.MUTED)


def reroute(cv, ctx):
    ep = ctx.ep
    u = ep.decisions["route_updates"][-1]
    title(cv, 30, 10, "Re-plan at link entry, t = 1055 s (escalated mode: time-only scoring)")
    c0, c1 = u["candidates"]
    rows = [("A  via FORTIS_RB_ARC_0 → 129041216#0", c0, D.AMB), ("B  via −129040925#4 → 1261385438 (incumbent)", c1, D.BASE)]
    y = 50
    for lab, c, col in rows:
        chip(cv, 30, y + 3, col)
        cv.text((52, y), lab, 16, D.INK, "semibold")
        cv.text((560, y), f"T̂ = {c['forecast_remaining_time_s']:.2f} s", 17, col, "bold")
        if c["chosen"]:
            tick(cv, (720, y + 10), True, 10)
        y += 34
    cv.eq((30, 130), r"$\mathrm{keep\ incumbent\ iff}\quad \hat T_{\mathrm{inc}}-\hat T_{\mathrm{best}}\ \leq\ 10\ \mathrm{s}\quad(\mathrm{switching\ hysteresis})$", 21)
    d = c1["forecast_remaining_time_s"] - c0["forecast_remaining_time_s"]
    cv.eq((30, 180), f"${c1['forecast_remaining_time_s']:.3f}-{c0['forecast_remaining_time_s']:.3f}={d:.3f}\\ \\mathrm{{s}}\\ >\\ 10\\ \\mathrm{{s}}\\ \\Rightarrow\\ \\mathrm{{switch\\ to\\ A}}$", 21)
    cv.text((30, 236), "The difference is confined to the last ≈300 m around the hospital roundabout; no city-wide detour occurs.", 14, D.MUTED)
    cv.text((30, 258), "ETA values are forecasts among evaluated candidates, not proof of a globally optimal route.", 14, D.MUTED)
    x = 1000
    speed = Chart(cv, (x + 60, 50, 1880, 280), (1040, 1090), (0, 60))
    k = ep.k(ctx.t)
    sl = slice(ep.k(1040), k + 1)
    title(cv, x, 12, "Speed on the hospital approach", "km/h")
    speed.axes(xticks=range(1040, 1091, 10), yticks=[0, 25, 50], xlabel="t (s)")
    speed.series(ep.t[sl], ep.amb_speed[sl], D.ESC, 2.6)
    speed.dot(ctx.t, ep.interp(ep.amb_speed, ctx.t), D.ESC)
