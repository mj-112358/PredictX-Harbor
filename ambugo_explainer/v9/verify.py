#!/usr/bin/env python3
"""Numerical verification: every number the film puts on screen, recomputed from the logs and the archive.

    python verify.py        # writes reports/verification.json and reports/VERIFICATION.md; exit 1 on any failure

Sources (all under data/):
  episodes/P3               instrumented replay of archived run seed 40, load 0.95, 1 mission, policy P3 (JSEL)
  episodes/P0               same scenario, policy P0 (S0, no priority)
  episodes/P3_withholdJ21   P3 with the single J21 request withheld from t = 4132 (explanatory counterfactual)
  archive/                  files copied unmodified from AmbuGo_First_Draft_Reproducibility_v9.zip
"""
from pathlib import Path
import csv
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data"
checks = []


def check(scene, label, shown, value, source, tol=0.051):
    """shown: the value as printed in the film (number or string); value: recomputed."""
    if isinstance(shown, (int, float)) and isinstance(value, (int, float, np.floating, np.integer)):
        ok = abs(float(shown) - float(value)) <= tol
    else:
        ok = shown == value
    checks.append(dict(scene=scene, label=label, shown=shown, recomputed=float(value) if isinstance(value, (np.floating, np.integer, float, int)) and not isinstance(value, bool) else value,
                       source=source, ok=bool(ok)))


def J(p):
    return json.loads((D / p).read_text())


# ------------------------------------------------------------------ inputs
P3, P0, WH = (J(f"episodes/{n}/result.json") for n in ("P3", "P0", "P3_withholdJ21"))
pol = J("archive/selected_policy.json")
arch = J("archive/records_s40_x095_m1.json")
main = J("archive/sumo_main_records.json")
agg = J("archive/sumo_analysis_aggregate.json")
gates = {(g["t"], g["junction"]): g for g in J("episodes/P3/gates.json")}
plans = J("episodes/P3/plans.json")
edges = {r["edge_id"]: r for r in csv.DictReader(open(D / "fixture/edges.csv"))}
nodes = list(csv.DictReader(open(D / "fixture/nodes.csv")))

# ------------------------------------------------------------------ reproduction
for n in ("P3", "P0"):
    ac = J(f"episodes/{n}/archive_check.json")
    check("all", f"instrumented {n} replay reproduces archived run on all {len(ac)} compared fields", True, all(ac.values()),
          f"episodes/{n}/archive_check.json")
    a = [r for r in arch if r["policy"] == n][0]
    r = J(f"episodes/{n}/result.json")
    check("all", f"{n} mission travel time equals archived record", a["mission_rows"][0]["travel_s"], r["mission_rows"][0]["travel_s"],
          "archive/records_s40_x095_m1.json", 0)
cfg = P3["config"]
check("routes", "harm budget B", 10000, cfg["harm_budget"], "episodes/P3/result.json config", 0)
check("routes", "epsilon", 0, cfg["epsilon"], "episodes/P3/result.json config", 0)
check("network", "progress threshold gamma (s)", 20, cfg["progress_guard_s"], "episodes/P3/result.json config", 0)
check("routes", "selected policy model hash = frozen controller", "471a22911a2de9773fcd839872b3ba1ec5e00412c0ecc79142503e72f41cecf2", pol["model_sha256"], "archive/selected_policy.json")
check("routes", "action set = strong {(0,0),(30,60)}", "strong", cfg["action_mode"], "episodes/P3/result.json config")
check("routes", "selected_policy.json harm budget", 10000, pol["selected_configuration"]["harm_budget"],
      "archive/selected_policy.json", 0)

# ------------------------------------------------------------------ district
tl = [n for n in nodes if n.get("node_type", n.get("type", "")).lower().startswith(("signal", "tl", "junction"))]
check("problem", "regular road links", 59, len(edges), "fixture/edges.csv", 0)
check("problem", "signalised junctions (J00..J33)", 16, len({e["to_node"] for e in edges.values() if e["to_node"].startswith("J")}), "fixture/edges.csv", 0)

# ------------------------------------------------------------------ route choice at dispatch
p0 = plans[0]
check("routes", "first plan time", 4065, p0["t"], "episodes/P3/plans.json[0]", 0)
by = {}
for c in p0["candidates"]:
    by.setdefault(tuple(c["route"]), {})[c["action"][1] > 0] = c
inc = [r for r, v in by.items() if v[True]["incumbent"]][0]
others = [r for r in by if r != inc]
order = [inc] + others
names = dict(zip("ABCD", order))
shown = {"A": (590.9, 250.8, 2528), "B": (590.9, 259.8, 2715), "C": (616.9, 248.0, 3330), "D": (590.9, 259.6, 1402)}
for n, (a, b, c) in shown.items():
    v = by[names[n]]
    check("routes", f"route {n} forecast, normal signals", a, v[False]["eta"], "plans.json[0]")
    check("routes", f"route {n} forecast, with request", b, v[True]["eta"], "plans.json[0]")
    check("routes", f"route {n} forecast burden (veh·s, rounded)", c, v[True]["burden"], "plans.json[0]", .5)
best = min(c["eta"] for c in p0["candidates"])
elig = [n for n in "ABCD" if by[names[n]][True]["eta"] <= best + cfg["epsilon"]]
check("routes", "eligible pairs at epsilon = 0", ["C"], elig, "plans.json[0]")
check("routes", "chosen route is C", list(names["C"]), p0["chosen_route"], "plans.json[0].chosen_route")
check("routes", "C is the slowest without priority", "C", max("ABCD", key=lambda n: by[names[n]][False]["eta"]), "plans.json[0]")

# ------------------------------------------------------------------ re-check at 4091
p1 = [p for p in plans if p["t"] == 4091][0]
req = {tuple(c["route"]): c for c in p1["candidates"] if c["action"][1] > 0}
i1 = [c for c in req.values() if c["incumbent"]][0]
b1 = min(req.values(), key=lambda c: c["eta"])
check("recheck", "route A forecast at 4091", 219.9, b1["eta"], "plans.json t=4091")
check("recheck", "route C (incumbent) forecast at 4091", 232.5, i1["eta"], "plans.json t=4091")
check("recheck", "gap (s)", 12.6, i1["eta"] - b1["eta"], "plans.json t=4091")
check("recheck", "best route at 4091 is route A", list(names["A"][1:]), b1["route"], "plans.json t=4091")
check("recheck", "switched to A (chosen_route)", list(names["A"][1:]), p1["chosen_route"], "plans.json t=4091")

# ------------------------------------------------------------------ gates
g = gates[(4065, "J10")]
check("travel", "J10 admitted", True, g["admitted"], "gates.json t=4065 J10")
check("travel", "J10 forecast gain (s)", 39, g["benefit"], "gates.json t=4065 J10", .5)
check("travel", "J10 burden (veh·s)", 227, g["burden"], "gates.json t=4065 J10", .5)

g = gates[(4091, "J11")]
check("refuse", "J11 admitted", False, g["admitted"], "gates.json t=4091 J11")
check("refuse", "J11 forecast gain (s)", 3.6, g["benefit"], "gates.json t=4091 J11")
check("refuse", "J11 gain fails the 5 s test", True, g["benefit"] < 5, "gates.json t=4091 J11")
check("refuse", "J11 Eq.19 left side (s)", 30.5, g["feasible_lhs"], "gates.json t=4091 J11")
check("refuse", "J11 Eq.19 recomputed [h+d-δ]+ + d + 2", round(g["feasible_lhs"], 6),
      round(max(0, 30 + g["d_hat"] - g["switch_delay"]) + g["d_hat"] + 2, 6), "gates.json t=4091 J11", 1e-6)
check("refuse", "J11 downstream count (veh)", 5, g["headroom_obs"], "gates.json t=4091 J11", 0)
check("refuse", "J11 headroom cap = 0.85 K(E_J11_J21)", 45.0, .85 * float(edges["E_J11_J21"]["usable_storage_veh"]), "fixture/edges.csv")
check("refuse", "J11 burden (veh·s)", 306, g["burden"], "gates.json t=4091 J11", .5)
check("refuse", "J11 pedestrian interval remaining (s)", 5, 25 - g["signal_elapsed"], "gates.json t=4091 J11", 0)
check("refuse", "J11 signal mode at request", "ped", g["signal_mode"], "gates.json t=4091 J11")
check("refuse", "ambulance ~27 s from stop line (free-flow)", 27, g["arrival"], "gates.json t=4091 J11", .5)
check("refuse", "normal cycle reaches ambulance ~3.6 s after arrival", 3.6, g["nominal_wait"], "gates.json t=4091 J11")

g = gates[(4132, "J21")]
A = {a["edge"]: a for a in g["arms"]}
check("burden", "J21 admitted", True, g["admitted"], "gates.json t=4132 J21")
Dv = min(60, max(5, 30 + g["d_hat"]))
check("burden", "D = min(60, max(5, 30 + d̂))", 44.2, Dv, "gates.json t=4132 J21")
check("burden", "logged D equals formula", round(g["D"], 9), round(Dv, 9), "gates.json t=4132 J21", 1e-9)
check("burden", "d̂ (s)", 14.2, g["d_hat"], "gates.json t=4132 J21")
check("burden", "own queue estimate q̂ (veh)", 5.9, g["q_hat_own"], "gates.json t=4132 J21")
tot = 0
for e, nm, sq, sa in [("E_J22_J21", "east", 342, 1020), ("E_J31_J21", "south", 443, 1249), ("E_J20_J21", "west", 34, 239)]:
    a = A[e]
    tq = a["q_hat"] * Dv
    ta = a["lam"] * Dv ** 2 / (2 * max(1 - a["X"], .05))
    check("burden", f"{nm}: q̂·D recomputed = logged", round(a["term_queue"], 6), round(tq, 6), "gates.json arms", 1e-6)
    check("burden", f"{nm}: λD²/(2max(1−X,.05)) recomputed = logged", round(a["term_arrival"], 6), round(ta, 6), "gates.json arms", 1e-6)
    check("burden", f"{nm}: waiting term shown", sq, tq, "gates.json arms", .5)
    check("burden", f"{nm}: arrivals term shown", sa, ta, "gates.json arms", .5)
    tot += tq + ta
check("burden", "east q̂ (veh)", 7.7, A["E_J22_J21"]["q_hat"], "gates.json arms")
check("burden", "east observed count", 13, A["E_J22_J21"]["n_obs"], "gates.json arms", 0)
check("burden", "east λ (veh/s)", 0.140, A["E_J22_J21"]["lam"], "gates.json arms", .0005)
check("burden", "east X", 0.87, A["E_J22_J21"]["X"], "gates.json arms", .005)
check("burden", "Ĉ = sum of arm terms", round(g["burden"], 6), round(tot, 6), "gates.json t=4132 J21", 1e-6)
check("burden", "Ĉ shown (veh·s)", 3326, g["burden"], "gates.json t=4132 J21", .5)
check("burden", "gain (s)", 43.4, g["benefit"], "gates.json t=4132 J21")
check("burden", "Eq.19 left side (s)", 50.4, g["feasible_lhs"], "gates.json t=4132 J21")
check("burden", "Eq.19 recomputed", round(g["feasible_lhs"], 6), round(max(0, 30 + g["d_hat"] - g["switch_delay"]) + g["d_hat"] + 2, 6), "gates.json", 1e-6)
check("burden", "downstream count", 8, g["headroom_obs"], "gates.json", 0)
check("burden", "headroom cap = 0.85 K(E_J21_J22)", 112.2, .85 * float(edges["E_J21_J22"]["usable_storage_veh"]), "fixture/edges.csv")
check("burden", "legal switch delay (s)", 10, g["switch_delay"], "gates.json", 0)
check("j21", "east green elapsed at 4132 (s)", 3, g["signal_elapsed"], "gates.json", 0)
check("j21", "east has green at 4132", "E_J22_J21", g["signal_edge"], "gates.json")
for t, j, lab in [(4175, "J22", 89), (4224, "J12", 87)]:
    gg = gates[(t, j)]
    check("network", f"{j} admitted", True, gg["admitted"], f"gates.json t={t} {j}")
    check("network", f"{j} gain (s)", lab, gg["benefit"], f"gates.json t={t} {j}", .5)
check("network", "J22 burden", 348, gates[(4175, "J22")]["burden"], "gates.json", .5)
check("network", "J12 burden", 185, gates[(4224, "J12")]["burden"], "gates.json", .5)

# ------------------------------------------------------------------ signals
def sig(name):
    s = {}
    for t, j, mode, e, el, rq, am in J(f"episodes/{name}/signals.json"):
        s[(t, j)] = (mode, e)
    return s


def first(s, j, t0, mode, edge=None):
    t = t0
    while (t, j) in s:
        m, e = s[(t, j)]
        if m == mode and (edge is None or e == edge):
            return t
        t += 1
    return None


sg, sw = sig("P3"), sig("P3_withholdJ21")
check("s1", "granted: east green ends (amber) at", 4137, first(sg, "J21", 4132, "amber"), "P3 signals.json", 0)
check("s1", "granted: all-red at", 4140, first(sg, "J21", 4132, "allred"), "P3 signals.json", 0)
check("s1", "granted: ambulance (north) green at", 4142, first(sg, "J21", 4132, "green", "E_J11_J21"), "P3 signals.json", 0)
check("compare", "withheld: east keeps green until (amber at)", 4158, first(sw, "J21", 4132, "amber"), "P3_withholdJ21 signals.json", 0)
check("compare", "withheld: ambulance (north) green at", 4228, first(sw, "J21", 4132, "green", "E_J11_J21"), "P3_withholdJ21 signals.json", 0)
check("cycle", "J11 pedestrian interval ends; north green at", 4096, first(sg, "J11", 4091, "green"), "P3 signals.json", 0)
check("cycle", "J11 west (ambulance) green at", 4122, first(sg, "J11", 4091, "green", "E_J10_J11"), "P3 signals.json", 0)

# legal stage durations over the whole logged window, every junction, all three runs
for name in ("P3", "P0", "P3_withholdJ21"):
    rows = J(f"episodes/{name}/signals.json")
    spans, cur = [], {}
    for t, j, mode, e, el, rq, am in rows:
        key = (mode, e)
        if j in cur and cur[j][0] == key and cur[j][2] == t - 1:
            cur[j][2] = t
        else:
            if j in cur:
                spans.append((j, cur[j][0][0], cur[j][1], cur[j][2]))
            cur[j] = [key, t, t]
    t_first, t_last = rows[0][0], rows[-1][0]
    full = [s for s in spans if s[2] > t_first]          # drop spans cut by the start of the log
    am = {s[3] - s[2] + 1 for s in full if s[1] == "amber"}
    ar = {s[3] - s[2] + 1 for s in full if s[1] == "allred"}
    pd = {s[3] - s[2] + 1 for s in full if s[1] == "ped"}
    gmin = min(s[3] - s[2] + 1 for s in full if s[1] == "green")
    check("cycle", f"{name}: every amber lasts 3 s", [3], sorted(am), f"episodes/{name}/signals.json")
    check("cycle", f"{name}: every all-red lasts 2 s", [2], sorted(ar), f"episodes/{name}/signals.json")
    check("cycle", f"{name}: every pedestrian interval lasts 25 s", [25], sorted(pd), f"episodes/{name}/signals.json")
    check("cycle", f"{name}: shortest green ≥ 8 s", True, gmin >= 8, f"episodes/{name}/signals.json (min {gmin} s)")
    nonlegal = 0
    for (j, m, a, b), (j2, m2, a2, b2) in zip(full, full[1:]):
        pass
    # red→green without amber + all-red: a green that follows a green of another approach directly
    by_j = {}
    for s in full:
        by_j.setdefault(s[0], []).append(s)
    for j, ss in by_j.items():
        for x, y in zip(ss, ss[1:]):
            if x[1] == "green" and y[1] == "green":
                nonlegal += 1
            if x[1] == "amber" and y[1] != "allred":
                nonlegal += 1
    check("cycle", f"{name}: no green→green or amber→(not all-red) transitions", 0, nonlegal, f"episodes/{name}/signals.json", 0)

# ------------------------------------------------------------------ trajectories: crossings, queues
sys.path.insert(0, str(HERE))


def traj(name):
    z = np.load(D / f"episodes/{name}/trajectories.npz")
    ix = json.loads((D / f"episodes/{name}/trajectory_index.json").read_text())
    lanes = np.array([l.rsplit("_", 1)[0] if not l.startswith(":") else ":" for l in ix["lanes"]])
    return z, lanes, ix["vehicles"].index("a000")


def amb_edges(name):
    z, lanes, amb = traj(name)
    out = []
    for k, t in enumerate(z["t"]):
        lo, hi = z["offsets"][k:k + 2]
        w = np.flatnonzero(z["vehicle"][lo:hi] == amb)
        if len(w):
            e = lanes[z["lane"][lo + w[0]]]
            if not out or out[-1][1] != e:
                out.append((float(t), e))
    return out


def leave(name, edge):
    seq = amb_edges(name)
    for (t, e), (t2, e2) in zip(seq, seq[1:]):
        if e == edge:
            return t2
    return None


check("s1", "granted: ambulance leaves J21 north approach at about", 4171, leave("P3", "E_J11_J21"), "P3 trajectories.npz", 1.0)
check("compare", "withheld: ambulance leaves J21 north approach at about", 4242, leave("P3_withholdJ21", "E_J11_J21"), "P3_withholdJ21 trajectories.npz", 1.0)
check("recheck", "ambulance enters E_J10_J11 at", 4091, [t for t, e in amb_edges("P3") if e == "E_J10_J11"][0], "P3 trajectories.npz", .5)


def stopped(name, arms, t0, t1):
    z, lanes, amb = traj(name)
    tot = {e: 0 for e in arms}
    mx = {e: 0 for e in arms}
    for k, t in enumerate(z["t"]):
        if not (t0 <= t <= t1) or t % 5:
            continue
        lo, hi = z["offsets"][k:k + 2]
        le, sp, v = lanes[z["lane"][lo:hi]], z["speed"][lo:hi], z["vehicle"][lo:hi]
        for e in arms:
            tot[e] += int(((le == e) & (sp < .1) & (v != amb)).sum()) * 5
            mx[e] = max(mx[e], int(((le == e) & (v != amb)).sum()))
    return tot, mx


arms = ["E_J22_J21", "E_J31_J21", "E_J20_J21", "E_J11_J21"]
sgr, mgr = stopped("P3", arms, 4125, 4300)
swh, mwh = stopped("P3_withholdJ21", arms, 4125, 4300)
for e, nm, a, b in [("E_J22_J21", "east", 1775, 670), ("E_J31_J21", "south", 1080, 530), ("E_J20_J21", "west", 110, 615), ("E_J11_J21", "north", 90, 605)]:
    check("compare", f"stopped veh·s {nm}, priority granted", a, sgr[e], "P3 trajectories.npz (5 s samples)", 0)
    check("compare", f"stopped veh·s {nm}, normal cycle", b, swh[e], "P3_withholdJ21 trajectories.npz (5 s samples)", 0)
peak = max(max(mgr.values()), max(mwh.values()))
check("compare", "most vehicles on one J21 approach at once (either replay)", 19, peak, "trajectories.npz", 0)
check("compare", "smallest J21 approach storage (veh)", 53, min(float(edges[e]["usable_storage_veh"]) for e in arms), "fixture/edges.csv", 0)
check("compare", "largest J21 approach storage (veh)", 132, max(float(edges[e]["usable_storage_veh"]) for e in arms), "fixture/edges.csv", 0)
check("compare", "no approach reached its storage", True,
      all(max(mgr[e], mwh[e]) < float(edges[e]["usable_storage_veh"]) for e in arms), "trajectories.npz + edges.csv")

# ------------------------------------------------------------------ occupancy contrasts
def occupancy(name):
    z = np.load(D / f"episodes/{name}/trace.npz")
    return z["counts"].astype(float), float(z["step"]), float(z["start"]), list(z["partition"])


def contrast(a, b, t0, t1):
    ca, dt, st, part = occupancy(a)
    cb, _, _, _ = occupancy(b)
    i0, i1 = int((t0 - st) / dt), int((t1 - st) / dt)
    d = ca[i0:i1] - cb[i0:i1]
    return float(np.maximum(d, 0).sum() * dt), float(d.sum() * dt), d, part


Apos, sgn, d, part = contrast("P3", "P3_withholdJ21", 4132, 4400)
check("result", "A (positive spatial exposure), granted − withheld, 4132–4400 (veh·s)", 5992, Apos, "trace.npz (Eq. 23 quadrature)", .5)
check("result", "signed residence change, same window (veh·s)", -74, sgn, "trace.npz", .5)
per = dict(zip(part, np.maximum(d, 0).sum(0) * .5))
check("network", "J22→J23 link carries extra occupancy (two junctions away)", True, per.get("E_J22_J23", 0) > 0, "trace.npz")
check("network", "J23→J22 link carries extra occupancy", True, per.get("E_J23_J22", 0) > 0, "trace.npz")

# ------------------------------------------------------------------ progress escalation
pg = P3["progress_guard_log"]
check("network", "one escalation in this episode", 1, len(pg), "result.json progress_guard_log", 0)
check("network", "escalation time", 4256, pg[0]["t"], "result.json progress_guard_log", 0)
check("network", "deficit at escalation (s)", 20.5, pg[0]["delay_s"], "result.json progress_guard_log")
check("network", "Eq.21 recomputed t − d − P", round(pg[0]["delay_s"], 9), round(pg[0]["t"] - pg[0]["dispatch"] - pg[0]["executed_freeflow_progress_s"], 9),
      "result.json progress_guard_log", 1e-9)
check("network", "J13 request at 4260 is in escalated (P2) mode", "P2", gates[(4260, "J13")]["policy"], "gates.json t=4260 J13")

# ------------------------------------------------------------------ outcomes of this episode
check("result", "JSEL arrival (s)", 4316.5, P3["mission_rows"][0]["arrival"], "result.json", 0)
check("result", "JSEL travel (s)", 252.5, P3["mission_rows"][0]["travel_s"], "result.json", 0)
check("result", "J21-withheld travel (s)", 323.5, WH["mission_rows"][0]["travel_s"], "P3_withholdJ21/result.json", 0)
check("result", "S0 travel (s)", 572.0, P0["mission_rows"][0]["travel_s"], "P0/result.json", 0)
check("problem", "dispatch time", 4064, P3["mission_rows"][0]["dispatch"], "result.json", 0)
check("problem", "origin", "BW1", P3["mission_rows"][0]["origin"], "result.json")
check("all", "same ordinary demand schedule in P3, P0 and the branch", True,
      P3["schedule_sha256"] == P0["schedule_sha256"] == WH["schedule_sha256"], "result.json schedule_sha256")

# ------------------------------------------------------------------ paper aggregates (archive)
a = agg["aggregate"]
check("result", "S0 balanced mean (s)", 401.57, a["P0"]["time_s"]["mean"], "archive/sumo_analysis_aggregate.json", .005)
check("result", "JFAST balanced mean (s)", 211.19, a["P2"]["time_s"]["mean"], "archive/sumo_analysis_aggregate.json", .005)
check("result", "JSEL balanced mean (s)", 222.29, a["P3"]["time_s"]["mean"], "archive/sumo_analysis_aggregate.json", .005)
check("result", "JSEL 44.6% below S0", 44.6, 100 * (1 - a["P3"]["time_s"]["mean"] / a["P0"]["time_s"]["mean"]), "archive", .05)
c = [z for z in agg["aggregate_contrasts"] if z["comparison"] == "P3-P2"][0]["metrics"]
for k, v, lab in [("time_s", (11.10, 9.47, 12.94), "JSEL − JFAST ambulance time (s)"), ("A_vh", (-3.84,), "positive spatial exposure (veh·h)"),
                  ("Dplus_vh", (-1.64,), "positive individual residence (veh·h)")]:
    check("result", lab + " mean", v[0], c[k]["mean"], "archive/sumo_analysis_aggregate.json", .005)
    if len(v) > 1:
        check("result", lab + " 95% CI low", v[1], c[k]["lo"], "archive", .005)
        check("result", lab + " 95% CI high", v[2], c[k]["hi"], "archive", .005)
check("result", "runs in the main grid", 1728, len(main), "archive/sumo_main_records.json", 0)
p3 = [r for r in main if r["policy"] == "P3"]
check("refuse", "archived JSEL runs in the main grid", 432, len(p3), "archive/sumo_main_records.json", 0)
check("refuse", "JSEL refusals on burden (ticks, all runs)", 0, sum(r["rejected_harm_ticks"] for r in p3), "archive/sumo_main_records.json", 0)
check("refuse", "JSEL refusals on downstream space (ticks, all runs)", 0, sum(r["rejected_headroom_ticks"] for r in p3), "archive/sumo_main_records.json", 0)
check("refuse", "JSEL refusals on gain/feasibility (ticks) > 0", True, sum(r["rejected_benefit_ticks"] for r in p3) > 0, "archive/sumo_main_records.json")
nm = sum(len(r["mission_rows"]) for r in p3)
ne = sum(len(r["progress_guard_log"]) for r in p3)
check("result", "JSEL mission evaluations", 2160, nm, "archive/sumo_main_records.json", 0)
check("result", "JSEL escalated share (%)", 51.6, 100 * ne / nm, "archive/sumo_main_records.json", .05)

# ------------------------------------------------------------------ short map film (mapfilm.py)
P2 = J("episodes/P2/result.json")
ac2 = J("episodes/P2/archive_check.json")
check("short", f"instrumented P2 (JFAST) replay reproduces archived run on all {len(ac2)} fields", True, all(ac2.values()),
      "episodes/P2/archive_check.json")
check("short", "P2 mission time equals archived record", [r for r in arch if r["policy"] == "P2"][0]["mission_rows"][0]["travel_s"],
      P2["mission_rows"][0]["travel_s"], "archive/records_s40_x095_m1.json", 0)
s2 = sig("P2")
check("short", "P2 and P3 signals identical up to the J11 decision (4091)", True,
      all(sg[k] == s2.get(k) for k in sg if k[0] <= 4091), "P3/P2 signals.json")
check("short", "JFAST: J11 ambulance (west) green at", 4096, first(s2, "J11", 4091, "green", "E_J10_J11"), "P2 signals.json", 0)
check("short", "JFAST: J11 cross road (north) green at", 4160, first(s2, "J11", 4091, "green", "E_J01_J11"), "P2 signals.json", 0)
check("short", "cross road green delayed by (s)", 64, first(s2, "J11", 4091, "green", "E_J01_J11") - first(sg, "J11", 4091, "green", "E_J01_J11"),
      "P2/P3 signals.json", 0)
la, lb = leave("P3", "E_J10_J11"), leave("P2", "E_J10_J11")
check("short", "ambulance time saved at J11 under JFAST (s)", 11, la - lb, "P3/P2 trajectories.npz", 0)


def stopped_half(name, e, t0, t1):
    z, lanes, amb = traj(name)
    tot = 0.0
    for k, t in enumerate(z["t"]):
        if t0 <= t <= t1:
            lo, hi = z["offsets"][k:k + 2]
            tot += float(((lanes[z["lane"][lo:hi]] == e) & (z["speed"][lo:hi] < .1) & (z["vehicle"][lo:hi] != amb)).sum()) * .5
    return tot


check("short", "cross-road stopped veh·s 4085–4200, AmbuGo", 79, stopped_half("P3", "E_J01_J11", 4085, 4200), "P3 trajectories (0.5 s)", .5)
check("short", "cross-road stopped veh·s 4085–4200, JFAST", 463, stopped_half("P2", "E_J01_J11", 4085, 4200), "P2 trajectories (0.5 s)", .5)
check("short", "'forty seconds later' (J21 decision − J11 decision, s)", 41, 4132 - 4091, "gates.json", 0)
check("short", "J21 normal-cycle wait ~43 s", 43, gates[(4132, "J21")]["nominal_wait"], "gates.json", .5)
check("short", "J11 normal-cycle wait shown (s)", 3.6, gates[(4091, "J11")]["nominal_wait"], "gates.json")
check("short", "J11 room ahead cap shown (rounded)", 45, gates[(4091, "J11")]["headroom_cap"], "gates.json", .5)
check("short", "J21 room ahead cap shown (rounded)", 112, gates[(4132, "J21")]["headroom_cap"], "gates.json", .5)
for e, v in [("E_J22_J21", 1362), ("E_J31_J21", 1692), ("E_J20_J21", 273)]:
    a = {x["edge"]: x for x in gates[(4132, "J21")]["arms"]}[e]
    check("short", f"J21 {e} extra waiting shown (veh·s)", v, a["term_queue"] + a["term_arrival"], "gates.json", .5)

# ------------------------------------------------------------------ report
fails = [c for c in checks if not c["ok"]]
out = HERE / "reports"
out.mkdir(exist_ok=True)
(out / "verification.json").write_text(json.dumps(dict(n=len(checks), failed=len(fails), checks=checks), indent=1, default=str))
lines = ["# Numerical verification report", "",
         f"Generated by `python verify.py`. **{len(checks)} checks, {len(fails)} failed.**", "",
         "Every number printed in the film is listed with the value recomputed from the logged replay or the archive.",
         "Tolerance is the display rounding unless the row is an exact recomputation of a logged quantity.", "",
         "| scene | quantity | shown | recomputed | source | ok |", "|---|---|---|---|---|---|"]
for c in checks:
    r = c["recomputed"]
    rs = f"{r:.4f}" if isinstance(r, float) else str(r)
    lines.append(f"| {c['scene']} | {c['label']} | {c['shown']} | {rs} | {c['source']} | {'✓' if c['ok'] else '**FAIL**'} |")
(out / "VERIFICATION.md").write_text("\n".join(lines) + "\n")
print(f"{len(checks)} checks, {len(fails)} failed")
for c in fails:
    print("FAIL", c)
sys.exit(1 if fails else 0)
