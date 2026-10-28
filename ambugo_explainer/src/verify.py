#!/usr/bin/env python3
"""Re-derive every number the explainer film states, from the raw replays and logs.

Exits non-zero if any claim made on screen does not hold. Writes output/verification.json.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Episode, ROOT  # noqa: E402

checks = {}


def check(name, ok, value=None):
    checks[name] = {"pass": bool(ok), "value": value}
    print(("PASS " if ok else "FAIL ") + name + ("" if value is None else f"  ->  {value}"))


def main():
    ep = Episode()
    ev = ep.ev
    near = lambda a, b, tol=1e-6: abs(a - b) <= tol

    # --- Eq. 20 and Eq. 19 reproduce the controller log exactly, for every logged request
    worst, feas_ok = 0.0, True
    for e in ep.all_events:
        h, l = e["action"]
        De = min(l, max(5, h + e["queue_clearance_s"]))
        tot = 0.0
        for a in e["conflicting_approaches"]:
            c = a["queue_proxy"] * De + a["arrival_rate"] * De ** 2 / (2 * max(1 - a["nominal_design_X"], .05))
            worst = max(worst, abs(c - a["burden_proxy_veh_s"]))
            tot += c
        worst = max(worst, abs(tot - e["raw_burden_proxy_veh_s"]))
        lhs = max(h + e["queue_clearance_s"] - e["switch_delay_s"], 0) + e["queue_clearance_s"] + 2
        feas_ok &= (lhs <= l) == e["pulse_feasible"]
    check("eq20_reproduces_log_all_events", worst < 1e-9, f"max abs error {worst:.2e} veh*s")
    check("eq19_reproduces_log_all_events", feas_ok)

    # --- scene 4-6: refusal and admission at PCL Chowk
    check("refused_785_to_791", all(ev[t]["reason"] == "burden_budget_exceeded" for t in range(785, 792)))
    check("admitted_792", ev[792]["reason"] == "selective_gate_passed")
    check("burden_785", near(ev[785]["raw_burden_proxy_veh_s"], 1044.521, 1e-3), round(ev[785]["raw_burden_proxy_veh_s"], 1))
    check("burden_787", near(ev[787]["raw_burden_proxy_veh_s"], 1184.43, 1e-1), round(ev[787]["raw_burden_proxy_veh_s"], 1))
    check("burden_792", near(ev[792]["raw_burden_proxy_veh_s"], 913.29, 1e-1), round(ev[792]["raw_burden_proxy_veh_s"], 1))
    De785 = min(60, max(5, 30 + ev[785]["queue_clearance_s"]))
    check("De_785_38.3", near(De785, 38.32, 1e-2), round(De785, 2))
    check("benefit_785_72.6", near(ev[785]["predicted_benefit_s"], 72.65, 1e-2))
    check("switch_delay_785_11_to_791_5", ev[785]["switch_delay_s"] == 11 and ev[791]["switch_delay_s"] == 5)
    q = lambda t, edge: next(a["queue_proxy"] for a in ev[t]["conflicting_approaches"] if a["edge"] == edge)
    check("SW_qhat_6.37_to_9.37_at_787", near(q(786, "172382686#4"), 6.3725, 1e-3) and near(q(787, "172382686#4"), 9.3725, 1e-3))
    check("NW_qhat_1.94_to_0.94_at_792", near(q(791, "172380615#1"), 1.938, 1e-3) and near(q(792, "172380615#1"), 0.938, 1e-3))
    check("SW_qhat_9.37_to_8.37_at_792", near(q(791, "172382686#4"), 9.3725, 1e-3) and near(q(792, "172382686#4"), 8.3725, 1e-3))
    check("own_qhat_4.78_to_2.78_dhat_10.52_to_6.12",
          near(ev[791]["queue_proxy"], 4.7829, 1e-3) and near(ev[792]["queue_proxy"], 2.7829, 1e-3)
          and near(ev[791]["queue_clearance_s"], 10.5223, 1e-3) and near(ev[792]["queue_clearance_s"], 6.1223, 1e-3))
    i785, i792 = ep.k(785), ep.k(792)
    idx = [ep.ei[e] for e in [ep.amb_app] + ep.conf]
    check("true_counts_unchanged_785_vs_792", np.array_equal(ep.c3[i785, idx], ep.c3[i792, idx]),
          ep.c3[i785, idx].tolist())
    check("free_flow_eta_785_3.0", near(ev[785]["free_flow_eta_to_stopline_s"], 3.03, 1e-2))

    # --- scene 3, 7, 8: signal timeline facts
    seg = lambda pol: ep.timeline[pol, ep.J1]
    ped = [s for s in seg("P3") if s[2] == 3 and s[0] <= 785 < s[1]]
    check("ped_reservation_771.5_to_796.5", len(ped) == 1 and near(ped[0][0], 771.5) and near(ped[0][1], 796.5))
    g = lambda pol, e: [s[0] for s in seg(pol) if s[2] == 0 and s[3] == e and 760 <= s[0] <= 1125]
    check("SE_green_796.5", g("P3", ep.amb_app)[0] == 796.5)
    shifts = {}
    for e in ep.conf:
        p3 = [x for x in g("P3", e) if x > 797][:3]
        p0 = [x for x in g("P0", e) if x > 790][:3]
        shifts[ep.app_name[e]] = [b - a for a, b in zip(p0, p3)]
    check("conflicting_greens_shifted_52s_every_cycle", all(all(near(v, 52) for v in s) for s in shifts.values()), shifts)
    check("SE_served_72s_earlier", near(g("P0", ep.amb_app)[0] - 796.5, 72), g("P0", ep.amb_app)[0])
    p0ped = [s[0] for s in seg("P0") if s[2] == 3 and 760 <= s[0] <= 1125]
    check("cycle_length_124s", all(near(b - a, 124) for a, b in zip(p0ped, p0ped[1:])), p0ped)
    se = [s for s in seg("P3") if s[0] == 796.5][0]
    check("inserted_block_51s", near((se[1] - se[0]) + 3 + 2 + 25, 51), se[1] - se[0])

    # --- scene 9 / 15 / summary: realised extra stopped time at PCL Chowk
    conf880 = sum(ep.app_extra[e][ep.k(880)] for e in ep.conf)
    check("realised_conflicting_extra_by_880_is_842.5", near(conf880, 842.5), conf880)
    check("SE_fewer_stopped_by_851", ep.app_extra[ep.amb_app][ep.k(851)] < 0, ep.app_extra[ep.amb_app][ep.k(851)])

    # --- scene 10: progress deficit (Eq. 21) recomputed from the trajectory
    log = ep.P3.meta["progress_guard_log"]
    g2 = next(x for x in log if x["ambulance"] == "a002")
    Da = float(ep.D_a[ep.k(851)])
    check("Da_851_matches_log_20.26", near(Da, g2["delay_s"], 1e-3), round(Da, 4))
    check("Pa_851_matches_log_70.74", near(float(ep.P_a[ep.k(851)]), g2["executed_freeflow_progress_s"], 1e-3))
    fb = ep.P3.a["fallback"][:, 2]
    check("escalation_flag_from_851.5", ep.t[np.argmax(fb > 0)] == 851.5)
    check("Da_rises_24_to_48_at_J2", near(float(ep.D_a[ep.k(970)]), 23.76, .01) and near(float(ep.D_a[ep.k(1000)]), 47.8, .01))
    check("a001_escalated_704", next(x for x in log if x["ambulance"] == "a001")["t"] == 704)

    # --- scene 12: J2
    j2 = ep.timeline["P3", ep.J2]
    gj = lambda e: [s for s in j2 if s[2] == 0 and s[3] == e and 950 <= s[0] <= 1000]
    check("J2_a001_green_955.5", gj("129126240#7")[0][0] == 955.5)
    check("J2_NW1_green_981.5_NW2_994.5", gj("129126240#7")[1][0] == 981.5 and gj("1215822559#0")[1][0] == 994.5)
    check("J2_switch_delay_971_10s", ev[971]["switch_delay_s"] == 10)
    lo = min(ev[t]["raw_burden_proxy_veh_s"] for t in range(971, 982))
    hi = max(ev[t]["raw_burden_proxy_veh_s"] for t in range(971, 982))
    check("J2_logged_burden_36.5_to_168.5", near(lo, 36.5, .05) and near(hi, 168.5, .05), [lo, hi])
    check("NW2_length_6.71m", near(ep.edges["1215822559#0"]["length_m"], 6.71))

    # --- scene 13: re-route and travel times
    u = ep.decisions["route_updates"][-1]
    a, b = (c["forecast_remaining_time_s"] for c in u["candidates"])
    check("reroute_gap_10.04_gt_10", near(b - a, 10.04, .005) and b - a > 10, round(b - a, 4))
    m3 = {m["id"]: m for m in ep.P3.meta["missions"]}
    m0 = {m["id"]: m for m in ep.P0.meta["missions"]}
    check("a002_travel_323.5_vs_516.5", m3["a002"]["travel_s"] == 323.5 and m0["a002"]["travel_s"] == 516.5)
    check("no_requests_after_arrival", not np.any(ep.P3.a["signal_request"][ep.k(1083.5):] >= 0))
    fd = ep.decisions["first_decision"]["unique_routes"]
    chosen = [r for r in fd if r["chosen"]][0]
    check("route_B_306.97_vs_500.24", chosen["route_id"] == "B" and near(chosen["forecast_with_priority_s"], 306.97, .01)
          and near(chosen["forecast_without_priority_s"], 500.24, .01))

    # --- scene 14: network totals
    k = ep.k(1125)
    d = ep.stop_cum3[k] - ep.stop_cum0[k]
    check("network_stopped_diff_760_1125_+307", near(d, 307.0), d)
    check("network_stopped_pct_5.2", near(d / ep.stop_cum0[k] * 100, 5.2, .05), round(d / ep.stop_cum0[k] * 100, 2))
    curve = json.loads((ROOT / "inputs/data/progression_data.json").read_text())["recorded"]
    t_s = np.array(curve["t_s"])
    sel = ep.k(t_s[0])
    check("network_stopped_matches_original_graph_data",
          np.array_equal(ep.net_stop3[sel:sel + len(t_s)], np.array(curve["ordinary_stopped_network"]))
          and np.array_equal(ep.net_stop0[sel:sel + len(t_s)], np.array(curve["routing_only_ordinary_stopped_network"])))
    check("A_ge_max_dJ_0", all(ep.A_cum[i] >= max(ep.sJ_cum[i], 0) - 1e-9 for i in range(len(ep.t))))

    # --- scene 1: experiment description
    check("414_vehicles_seed_19", len(ep.vehicles) == 414 and ep.P3.meta["seed"] == 19)
    check("dispatches_600_660_760", [m["depart"] for m in ep.P3.meta["missions"]] == [600, 660, 760])
    check("budget_B_1000", ep.P3.meta["configuration"]["harm_budget"] == 1000.0)
    check("seven_signals", len(ep.signals) == 7)

    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    ok = all(c["pass"] for c in checks.values())
    (out / "verification.json").write_text(json.dumps({"all_pass": ok, "checks": checks}, indent=2, default=float))
    print(f"\n{sum(c['pass'] for c in checks.values())}/{len(checks)} checks pass")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
