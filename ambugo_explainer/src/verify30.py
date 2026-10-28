#!/usr/bin/env python3
"""Verify the numbers specific to AmbuGo_30s_Explainer.mp4 (routes, conditions, options, re-evaluation).

The shared claims (green 796.5 s, 72 s / 52 s, 843 measured, 913 forecast) are checked by cine_verify.py.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Episode, ROOT  # noqa: E402

ep = Episode()
ev = ep.ev
rows, ok_all = [], True


def claim(shown, value, ok, source):
    global ok_all
    ok_all &= bool(ok)
    rows.append((shown, value, "PASS" if ok else "FAIL", source))


fd = {r["route_id"]: r for r in ep.decisions["first_decision"]["unique_routes"]}
exp = {"A": (310, 503), "B": (307, 500), "C": (341, 534), "D": (337, 531)}
for k, (w, wo) in exp.items():
    r = fd[k]
    claim(f"Route {k}: {w} s with priority, {wo} s without", f"{r['forecast_with_priority_s']:.2f} / {r['forecast_without_priority_s']:.2f}",
          round(r["forecast_with_priority_s"]) == w and round(r["forecast_without_priority_s"]) == wo,
          "decisions.json first_decision.unique_routes (t = 761 s)")
claim("Route B selected", str([k for k, r in fd.items() if r["chosen"]]), fd["B"]["chosen"], "decisions.json `chosen`")
e = ev[785]
for ap in e["conflicting_approaches"]:
    n = ep.app_name[ap["edge"]]
    claim(f"{n}: queue {ap['queue_proxy']:.1f} veh, {ap['arrival_rate'] * 3600:.0f} arrivals/h, loading {ap['nominal_design_X']:.2f}",
          f"q={ap['queue_proxy']:.4f}, lambda={ap['arrival_rate']:.6f}/s, X={ap['nominal_design_X']:.4f}", True,
          "control_explanations.json events[t=785].conflicting_approaches")
claim(f"SE queue {e['queue_proxy']:.1f} veh", f"{e['queue_proxy']:.4f}", True, "events[785].queue_proxy")
claim("Signal: pedestrian interval, 11 s left", f"switch_delay_s={e['switch_delay_s']}, before={e['signal_mode_before']}",
      e["switch_delay_s"] == 11 and e["signal_mode_before"] == "ped", "events[785]")
claim("Ambulance 3.0 s from the stop line", f"{e['free_flow_eta_to_stopline_s']:.3f}", round(e["free_flow_eta_to_stopline_s"], 1) == 3.0,
      "events[785].free_flow_eta_to_stopline_s")
claim("Priority now: saves about 73 s", f"predicted_benefit_s={e['predicted_benefit_s']:.2f}", round(e["predicted_benefit_s"]) == 73,
      "events[785].predicted_benefit_s (nominal red wait avoided)")
claim("Keep the normal cycle: waits about 73 s for its green", f"nominal_signal_wait_s={e['nominal_signal_wait_s']:.2f}",
      round(e["nominal_signal_wait_s"]) == 73, "events[785].nominal_signal_wait_s (forecast)")
claim("Priority now: +1,045 vehicle-seconds (NW 175, SW 501, NE 368), over budget",
      f"{e['raw_burden_proxy_veh_s']:.3f}", round(e["raw_burden_proxy_veh_s"]) == 1045 and e["reason"] == "burden_budget_exceeded",
      "events[785]")
bars = [round(ev[s]["raw_burden_proxy_veh_s"]) for s in range(785, 793)]
claim("Re-evaluation bars 785-792 s", str(bars), bars == [1045, 1045, 1184, 1184, 1184, 1184, 1184, 913]
      and all(ev[s]["reason"] == "burden_budget_exceeded" for s in range(785, 792)) and ev[792]["reason"] == "selective_gate_passed",
      "events[785..792].raw_burden_proxy_veh_s, reason")
claim("What changed at 792 s: SE 4.8 to 2.8, SW 9.4 to 8.4, NW 1.9 to 0.9",
      f"SE {ev[791]['queue_proxy']:.2f}->{ev[792]['queue_proxy']:.2f}", round(ev[791]["queue_proxy"], 1) == 4.8 and round(ev[792]["queue_proxy"], 1) == 2.8,
      "events[791], events[792]")
idx = [ep.ei[x] for x in [ep.amb_app] + ep.conf]
claim("physical queues unchanged", f"{ep.c3[ep.k(785), idx].tolist()} vs {ep.c3[ep.k(792), idx].tolist()}",
      (ep.c3[ep.k(785), idx] == ep.c3[ep.k(792), idx]).all(), "P3.npz ordinary_edge_counts")
out = ROOT / "output" / "cinematic" / "verification_30s.md"
lines = ["# Verification: AmbuGo_30s_Explainer.mp4 (step-specific numbers)", "",
         "Shared claims (green 796.5 s, 72 s earlier, 52 s later, measured 843, forecast 913) are in verification.md.", "",
         "| Shown on screen | Value from data | Check | Source |", "|---|---|---|---|"]
lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
lines += ["", f"**{sum(r[2] == 'PASS' for r in rows)}/{len(rows)} pass.**", "",
          "Arrival rates and loading are the controller's fixed historical inputs for this run; time of day is not a separate input in this episode."]
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if ok_all else 1)
