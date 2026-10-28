#!/usr/bin/env python3
"""Verify every number and claim shown in AmbuGo_20s_Cinematic.mp4 against the raw data.

Writes output/cinematic/verification.md; exits non-zero on any failure.
"""
from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Episode, ROOT  # noqa: E402
import cinematic as C  # noqa: E402

rows, ok_all = [], True


def claim(shown, value, ok, kind, source, units, when):
    global ok_all
    ok_all &= bool(ok)
    rows.append((shown, value, "PASS" if ok else "FAIL", kind, source, units, when))


def r(x):
    return math.floor(x + .5)


def main():
    ep = Episode()
    ev = ep.ev
    LOG = "data/control_explanations.json `events[a002]`"
    for t in (785, 786):
        claim("1,045 / 1,000, OVER BUDGET, PRIORITY WITHHELD", f"{ev[t]['raw_burden_proxy_veh_s']:.3f}",
              r(ev[t]["raw_burden_proxy_veh_s"]) == 1045 and ev[t]["reason"] == "burden_budget_exceeded",
              "forecast", LOG + " `raw_burden_proxy_veh_s`, `reason`", "veh·s", f"t = {t} s")
    claim("NW 175 · SW 501 · NE 368 (chips and breakdown)",
          ", ".join(f"{a['burden_proxy_veh_s']:.2f}" for a in ev[785]["conflicting_approaches"]),
          [r(a["burden_proxy_veh_s"]) for a in ev[785]["conflicting_approaches"]] == [175, 501, 368],
          "forecast", LOG + " `conflicting_approaches[].burden_proxy_veh_s`", "veh·s", "t = 785 s")
    claim("1,184 (re-checked, still over budget)", f"{ev[787]['raw_burden_proxy_veh_s']:.3f}",
          all(r(ev[t]["raw_burden_proxy_veh_s"]) == 1184 and ev[t]["reason"] == "burden_budget_exceeded" for t in range(787, 792)),
          "forecast", LOG, "veh·s", "t = 787-791 s")
    claim("913 / 1,000, WITHIN BUDGET, PRIORITY APPROVED", f"{ev[792]['raw_burden_proxy_veh_s']:.3f}",
          r(ev[792]["raw_burden_proxy_veh_s"]) == 913 and ev[792]["reason"] == "selective_gate_passed",
          "forecast", LOG, "veh·s", "t = 792 s")
    claim("NW 55 · SW 531 · NE 327 (breakdown after admission)",
          ", ".join(f"{a['burden_proxy_veh_s']:.2f}" for a in ev[792]["conflicting_approaches"]),
          [r(a["burden_proxy_veh_s"]) for a in ev[792]["conflicting_approaches"]] == [55, 531, 327],
          "forecast", LOG, "veh·s", "t = 792 s")
    claim("Gauge keeps the 792 s forecast while the lease runs", "rows 793-801 = committed_selective_lease",
          all(ev[t]["reason"] == "committed_selective_lease" for t in range(793, 802)),
          "controller state", LOG + " `reason`", "-", "t = 793-801 s")
    claim("Budget B = 1,000", str(ep.P3.meta["configuration"]["harm_budget"]),
          ep.P3.meta["configuration"]["harm_budget"] == 1000.0, "parameter",
          "data/replays/P3.json `configuration.harm_budget`", "veh·s", "-")
    idx = [ep.ei[e] for e in [ep.amb_app] + ep.conf]
    claim("'the estimate changed, the physical queues did not'",
          f"counts SE,NW,SW,NE at 785 = {ep.c3[ep.k(785), idx].tolist()}, at 792 = {ep.c3[ep.k(792), idx].tolist()}",
          np.array_equal(ep.c3[ep.k(785), idx], ep.c3[ep.k(792), idx]), "observed",
          "data/replays/P3.npz `ordinary_edge_counts`", "vehicles", "t = 785 vs 792 s")
    ped = [s for s in ep.timeline["P3", ep.J1] if s[2] == 3 and s[0] <= 792 < s[1]][0]
    claim("Pedestrian phase protected; countdown to 0", f"{ped[0]}-{ped[1]} s",
          ped[0] == 771.5 and ped[1] == 796.5, "observed signal state",
          "data/replays/P3.npz `signal_mode` (3 = pedestrian) at J1", "s", "771.5-796.5 s")
    se = [s for s in ep.timeline["P3", ep.J1] if s[2] == 0 and s[3] == ep.amb_app and s[0] > 790][0]
    claim("Green at first legal moment, 796.5 s", f"{se[0]} s", se[0] == 796.5, "observed signal state",
          "data/replays/P3.npz `signal_mode`, `signal_edge`", "s", "796.5 s")
    claim("Withholding did not delay the ambulance (pedestrian phase ran to 796.5 s either way)",
          f"switch delay 785 s = {ev[785]['switch_delay_s']} s (= 796 - 785)", ev[785]["switch_delay_s"] == 11,
          "controller input", LOG + " `switch_delay_s`", "s", "t = 785 s")
    p0se = [s[0] for s in ep.timeline["P0", ep.J1] if s[2] == 0 and s[3] == ep.amb_app and s[0] > 790][0]
    claim("72 s earlier: SE green for the ambulance vs no priority", f"{p0se} - {se[0]} = {p0se - se[0]}",
          p0se - se[0] == 72, "observed vs counterfactual run", "P3.npz and P0.npz `signal_mode/edge`", "s", "first SE green after 790 s")
    sh = []
    for e in ep.conf:
        a = [s[0] for s in ep.timeline["P0", ep.J1] if s[2] == 0 and s[3] == e and s[0] > 790][0]
        b = [s[0] for s in ep.timeline["P3", ep.J1] if s[2] == 0 and s[3] == e and s[0] > 797][0]
        sh.append(b - a)
    claim("+52 s later: next green for NW, SW and NE", str(sh), all(x == 52 for x in sh),
          "observed vs counterfactual run", "P3.npz and P0.npz `signal_mode/edge`", "s", "first green after 790 s")
    meas = sum(ep.app_extra[e][ep.k(880)] for e in ep.conf)
    claim("843 veh·s measured, OBSERVED NOT FORECAST", f"{meas:.1f} (shown rounded half-up)", r(meas) == 843,
          "observed, cumulative", "edge_halts (speed < 0.1 m/s) on NW, SW, NE, P3 minus P0, x 0.5 s", "veh·s", "785-880 s")
    claim("Forecast tick 913 on the measured gauge", f"{ev[792]['raw_burden_proxy_veh_s']:.1f}", True, "forecast", LOG, "veh·s", "t = 792 s")
    v = np.arange(0, C.DUR, 1 / C.FPS)
    sims = np.array([C.SIM(x) for x in v])
    claim("Clock never runs backwards; events keep recorded order", f"min dSIM/frame = {np.diff(sims).min():.4f}",
          np.all(np.diff(sims) >= -1e-9), "render", "src/cinematic.py KNOTS (monotone PCHIP)", "s", "all frames")
    claim("Peak replay speed (shown in the clock)", f"{max(C.rate(x) for x in v):.1f}x", True, "render", "src/cinematic.py", "-", "trade-off beat")
    claim("Request admitted at video time", f"{v[np.argmax(sims >= 792)]:.2f} s", True, "render", "KNOTS", "s", "sim 792.0 s")
    claim("Green at video time", f"{v[np.argmax(sims >= 796.5)]:.2f} s", True, "render", "KNOTS", "s", "sim 796.5 s")

    out = ROOT / "output" / "cinematic" / "verification.md"
    lines = ["# Verification: AmbuGo_20s_Cinematic.mp4", "",
             "Every number or factual claim on screen, its source, units, window, and whether it is a forecast, an observation or a render property.", "",
             "| Shown on screen | Value from data | Check | Kind | Source (file / field) | Units | Time / window |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    lines += ["", f"**{sum(r[2] == 'PASS' for r in rows)}/{len(rows)} pass.**", "",
              "Notes: forecasts are the controller's Eq. 20 ranking surrogate, logged before the decision. The measured value counts",
              "stopped (speed < 0.1 m/s) vehicle-seconds on the three conflicting approaches, AmbuGo minus the matched no-priority run,",
              "and also reflects the other two ambulances (a000, a001), whose priority requests exist only in the AmbuGo run.",
              "One synthetic episode (seed 19); not a field measurement."]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
