#!/usr/bin/env python3
"""Pass 1 for the cinematic cut: an event table for PCL Chowk, 776-885 s, read from the raw data.

Writes output/cinematic/event_table.csv (every 0.5 s sample) and prints the key events.
Columns: simulation time; a002 position/speed/edge; J1 signal (AmbuGo and no-priority);
active request; controller decision and forecast; queue proxies; true halted vehicles per
approach in both runs; cumulative measured extra stopped vehicle-seconds.
"""
from pathlib import Path
import csv
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Episode, ROOT  # noqa: E402

MODES = {0: "green", 1: "amber", 2: "all-red", 3: "pedestrian"}


def table(ep, t0=776.0, t1=885.0):
    rows = []
    for k in range(ep.k(t0), ep.k(t1) + 1):
        t = float(ep.t[k])
        p = ep.amb_pos(t)
        m3, s3, r3 = ep.signal_state("P3", ep.J1, t)
        m0, s0, _ = ep.signal_state("P0", ep.J1, t)
        e = ep.ev.get(int(t)) if t == int(t) else None
        row = {
            "t_s": t,
            "amb_x_merc": round(float(p[0]), 2) if np.isfinite(p).all() else "",
            "amb_y_merc": round(float(p[1]), 2) if np.isfinite(p).all() else "",
            "amb_speed_kmh": round(float(ep.amb_speed[k]), 2) if np.isfinite(ep.amb_speed[k]) else "",
            "amb_edge": ep.edge_ids[ep.amb_edge[k]] if ep.amb_edge[k] >= 0 else "internal",
            "J1_ambugo": MODES[m3] + (f":{ep.app_name.get(s3, s3)}" if s3 else ""),
            "J1_no_priority": MODES[m0] + (f":{ep.app_name.get(s0, s0)}" if s0 else ""),
            "J1_request_edge": ep.app_name.get(r3, r3) if r3 else "",
            "decision": e["reason"] if e else "",
            "forecast_veh_s": round(e["raw_burden_proxy_veh_s"], 3) if e else "",
            "switch_delay_s": e["switch_delay_s"] if e else "",
            "qhat_SE": round(e["queue_proxy"], 3) if e else "",
        }
        for a in (e["conflicting_approaches"] if e else []):
            row[f"qhat_{ep.app_name[a['edge']]}"] = round(a["queue_proxy"], 3)
            row[f"forecast_{ep.app_name[a['edge']]}"] = round(a["burden_proxy_veh_s"], 3)
        for edge in [ep.amb_app] + ep.conf:
            i = ep.ei[edge]
            n = ep.app_name[edge]
            row[f"halted_{n}_ambugo"] = int(ep.h3[k, i])
            row[f"halted_{n}_nopriority"] = int(ep.h0[k, i])
        row["measured_extra_conflicting_veh_s"] = float(sum(ep.app_extra[e2][k] for e2 in ep.conf))
        rows.append(row)
    return rows


def main():
    ep = Episode()
    rows = table(ep)
    out = ROOT / "output" / "cinematic"
    out.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with (out / "event_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    prev = None
    for r in rows:
        sig = (r["J1_ambugo"], r["decision"], r["amb_edge"], r["J1_request_edge"])
        if sig != prev:
            print(r["t_s"], r["amb_edge"], r["amb_speed_kmh"], r["J1_ambugo"], "| base", r["J1_no_priority"],
                  "| req", r["J1_request_edge"], "|", r["decision"], r["forecast_veh_s"], "| extra", r["measured_extra_conflicting_veh_s"])
        prev = sig
    print("rows", len(rows), "->", out / "event_table.csv")


if __name__ == "__main__":
    main()
