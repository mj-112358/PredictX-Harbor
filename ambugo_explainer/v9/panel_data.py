"""Time series for the live panel beside AmbuGo_v9_Plan.mp4, computed from the same recorded runs as the map.

    python panel_data.py      # writes data/panel_series.json

For each run, every 0.5 s from dispatch (4,064 s) to dispatch + 320 s:
  amb_kmh  ambulance speed (km/h), null before insertion and after arrival
  stopped  ordinary vehicles at a standstill (speed < 0.1 m/s) anywhere in the 16-junction district
"""
from pathlib import Path
import json

import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / "data" / "episodes"
DISPATCH, SPAN = 4064.0, 320.0


def series(name):
    z = np.load(D / name / "trajectories.npz")
    ix = json.loads((D / name / "trajectory_index.json").read_text())
    amb = ix["vehicles"].index("a000")
    T, kmh, stopped = [], [], []
    for k, t in enumerate(z["t"]):
        if not (DISPATCH <= t <= DISPATCH + SPAN):
            continue
        lo, hi = z["offsets"][k:k + 2]
        v, sp = z["vehicle"][lo:hi], z["speed"][lo:hi]
        w = np.flatnonzero(v == amb)
        T.append(round(float(t) - DISPATCH, 2))
        kmh.append(round(float(sp[w[0]]) * 3.6, 2) if len(w) else None)
        stopped.append(int(((sp < .1) & (v != amb)).sum()))
    arrival = json.loads((D / name / "result.json").read_text())["mission_rows"][0]["travel_s"]
    return dict(t=T, amb_kmh=kmh, stopped=stopped, travel_s=arrival)


if __name__ == "__main__":
    out = {n: series(n) for n in ("P3", "P0", "P2")}
    out["meta"] = dict(dispatch=DISPATCH, span=SPAN, stopped_threshold_mps=0.1,
                       runs={"P3": "AmbuGo (JSEL), archived", "P0": "no signal priority (S0), archived, same traffic",
                             "P2": "always change the light (JFAST), archived, same traffic"})
    (HERE / "data" / "panel_series.json").write_text(json.dumps(out))
    for n in ("P3", "P0", "P2"):
        s = out[n]
        print(n, len(s["t"]), "travel", s["travel_s"], "stopped range", min(s["stopped"]), max(s["stopped"]),
              "amb samples", sum(x is not None for x in s["amb_kmh"]))
