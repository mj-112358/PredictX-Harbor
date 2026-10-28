# AmbuGo explainer film

A slow, chaptered 1920×1080 explainer (≈10.4 min, 30 fps). It replaces the 20-second
`01_Google_Maps_Journey.mp4` / `02_Progression.mp4` pair with one film. In it every visual
change on the map is shown next to the mathematics that caused it.

It uses the **same frozen episode** as the original package (case B1000, ambulance
`a002`, seed 19, AmbuGo run `P3` paired with the no-priority run `P0`). Nothing is
re-simulated. Vehicles, signals, routes and controller numbers are read from the recorded
replays and the controller's decision log in `inputs/data/`.

```
output/AmbuGo_Explainer.mp4   the film
output/chapters.txt           chapter start times
output/verification.json      46 re-derived on-screen claims, all passing
output/poster.jpg             still from chapter 4
```

## Layout of every frame

| Region | Content |
|---|---|
| Header | Chapter title, simulation clock `t`, time since dispatch, playback rate (`PAUSED · explanation` or `▶ 0.33× real time` …), chapter progress bar |
| Map (top left) | Fixed Google Maps captures, desaturated, under one Web Mercator camera. On top: simulated road links, colour-coded approach lanes, every recorded vehicle, signal heads with their recorded state, routes, ambulance, annotation tags |
| Right panel | Narration: what to look at and why. Items appear at reading pace (3.2 words/s); chapters are lengthened automatically so they can be read |
| Bottom panel | The live mathematics or charts for that instant (equations with the logged numbers, gate checklist, signal timelines, queue charts, network totals) |

### Colour code (fixed for the whole film)

* **Blue**: ambulance a002, its route, its approach at PCL Chowk (SE); AmbuGo series in charts
* **Violet / orange / teal**: the NW / SW / NE conflicting approaches at PCL Chowk, and the cars on them
* **Orange-red ring**: escalated (aggressive) priority mode after 851 s
* **Grey**: the no-priority baseline run
* **Red rear dots**: stopped car (speed < 0.1 m/s, the SUMO halting threshold)
* **Chapter 9 car colours**: each car's extra stopped time since dispatch versus *the same car*
  in the no-priority run (blue less, grey equal, red more), with labels in seconds.
  Hollow outlines show where the same cars are in the no-priority run.
* **Chapter 14 link colours**: vehicles present per link, AmbuGo minus no priority
  (the quantity integrated in the paper's spatial-exposure metric A, Eq. 23)

## Chapters and what each one proves

| # | Chapter | Mechanism shown | Bottom panel |
|---|---|---|---|
| 1 | Network and paired experiment | What is real (OSM geometry, map) vs simulated (demand, signals, missions); the paired design | Colour legend and notation |
| 2 | Route choice | 4 routes × 2 actions, ε = 0 eligibility, route B chosen (306.97 s vs 500.24 s without priority) | Candidate table and selection rule |
| 3 | Approach to PCL Chowk | Pedestrian all-red reservation 771.5–796.5 s fixes the earliest legal switch | Speed and progress deficit |
| 4 | Request evaluated (paused, t = 785 s) | Trigger, displaced green D_e = 38.32 s, per-approach burden Ĉ_j | Eq. 20 with every term; Eq. 19; stacked gauge ΣĈ = 1044.5 > B; gate checklist |
| 5 | Withheld 785–791 s | Re-evaluated each second; a sample at 787 s raises SW q̂ to 9.37, so ΣĈ = 1184. The refusal costs nothing because the pedestrian reservation blocks the switch anyway | Live Eq. 20 table with changed cells highlighted |
| 6 | Admitted (paused, t = 792 s) | New sample: own q̂ 4.78→2.78 so D_e 40.52→36.12; NW 1.94→0.94, SW 9.37→8.37; ΣĈ = 913.3 ≤ B. **True counts are unchanged**; only the estimate moved | Same, with old→new values |
| 7 | Legal transition | Pedestrian phase completes, SE green at 796.5 s (normal order skipped once), ambulance crosses ≈801.5 s | Signal timeline, both runs, revealed live |
| 8 | Cycle shift (paused) | 51 s inserted ahead of NW/SW/NE, so their greens come **52 s later in every later cycle**; SE served 72 s earlier | Full timeline with shift arrows |
| 9 | What the other approaches pay | Car-level extra waiting vs baseline; queue growth on NW/SW/NE | Stopped vehicles per approach, both runs; realised extra veh·s vs forecast 913 |
| 10 | Escalation (t ≈ 851 s) | Eq. 21: 91.0 − 70.74 = 20.26 s > γ = 20 s, so escalation is irreversible | Progress deficit chart |
| 11 | Corridor | No decisions; sped-up playback | Speed / deficit |
| 12 | Junction J2 | Escalated requests: Ĉ still logged (36.5–168.5) but B not applied; legal clearances still cost time (δ = 10 s) | J2 timeline and controller log |
| 13 | Hospital approach | Re-route by 10 s hysteresis (gap 10.04 s > 10 s); arrival 323.5 s vs 516.5 s without priority | Re-route rule; travel-time bars for all three missions |
| 14 | Whole network | Disturbance travels with platoons; rings mark when each link first departs from the baseline | Network stopped vehicles, both runs; accumulated A and ΔJ |
| 15 | After arrival | Ordinary queues discharge; the 52 s shift persists | Approach charts to 1125 s |
| 16 | Summary | Causal chain, episode numbers, paper-level evidence, deployment requirements, limitations | Limitations and reproduction commands |

## Rigour notes

* `src/verify.py` recomputes Eq. 20 for **every** logged request (max error 1.1e-13 veh·s).
  It also recomputes Eq. 19 feasibility, recomputes Eq. 21 from the trajectory (20.2636 s
  vs the logged 20.2636 s), and checks every number quoted in the narration. The network
  stopped-vehicle curves match the original package's `progression_data.json` exactly.
* Positions and headings are interpolated between 0.5 s samples for smooth slow motion.
  Speed, stopped state, link membership and signal state are never interpolated.
* The film states its limits on screen. It is one synthetic episode. Ĉ is a ranking
  surrogate. Comparisons against the no-priority run include all three missions. It makes
  no claim of optimality or full recovery.

## Reproduce

```bash
pip install -r requirements.txt
python src/verify.py                 # re-derives every on-screen number
python src/render.py --preview       # one still per chapter -> output/preview/
python src/render.py --stills 180 330  # stills at chosen video seconds
python src/render.py --jobs 4        # full film -> output/AmbuGo_Explainer.mp4
```

Needs Python ≥ 3.10, NumPy, Pillow, Matplotlib (for equation typesetting) and an `ffmpeg`
binary with libx264 on `PATH`. The Inter font is used if installed; otherwise DejaVu Sans.
A full render takes about 40 min on 4 cores.

## Inputs and attribution

`inputs/` is copied unchanged from `AmbuGo_20s_Reproducible` (replays, network geometry,
connectors, decision and controller logs, six fixed Google Maps captures). Google Maps
imagery keeps its native logo and copyright wording in the strip under the map; see
`THIRD_PARTY_NOTICES.txt`. Road geometry is OpenStreetMap-derived.
