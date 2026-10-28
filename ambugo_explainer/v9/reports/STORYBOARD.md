# AmbuGo v9 explainer: storyboard

* Film: `output/AmbuGo_v9_Explainer.mp4`, 1920×1080, 30 fps, 2:25.5 (145.5 s), no audio.
* Episode: archived main-grid run seed 40, load 0.95, one mission. The ambulance is dispatched from BW1 at t = 4,064 s and arrives at BE1.
* Policy: JSEL (P3), frozen configuration B = 10,000 veh·s, ε = 0 s, strong actions {(0,0), (30,60)}, γ = 20 s.
* Counterfactual: one labelled single-decision branch, "J21 request withheld from t = 4,132 s" (see `COUNTERFACTUALS.md`).
* Timing: simulation time maps to video time piecewise-linearly. The top-right badge always shows the simulation clock and either the playback rate or "paused".
* Overlays: every number on screen is checked by `verify.py` (154 checks, see `VERIFICATION.md`).

## Fixed screen furniture

* Top left: a chapter pill.
* Top right: the simulation clock, dispatch-relative time and playback rate, with the caption "synthetic SUMO district (AmbuGo v9 fixture) · seed 40, load 0.95 · not a calibrated city" beneath.
* Lower left: one caption box, two or three lines. It moves above the signal strip in scenes that show one.
* Signal colours:
  * green, amber and red come only from the logged controller state;
  * a stop-line bar is drawn for every approach;
  * three-lamp heads are drawn at focus junctions;
  * the pedestrian interval is shown as red.
* Vehicles:
  * every vehicle is drawn at its recorded SUMO position and heading;
  * the class is shown by colour and size: car, motorcycle, autorickshaw, bus, goods;
  * in overview shots, vehicles are drawn at a minimum of 10 px long so they stay visible (size exaggerated, positions not);
  * the ambulance is white with blue stripes, a red cross and a blue ring.
* The film shows no collisions or risk scores (none occur in the logs, and none are modelled).

| # | Video | Sim time (rate) | Chapter | What is shown | Key on-screen values (all logged) |
|---|---|---|---|---|---|
| A | 0:00.0–0:11.0 | 3,992 → 4,066 (6.7×) | The problem | The whole district (16 signalised junctions, 59 links, junction IDs, city blocks) with the ambulance base BW1 and hospital BE1. Traffic is moving. The ambulance appears at dispatch, with a tag. | dispatch 4,064 s at BW1 |
| B | 0:11.0–0:25.0 | paused at 4,065.5 | 1 · Choosing a route and a priority action together | The four candidate routes A–D are drawn one by one in muted colours (not signal colours). A table shows, per route, the forecast with normal signals, the forecast with a request, the forecast burden, and whether the route is eligible at ε = 0. Route C is highlighted. | A 590.9 / 250.8 / 2,528; B 590.9 / 259.8 / 2,715; **C 616.9 / 248.0 / 3,330 (eligible)**; D 590.9 / 259.6 / 1,402 |
| C | 0:25.0–0:32.0 | 4,065.5 → 4,091 (3.6×) | 2 · Every road link: check the route, then the request | The camera follows the ambulance from BW1 to J10–J11. The J10 request is tagged as admitted and J10's logged signal turns green for it. | J10: gain 39 s, burden 227 veh·s |
| D | 0:32.0–0:37.5 | paused at 4,091 | (same) | Re-plan on entering E_J10_J11: route A is compared with the incumbent C. | A 219.9 s vs C 232.5 s; gap 12.6 s > 10 s hysteresis → switch to A |
| E | 0:37.5–0:46.5 | paused at 4,091 | 3 · A priority request that is withheld | Close-up of J11 with labelled approaches (west = ambulance). A gate checklist marks the one failing gate in red, then the banner "PRIORITY WITHHELD: too little to gain". | gain 3.6 s < 5 s ✗; Eq. 19: 30.5 ≤ 60 ✓; downstream 5 < 45.0 ✓; burden 306 ≤ 10,000 ✓; pedestrian interval with 5 s left |
| F | 0:46.5–0:59.5 | 4,091 → 4,130.5 (3.0×) | 4 · How a normal signal cycle runs | J11 runs its normal cycle with live signal heads. The recorded J11 stage strip (4,070–4,150) has a moving cursor. A "Fixed rules" panel lists the cycle rules. | ped → north green 4,096 → amber → all-red → west green 4,122; minimum green 8 s, amber 3 s, all-red 2 s, pedestrian 25 s, 150 s service-age guard (not a per-driver bound) |
| G | 0:59.5–1:04.5 | 4,130.5 → 4,132, then paused | 5 · The decision at J21 | The camera moves to J21 and labels its approaches. The question is posed. | east green, elapsed 3 s |
| H | 1:04.5–1:16.5 | 4,132 → 4,180 (4.0×) | Scenario 1, give the ambulance priority (what AmbuGo did) | The recorded JSEL run. East finishes its minimum green, then amber and all-red, then north (ambulance) green. Stopped-vehicle counts per approach are shown live, with the recorded J21 strip below. | east amber 4,137; all-red 4,140; north green 4,142; ambulance clears ~4,171 |
| I | 1:16.5–1:35.5 | paused at 4,132 (labelled rewind "back to t = 4,132 s, before the decision") | 6 · How AmbuGo forecast that cost (Eq. 20) | The east approach is highlighted with a tag explaining q̂ (an estimate from delayed, noisy counts, not the visible stopped count), then λ and X. The Eq. 20 panel shows the formula and D. The per-arm table and total follow. The counterfactual normal-cycle strip below marks the "green displaced" window D. Last, the admission checklist (all gates pass) and "PRIORITY ADMITTED". | D = min(60, max(5, 30 + 14.2)) = 44.2 s; east 342 + 1,020 = 1,362; south 443 + 1,249 = 1,692; west 34 + 239 = 273; Ĉ = 3,326 veh·s; gain 43.4 s; Eq. 19 50.4 ≤ 60; downstream 8 < 112.2; legal switch delay 10 s |
| J | 1:35.5–1:52.5 | 4,132 → 4,236 (7.0×), shared clock | Scenario 1 vs Scenario 2, same moment, same traffic | Side by side: left is Scenario 1, priority granted (the archived decision); right is Scenario 2, normal cycle kept (counterfactual replay). Both have identical cameras and clock, and live stopped counts on all four approaches. A summary table closes the scene. | ambulance green 4,142 vs 4,228; stopped veh·s 4,125–4,300: east 1,775 vs 670, south 1,080 vs 530, west 110 vs 615, north 90 vs 605; at most 19 vehicles on any approach (storage 53–132): no spillback |
| K | 1:52.5–2:13.5 | 4,171 → 4,255 (8×), 4,256 paused ~2 s, → 4,316.5 (11×) | 7 · One junction, several roads, and the rest of the trip | Zoom out over J11–J13 / J21–J23. Links are tinted where occupancy is higher (violet) or lower (teal) than in the normal-cycle replay at the same time (±10 s mean, colours not used for signals). J22 and J12 are tagged as admitted. The Eq. 21 progress gauge rises past the 20 s line; at 4,256 the "Stronger priority mode (JFAST)" tag appears. Then arrival. | J22 gain 89 s / burden 348; J12 gain 87 s / burden 185; deficit 20.5 s at 4,256 s; arrival 4,316.5 s (252.5 s) |
| L | 2:13.5–2:25.5 | paused | Results | Zoom out to the district. The results panel separates this episode from the paper aggregates. | episode: JSEL 252.5 s, J21-withheld 323.5 s, S0 572.0 s; A = 5,992 veh·s, signed −74 veh·s (J21 branch, 4,132–4,400). Paper (1,728 runs, balanced means): S0 401.57, JFAST 211.19, JSEL 222.29 s; JSEL − JFAST +11.10 s [9.47, 12.94]; A −3.84 veh·h; D⁺ (vs S0) −1.64 veh·h; 44.6 % below S0; 51.6 % of JSEL missions escalated; "synthetic, uncalibrated" |

## Ordering note

Scenario 1 (H) is shown before its forecast (I) so that the viewer first sees the physical consequence. Scene I is explicitly labelled as a rewind to t = 4,132 s, and its signal strip is labelled "counterfactual replay". No event from one run is shown with the clock of another. In J the two panels share one clock and come from the same initial state (see `COUNTERFACTUALS.md`).
