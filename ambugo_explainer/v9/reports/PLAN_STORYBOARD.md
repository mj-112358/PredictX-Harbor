# AmbuGo v9 plan film: storyboard and voiceover

* **File:** `output/AmbuGo_v9_Plan.mp4`, 1920×1080, 30 fps, 2:20.5 (140.5 s), no audio. Render it with `python3 planfilm.py --jobs 4`.
* **What it adds over the simple cut:**
  * the trip is planned at dispatch: route and signal changes are chosen together, using the delay forecast;
  * the forecast model itself (Eq. 20) is shown term by term with the logged values.
* **Design:** same rule as the simple cut. Each beat has one caption and at most one highlighted thing.

## How the plan is chosen (from the frozen controller, `Controller.plan` and `estimate`)

* **What is forecast.** For every candidate route, and for both "no signal changes" and "request changes", AmbuGo forecasts the trip junction by junction.
* **Where changes are planned.** At each junction it runs the same admission test it uses on the road: forecast gain ≥ 5 s, forecast extra waiting ≤ 10,000 veh·s, and room downstream. A change is planned only where that test passes. The forecast extra waiting of the planned changes is added up per route.
* **How the plan is picked.** Among plans whose forecast time is within ε of the fastest, it takes the one with the least extra waiting. In the paper's selected setting ε = 0, so the fastest plan wins and the delay forecast decides only between equally fast plans. The film says exactly this; it does **not** claim the chosen route minimises delay. Route D had less forecast waiting (1,402 veh·s) than route C (3,330 veh·s), and C was chosen because it was fastest.
* **Re-planning.** The plan is redone on entering every road. Half a minute later, route A had become both faster and lower-delay, so the ambulance switched to it.

## Beats

| # | Time | View | Caption | Highlight |
|---|---|---|---|---|
| 1 | 0:00–0:05 | plan diagram over a faded Mohali map | When the ambulance is dispatched, AmbuGo plans the trip before it sets off. | base and hospital |
| 2 | 0:05–0:10 | plan | It compares several routes to the hospital. | the four routes (faint) |
| 3 | 0:10–0:16 | plan | For each route it forecasts which signals to change, how long the trip takes, and how much extra waiting it causes. | route A: 251 s · 5 planned signal changes · others wait +2,528 veh·s |
| 4 | 0:16–0:20.5 | plan | (same) | route B: 260 s · 6 · +2,715 |
| 5 | 0:20.5–0:25 | plan | (same) | route C: 248 s · 5 · +3,330 |
| 6 | 0:25–0:29.5 | plan | (same) | route D: 260 s · 6 · +1,402 |
| 7 | 0:29.5–0:36.5 | plan | A signal change is planned only where its forecast delay for other traffic stays within AmbuGo's limit. | none |
| 8 | 0:36.5–0:43.5 | plan | It takes the fastest plan: route C, 248 s. Forecast delay decides only between equally fast plans. | route C + verdict "Route C · 248 s" |
| 9 | 0:43.5–0:51.5 | plan, t = 4,091 | The plan is redone on every road. Half a minute later, route A had become faster and caused less waiting, so it switched. | route A from the ambulance's position: A now 220 s · +3,042; C now 233 s · +4,126 |
| 10 | 0:51.5–0:56.5 | Junction 1 map, 4,079 → 4,091 | Along the route, each signal is asked about 30 seconds before the ambulance gets there. | ambulance |
| 11 | 0:56.5–1:02.5 | J1, paused | Here a change would save only 3.6 seconds, under the 5-second minimum. | ambulance's road "saves 3.6 s" + verdict "No change" |
| 12 | 1:02.5–1:10.5 | J1, JFAST run 4,091 → 4,170 | In a recorded run where it was changed anyway, the cross road waited 64 s longer so the ambulance passed 11 s sooner. | cross road "+64 s wait" |
| 13 | 1:10.5–1:15.5 | Junction 2 map, 4,120 → 4,132 | At the next signal, the forecast is made about 50 seconds before the ambulance arrives. | ambulance |
| 14 | 1:15.5–1:24.5 | card over blurred map | (card) How AmbuGo forecasts the extra waiting on each other road: Eq. 20, with D (green lost, ≈ 44 s), q̂·D (cars queued × green lost), λD²/2max(1−X, 0.05) (expected arrivals, growing as the road nears capacity) | the card |
| 15 | 1:24.5–1:30.5 | J2, paused 4,132 | East road: about 7.7 cars already queued, estimated from counts that arrive late and with noise. | east road "≈ 7.7 queued" |
| 16 | 1:30.5–1:36.5 | J2 | About 0.14 more cars per second are expected, on a road already 87% busy. | east road "0.14 cars/s · 87% busy" |
| 17 | 1:36.5–1:41.5 | J2 | Expected extra waiting on the east road: 1,362 vehicle-seconds. | east road "+1,362 veh·s" |
| 18 | 1:41.5–1:46.5 | J2 | The same for the south and west roads. | south "+1,692", west "+273" |
| 19 | 1:46.5–1:53.5 | J2 | Total forecast: 3,326 vehicle-seconds, within the limit of 10,000, and the ambulance saves 43 seconds. | verdict "Change the light" |
| 20 | 1:53.5–2:01.5 | J2, 4,132 → 4,146 | The change is made safely: minimum green, then amber, then all-red. | junction ring |
| 21 | 2:01.5–2:08.5 | J2, 4,146 → 4,180 | The ambulance gets green 86 seconds sooner than without the change. | ambulance |
| 22 | 2:08.5–2:14.5 | blurred | AmbuGo plans the route and the signal changes together, forecasting the delay every change would cause. | — |
| 23 | 2:14.5–2:20.5 | blurred | Across 1,728 simulations: trips 44.6% faster than with no priority, with less extra congestion than changing every light. | — |

**Sources.**

* The plan numbers come from `plans.json` (the logged candidate table at 4,065 s and 4,091 s).
* The forecast-model numbers come from `gates.json` (J21 at 4,132 s).
* All numbers are checked by `verify.py`, section `plan` (214 checks in total, 0 failed).

**The planning diagram.** It is drawn over a faded Google Maps view of Mohali as a schematic of the 16-junction district, not to map scale.

## Voiceover (one sentence or two per beat)

1. When an ambulance is dispatched, AmbuGo plans the whole trip before it sets off.
2. It compares several routes to the hospital.
3. For each one, it forecasts which traffic lights to change, how long the trip will take, and how much extra waiting those changes would cause for everyone else.
4. to 6. (Let the route tags read; optionally: "Route B… route C… route D.")
7. A light change is only planned where its forecast delay for other traffic stays within AmbuGo's limit.
8. AmbuGo takes the fastest plan, route C. The delay forecast decides between plans that are equally fast.
9. And the plan is updated on every road. Half a minute later, route A had become faster and less disruptive, so the ambulance switched.
10. Each light along the route is asked about thirty seconds before the ambulance gets there.
11. At this one, a change would save less than four seconds, not enough to be worth it.
12. In a recorded run where the light was changed anyway, the cross road waited an extra minute, to get the ambulance through eleven seconds sooner.
13. At the next light, the forecast is made almost a minute ahead.
14. Here is the forecast. For every other road, it multiplies the cars already queued by the green time they would lose. Then it adds the cars expected to arrive in that time, which counts more heavily on roads that are already close to capacity.
15. On the east road, about eight cars are already waiting. AmbuGo estimates this from sensor counts that arrive late and with noise.
16. About one more car arrives every seven seconds, on a road that is already almost ninety percent busy.
17. Together, that is about thirteen hundred and sixty vehicle-seconds of extra waiting.
18. The same calculation is made for the south and west roads.
19. In total about thirty-three hundred vehicle-seconds, well within the limit, and the ambulance saves forty-three seconds. So the light changes.
20. It changes safely: the current green finishes its minimum, then amber, then all-red.
21. The ambulance gets its green eighty-six seconds sooner.
22. So AmbuGo plans the route and the light changes together, and forecasts the delay every change would cause.
23. Across all the simulations, trips were forty-five percent faster than with no priority, with less extra congestion than changing every light.
