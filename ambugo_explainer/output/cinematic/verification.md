# Verification: AmbuGo_20s_Cinematic.mp4

Every number or factual claim on screen, its source, units, window, and whether it is a forecast, an observation or a render property.

| Shown on screen | Value from data | Check | Kind | Source (file / field) | Units | Time / window |
|---|---|---|---|---|---|---|
| 1,045 / 1,000, OVER BUDGET, PRIORITY WITHHELD | 1044.521 | PASS | forecast | data/control_explanations.json `events[a002]` `raw_burden_proxy_veh_s`, `reason` | veh·s | t = 785 s |
| 1,045 / 1,000, OVER BUDGET, PRIORITY WITHHELD | 1044.521 | PASS | forecast | data/control_explanations.json `events[a002]` `raw_burden_proxy_veh_s`, `reason` | veh·s | t = 786 s |
| NW 175 · SW 501 · NE 368 (chips and breakdown) | 175.06, 501.16, 368.30 | PASS | forecast | data/control_explanations.json `events[a002]` `conflicting_approaches[].burden_proxy_veh_s` | veh·s | t = 785 s |
| 1,184 (re-checked, still over budget) | 1184.428 | PASS | forecast | data/control_explanations.json `events[a002]` | veh·s | t = 787-791 s |
| 913 / 1,000, WITHIN BUDGET, PRIORITY APPROVED | 913.295 | PASS | forecast | data/control_explanations.json `events[a002]` | veh·s | t = 792 s |
| NW 55 · SW 531 · NE 327 (breakdown after admission) | 55.34, 530.73, 327.22 | PASS | forecast | data/control_explanations.json `events[a002]` | veh·s | t = 792 s |
| Gauge keeps the 792 s forecast while the lease runs | rows 793-801 = committed_selective_lease | PASS | controller state | data/control_explanations.json `events[a002]` `reason` | - | t = 793-801 s |
| Budget B = 1,000 | 1000.0 | PASS | parameter | data/replays/P3.json `configuration.harm_budget` | veh·s | - |
| 'the estimate changed, the physical queues did not' | counts SE,NW,SW,NE at 785 = [2, 2, 8, 0], at 792 = [2, 2, 8, 0] | PASS | observed | data/replays/P3.npz `ordinary_edge_counts` | vehicles | t = 785 vs 792 s |
| Pedestrian phase protected; countdown to 0 | 771.5-796.5 s | PASS | observed signal state | data/replays/P3.npz `signal_mode` (3 = pedestrian) at J1 | s | 771.5-796.5 s |
| Green at first legal moment, 796.5 s | 796.5 s | PASS | observed signal state | data/replays/P3.npz `signal_mode`, `signal_edge` | s | 796.5 s |
| Withholding did not delay the ambulance (pedestrian phase ran to 796.5 s either way) | switch delay 785 s = 11 s (= 796 - 785) | PASS | controller input | data/control_explanations.json `events[a002]` `switch_delay_s` | s | t = 785 s |
| 72 s earlier: SE green for the ambulance vs no priority | 868.5 - 796.5 = 72.0 | PASS | observed vs counterfactual run | P3.npz and P0.npz `signal_mode/edge` | s | first SE green after 790 s |
| +52 s later: next green for NW, SW and NE | [52.0, 52.0, 52.0] | PASS | observed vs counterfactual run | P3.npz and P0.npz `signal_mode/edge` | s | first green after 790 s |
| 843 veh·s measured, OBSERVED NOT FORECAST | 842.5 (shown rounded half-up) | PASS | observed, cumulative | edge_halts (speed < 0.1 m/s) on NW, SW, NE, P3 minus P0, x 0.5 s | veh·s | 785-880 s |
| Forecast tick 913 on the measured gauge | 913.3 | PASS | forecast | data/control_explanations.json `events[a002]` | veh·s | t = 792 s |
| Clock never runs backwards; events keep recorded order | min dSIM/frame = -0.0000 | PASS | render | src/cinematic.py KNOTS (monotone PCHIP) | s | all frames |
| Peak replay speed (shown in the clock) | 17.7x | PASS | render | src/cinematic.py | - | trade-off beat |
| Request admitted at video time | 6.90 s | PASS | render | KNOTS | s | sim 792.0 s |
| Green at video time | 9.53 s | PASS | render | KNOTS | s | sim 796.5 s |

**20/20 pass.**

Notes: forecasts are the controller's Eq. 20 ranking surrogate, logged before the decision. The measured value counts
stopped (speed < 0.1 m/s) vehicle-seconds on the three conflicting approaches, AmbuGo minus the matched no-priority run,
and also reflects the other two ambulances (a000, a001), whose priority requests exist only in the AmbuGo run.
One synthetic episode (seed 19); not a field measurement.
