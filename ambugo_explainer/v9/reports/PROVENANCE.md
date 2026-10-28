# Event-by-event provenance

Every event in the film, in film order, with the run and logged record it comes from.

* **Paths.** All paths are relative to `ambugo_explainer/v9/data/`.
* **Times.** "t" is simulation time in seconds.
* **Machine-readable form.** `reports/verification.json` holds each displayed value with its source and the recomputed value.
* **Run labels.** `P3` and `P0` are instrumented re-executions shown to reproduce the archived runs exactly. `WH` is the newly executed counterfactual `P3_withholdJ21` (see `COUNTERFACTUALS.md`).
* **No splicing.** Each frame draws one run only, at one simulation time; the split screen draws two runs at the same time. No event from another episode, seed or load is used anywhere.

| # | Film time | t | Event | Run | Record |
|---|---|---|---|---|---|
| 1 | 0:00–0:11 | 3,992–4,066 | Background traffic: all vehicles, positions, headings, classes | P3 | `episodes/P3/trajectories.npz` (every 0.5 s physical step; positions are interpolated linearly between steps for display only) |
| 2 | 0:00– | all | Network geometry: lanes, internal lanes, junction shapes | – | `network/district.net.xml` (archive, sha `da9b1719…`) |
| 3 | 0:00– | all | Signal colours at every stop line | the run drawn | `episodes/<run>/signals.json`, controller state per 1 s tick (mode, served edge, elapsed) |
| 4 | 0:09 | 4,064 | Ambulance dispatched at BW1 | P3 | `result.json` `mission_rows[0]` (dispatch 4064, origin BW1, insertion 4064.5) |
| 5 | 0:11–0:25 | 4,065 | Four candidate routes × two actions; table values; choice of route C | P3 | `plans.json[0]` (t 4065): `candidates[*].eta`, `.burden`, `.incumbent`; `chosen_route` |
| 6 | 0:25–0:32 | 4,065 | J10 request admitted (gain 39 s, burden 227) | P3 | `gates.json` (t 4065, J10); `requests.json` (t 4065) lease `a000\|E_BW1_J10` |
| 7 | 0:26–0:31 | 4,066–4,088 | J10 green for the ambulance | P3 | `signals.json` J10 |
| 8 | 0:32–0:37 | 4,091 | Re-plan on entering E_J10_J11; switch C → A (gap 12.6 s > 10 s) | P3 | `plans.json` (t 4091); trajectories confirm link entry at 4,091 |
| 9 | 0:37–0:46 | 4,091 | J11 request refused: gain 3.578 s < 5 s; other gates pass | P3 | `gates.json` (t 4091, J11): `benefit`, `feasible_lhs`, `headroom_obs/cap`, `burden`, `signal_mode` = ped, `signal_elapsed` 20, `arrival` 27.4 |
| 10 | 0:44 | – | "In all 432 archived JSEL runs every refusal was too little gain or no feasible pulse; none on burden or space" | archive | `archive/sumo_main_records.json`: P3 rows (432), Σ`rejected_harm_ticks` = 0, Σ`rejected_headroom_ticks` = 0, Σ`rejected_benefit_ticks` = 59,248 |
| 11 | 0:46–0:59 | 4,091–4,130 | J11 normal cycle: ped → north green 4,096 → amber 4,117 → all-red 4,120 → west green 4,122 → amber 4,139 → all-red 4,142 → south green 4,144 | P3 | `signals.json` J11 |
| 12 | 0:50 | – | Fixed rules (8 s, 3 s, 2 s, 25 s, 150 s guard) | frozen source + logs | `sumo_study.py` constants; stage durations re-measured over every junction in all three runs by `verify.py` (amber always 3 s, all-red 2 s, pedestrian 25 s, green ≥ 8 s, no green→green) |
| 13 | 0:57 | 4,128 | Ambulance crosses J11 | P3 | trajectories (lane changes from E_J10_J11 to internal lane) |
| 14 | 0:59–1:04 | 4,132 | J21 state at decision: east green, elapsed 3 s | P3 | `gates.json` (t 4132, J21) `signal_edge`, `signal_elapsed`; `signals.json` |
| 15 | 1:04–1:16 | 4,132–4,180 | J21 east amber 4,137, all-red 4,140, north green 4,142; ambulance leaves the approach ~4,171 | P3 | `signals.json` J21; trajectories |
| 16 | 1:04–1:16 | 4,132–4,180 | Live "n stopped" per approach | P3 | trajectories: vehicles on the approach's lanes with speed < 0.1 m/s, ambulance excluded |
| 17 | 1:16–1:35 | 4,132 | Eq. 20 inputs and terms per arm; total Ĉ = 3,326.42 | P3 | `gates.json` (t 4132, J21) `arms[*]` (`q_hat`, `n_obs`, `lam`, `X`, `term_queue`, `term_arrival`), `D`, `d_hat`, `q_hat_own`, `burden`; `verify.py` recomputes every term from the formula |
| 18 | 1:16–1:35 | 4,100–4,260 | Normal-cycle strip with displaced-green window | **WH** | `episodes/P3_withholdJ21/signals.json` J21, labelled counterfactual |
| 19 | 1:31–1:35 | 4,132 | Admission checklist; PRIORITY ADMITTED | P3 | `gates.json` (t 4132, J21) `admitted` true, `benefit`, `feasible_lhs`, `headroom_*`, `switch_delay` |
| 20 | 1:35–1:52 | 4,132–4,236 | Left: granted run | P3 | trajectories + signals |
| 21 | 1:35–1:52 | 4,132–4,236 | Right: normal cycle kept; ambulance green 4,228; leaves ~4,242 | **WH** | `episodes/P3_withholdJ21/` trajectories + signals; `branch_events.json` (withheld ticks from 4,132) |
| 22 | 1:50–1:52 | 4,125–4,300 | Stopped veh·s table; at most 19 vehicles on an approach vs storage 53–132 | P3, WH | trajectories at 5 s samples; `fixture/edges.csv` `usable_storage_veh` |
| 23 | 1:52–2:13 | 4,171–4,320 | Link tint: occupancy P3 − WH, ±10 s mean | P3, WH | `trace.npz` `counts` (68 groups, 0.5 s) |
| 24 | 1:57 | 4,175 | J22 admitted (gain 89, burden 348) | P3 | `gates.json` (t 4175, J22) |
| 25 | 2:03 | 4,224 | J12 admitted (gain 87, burden 185) | P3 | `gates.json` (t 4224, J12) |
| 26 | 1:58–2:11 | 4,180–4,300 | Progress gauge D = t − d − P | P3 | `requests.json` `progress.a000.D` per tick (computed as in `progress_mode`) |
| 27 | 2:03 | 4,256 | Escalation: D = 20.48 s > 20 s → JFAST mode | P3 | `result.json` `progress_guard_log[0]`; next request (J13, t 4260) has `policy` P2 in `gates.json` |
| 28 | 2:11 | 4,316.5 | Arrival at BE1 (252.5 s) | P3 | `result.json` `mission_rows[0]` = archived record |
| 29 | 2:13–2:25 | – | Episode results 252.5 / 323.5 / 572.0 s | P3 / WH / P0 | `result.json` of each run |
| 30 | 2:13–2:25 | 4,132–4,400 | A = 5,992 veh·s, signed −74 veh·s | P3, WH | `trace.npz`, using the same quadrature as the archive's `analysis.cumulative_metrics` (Σ max(Δ, 0)·dt; Σ Δ·dt) |
| 31 | 2:13–2:25 | – | Paper aggregates (balanced means and CIs), 44.6 %, 51.6 % escalated | archive | `archive/sumo_analysis_aggregate.json` (P0/P2/P3 `time_s`; contrast P3−P2 `time_s`, `A_vh`, `Dplus_vh`); `archive/sumo_main_records.json` (1,114 of 2,160 JSEL missions escalated); `archive/paper_sumo_results.txt` |

## Things deliberately not shown

* **Crashes and collision-risk scores.** None are modelled. The archive records zero collision events, and SUMO's collision checks are model diagnostics, not road safety.
* **Spillback.** It does not occur at J21 in either replay, and the film says so instead of drawing it.
* **True queue length as the controller's input.** The controller only sees delayed, noisy counts. The film labels q̂ as an estimate and keeps it separate from the visible stopped-vehicle count.
