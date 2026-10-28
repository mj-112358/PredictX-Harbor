# Archived scenes vs newly executed counterfactual scenes

## Where the data comes from

| Run (data/episodes/…) | What it is | Archived? | How it was checked |
|---|---|---|---|
| `P3` | JSEL (P3), seed 40, load 0.95, 1 mission: the policy's actual decisions | **Yes.** This exact run is in the v9 archive (`work/v9/sumo_main/records.json`) | Re-executed with log-only instrumentation. All 16 compared scalar fields equal the archived record: schedule hash, ordinary J, completions, mission rows, route changes, fallbacks, request ticks, every rejection counter, admissions, guard overrides, timing violations, expired leases, action counts (`archive_check.json`). |
| `P0` | S0 (no priority), same seed, load and missions | **Yes** | Same check, all fields equal |
| `P3_withholdJ21` | P3 with one change: the J21 request for approach E_J11_J21 is dropped from t = 4,132 s on that link | **No.** Newly executed explanatory counterfactual | Identical to `P3` before 4,132 s: same schedule hash and the same deterministic replay up to the branch tick. `branch_events.json` lists every withheld tick. |
| `P3_grantJ11` | P3 with the refused J11 request forced through at 4,091 s | **No.** Newly executed | Not used in the film (see below) |

The archive contains no per-vehicle trajectories, per-tick signal states, gate inputs or candidate tables. Its run outputs are scalar records and paired occupancy traces. Everything drawn on the map therefore comes from the instrumented re-execution. That re-execution is shown to be the archived run by the field-by-field match above, together with `verify.py`'s mission-time checks.

**Versions.** SUMO 1.28.0 (libsumo), as in the archive. Python 3.13.17 here, against 3.12.14 in the archive. The match was nevertheless exact.

**Frozen source.** `work/v9/sumo_study.py`, SHA-256 `471a2291…cecf2`, was imported unmodified from a separate working copy. `instrument.py` only replaces module attributes (`traci`, `Signal`, `Controller`) with subclasses or proxies that read state and log it. The one exception is the `--branch` option, which removes the single request.

## Which scenes use which run

| Scene | Run | Status shown on screen |
|---|---|---|
| A–H, K | `P3` | the archived decision ("what AmbuGo did") |
| I, strip | `P3_withholdJ21` signal log | strip labelled "J21 if the normal cycle had continued (counterfactual replay)". The paused map is at 4,132 s, where both runs are identical. |
| I, Eq. 20 values | `P3` gate log at 4,132 s | values the controller actually computed |
| J, left | `P3` | "Scenario 1 — priority granted (AmbuGo's decision)" |
| J, right | `P3_withholdJ21` | "Scenario 2 — normal cycle kept (counterfactual replay)" |
| K, link tint | `P3` − `P3_withholdJ21` occupancy (trace.npz) | legend "priority vs normal cycle … not a signal colour" |
| L, episode rows | `P3`, `P3_withholdJ21` ("one-decision counterfactual"), `P0` | labelled per row |
| L, paper rows | archived `sumo_analysis.json` aggregates | "Across the paper's 1,728 microscopic runs (balanced means)" |

## Why "withhold at J21" is the counterfactual

* **One decision.** It isolates a single decision. Both branches share the state at t = 4,132 s: vehicles, signal phases, sensor history, controller memory and the RNG streams. Sensor draws are keyed by (seed, time, edge), so both branches observe identically until their physical states diverge.
* **The decision was admitted.** J21 is the decision in this episode where a priority request was admitted against a sizeable forecast burden (3,326 veh·s), so the comparison shows a real trade-off.
* **Later decisions follow the policy.** After 4,132 s the counterfactual is not frozen. Later decisions (J22, J12, J13, escalation) are made by the same policy on the new state. The 323.5 s mission time is therefore "what JSEL would have done had J21 been withheld", not "the same later decisions with a delay".

## Why the J11 forced-grant branch is not in the film

Forcing the J11 request through moved the ambulance across J11 11 s earlier, against a forecast of 3.6 s. It also changed the later route choice (via J23); arrival was 243.5 s, and the north approach at J11 waited 64 s longer. That branch is useful as a check that the benefit forecast is conservative here. However, it shows something AmbuGo did not do, and it changes two decisions at once (the grant and a later re-route), so it is not a clean single-decision illustration. It is reported here and kept in `data/episodes/P3_grantJ11`. It is not shown.

## What is not a counterfactual

* The whole-episode S0 result (572.0 s) is an archived whole-policy comparison under identical demand, not a single-decision counterfactual. The film labels it "no signal priority (S0)".
* The paper aggregates are never presented as this episode's outcome.
