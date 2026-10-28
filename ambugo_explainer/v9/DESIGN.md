# AmbuGo v9 explainer — internal design document (Part I)

Sources read: manuscript *AmbuGo First Empirical Draft v9* (Sections 3–5, Eqs. 19–23, Algorithm) and the
frozen controller `work/v9/sumo_study.py`. The copy available here is
`AmbuGo_20s_Reproducible/simulation/upstream/sumo_study.py`, SHA-256
`471a22911a2de9773fcd839872b3ba1ec5e00412c0ecc79142503e72f41cecf2`. That is byte-identical to the hash the
paper cites. Line references below are to that file. The JSEL policy is `P3`, JFAST is `P2`, and S0 is `P0`.
The selected configuration (paper §5) is B = 10,000 veh·s, ε = 0 s, strong actions {(0,0),(30,60)}, and γ = 20 s.

**Status: blocked on inputs.** `public_model()` reads `nodes.csv`, `edges.csv`, `vehicle_classes.csv` and
`od_probabilities.csv` from `outputs/AmbuGo_v7_Synthetic_Data/`. The district geometry, storage, saturation
flows, vehicle-class parameters and OD weights exist only there. None of these files is in the material
supplied so far. The paper gives ranges (links 260–472 m, 1–2 lanes, 8.33/11.11 m/s), not the values.
The district must not be re-invented, so items 9–10 are stated conditionally.

## 1. What the controller knows at each decision (1 s ticks)

* **Link counts** for all 59 links (`Controller.observe`):
  * sampled every 5 s and received 2 s later;
  * noise of −2…+2 vehicles, with probabilities 1/7, 1/7, 3/7, 1/7, 1/7;
  * two-state dropout (available→missing 0.015, missing→available 0.25);
  * a missing sample keeps the last received value.
  * Draws are keyed by `stable_rng(seed,'sensor',t,edge)`, so they are identical across policies.
* **Queue proxy** q̂ₑ = [n̂ₑ − λₑ·Lₑ/vₑ]₊ (`q()`). Here λₑ is the *design* expected flow, not a measurement.
  It counts moving vehicles too, so it is not the visible stopped-car count.
* **Exact ambulance state**: link and longitudinal position at each tick.
* **Each junction's current signal mode and elapsed time**, plus its fixed plan:
  * arms, greens and cycle;
  * nominal per-arm loading X from `signal_design`.
* **Public geometry**: lengths, speeds, saturation flow, storage K.
* The controller has **no** future arrivals, **no** paired no-priority run and **no** true queue.

## 2. Alternatives evaluated (`plan`, `candidates`)

* **When**: at every *regular-link entry*, and immediately when the ambulance escalates.
* **Routes**:
  * up to 4 loopless continuations to BE1 within 1.5× the minimum free-flow time;
  * no immediate U-turn;
  * the legal incumbent continuation replaces the 4th candidate if it is missing.
* **Actions in selective mode**: (h, l) ∈ {(0,0), (30,60)}, meaning look-ahead h and requested hold l in seconds.
* **Actions in escalated (P2) mode**: only (30,60).
* Every route × action pair is scored. That gives up to 8 candidates in selective mode.

## 3. ETA forecast (`estimate`)

The forecast walks the route link by link and adds:

1. free-flow time (remaining distance on the current link, full length on later links);
2. the queue discharge estimate q̂/(s/p);
3. the signal wait at that link's junction.

The signal wait is `nominal_wait`, a cyclic continuation from the *current* stage with no future priority.
For a priority action it is reduced by the admitted benefit from `pulse`.
This is a recursive approximation, not a future simulation.

## 4. Burden forecast, Eq. 20 (`pulse`)

* Displaced green: Dₑ = min(l, max(5, h + d̂ₑ)), where d̂ₑ = q̂ₑ/(sₑ/p).
* Burden: Ĉₑ = Σⱼ≠ₑ [ q̂ⱼ·Dₑ + λⱼ·Dₑ² / (2·max(1 − Xⱼ, 0.05)) ].
* The sum runs over the junction's other arms. λⱼ is the design expected flow and Xⱼ is the design loading.
* Units are vehicle-seconds. The paper calls it a ranking surrogate, not a harm bound.
* A route's burden is the sum of its *admitted* junction burdens.

## 5. Accept or reject (`pulse` and `requests_at`, P3 only)

* **Trigger**: a request is considered only when the free-flow ETA to the stop line is ≤ h + d̂ₑ.
* **Eq. 19 feasibility**: [h + d̂ₑ − δₑ]₊ + d̂ₑ + 2 ≤ l, where δₑ is the current legal switch delay.
* **Headroom**: n̂_out < 0.85·K_out.
* **Benefit**: bₑ = nominal wait − max(0, δₑ − arrival). It is set to 0 if the pulse is infeasible or headroom fails.
* **Admission** requires bₑ ≥ 5 s, Ĉₑ ≤ B and headroom.
* **Rejection reason** is recorded in the code's order: headroom first, then benefit (which also covers an
  infeasible pulse), then burden.
* With B = 10,000, burden refusals should be rare. **The film must report whichever gate actually fails.**
* Once admitted, a **lease** forwards the request. It is not withdrawn because its own green removed the benefit.
* A lease ends at the earliest of:
  * l seconds after the approach's green starts;
  * 120 s after admission;
  * loss of headroom;
  * the ambulance leaving the link.
* A lease never renews on the same link.

## 6. Request → signal (`Signal.update`)

* **Arbitration**: requests are formed in (dispatch time, ambulance id) order. The first request per junction wins.
* **Requested approach already green**: the green is held to max(planned green, min(l, 90)).
* **Another approach green**: it is truncated only after 8 s of minimum green, followed by amber 3 s and
  all-red 2 s. Then the requested approach gets green.
* **Pedestrian all-red opportunity (25 s)**: always completed before the switch.
* **Service-age guard**: an *observed-occupied* approach, or the pedestrian opportunity, whose service age
  is ≥ 150 s takes precedence; the most overdue wins.
  This is a service-age guard, not a bound on any driver's wait.
* **After the green**: the cyclic pointer is synchronised to the served approach, so the cycle is permanently shifted.

## 7. Neighbouring junctions

* Each of the 16 junctions runs its own fixed-time automaton with a seeded stagger offset.
* There is no coordination and no green wave.
* An intervention reaches other junctions only physically:
  * longer queues occupy link storage, and full storage blocks insertion from upstream (`--ignore-junction-blocker -1`, no teleports);
  * platoons are released at different times;
  * other ambulances' requests also affect neighbours.
* Evaluation pairs each run with S0 using the per-link occupancy history (59 links + internal + 8 sources).
  The film should use **positive spatial exposure A** (Eq. 23). It counts extra occupancy where it appears,
  and is not the same thing as extra individual travel time.

## 8. Progress escalation (`progress_mode`, Eq. 21)

* Pₐ is the executed free-flow progress on traversed regular links (no credit on internal connectors).
* Dₐ = [t − dₐ − Pₐ]₊.
* If Dₐ > 20 s, that ambulance switches irreversibly to P2: its leases are cleared, it replans at once, and it
  uses (30,60) with no burden or benefit gate.
* Legality, headroom in SUMO, arbitration and the service-age guard still apply.
* This is a control trigger, not a clinical threshold. It does not guarantee being within 20 s of JFAST.

## 9. What a replay can supply

| Item | Archived `run_case` output | Faithful instrumented re-run |
|---|---|---|
| Per-link occupancy history (paired) | yes (`.npz`) | yes |
| Ordinary vehicle event ledger | yes (`_ordinary_events.csv`) | yes |
| Mission rows, escalation log | yes | yes |
| Plan log (chosen pair, T, A, global min) | first 100 only | all, with every candidate |
| Vehicle trajectories, classes, lanes | **no** | yes (TraCI subscription) |
| Signal states per tick | **no** (aggregate spans only) | yes |
| Per-request gate values (q̂, d̂, δ, b, Ĉⱼ, headroom) | **no** | yes (wrap `pulse` and `requests_at`) |
| Lease states, arbitration losers | **no** | yes |

**Instrumentation plan.** Run on a separate working copy. Subclass `Controller` and `Signal` and log only;
the decision logic stays byte-identical. Check reproduction against the archived per-run JSON:

* mission travel times;
* J;
* admission counts;
* rejection-tick counts;
* fallbacks.

Python 3.13 here vs 3.12.14 archived, so any mismatch must be reported, not hidden.

**Scenario search.** Do this over the locked main grid (seeds 0–47, loads {0.6, 0.8, 0.95}, missions
{1, 4, 10}), looking for, in one episode where possible:

* a route choice where candidates differ;
* a refusal with its actual failing gate;
* an admission;
* a measurable change in A;
* escalation;
* (optionally) arbitration.

## 10. What needs a new controlled simulation

* **Scenario 1 vs Scenario 2 at one decision** (forced grant vs forced normal cycle from an *identical* state).
  The archived S0/JSEL comparison is a whole-policy comparison, not a single-decision counterfactual.
  * Mechanism: `traci.simulation.saveState`/`loadState` at the decision tick, plus a deep copy of
    `Controller`/`Signal` state. Sensor draws are stateless keyed RNG, so branches see identical observations.
  * Each branch must be labelled as an explanatory counterfactual, not a policy result.
* **Spillback or receiving-space restriction**: shown only if it appears in the chosen run or branch.
* **Aggregate results**: taken from the archived analysis, never from the animated episode:
  * S0 401.57 s, JFAST 211.19 s, JSEL 222.29 s;
  * JSEL − JFAST = +11.10 [9.47, 12.94] s;
  * A −3.84 veh·h;
  * D⁺ −1.64 veh·h.

## Inputs required to proceed

1. `AmbuGo_First_Draft_Reproducibility_v9.zip`. At minimum, `outputs/AmbuGo_v7_Synthetic_Data/{nodes,edges,vehicle_classes,od_probabilities}.csv`.
2. Preferably also:
   * `work/v9/selected_policy.json`;
   * `work/v9/sumo_main/` (archived per-run JSON, for checking that the replay reproduces it);
   * `work/v9/sumo_network/district.net.xml` (to confirm the netconvert output).
