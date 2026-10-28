# Verification: AmbuGo_30s_Explainer.mp4 (step-specific numbers)

Shared claims (green 796.5 s, 72 s earlier, 52 s later, measured 843, forecast 913) are in verification.md.

| Shown on screen | Value from data | Check | Source |
|---|---|---|---|
| Route A: 310 s with priority, 503 s without | 310.09 / 503.37 | PASS | decisions.json first_decision.unique_routes (t = 761 s) |
| Route B: 307 s with priority, 500 s without | 306.97 / 500.24 | PASS | decisions.json first_decision.unique_routes (t = 761 s) |
| Route C: 341 s with priority, 534 s without | 340.59 / 533.87 | PASS | decisions.json first_decision.unique_routes (t = 761 s) |
| Route D: 337 s with priority, 531 s without | 337.47 / 530.74 | PASS | decisions.json first_decision.unique_routes (t = 761 s) |
| Route B selected | ['B'] | PASS | decisions.json `chosen` |
| NW: queue 3.9 veh, 68 arrivals/h, loading 0.43 | q=3.9381, lambda=0.018809/s, X=0.4280 | PASS | control_explanations.json events[t=785].conflicting_approaches |
| SW: queue 6.4 veh, 237 arrivals/h, loading 0.81 | q=6.3725, lambda=0.065830/s, X=0.8119 | PASS | control_explanations.json events[t=785].conflicting_approaches |
| NE: queue 0.0 veh, 271 arrivals/h, loading 0.85 | q=0.0000, lambda=0.075234/s, X=0.8500 | PASS | control_explanations.json events[t=785].conflicting_approaches |
| SE queue 3.8 veh | 3.7829 | PASS | events[785].queue_proxy |
| Signal: pedestrian interval, 11 s left | switch_delay_s=11, before=ped | PASS | events[785] |
| Ambulance 3.0 s from the stop line | 3.032 | PASS | events[785].free_flow_eta_to_stopline_s |
| Priority now: saves about 73 s | predicted_benefit_s=72.65 | PASS | events[785].predicted_benefit_s (nominal red wait avoided) |
| Keep the normal cycle: waits about 73 s for its green | nominal_signal_wait_s=72.65 | PASS | events[785].nominal_signal_wait_s (forecast) |
| Priority now: +1,045 vehicle-seconds (NW 175, SW 501, NE 368), over budget | 1044.521 | PASS | events[785] |
| Re-evaluation bars 785-792 s | [1045, 1045, 1184, 1184, 1184, 1184, 1184, 913] | PASS | events[785..792].raw_burden_proxy_veh_s, reason |
| What changed at 792 s: SE 4.8 to 2.8, SW 9.4 to 8.4, NW 1.9 to 0.9 | SE 4.78->2.78 | PASS | events[791], events[792] |
| physical queues unchanged | [2, 2, 8, 0] vs [2, 2, 8, 0] | PASS | P3.npz ordinary_edge_counts |

**17/17 pass.**

Arrival rates and loading are the controller's fixed historical inputs for this run; time of day is not a separate input in this episode.
