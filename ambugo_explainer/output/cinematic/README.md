# AmbuGo — 20-second cinematic cut

`AmbuGo_20s_Cinematic.mp4`: 1920×1080, 30 fps, 600 frames, exactly 20.0 s, H.264 (CRF 16), about 4.7 MB, no audio.

```bash
python src/cinematic.py            # render (≈3 min on 4 cores)
python src/cine_verify.py          # verification.md, 20/20 claims
python src/cine_events.py          # event_table.csv (PCL Chowk, 776–885 s, every 0.5 s)
python src/cinematic.py --contact  # contact_sheet.jpg, contact_transitions.jpg
```

## Storyboard (final)

One intersection (PCL Chowk), one ambulance (a002), one decision. The simulated clock runs 777 → 880 s.
Its speed is a smooth, monotone time-warp, so events keep their recorded order and the clock never jumps.
The clock in the corner always shows the simulation time and the current replay speed.

| Video | Sim | Stage | On screen |
|---|---|---|---|
| 0.0–2.5 | 777–785 | — | Ambulance on its recorded track; pedestrian countdown at the junction. *An ambulance approaches* |
| 2.5–4.0 | 785–786 | 01 PREDICT | NW / SW / NE approaches light up with their forecast contributions (175 · 501 · 368); the gauge fills. *Predict the cost* |
| 4.0–5.5 | 786–787 | 02 DECIDE | Gauge passes the fixed budget tick: **1,045 / 1,000**, coral overflow arc, PRIORITY WITHHELD; dashed coral request line from the ambulance to its stop line. *Over budget* |
| 5.5–7.05 | 787–792 | 02 DECIDE | Logged re-checks: 1,184, still withheld. *Re-checked every second* |
| 7.05–8.0 | 792–793 | 02 DECIDE | **913 / 1,000**, PRIORITY APPROVED, request line turns emerald. *Within budget: the estimate changed, the physical queues did not* |
| 8.0–9.7 | 793–796.5 | 03 ACT | Pedestrian ring counts down to 0 while the ambulance is stopped. *Pedestrian phase protected* |
| 9.7–10.5 | 796.5–797.6 | 03 ACT | SE stop line turns green with a pulse. *Green at first legal moment* |
| 10.5–12.4 | 797.6–804 | 03 ACT | Ambulance accelerates through; card shows both runs' green phases: **72 s earlier**. *Ambulance through* |
| 12.4–14.9 | 804–840 | 04 TRADE-OFF | Conflicting approaches and their queued cars tinted; timeline arrows NW/SW/NE: **+52 s later**. *Other approaches wait longer* |
| 14.9–17.4 | 840–880 | 04 TRADE-OFF | Same gauge, now MEASURED: extra stopped vehicle-seconds grows to **843**, with ticks for FORECAST 913 and BUDGET; OBSERVED, NOT FORECAST. *The cost, measured* |
| 17.4–20.0 | 880 (hold) | — | Card clears; hero view, route trace leaving frame. **Predict. Constrain. Prioritize.** |

## What changed and why it reads better

* **One subject.** The whole film stays on PCL Chowk. Escalation, the second junction, the re-route and the network replay are cut. The viewer follows one causal chain: forecast → budget → decision → legal transition → passage → cost.
* **A designed map.** The Google capture is blurred and tone-mapped into a narrow navy band, so geography stays and place labels sink. The roads on top are the actual SUMO lane geometry drawn crisply, so a vector layer adds sharpness without inventing roads. Attribution is kept bottom-right.
* **Colour carries meaning, consistently.** Cyan is always the ambulance (route, trail, halo, its approach). Violet, amber and teal are always NW, SW and NE (map glow, chips, gauge segments, timeline blocks). Coral means over budget or a cost; emerald means approved or green. Slate is ordinary traffic.
* **One instrument, reused.** A single 240° gauge with a fixed BUDGET tick holds the forecast (segmented by approach), flips its decision pill, and later returns as the *measured* gauge. There it shows the 913 forecast as a tick, so forecast and observation share one scale without being conflated. The timeline card in between is the only other graphic.
* **The causal link is drawn.** Forecast chips sit on the approaches they come from. A dashed request line runs from the ambulance to its stop line and changes from coral to emerald at the moment of admission. The pedestrian countdown explains why the light stays red after approval.
* **Typography.** Manrope 800/500 for headlines (60–66 px) and supporting lines (26 px). IBM Plex Mono for the clock, stage indicator and every number (gauge numerals 76 px). One headline at a time, 2–5 words.
* **Pacing.** The opening runs at ~3× so the ambulance arrives quickly. The decision runs at 0.6–0.9× while the numbers matter. The stopped period is used for the pedestrian countdown. The passage runs at ~3×. The trade-off runs at up to 19×, while vehicles are mostly queued, and is held for 2.6 s at the end.

## Integrity notes

* Withholding at 785–791 s did not delay the ambulance: the pedestrian reservation held every approach red until 796.5 s either way (switch delay 11 s at 785 s). The film never shows otherwise.
* After admission, the controller log's later rows (`committed_selective_lease`) carry recomputed burdens (e.g. 1,136 at 797 s). These are not new decisions. The gauge keeps the 792 s admission forecast, as the lease does.
* 843 is cumulative stopped vehicle-seconds, AmbuGo minus no-priority, on NW+SW+NE over 785–880 s (842.5, rounded half-up). It also reflects the other two ambulances' requests, which exist only in the AmbuGo run.
* Fonts: Manrope and IBM Plex Mono, SIL Open Font License, in `inputs/fonts/`.
