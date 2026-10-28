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

| Video | Sim | Stage | Headline / subline |
|---|---|---|---|
| 0.0-2.5 | 777.0-785.0 | - | Ambulance approaching PCL Chowk / Recorded traffic simulation on the Mohali road network |
| 2.5-6.9 | 785.0-792.0 | FORECAST | Forecast exceeds the budget / Expected delay to the other approaches is above 1,000 vehicle-seconds: request withheld |
| 6.9-9.6 | 792.0-796.6 | DECISION | Forecast within the budget / 913 vehicle-seconds: request admitted; the pedestrian interval completes first |
| 9.6-12.6 | 796.6-804.5 | ACTUATION | Green at the earliest legal time / The ambulance is served 72 s earlier than in the no-priority run |
| 12.6-16.6 | 804.5-862.0 | CONSEQUENCES | Effect on adjacent roads / Conflicting approaches wait 52 s longer; traffic shifts across the junction |
| 16.6-20.0 | 862.0-880.0 | - | Priority, granted within a forecast budget / Observed cost: 843 additional stopped vehicle-seconds, against a forecast of 913 |

Card sequence: forecast gauge (2.7-9.6 s), signal-service timeline (9.6-12.6 s), adjacent-roads table (12.6-16.6 s), observed-cost gauge (16.6-20 s).

## Revision 3 (this version)

* Real Google Maps imagery with place and road names, lightly desaturated; headline, clock and card sit on dark panels above it. The vector-road overlay was removed.
* Six headlines instead of twelve; each stays 2.5-4.4 s, with 0.5-0.6 s cross-fades. The card changes four times.

## Revision 2 (this version)

* Headlines rewritten in plain academic register (e.g. *Forecast exceeds the budget*, *Pedestrian interval completes first*, *Priority, granted within a forecast budget*); stage labels FORECAST / DECISION / ACTUATION / CONSEQUENCES.
* Vehicles drawn ~40 % larger; ambulance marker ~50 % larger with a solid halo.
* New beat *Effect on adjacent roads*: seven roads around PCL Chowk (four approaches, three exits) coloured and labelled by vehicles present, AmbuGo minus no priority (red more, blue fewer), with a matching table in the card. Shows one signal change altering conditions on several roads at once.
* Glow reduced to a single pass.

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
