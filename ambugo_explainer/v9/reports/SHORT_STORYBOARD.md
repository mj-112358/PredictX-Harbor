# AmbuGo v9 short film: storyboard and voiceover

* **File:** `output/AmbuGo_v9_Short.mp4`, 1920×1080, 30 fps, 1:24 (84 s), no audio. Render it with `python3 mapfilm.py --jobs 4`.
* **Focus:** two junctions and two decisions. One request is withheld (J11) and one is granted (J21), and the film shows how the after-effects are kept small.
* **Captions:** at the top, in EB Garamond.
* **Backdrop:** Google Maps captures of two Mohali (Punjab) intersections (`data/maps/capture_manifest.json`).
* **Traffic:** the recorded v9 SUMO trajectories, placed on those roads by `mapfit.py`.

| # | Video | Sim time | Shown | Top caption |
|---|---|---|---|---|
| 1 | 0:00–0:07 | 4,078 → 4,091 s (≈1.9×) | Junction 1 (Sarovar Path capture): the ambulance approaches on its road while live signals run | "An ambulance is on its way to hospital. At each signal it may ask for the light to be changed. AmbuGo decides, junction by junction, whether that is worth it." |
| 2 | 0:07–0:19 | paused 4,091 | Gate panel, rows revealed one by one: time saved 3.6 s ✗ (≥ 5 s); extra waiting 306 veh·s ✓; room 5 of 45 ✓; within 60 s hold 30.5 s ✓; then "No change: normal signal cycle" | "…the light would turn green for the ambulance only about 3.6 seconds after it arrives…" → "…less than the 5-second minimum, so AmbuGo leaves the signal on its normal cycle." |
| 3 | 0:19–0:34 | 4,091 → 4,170 (≈6.8×) | Split screen, same clock. Left: AmbuGo (archived P3). Right: always-priority controller JFAST (archived P2, identical state up to 4,091 s). Live "stopped" counts for the cross road and the ambulance's road | "…a controller that always changes the light…" → "Changing the light saves the ambulance 11 s here, but the cross road waits 64 s longer for its green: 463 instead of 79 vehicle-seconds of stopped time." |
| 4 | 0:34–0:40 | 4,118 → 4,132 (≈2.3×) | Junction 2 (3B2 / Mohali Bypass capture): the ambulance approaches | "Forty seconds later, the next junction. Here the ambulance would otherwise wait about 43 seconds for its green." |
| 5 | 0:40–0:55 | paused 4,132 | Estimated extra waiting tagged on each other road (east +1,362, south +1,692, west +273 veh·s). Gate panel: 43.4 s ✓, 3,326 veh·s ✓, 8 of 112 ✓, 50.4 s ✓, then "Change the light: priority given" | "Before changing the light, AmbuGo estimates the extra waiting…" → "About 3,326 vehicle-seconds in total, well within its limit of 10,000, and the ambulance saves about 43 s. So the light is changed." |
| 6 | 0:55–1:12 | 4,132 → 4,236 (≈7×) | Split screen, same clock. Left: AmbuGo, priority given (P3). Right: the same moment with the light not changed (labelled replay, `P3_withholdJ21`). Live stopped counts on all four roads | "The change is made safely: 8-second minimum green, then amber and all-red…" → "…green at 4,142 s instead of 4,228 s…east and south roads…" → "No queue comes close to filling its road (at most 19 vehicles), and the normal cycle resumes once the ambulance has passed." |
| 7 | 1:12–1:24 | paused | Blurred map, summary card. "How AmbuGo keeps the after-effects small": three rules with the junction values. Then the study averages: 44.6% faster than no priority; 11.1 s slower than JFAST, with 3.84 vehicle-hours less extra road occupancy per run | none |

Every on-screen number is checked by `verify.py` (section `short`; 171 checks in total, 0 failed).

## Voiceover (about 190 words, measured pace)

**0:00** An ambulance is on its way to hospital. At each traffic signal it can ask for the light to be changed. AmbuGo decides, junction by junction, whether that is worth doing.

**0:07** At this first junction, the light will turn green for the ambulance only three or four seconds after it arrives anyway. Stopping the cross traffic would gain less than AmbuGo's five-second minimum. So the signal stays on its normal cycle.

**0:19** Compare a controller that always changes the light. It saves the ambulance eleven seconds here, but the cross road waits more than a minute longer for its green.

**0:34** At the next junction it is different: the ambulance would otherwise wait over forty seconds.

**0:40** AmbuGo first estimates how much extra waiting the change would cause on the other three roads. It is about three thousand three hundred vehicle-seconds, well within its limit. So the light is changed.

**0:55** It changes safely. The current green finishes its minimum time, then amber and all-red. The ambulance gets green eighty-six seconds earlier. The cost falls on the east and south roads, and no queue comes close to filling its road.

**1:12** Across the study's simulations, this selective approach was slightly slower than changing every light, but it spread much less extra congestion.
