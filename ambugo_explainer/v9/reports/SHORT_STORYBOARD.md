# AmbuGo v9 short film: storyboard and voiceover

* **File:** `output/AmbuGo_v9_Short.mp4`, 1920×1080, 30 fps, 1:36 (96 s), no audio. Render it with `python3 mapfilm.py --jobs 4`.
* **Focus:** two junctions and two decisions. The request is withheld at Junction 1 (J11) and granted at Junction 2 (J21), and the film shows what each choice does to every road.
* **Captions:** at the top, in EB Garamond.
* **Backdrop:** Google Maps captures of two Mohali (Punjab) intersections (`data/maps/capture_manifest.json`).
* **Traffic:** the recorded v9 SUMO trajectories, placed on those roads by `mapfit.py`.

| # | Video | Sim time | Shown | Top caption |
|---|---|---|---|---|
| 1 | 0:00–0:07 | 4,078 → 4,091 s (≈1.9×) | Junction 1 (Sarovar Path capture), live signals. The ambulance's position is marked at the screen edge while it is still off-screen. | "An ambulance is on its way to hospital. At each signal it may ask for the light to be changed. AmbuGo decides, junction by junction, whether that is worth it." |
| 2 | 0:07–0:22 | paused 4,091 | **Forecast per road**, tagged on each other road: cross road +179 veh·s, south road +126 veh·s. The gate panel reveals: time saved 3.6 s ✗ (needs ≥ 5 s); extra waiting 306 veh·s ✓; room 5 of 45 ✓; 30.5 s within the 60 s hold ✓. Then "No change: normal signal cycle". Marker: "Ambulance, about 27 s away". | (1) "…would turn green for the ambulance only about 3.6 seconds after it arrives…" (2) "If the light were changed anyway, AmbuGo estimates the cross road would wait an extra 179 vehicle-seconds and the south road 126: about 306 in total." (3) "That cost would buy the ambulance less than the 5-second minimum, so AmbuGo leaves the signal on its normal cycle." |
| 3 | 0:22–0:46 | 4,091 → 4,170 (≈5×), then paused | Split screen, same clock. Left: AmbuGo (archived P3). Right: the always-priority controller JFAST (archived P2, identical up to 4,091 s). Live stopped counts on all **three** roads. Each side has an **"Order of green lights from now"** card that highlights the road currently green. Left: cross road 4,096 → ambulance's road 4,122 → south road 4,144. Right: ambulance's road 4,096 → south road 4,122 → cross road 4,160. The scene ends on the **per-road table**. | (1) "The same moment, the same traffic…" (2) "Changing the light reorders the greens: the ambulance's road goes first, the south road moves up, and the cross road is pushed to the back." (3) "So the cross road waits 64 s longer for its green, while the south road gets green 22 s sooner and the ambulance passes 11 s earlier." (4) "In total: 296 more vehicle-seconds of waiting, almost all on the cross road, to save the ambulance 11 seconds. That is why AmbuGo did not change this light." |
| 4 | 0:46–0:52 | 4,118 → 4,132 (≈2.3×) | Junction 2 (3B2 / Mohali Bypass capture): the ambulance approaches | "Forty seconds later, the next junction. Here the ambulance would otherwise wait about 43 seconds for its green." |
| 5 | 0:52–1:07 | paused 4,132 | Forecast per road: east +1,362, south +1,692, west +273 veh·s. Gate panel: 43.4 s ✓, 3,326 veh·s ✓, 8 of 112 ✓, 50.4 s ✓. Then "Change the light: priority given". | "Before changing the light, AmbuGo estimates the extra waiting…" → "About 3,326 vehicle-seconds in total, well within its limit of 10,000, and the ambulance saves about 43 s. So the light is changed." |
| 6 | 1:07–1:24 | 4,132 → 4,236 (≈7×) | Split screen, same clock. Left: AmbuGo, priority given (P3). Right: the same moment with the light not changed (labelled replay, `P3_withholdJ21`). Live stopped counts on all four roads. | "The change is made safely…" → "…green at 4,142 s instead of 4,228 s…east and south roads…" → "No queue comes close to filling its road (at most 19 vehicles), and the normal cycle resumes once the ambulance has passed." |
| 7 | 1:24–1:36 | paused | Summary card: three rules with the junction values, then the study averages | none |

**The per-road table at the end of scene 3** (all values measured in the two archived runs):

| Road | Next green: no change → change | Stopped vehicle-seconds, 4,085–4,200 s |
|---|---|---|
| Cross road | 4,096 → 4,160 s (64 s later) | 79 → 463 |
| South road | 4,144 → 4,122 s (22 s sooner) | 242 → 195 |
| Ambulance's road | 4,122 → 4,096 s (26 s sooner) | 173 → 131 |
| All three roads | ambulance through 11 s earlier | 493 → 789 (+296) |

The table's footnote reads: "AmbuGo's advance estimate was +306 vehicle-seconds."

Every on-screen number is checked by `verify.py` (sections `short` and `short-J1`; 189 checks in total, 0 failed).

## Voiceover (about 240 words, measured pace)

**0:00** An ambulance is on its way to hospital. At each traffic signal it can ask for the light to be changed. AmbuGo decides, junction by junction, whether that is worth doing.

**0:07** At this first junction, the light will turn green for the ambulance only three or four seconds after it arrives anyway. Before deciding, AmbuGo estimates what a change would cost the other roads: about 180 extra vehicle-seconds of waiting on the cross road, and about 130 on the south road. All of that would save the ambulance less than five seconds, so the signal stays on its normal cycle.

**0:22** Here is what changing the light would actually have done. On the right, a controller that always gives the ambulance green. The order of the greens changes: the ambulance's road goes first, the south road moves up, and the cross road drops to the back. The cross road waits more than a minute longer. In total there are nearly three hundred more vehicle-seconds of waiting, to save the ambulance eleven seconds.

**0:46** At the next junction it is different: the ambulance would otherwise wait over forty seconds.

**0:52** Again AmbuGo estimates the cost on each road: about three thousand three hundred vehicle-seconds, well within its limit. So here the light is changed.

**1:07** It changes safely, through minimum green, amber and all-red. The ambulance gets green eighty-six seconds earlier. The cost falls on the east and south roads, and no queue comes close to filling its road.

**1:24** Across the study's simulations, this selective approach was slightly slower than changing every light, but it spread much less extra congestion.
