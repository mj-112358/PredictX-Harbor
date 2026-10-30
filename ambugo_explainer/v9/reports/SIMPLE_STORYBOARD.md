# AmbuGo v9 simple film: storyboard and voiceover

* **File:** `output/AmbuGo_v9_Simple.mp4`, 1920×1080, 30 fps, 1:46 (106 s), no audio. Render it with `python3 simplefilm.py --jobs 4`.
* **Design rule:** the viewer never has to look at more than one thing at a time. Each beat has:
  * one short caption at the top;
  * at most one highlighted thing on the map: a road (or one group of roads), the ambulance, or the junction;
  * or a single verdict pill, or a single centred sentence.
* **Nothing else:** no split screens, panels, tables, clocks or counters.
* **Map:** the Google Maps backdrop is faded so the highlight stands out.
* **Colours:**
  * blue = the ambulance's road;
  * orange = roads that would pay for a change;
  * green = a road that gains.
* **Signals:** the coloured stop lines at the junction are the logged signal states.

| # | Time | Run (sim time) | Caption | Highlight |
|---|---|---|---|---|
| 1 | 0:00–0:05 | P3, 4,079 → 4,091 | An ambulance is on its way to hospital. | ambulance marker |
| 2 | 0:05–0:09 | P3, paused 4,091 | At this junction it asks for a green light. | ambulance's road (blue) |
| 3 | 0:09–0:14 | P3, paused | Without any change, its light turns green only 3.6 seconds after it arrives. | ambulance's road, "wait: 3.6 s" |
| 4 | 0:14–0:18 | P3, paused | Changing the light would make the cross road wait. | cross road (orange) |
| 5 | 0:18–0:24 | P3, paused | AmbuGo changes a light only if it saves the ambulance at least 5 seconds. | verdict "3.6 s saved → no change" |
| 6 | 0:24–0:28 | P2, paused 4,091 | What if the light had been changed anyway? | junction ring |
| 7 | 0:28–0:39 | P2, 4,091 → 4,170 (≈7×) | Recorded run of a controller that always changes the light. | cross road (orange) |
| 8 | 0:39–0:43 | P2, paused 4,170 | The cross road waited 64 seconds longer for green. | cross road, "+64 s wait" |
| 9 | 0:43–0:47 | P2, paused | The south road got its green 22 seconds sooner. | south road (green), "22 s sooner" |
| 10 | 0:47–0:51 | P2, paused | The ambulance got through just 11 seconds sooner. | ambulance's road, "11 s sooner" |
| 11 | 0:51–0:57 | P2, paused | In total, 296 more vehicle-seconds of waiting to save 11 seconds. | verdict "Not worth it → no change" |
| 12 | 0:57–1:02 | P3, 4,120 → 4,132 | At the next junction, the ambulance would wait 43 seconds for green. | ambulance's road, "wait: 43 s" |
| 13 | 1:02–1:08 | P3, paused 4,132 | A change would make the other three roads wait: about 3,300 vehicle-seconds in total. | the three other roads (orange) |
| 14 | 1:08–1:14 | P3, paused | That is within AmbuGo's limit of 10,000, and it saves the ambulance 43 seconds. | verdict "43 s saved → change the light" |
| 15 | 1:14–1:22 | P3, 4,132 → 4,146 (≈1.75×) | The change is made safely: minimum green, then amber, then all-red. | junction ring (watch the stop lines) |
| 16 | 1:22–1:29 | P3, 4,146 → 4,180 (≈5×) | The ambulance gets green 86 seconds sooner than without the change. | ambulance marker |
| 17 | 1:29–1:35 | P3, 4,180 → 4,205 | The east and south roads wait a little longer; no queue comes close to filling its road. | east and south roads (orange) |
| 18 | 1:35–1:40 | blurred | AmbuGo changes a light only when it clearly helps the ambulance and the cost to others stays small. | none |
| 19 | 1:40–1:46 | blurred | Across 1,728 simulations: trips 44.6% faster than with no priority, with less extra congestion than changing every light. | none |

**Runs.**

* P3 is AmbuGo (JSEL), archived.
* P2 is the always-priority controller (JFAST), archived. It is identical to P3 up to 4,091 s.

**Sources of the numbers.**

* The "86 seconds sooner" figure compares P3 with the one-decision replay `P3_withholdJ21`.
* Every number is checked by `verify.py` (192 checks, 0 failed).

## Voiceover (about 210 words, measured pace; one sentence per beat)

1. An ambulance is on its way to hospital.
2. At this junction, it asks for a green light.
3. Without any change, its light would turn green only about four seconds after it arrives.
4. Changing the light would make the cross road wait.
5. AmbuGo only changes a light if it saves the ambulance at least five seconds. Here it would not, so the light stays as it is.
6. What if it had been changed anyway?
7. This is a recorded run of a controller that always changes the light.
8. The cross road waited a whole minute longer for its green.
9. The south road did get its green a little sooner.
10. And the ambulance got through just eleven seconds sooner.
11. Nearly three hundred extra vehicle-seconds of waiting, to save eleven seconds. Not worth it.
12. At the next junction, the ambulance would wait forty-three seconds.
13. A change here would make the other roads wait, about three thousand three hundred vehicle-seconds in total.
14. That is well within AmbuGo's limit, and it saves the ambulance forty-three seconds. So the light changes.
15. It changes safely: the current green finishes, then amber, then all-red.
16. The ambulance gets its green eighty-six seconds sooner.
17. The other roads wait a little longer, but no queue gets close to filling its road.
18. So AmbuGo changes a light only when it clearly helps the ambulance, and the cost to everyone else stays small.
19. Across all the simulations, trips were forty-five percent faster than with no priority, with less extra congestion than changing every light.
