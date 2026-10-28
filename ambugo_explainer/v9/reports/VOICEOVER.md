# Voiceover script

Written for a human narrator at a measured pace, about 2.3 words per second. Each block is timed to its scene.
It can be read slower; the pauses inside the scenes leave room. Numbers are spoken as rounded on screen.

**A · 0:00–0:11. The problem**
This is a synthetic test district: sixteen signal-controlled junctions and fifty-nine one-way roads, with cars, motorcycles, autorickshaws, buses and goods vehicles. Every vehicle you see is a recorded microsimulation trajectory. An ambulance has just been dispatched from the base on the left. It needs to reach the hospital on the right.

**B · 0:11–0:25. Route and action together**
At dispatch, AmbuGo forecasts the trip for four candidate routes. It does this twice for each: once with normal signals, and once with a priority request at each junction. Route C is the slowest under normal signals, but the fastest with a request. So the route and the signal action have to be chosen together. With the tolerance set to zero, only the fastest pair is eligible, and the burden on other traffic is used only to break ties.

**C · 0:25–0:32. On the way**
Each time the ambulance enters a new road, AmbuGo checks the route again and tests the request at the next junction. At J10 the request is admitted, and the light turns green for the ambulance.

**D · 0:32–0:37. Re-check**
Newer counts now make route A about thirteen seconds faster. That is more than the ten-second hysteresis, so the plan switches back to A.

**E · 0:37–0:46. A withheld request**
At J11 the ambulance is about twenty-seven seconds from the stop line. The normal cycle would reach its approach only three or four seconds after it arrives. A priority pulse would interrupt the other approaches for less than five seconds of gain, so AmbuGo withholds it. In the archived JSEL runs, every refusal was of this kind; none was refused on the burden budget.

**F · 0:46–0:59. The normal cycle**
So J11 keeps its normal cycle. The pedestrian interval finishes, north gets green, then amber, all-red, and then the ambulance's approach. These rules hold at every junction, and they hold for the ambulance too. There is a minimum green, amber and all-red, a full pedestrian interval, and a guard that serves an approach that has waited a long time.

**G · 0:59–1:04. J21**
At J21 the ambulance asks again. East has just started its green, and south and west are waiting.

**H · 1:04–1:16. Scenario one: priority granted**
This is what AmbuGo actually did. East keeps its minimum eight seconds of green, then amber, then all-red. The ambulance's approach turns green at 4,142 seconds. East and south now wait longer, and their queues grow.

**I · 1:16–1:35. How the cost was forecast**
Back to the moment of decision. AmbuGo cannot see the true queues. It estimates them from link counts that arrive late and with noise. For each other approach, it multiplies the vehicles already waiting by the green time they would lose: about forty-four seconds here. Then it adds the vehicles that would arrive meanwhile, weighted more heavily on approaches that are already close to capacity. The total is about 3,300 vehicle-seconds, well within the budget of 10,000. Every gate passes, so the request is admitted.

**J · 1:35–1:52. Granted versus withheld**
Here is the same moment and the same traffic, side by side. On the left is the decision AmbuGo made. On the right is a replay in which only that one request is withheld. With priority, the ambulance gets green at 4,142 seconds. Without it, at 4,228. The cost moves from the ambulance's road to the east and south approaches. No queue comes near filling its road.

**K · 1:52–2:13. The wider network and escalation**
The change spreads beyond J21. Platoons leave at different times, so roads beyond the next junction carry more or fewer vehicles than they otherwise would. AmbuGo also tracks the ambulance's progress against the free-flow time of the roads it has actually driven. At 4,256 seconds the shortfall passes twenty seconds. For the rest of this trip it switches to the stronger priority mode, though the legal signal stages still apply. The ambulance reaches the hospital 252.5 seconds after dispatch.

**L · 2:13–2:25. Results**
In this episode, with identical demand, the trip took 252.5 seconds. Without signal priority it took 572. Across the paper's 1,728 runs, selective control was about eleven seconds slower than aggressive control, and it spread less extra occupancy over the network. This is a synthetic, uncalibrated network: a reproducible method, not a field result.
