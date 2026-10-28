"""Load the frozen AmbuGo B1000 episode and precompute every series the explainer shows.

Nothing here simulates traffic. All values are read from the recorded SUMO replays
(P3 = AmbuGo selective controller, budget B = 1000 veh*s; P0 = matched routing-only,
no-priority run with identical network, demand seed and missions) and from the
controller's own decision log. Derived quantities (progress deficit, cumulative
stopped time, occupancy differences) are recomputed from those arrays and checked
against the logged values in verify.py.
"""
from pathlib import Path
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "inputs"
DATA = INP / "data"
R_EARTH = 6378137.0
DT = 0.5  # recorded sample interval, seconds
DISPATCH = 760.0  # a002 desired dispatch, simulation seconds


def llxy(lon, lat):
    """Web Mercator (EPSG:3857) metres, the projection shared with the map captures."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    return np.stack((R_EARTH * np.radians(lon),
                     R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))), axis=-1)


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


class Replay:
    """Ragged per-sample vehicle records with display interpolation.

    Positions and headings are interpolated between 0.5 s samples for smooth slow
    playback; speed, edge membership and stopped state use the previous sample
    (zero-order hold), so no stop or movement is invented.
    """

    def __init__(self, policy):
        self.policy = policy
        self.meta = json.loads((DATA / "replays" / f"{policy}.json").read_text())
        with np.load(DATA / "replays" / f"{policy}.npz") as z:
            self.a = {k: z[k] for k in z.files}
        a = self.a
        self.t = a["t"]
        self.off = a["offsets"]
        self.xy = llxy(a["lon"], a["lat"])
        self.amxy = llxy(a["ambulance_lon"], a["ambulance_lat"])
        self.types = [v["type"] for v in self.meta["vehicles"]]
        self.amb_idx = [i for i, v in enumerate(self.types) if v == "ambulance"]
        dims = {"car": (4.5, 1.8), "motorcycle": (2.1, 0.8), "bus": (12, 2.5),
                "goods": (8, 2.3), "ambulance": (5, 2)}
        self.dims = np.array([dims[v] for v in self.types])
        self.frame_of = np.repeat(np.arange(len(self.t)), np.diff(self.off))

    def k(self, t):
        return int(np.clip(np.searchsorted(self.t, t + 1e-9, side="right") - 1, 0, len(self.t) - 1))

    def sample(self, k):
        lo, hi = self.off[k:k + 2]
        a = self.a
        return (a["vehicle"][lo:hi], self.xy[lo:hi], a["angle"][lo:hi], a["speed"][lo:hi], a["edge"][lo:hi])

    def positions(self, t):
        k = self.k(t)
        vid, xy, ang, spd, edge = self.sample(k)
        xy = xy.copy()
        ang = ang.astype(float).copy()
        if k + 1 < len(self.t):
            vid2, xy2, ang2, _, _ = self.sample(k + 1)
            _, i, j = np.intersect1d(vid, vid2, return_indices=True)
            u = float(np.clip((t - self.t[k]) / (self.t[k + 1] - self.t[k]), 0, 1))
            xy[i] = xy[i] * (1 - u) + xy2[j] * u
            ang[i] = ang[i] + _wrap(ang2[j] - ang[i]) * u
        return vid, xy, ang, spd, edge


class Episode:
    def __init__(self):
        self.P3 = Replay("P3")
        self.P0 = Replay("P0")
        assert np.array_equal(self.P3.t, self.P0.t)
        assert self.P3.meta["vehicles"] == self.P0.meta["vehicles"]
        self.t = self.P3.t
        m = self.P3.meta
        self.edge_ids = m["edge_ids"]
        self.ei = {e: i for i, e in enumerate(self.edge_ids)}
        self.junction_ids = m["junction_ids"]
        self.vehicles = m["vehicles"]
        self.focus = next(i for i, v in enumerate(self.vehicles) if v["id"] == "a002")
        self.focus_col = 2  # column of a002 in the ambulance_* arrays

        net = json.loads((DATA / "network_geo.json").read_text())
        self.edges = {e["id"]: e for e in net["edges"]}
        self.exy = {e["id"]: llxy(*np.array(e["lonlat"]).T) for e in net["edges"]}
        self.signals = {s["id"]: s for s in net["signals"]}
        self.connectors = json.loads((DATA / "connectors.json").read_text())
        self.decisions = json.loads((DATA / "decisions.json").read_text())
        self.control = json.loads((DATA / "control_explanations.json").read_text())
        self.story = json.loads((DATA / "story.json").read_text())

        self.ev = {e["t"]: e for e in self.control["events"] if e["ambulance"] == "a002"}
        self.all_events = self.control["events"]

        # Junctions of interest.
        self.J1 = next(j for j in self.junction_ids if "1424395831" in j)   # PCL Chowk
        self.J2 = next(j for j in self.junction_ids if j.startswith("joinedS_1424619181"))
        self.J3 = "3867217322"
        self.j_index = {j: i for i, j in enumerate(self.junction_ids)}
        self.J1xy = llxy(self.signals[self.J1]["lon"], self.signals[self.J1]["lat"])
        self.J2xy = llxy(self.signals[self.J2]["lon"], self.signals[self.J2]["lat"])
        self.J3xy = llxy(self.signals[self.J3]["lon"], self.signals[self.J3]["lat"])
        self.hosp = llxy(76.73095468333334, 30.69473045)

        # PCL Chowk approaches, named by the compass side the traffic comes from.
        self.amb_app = "335061218"
        self.conf = ["172380615#1", "172382686#4", "172660580#0"]
        self.app_name = {"335061218": "SE", "172380615#1": "NW", "172382686#4": "SW", "172660580#0": "NE"}

        self._series()
        self._signal_timelines()
        self._progress()
        self._stopped_time()

    # ------------------------------------------------------------------ series
    def _series(self):
        a3, a0 = self.P3.a, self.P0.a
        self.h3 = a3["edge_halts"].astype(np.int32)
        self.h0 = a0["edge_halts"].astype(np.int32)
        self.c3 = a3["ordinary_edge_counts"].astype(np.int32)
        self.c0 = a0["ordinary_edge_counts"].astype(np.int32)

        def stopped(rep):
            ordinary = ~np.isin(rep.a["vehicle"], rep.amb_idx)
            mask = ordinary & (rep.a["speed"] < 0.1)
            return np.bincount(rep.frame_of[mask], minlength=len(rep.t))

        self.net_stop3 = stopped(self.P3)
        self.net_stop0 = stopped(self.P0)
        start = self.k(DISPATCH)
        d = self.c3 - self.c0
        pos = np.clip(d, 0, None).sum(1) * DT
        self.A_cum = np.zeros(len(self.t))
        self.A_cum[start:] = np.cumsum(pos[start:])
        self.sJ_cum = np.zeros(len(self.t))
        self.sJ_cum[start:] = np.cumsum(d.sum(1)[start:] * DT)
        self.stop_cum3 = np.zeros(len(self.t))
        self.stop_cum0 = np.zeros(len(self.t))
        self.stop_cum3[start:] = np.cumsum(self.net_stop3[start:] * DT)
        self.stop_cum0[start:] = np.cumsum(self.net_stop0[start:] * DT)

        # Ambulance speed (km/h) for a002.
        r = self.P3
        obs = np.flatnonzero(r.a["vehicle"] == self.focus)
        self.amb_speed = np.full(len(self.t), np.nan)
        self.amb_speed[r.frame_of[obs]] = r.a["speed"][obs] * 3.6
        self.amb_edge = np.full(len(self.t), -1)
        self.amb_edge[r.frame_of[obs]] = r.a["edge"][obs]
        self.amb_x = np.full(len(self.t), np.nan)
        self.amb_y = np.full(len(self.t), np.nan)
        self.amb_x[r.frame_of[obs]] = r.a["x"][obs]
        self.amb_y[r.frame_of[obs]] = r.a["y"][obs]

        # Realised extra stopped vehicle-seconds on PCL Chowk approaches vs no priority.
        s = self.k(785)
        self.app_extra = {}
        for e in [self.amb_app] + self.conf:
            i = self.ei[e]
            dd = (self.h3[:, i] - self.h0[:, i]).astype(float) * DT
            c = np.zeros(len(self.t))
            c[s:] = np.cumsum(dd[s:])
            self.app_extra[e] = c

    def k(self, t):
        return self.P3.k(t)

    # --------------------------------------------------------------- signals
    def _signal_timelines(self):
        self.timeline = {}
        for pol, rep in (("P3", self.P3), ("P0", self.P0)):
            for j in (self.J1, self.J2):
                ji = self.j_index[j]
                mode = rep.a["signal_mode"][:, ji]
                edge = rep.a["signal_edge"][:, ji]
                segs = []
                start = 0
                for k in range(1, len(self.t) + 1):
                    if k == len(self.t) or mode[k] != mode[start] or edge[k] != edge[start]:
                        e = self.edge_ids[edge[start]] if edge[start] >= 0 else None
                        segs.append((float(self.t[start]), float(self.t[min(k, len(self.t) - 1)]), int(mode[start]), e))
                        start = k
                self.timeline[pol, j] = segs

    def signal_state(self, pol, j, t):
        rep = self.P3 if pol == "P3" else self.P0
        k = rep.k(t)
        ji = self.j_index[j]
        mode = int(rep.a["signal_mode"][k, ji])
        e = int(rep.a["signal_edge"][k, ji])
        req = int(rep.a["signal_request"][k, ji])
        return mode, (self.edge_ids[e] if e >= 0 else None), (self.edge_ids[req] if req >= 0 else None)

    # -------------------------------------------------------------- progress
    def _edge_pos(self, eid, x, y):
        p = np.array(self.edges[eid]["xy"])
        best, acc, along = 1e18, 0.0, 0.0
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            d = b - a
            L = float(np.hypot(*d))
            u = float(np.clip(np.dot(np.array([x, y]) - a, d) / max(L * L, 1e-9), 0, 1))
            q = a + u * d
            dd = float(np.hypot(*(q - np.array([x, y]))))
            if dd < best:
                best, along = dd, acc + u * L
            acc += L
        return along * self.edges[eid]["length_m"] / max(acc, 1e-9)

    def _progress(self):
        """Paper Eq. 21: P_a(t) executed free-flow progress, D_a(t)=[t-d_a-P_a]_+."""
        done, cur = [], None
        self.P_a = np.full(len(self.t), np.nan)
        self.D_a = np.full(len(self.t), np.nan)
        self.P_done = np.full(len(self.t), np.nan)
        lastP = 0.0
        for k in range(len(self.t)):
            tk = self.t[k]
            if tk < 760.5 or tk > 1083.5:
                continue
            ei = self.amb_edge[k]
            if ei < 0 or not np.isfinite(self.amb_x[k]):
                # internal connector: no free-flow credit while elapsed time grows
                self.P_a[k] = lastP
                self.D_a[k] = max(0.0, tk - DISPATCH - lastP)
                continue
            e = self.edge_ids[ei]
            if e != cur:
                if cur is not None:
                    done.append(cur)
                cur = e
            x = self._edge_pos(e, self.amb_x[k], self.amb_y[k])
            base = sum(self.edges[f]["length_m"] / self.edges[f]["speed_mps"] for f in done)
            P = base + max(0.0, x) / self.edges[e]["speed_mps"]
            self.P_done[k] = base
            self.P_a[k] = P
            self.D_a[k] = max(0.0, tk - DISPATCH - P)
            lastP = P

    # ------------------------------------------------- per-vehicle waiting
    def _stopped_time(self):
        """Cumulative stopped seconds since dispatch for every vehicle in both runs."""
        nv = len(self.vehicles)
        start = self.k(DISPATCH)
        self.W = {}
        for pol, rep in (("P3", self.P3), ("P0", self.P0)):
            M = np.zeros((len(self.t), nv), dtype=np.float32)
            st = rep.a["speed"] < 0.1
            fr = rep.frame_of
            inc = np.zeros((len(self.t), nv), dtype=np.float32)
            sel = st & (fr >= start)
            np.add.at(inc, (fr[sel], rep.a["vehicle"][sel]), DT)
            M = np.cumsum(inc, axis=0)
            self.W[pol] = M
        self.dW = self.W["P3"] - self.W["P0"]

    def extra_wait(self, t):
        k = self.k(t)
        return self.dW[k]

    # ----------------------------------------------------------- utilities
    def route_points(self, ids):
        pts = []
        for j, e in enumerate(ids):
            pts.extend(self.exy[e])
            if j + 1 < len(ids):
                opts = self.connectors.get(e + "|" + ids[j + 1], [])
                if opts:
                    pts.extend(llxy(*np.array(opts[0]["shape_lonlat"]).T))
        return np.array(pts)

    def amb_pos(self, t, col=None):
        col = self.focus_col if col is None else col
        rep = self.P3
        k = rep.k(t)
        p = rep.amxy[k, col].copy()
        if k + 1 < len(self.t):
            q = rep.amxy[k + 1, col]
            if np.isfinite(p).all() and np.isfinite(q).all():
                u = float(np.clip((t - self.t[k]) / DT, 0, 1))
                p = p * (1 - u) + q * u
        return p

    def amb_route(self, t):
        rid = int(self.P3.a["ambulance_route"][self.k(t), self.focus_col])
        return self.P3.meta["routes"][rid] if rid >= 0 else []

    def event_at(self, t):
        """Controller record for a002 at the last integer decision second <= t."""
        s = int(math.floor(t + 1e-9))
        return self.ev.get(s)

    def interp(self, arr, t):
        return float(np.interp(t, self.t, arr))
