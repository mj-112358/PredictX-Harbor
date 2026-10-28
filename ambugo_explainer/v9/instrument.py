#!/usr/bin/env python3
"""Log-only instrumented replay of the frozen AmbuGo v9 SUMO study.

The frozen source (work/v9/sumo_study.py, SHA-256 471a2291...) is imported, never edited.
Logging is added by swapping module attributes at run time:
  * Signal     -> LoggedSignal      (records mode/stage after every update)
  * Controller -> LoggedController  (records candidate tables, gate inputs, leases)
  * traci      -> TraciProxy        (records vehicle positions after every physical step)
All hooks only read simulator state, so decisions are unchanged; this is checked by comparing
the instrumented run's scalar results with the archived record (see `verify_against_archive`).

Optional single-decision counterfactual (`--branch`):
  withhold:J:EDGE:T  from integer tick T, the request for approach EDGE at junction J is dropped
                     (and not re-issued on that link), i.e. the junction keeps its normal cycle
  grant:J:EDGE:T     from tick T, a request that the policy refused is forwarded anyway with the
                     selected strong action (30 s look-ahead, 60 s hold)
Everything before tick T is the archived trajectory (deterministic replay), so both branches start
from an identical state with identical exogenous demand and sensor draws.

Usage (from the replay working copy's work/v9 directory):
  python instrument.py --seed 40 --load 0.95 --missions 1 --policy P3 --out OUT [--branch ...]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import sumo_study as S  # noqa: E402  (frozen source, unmodified)

REAL = S.traci
MAIN = dict(step=.5, warmup=1800, horizon=7200, mission_end=5400, save_trace=True)
JSEL = dict(harm_budget=10000, epsilon=0, action_mode='strong', progress_guard_s=20)


class Log:
    def __init__(self):
        self.window = (0, 1e9)
        self.frames = []          # (t, ids, x, y, angle, speed, lane, type)
        self.signals = []         # (t, j, mode, stage_edge, elapsed, request_edge)
        self.plans = []           # one entry per plan() call, with every candidate
        self.gates = []           # one entry per request-gate pulse() call
        self.requests = []        # per tick: junction -> request dict, leases
        self.branch = None
        self.branch_events = []


LOG = Log()


class TraciProxy:
    """Forwards everything to libsumo; after each physical step records vehicle states."""

    def __getattr__(self, k):
        return getattr(REAL, k)

    def simulationStep(self, *a):
        REAL.simulationStep(*a)
        t = REAL.simulation.getTime()
        if not (LOG.window[0] <= t <= LOG.window[1]):
            return
        V = REAL.vehicle
        ids = V.getIDList()
        rec = [(i, *V.getPosition(i), V.getAngle(i), V.getSpeed(i), V.getLaneID(i), V.getTypeID(i)) for i in ids]
        LOG.frames.append((t, rec))


class LoggedSignal(S.Signal):
    def update(self, t, request, observed, base='fixed'):
        super().update(t, request, observed, base)
        if LOG.window[0] - 300 <= t <= LOG.window[1]:
            LOG.signals.append((t, self.j, self.mode, self.arms[self.stage] if self.mode in ('green', 'amber') else None,
                                self.elapsed, request['edge'] if request else None, request['ambulance'] if request else None))


class LoggedController(S.Controller):
    _in_plan = False
    _tick = None

    # ---- candidate tables (read-only recomputation with the frozen estimate())
    def plan(self, aid, e, remaining, t, policy=None):
        policy = policy or self.policy
        pp = list(self.candidates(e))
        current = list(S.traci.vehicle.getRoute(aid)); ix = S.traci.vehicle.getRouteIndex(aid); cur = current[max(ix, 0):]
        if cur and cur[0] == e and cur not in pp:
            pp = pp[:3] + [cur]
        p3actions = [(0., 0.), (30., 60.)] if self.c['action_mode'] == 'strong' else [(0., 0.), (10., 20.), (20., 40.), (30., 60.)]
        actions = [(0., 0.)] if policy == 'P0' else ([(self.c['trigger'], self.c['hold'])] if policy in ('P1', 'P2') else p3actions)
        self._in_plan = True
        try:
            table = []
            for path in pp:
                for a in actions:
                    T, A, n = self.estimate(path, a, remaining, policy=policy)
                    table.append(dict(route=list(path), action=list(a), eta=T, burden=A, admitted_junctions=n,
                                      incumbent=(list(path) == cur)))
            before = list(cur)
            super().plan(aid, e, remaining, t, policy=policy)
        finally:
            self._in_plan = False
        after = list(S.traci.vehicle.getRoute(aid))[S.traci.vehicle.getRouteIndex(aid):]
        LOG.plans.append(dict(t=t, ambulance=aid, edge=e, mode=policy, remaining_m=remaining, candidates=table,
                              chosen_route=after, chosen_action=list(self.amaction[aid]), previous_route=before,
                              changed=after != before, epsilon=self.c['epsilon']))

    # ---- request-gate inputs (pulse calls made outside plan())
    def pulse(self, e, out, a, arrival, policy=None):
        res = super().pulse(e, out, a, arrival, policy)
        if not self._in_plan and self._tick is not None:
            policy = policy or self.policy
            ee = self.m['edges'][e]; j = ee['to']
            if j in self.signals:
                si = self.signals[j]
                qclear = self.q(e) / max(ee['s'] / self.m['meanpcu'], .1)
                lost = min(a[1], max(5., a[0] + qclear))
                arms = []
                for other in si.arms:
                    if other == e:
                        continue
                    lam = self.design['expected_veh_per_s'].get(other, 0); X = self.design['plans'][j]['X'].get(other, 0)
                    arms.append(dict(edge=other, q_hat=self.q(other), n_obs=self.obs.get(other, 0), lam=lam, X=X,
                                     term_queue=self.q(other) * lost, term_arrival=lam * lost * lost / (2 * max(1 - X, .05))))
                switch = si.switch_delay(e)
                LOG.gates.append(dict(t=self._tick, edge=e, junction=j, out=out, action=list(a), policy=policy,
                                      arrival=arrival, q_hat_own=self.q(e), d_hat=qclear, switch_delay=switch,
                                      nominal_wait=si.nominal_wait(e, arrival),
                                      feasible_lhs=max(0, a[0] + qclear - switch) + qclear + 2.,
                                      headroom_obs=None if out is None else self.obs.get(out, 0),
                                      headroom_cap=None if out is None else self.c['headroom'] * self.m['edges'][out]['K'],
                                      D=lost, arms=arms, burden=sum(x['term_queue'] + x['term_arrival'] for x in arms),
                                      wait_after=res[0], burden_returned=res[1], admitted=res[2], benefit=res[3],
                                      signal_mode=si.mode, signal_edge=si.edge, signal_elapsed=si.elapsed))
        return res

    def requests_at(self, t, active):
        self._tick = t
        req = super().requests_at(t, active)
        b = LOG.branch
        if b is not None and t >= b['t']:
            if b['kind'] == 'withhold' and b['j'] in req and req[b['j']]['edge'] == b['edge']:
                dropped = req.pop(b['j'])
                LOG.branch_events.append(dict(t=t, action='withheld', request=dropped))
            if b['kind'] == 'grant' and b['j'] not in req:
                for a in active:
                    aid = a['id']
                    if S.traci.vehicle.getRoadID(aid) == b['edge'] and not b.get('done'):
                        if t - b['t'] < b.get('hold', 60):
                            req[b['j']] = dict(edge=b['edge'], hold=60., ambulance=aid, dispatch=a['depart'],
                                               predicted_benefit=None, predicted_burden=None, forced=True)
                            LOG.branch_events.append(dict(t=t, action='forced_grant', ambulance=aid))
        if active:
            prog = {}
            for a in active:
                aid = a['id']; road = S.traci.vehicle.getRoadID(aid)
                st = self.progress.get(aid)
                if st is not None and road in self.m['edges'] and st['last_regular'] == road:
                    pos = S.traci.vehicle.getLanePosition(aid)
                    pff = st['completed_freeflow'] + max(0., pos) / self.m['edges'][road]['v']
                    prog[aid] = dict(edge=road, pos=pos, completed_ff=st['completed_freeflow'], P=pff,
                                     D=max(0., t - a['depart'] - pff))     # Eq. 21, as in progress_mode()
            LOG.requests.append(dict(t=t, requests={j: {k: v for k, v in r.items()} for j, r in req.items()},
                                     leases={f'{k[0]}|{k[1]}': dict(v) for k, v in self.leases.items()},
                                     fallback=sorted(self.fallback_ids), progress=prog))
        self._tick = None
        return req


def run(seed, load, missions, policy, out, branch=None, window=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    S.traci = TraciProxy(); S.Signal = LoggedSignal; S.Controller = LoggedController
    kw = dict(MAIN)
    if policy == 'P3':
        kw.update(JSEL)
    sc = S.make_scenario(seed, load, missions, warmup=kw['warmup'], horizon=kw['horizon'], mission_end=kw['mission_end'])
    d0 = min(a['depart'] for a in sc['ambulances'])
    LOG.window = window or (d0 - 120, d0 + 1200)
    LOG.branch = branch
    t0 = time.time()
    r = S.run_case(seed, load, missions, policy, output=str(out / 'runcase'), **kw)
    S.traci = REAL
    meta = {k: v for k, v in r.items() if k not in ('history', 'partition', 'whole_episode_population')}
    meta['instrumentation'] = dict(window=LOG.window, branch=branch, wall_s=time.time() - t0, frames=len(LOG.frames))
    (out / 'result.json').write_text(json.dumps(meta, indent=1, default=float))
    # trajectories as ragged arrays
    types, lanes = {}, {}
    T, off, vid, X, Y, A, V, L, C = [], [0], [], [], [], [], [], [], []
    vnames = {}
    for t, rec in LOG.frames:
        T.append(t)
        for i, x, y, ang, sp, lane, ty in rec:
            vid.append(vnames.setdefault(i, len(vnames))); X.append(x); Y.append(y); A.append(ang); V.append(sp)
            L.append(lanes.setdefault(lane, len(lanes))); C.append(types.setdefault(ty, len(types)))
        off.append(len(vid))
    np.savez_compressed(out / 'trajectories.npz', t=np.array(T), offsets=np.array(off), vehicle=np.array(vid, np.int32),
                        x=np.array(X, np.float32), y=np.array(Y, np.float32), angle=np.array(A, np.float32),
                        speed=np.array(V, np.float32), lane=np.array(L, np.int32), vtype=np.array(C, np.int16))
    (out / 'trajectory_index.json').write_text(json.dumps(dict(vehicles=list(vnames), lanes=list(lanes), types=list(types))))
    (out / 'signals.json').write_text(json.dumps(LOG.signals))
    (out / 'plans.json').write_text(json.dumps(LOG.plans, default=float))
    (out / 'gates.json').write_text(json.dumps(LOG.gates, default=float))
    (out / 'requests.json').write_text(json.dumps(LOG.requests, default=float))
    (out / 'branch_events.json').write_text(json.dumps(LOG.branch_events, default=float))
    (out / 'trace.npz').write_bytes(Path(r['trace_file']).read_bytes())
    return r


KEYS = ['schedule_sha256', 'n_ordinary', 'ordinary_J_vehicle_s', 'ordinary_completed', 'mission_rows', 'route_changes',
        'progress_guard_fallbacks', 'priority_request_ticks', 'rejected_harm_ticks', 'rejected_headroom_ticks',
        'rejected_benefit_ticks', 'priority_phase_admissions', 'guard_overrides', 'signal_timing_violations',
        'expired_priority_leases', 'selected_action_counts']


def verify_against_archive(r, records, seed, load, missions, policy):
    arch = [x for x in json.load(open(records)) if (x['seed'], x['load'], x['missions'], x['policy']) == (seed, load, missions, policy)][0]
    return {k: (r[k] == arch[k]) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--load', type=float, required=True)
    ap.add_argument('--missions', type=int, default=1); ap.add_argument('--policy', default='P3')
    ap.add_argument('--out', required=True); ap.add_argument('--records')
    ap.add_argument('--branch'); ap.add_argument('--window', type=float, nargs=2)
    a = ap.parse_args()
    br = None
    if a.branch:
        kind, j, edge, t = a.branch.split(':')
        br = dict(kind=kind, j=j, edge=edge, t=int(t))
    r = run(a.seed, a.load, a.missions, a.policy, a.out, br, tuple(a.window) if a.window else None)
    if a.records and not br:
        chk = verify_against_archive(r, a.records, a.seed, a.load, a.missions, a.policy)
        Path(a.out, 'archive_check.json').write_text(json.dumps(chk, indent=1))
        print('archive match:', all(chk.values()), [k for k, v in chk.items() if not v])
    print('missions', r['mission_rows'])


if __name__ == '__main__':
    main()
