"""What time scale does collaboration live on?

1. Event study around requests. For every request (asker A, askee B, time tau)
   in the first slice, count A-B *handover events* (one agent opens a trace
   session on a trace the other had a session on within the previous two hours)
   at offsets from tau, in 15-minute bins over +-8 hours. The same count for
   every other pair active that day is the baseline. The excess profile shows
   how far before and after a request the shared work happens.

2. Window length. Pair detection is rebuilt with windows of 1, 2, 4 and 8 hours
   (Pacific office hours start at 9:00) instead of days: same features, a window
   counts as labelled when the pair has a request inside it, train on 6-31 July,
   validate on 3-14 August.

Run: python -I -m swarmcollab.timescale   (first slice only)
"""
import gzip
import json
import time

import numpy as np
import pandas as pd

from . import config
from . import labels as L
from . import metrics as M
from . import pairs as P
from . import spread as S

OUT = config.OUT / 'timescale'
BIN = 900.0
SPAN = 8 * 3600.0
HANDOVER_GAP = 2 * 3600.0
WAITING = {'pause', 'wait', 'empty', 'restart'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def handover_events(e):
    """(src, dst, t): dst opens a session on a trace src had a session on within HANDOVER_GAP before."""
    e = e.sort_values(['tid', 't']).reset_index(drop=True)
    out = []
    tid, ag, t = e.tid.values, e.agent.values, e.t.values
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(e)]
    for s0, e0 in zip(starts, ends):
        if e0 - s0 < 2:
            continue
        for k in range(s0 + 1, e0):
            j = k - 1
            seen = set()
            while j >= s0 and t[k] - t[j] <= HANDOVER_GAP and k - j <= 60:
                if ag[j] != ag[k] and ag[j] not in seen:
                    seen.add(ag[j])
                    out.append((ag[j], ag[k], t[k]))
                j -= 1
    return pd.DataFrame(out, columns=['src', 'dst', 't'])


def event_study(h, requests, active_by_day):
    """Excess handover rate of the requesting pair around each request, against other active pairs."""
    h = h.assign(a=np.where(h.src < h.dst, h.src, h.dst), b=np.where(h.src < h.dst, h.dst, h.src))
    by_pair = {k: np.sort(g.t.values) for k, g in h.groupby(['a', 'b'])}
    all_t = np.sort(h.t.values)
    edges = np.arange(-SPAN, SPAN + BIN, BIN)
    own = np.zeros(len(edges) - 1)
    base = np.zeros(len(edges) - 1)
    n = 0
    for r in requests:
        a, b = sorted((r['from'], r['to']))
        act = active_by_day.get(r['day'].isoformat(), set())
        if a not in act or b not in act:
            continue
        tau = r['utc'].timestamp()
        ts = by_pair.get((a, b), np.array([]))
        own += np.histogram(ts - tau, bins=edges)[0]
        n_pairs = len(act) * (len(act) - 1) / 2
        sel = all_t[(all_t >= tau - SPAN) & (all_t < tau + SPAN)]
        base += np.histogram(sel - tau, bins=edges)[0] / max(n_pairs, 1)
        n += 1
    centers = (edges[:-1] + edges[1:]) / 2 / 3600.0
    prof = pd.DataFrame({'offset_hours': centers, 'requesting_pair': own / max(n, 1), 'average_pair': base / max(n, 1)})
    prof['excess'] = prof.requesting_pair - prof.average_pair
    return prof, n


def working_hours(days):
    """Per agent: Pacific (day, hour) -> working steps, read straight from the step files."""
    with open(config.STEPS / 'summary.json', encoding='utf-8') as f:
        summary = json.load(f)
    lo = pd.Timestamp(min(days), tz='America/Los_Angeles').tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
    hi = (pd.Timestamp(max(days), tz='America/Los_Angeles') + pd.Timedelta(days=1)).tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S')
    rows = []
    for agent, info in summary['agents'].items():
        ts = []
        with gzip.open(config.STEPS / info['file'], 'rt', encoding='utf-8') as f:
            for line in f:
                k = line.find('"t": "')
                stamp = line[k + 6:k + 32]
                if not (lo <= stamp < hi):
                    continue
                kk = line.find('"k": "')
                kind = line[kk + 6:line.find('"', kk + 6)]
                if kind in WAITING:
                    continue
                ts.append(stamp.split('"')[0])
        if ts:
            t = pd.to_datetime(pd.Series(ts), utc=True, format='ISO8601').dt.tz_convert('America/Los_Angeles')
            rows.append(pd.DataFrame({'agent': agent, 'day': t.dt.strftime('%Y-%m-%d'), 'hour': t.dt.hour}))
    w = pd.concat(rows, ignore_index=True)
    w = w[w.day.isin(days)]
    return w.groupby(['agent', 'day', 'hour']).size().rename('working').reset_index()


def window_key(day, hour, width):
    """Office-hour windows; out-of-hours observations have no window.

    In particular 08:59 and 17:00 are not silently folded into edge windows.
    """
    if width not in (1, 2, 4, 8):
        raise ValueError('width must partition the eight-hour 09:00–17:00 interval')
    hours = np.asarray(hour)
    slot = (hours - 9) // width
    keys = pd.Series(day).astype(str).values + '#' + slot.astype(int).astype(str)
    return np.where((hours >= 9) & (hours < 17), keys, None)


def window_eval(m, wh, requests, width, days):
    """Pair-window features and validation AUCs for windows of `width` hours."""
    pt = pd.to_datetime(m.t, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    mw = m.copy()
    mw['day'] = pd.Categorical(window_key(mw.day.astype(str).values, pt.dt.hour.values, width))
    mw = mw[mw.day.notna()].copy()
    act = wh.assign(win=window_key(wh.day.values, wh.hour.values, width)).groupby(['agent', 'win']).working.sum().reset_index()
    act = act.rename(columns={'win': 'day'})
    U = P.universe(act)
    agents = sorted(act.agent.unique())
    tm = mw[mw.type.isin(P.TRACE_TYPES)].copy()
    tm['tid'] = tm['tid'].astype(str).astype('category')
    training_dates = {d for d in days if d <= config.TRAIN_END.isoformat()}
    reference = m[m.type.isin(P.TRACE_TYPES)].copy()
    training_agents = wh.loc[wh.day.isin(training_dates), 'agent'].unique()
    idf = P.trace_weights(reference, training_agents, training_dates)
    cos, sync = P.p1_features(tm, idf, agents)
    hand, pou, _, _ = P.p2_features(tm, idf)
    names, _ = P.name_features(mw)
    F = U.copy()
    for part in (cos, sync, hand, pou, names):
        part = part.copy()
        part['day'] = part['day'].astype(str)
        F = F.merge(part, on=['a', 'b', 'day'], how='left')
    F = F.fillna(0.0)
    wa = act.set_index(['agent', 'day']).working
    w1 = wa.reindex(list(zip(F.a, F.day))).values
    w2 = wa.reindex(list(zip(F.b, F.day))).values
    F['act_min'] = np.log1p(np.minimum(w1, w2))
    F['act_max'] = np.log1p(np.maximum(w1, w2))
    lab = set()
    for r in requests:
        local = r['utc'].astimezone(L.PACIFIC)
        key = window_key([local.date().isoformat()], [local.hour], width)[0]
        if key is not None:
            lab.add((*sorted((r['from'], r['to'])), key))
    F['y'] = [(a, b, d) in lab for a, b, d in zip(F.a, F.b, F.day)]
    F['date'] = F.day.str[:10]
    tr = F.date.isin([d for d in days if d <= config.TRAIN_END.isoformat()]).values
    va = F.date.isin([d for d in days if d >= config.VALID_START.isoformat()]).values
    res = {'width_hours': width, 'time_support': '[09:00,17:00) America/Los_Angeles; other hours excluded',
           'feature_fit': 'training only', 'pair_windows': len(F), 'labelled': int(F.y.sum()),
           'base_rate': round(float(F.y.mean()), 4)}
    for name, extra in (('activity only', []),
                        ('behaviour only', P.SETS['pipelines 1 + 2 (behaviour only)']),
                        ('with names', P.SETS['pipelines 1 + 2 (behaviour only)'] + P.SETS['+ all names'])):
        cols = ['act_min', 'act_max'] + [c for c in extra if c in F]
        X = np.log1p(F[cols].clip(lower=0).values)
        s = M.Logistic().fit(X[tr], F.y.values[tr]).decision(X[va])
        res[name] = {'auc': round(M.auc(s, F.y.values[va]), 3),
                     'recall_at_10pct_fa': round(M.recall_at_fpr(s, F.y.values[va]), 3)}
    res['p1_cos_auc_validation'] = round(M.auc(F.p1_cos.values[va], F.y.values[va]), 3)
    return res


def main():
    days = [d.isoformat() for d in pd.date_range(config.DEV_START, config.DEV_END).date]
    config.check_unsealed(days)
    rows, _, _ = L.load_requests()
    dev = [r for r in rows if config.DEV_START <= r['day'] <= config.DEV_END and r['from'] != r['to']]
    m, _ = P.prepare_mentions(set(days))
    tm = m[m.type.isin(P.TRACE_TYPES)]
    e = S.trace_sessions(tm)
    log(f'{len(e)} trace sessions')
    h = handover_events(e)
    log(f'{len(h)} handover events')
    act = P.load_activity(set(days))
    active_by_day = {d: set(g.agent[g.working > 0]) for d, g in act.groupby('day')}
    prof, n = event_study(h, dev, active_by_day)
    ex = prof.excess.clip(lower=0)
    total = ex.sum()
    shares = {f'within_{w}h': round(float(ex[prof.offset_hours.abs() <= w].sum() / total), 3) for w in (0.5, 1, 2, 4)}
    before = round(float(ex[prof.offset_hours < 0].sum() / total), 3)
    peak = float(prof.offset_hours[prof.excess.idxmax()])
    res = {'requests_used': n, 'handover_events': len(h), 'excess_mass_share': shares,
           'excess_mass_before_request': before, 'peak_offset_hours': peak,
           'excess_ratio_within_1h': round(float(prof.requesting_pair[prof.offset_hours.abs() <= 1].sum()
                                                 / max(prof.average_pair[prof.offset_hours.abs() <= 1].sum(), 1e-9)), 2),
           'excess_ratio_beyond_4h': round(float(prof.requesting_pair[prof.offset_hours.abs() > 4].sum()
                                                 / max(prof.average_pair[prof.offset_hours.abs() > 4].sum(), 1e-9)), 2)}
    log('event study done', json.dumps(res))
    wh = working_hours(days)
    log('working hours read')
    res['windows'] = []
    for width in (1, 2, 4, 8):
        r = window_eval(m, wh, dev, width, days)
        log(json.dumps(r))
        res['windows'].append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    prof.to_csv(OUT / 'event_study_dev.csv', index=False)
    with open(OUT / 'summary_dev.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
