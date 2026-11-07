"""Stage E: how collaboration spreads, with a multivariate Hawkes process.

Each agent is a point process of *work-block starts*: the first working step
after at least BLOCK_GAP seconds without one. A Hawkes process with exponential
kernels explains every event either by the agent's own baseline or as triggered
by an earlier event of some agent j:

    lambda_i(t) = mu_i * g(hour of t) + sum_j alpha_ij * sum_{t_k^j < t} beta * exp(-beta (t - t_k^j))

alpha_ij is the expected number of i-events each j-event triggers; g is the
village's shared hour-of-day profile, so the common office-hours rhythm is not
read as influence. Fitting is EM (Lewis & Mohler style) with a fixed decay and a
truncation window; Xu et al. (2016) learn the same alpha network with sparsity
priors for Granger causality.

Influence that is real must beat a *day-swap null*: each agent's day of events is
moved to another of its active days, which keeps every agent's own rhythm and
volume but breaks same-day coordination. The fitted network is then compared
with who asked whom in the request records (held out until here).

Per window (day), the share of events triggered by *other* agents measures how
collaborative the village was that day.

Run: python -I -m swarmcollab.hawkes   (first slice only)
"""
import gzip
import json
import time

import numpy as np
import pandas as pd

from . import config
from . import labels as L
from . import metrics as M

HAWKES = config.OUT / 'hawkes'
BLOCK_GAP = 300.0          # seconds without a working step before a new block starts
BETA = 1.0 / 1800.0        # kernel decay: 30-minute mean delay
WINDOW = 2 * 3600.0        # truncation: ignore triggers older than 2 hours (98% of the kernel mass)
EM_ITERS = 100
NULL_FITS = 10
WAITING = {'pause', 'wait', 'empty', 'restart'}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def block_starts(days):
    """agent -> sorted array of block-start times (epoch seconds), first slice only."""
    with open(config.STEPS / 'summary.json', encoding='utf-8') as f:
        summary = json.load(f)
    keep = set(days)
    out = {}
    for agent, info in summary['agents'].items():
        times = []
        with gzip.open(config.STEPS / info['file'], 'rt', encoding='utf-8') as f:
            for line in f:
                # cheap pre-filter on the timestamp before parsing the record
                k = line.find('"t": "')
                stamp = line[k + 6:k + 32]
                if stamp[:10] < min(keep) or stamp[:10] > max(keep):
                    continue
                r = json.loads(line)
                if r['k'] in WAITING:
                    continue
                times.append(config.parse_utc(r['t']).timestamp())
        if not times:
            continue
        t = np.array(times)
        t.sort()
        day = pd.to_datetime(t, unit='s', utc=True).tz_convert('America/Los_Angeles').strftime('%Y-%m-%d').values
        ok = np.isin(day, list(keep))
        t = t[ok]
        if len(t) == 0:
            continue
        starts = np.concatenate([[True], np.diff(t) > BLOCK_GAP])
        out[agent] = t[starts]
    return out


def pacific_day(t):
    return pd.to_datetime(t, unit='s', utc=True).tz_convert('America/Los_Angeles').strftime('%Y-%m-%d').values


def hour_profile(events):
    """Village-wide share of events per Pacific hour, as a rate multiplier with mean 1 over active hours."""
    allt = np.concatenate(list(events.values()))
    hrs = pd.to_datetime(allt, unit='s', utc=True).tz_convert('America/Los_Angeles').hour.values
    counts = np.bincount(hrs, minlength=24).astype(float)
    active = counts > 0
    g = np.zeros(24)
    g[active] = counts[active] / counts[active].mean()
    return g


class Hawkes:
    def __init__(self, agents, beta=BETA, window=WINDOW):
        self.agents = list(agents)
        self.K = len(agents)
        self.beta = beta
        self.window = window

    def _prepare(self, events, g):
        idx = {a: i for i, a in enumerate(self.agents)}
        t = np.concatenate([events[a] for a in self.agents if a in events])
        i = np.concatenate([np.full(len(events[a]), idx[a]) for a in self.agents if a in events])
        order = np.argsort(t, kind='mergesort')
        t, i = t[order], i[order]
        lo = np.searchsorted(t, t - self.window, side='left')
        n_pairs = int((np.arange(len(t)) - lo).sum())
        tgt = np.repeat(np.arange(len(t)), np.arange(len(t)) - lo)
        src = np.concatenate([np.arange(a, b) for a, b in zip(lo, np.arange(len(t)))]) if n_pairs else np.array([], int)
        dt = t[tgt] - t[src]
        keep = dt > 0
        tgt, src, dt = tgt[keep], src[keep], dt[keep]
        self.t, self.i = t, i
        self.tgt, self.src = tgt, src
        self.kern = self.beta * np.exp(-self.beta * dt)
        hrs = pd.to_datetime(t, unit='s', utc=True).tz_convert('America/Los_Angeles').hour.values
        self.g_at = g[hrs]
        # exposure for the baseline: active hours per day for each agent, weighted by g
        days = pacific_day(t)
        self.days = days
        self.exposure = np.zeros(self.K)
        for a in self.agents:
            if a not in events:
                continue
            k = idx[a]
            n_days = len(set(days[i == k]))
            self.exposure[k] = n_days * g.sum() * 3600.0
        # kernel mass each source event can spend before its day ends (truncation included)
        self.mass = np.ones(len(t))
        return self

    def fit(self, events, g, iters=EM_ITERS):
        self._prepare(events, g)
        K = self.K
        counts = np.bincount(self.i, minlength=K).astype(float)
        mu = counts / np.maximum(self.exposure, 1.0) * 0.5
        alpha = np.full((K, K), 0.01)
        src_counts = np.bincount(self.i, weights=self.mass, minlength=K)
        pair_code = self.i[self.tgt] * K + self.i[self.src]
        for _ in range(iters):
            trig = alpha[self.i[self.tgt], self.i[self.src]] * self.kern
            lam = mu[self.i] * self.g_at + np.bincount(self.tgt, weights=trig, minlength=len(self.t))
            r_bg = mu[self.i] * self.g_at / lam
            r = trig / lam[self.tgt]
            mu = np.bincount(self.i, weights=r_bg, minlength=K) / np.maximum(self.exposure, 1.0)
            num = np.bincount(pair_code, weights=r, minlength=K * K).reshape(K, K)
            alpha = num / np.maximum(src_counts[None, :], 1.0)
        self.mu, self.alpha = mu, alpha
        self.r_bg = r_bg
        self.r = r
        self.loglik = float(np.log(lam).sum() - (mu * self.exposure).sum() - (alpha * src_counts[None, :]).sum())
        return self

    def triggered_by_others(self):
        """Per event: probability it was triggered by another agent's event."""
        other = self.i[self.tgt] != self.i[self.src]
        return np.bincount(self.tgt[other], weights=self.r[other], minlength=len(self.t))

    def branching_ratio(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.alpha))))


def day_swap(events, rng):
    """Move each agent's events of each day to another of its active days (time of day kept)."""
    out = {}
    for a, t in events.items():
        days = pacific_day(t)
        uniq = sorted(set(days))
        target = [str(x) for x in rng.permutation(uniq)]
        offset = {d: (pd.Timestamp(x) - pd.Timestamp(d)).total_seconds() for d, x in zip(uniq, target)}
        out[a] = np.sort(t + np.array([offset[d] for d in days]))
    return out


def main():
    rng = np.random.default_rng(20261106)
    days = [d.isoformat() for d in pd.date_range(config.DEV_START, config.DEV_END).date]
    config.check_unsealed(days)
    log('reading block starts ...')
    events = block_starts(days)
    agents = sorted(events)
    log(f'{sum(len(v) for v in events.values())} block starts for {len(agents)} agents')
    g = hour_profile(events)
    model = Hawkes(agents).fit(events, g)
    log(f'fitted: branching ratio {model.branching_ratio():.3f}, {len(model.tgt)} candidate trigger pairs')
    A = model.alpha
    nulls = []
    for s in range(NULL_FITS):
        nm = Hawkes(agents).fit(day_swap(events, rng), g)
        nulls.append(nm.alpha)
        log(f'null {s + 1}/{NULL_FITS}: branching ratio {nm.branching_ratio():.3f}')
    N = np.array(nulls)
    excess = A - N.mean(0)
    z = excess / np.maximum(N.std(0), 1e-9)
    off = ~np.eye(len(agents), dtype=bool)
    # compare with requests: j -> i in alpha_ij means j's events trigger i's
    rows, _, _ = L.load_requests()
    dev = [r for r in rows if config.DEV_START <= r['day'] <= config.DEV_END]
    asked = np.zeros_like(A, dtype=bool)        # asked[i, j]: j asked i
    idx = {a: k for k, a in enumerate(agents)}
    for r in dev:
        if r['from'] in idx and r['to'] in idx and r['from'] != r['to']:
            asked[idx[r['to']], idx[r['from']]] = True
    undirected = asked | asked.T
    res = {
        'events': int(sum(len(v) for v in events.values())),
        'agents': len(agents),
        'branching_ratio': round(model.branching_ratio(), 3),
        'branching_ratio_null_mean': round(float(np.mean([np.max(np.abs(np.linalg.eigvals(x))) for x in N])), 3),
        'self_excitation_mean': round(float(np.diag(A).mean()), 4),
        'cross_excitation_sum_mean': round(float((A * off).sum(1).mean()), 4),
        'cross_excitation_sum_null_mean': round(float((N * off).sum(2).mean()), 4),
        'auc_excess_vs_any_request_pair': round(M.auc(excess[off], undirected[off]), 3),
        'auc_excess_vs_asker_triggers_askee': round(M.auc(excess[off], asked[off]), 3),
        'auc_excess_vs_askee_triggers_asker': round(M.auc(excess[off], asked.T[off]), 3),
        'auc_raw_alpha_vs_any_request_pair': round(M.auc(A[off], undirected[off]), 3),
        'share_of_pairs_with_requests': round(float(undirected[off].mean()), 3),
    }
    other = model.triggered_by_others()
    per_day = pd.DataFrame({'day': model.days, 'other': other}).groupby('day').other.agg(['mean', 'size'])
    per_day.columns = ['share_triggered_by_others', 'events']
    HAWKES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=agents, columns=agents).to_csv(HAWKES / 'alpha_dev.csv')
    pd.DataFrame(excess, index=agents, columns=agents).to_csv(HAWKES / 'alpha_excess_dev.csv')
    per_day.to_csv(HAWKES / 'days_dev.csv')
    with open(HAWKES / 'summary_dev.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
