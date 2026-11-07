"""Stage E2: spread through shared artifacts, with a marked Hawkes process.

Events are *trace sessions*: an agent's first touch of a trace after at least
TRACE_GAP seconds without touching it, on traces that two or more agents touch.
An event of agent i on trace c at time t is either background or triggered by
an earlier event on the *same trace*:

    lambda_{i,c}(t) = mu_i g(hour) p_i(c) + sum_{m on c, t_m < t} alpha_{i j_m} beta exp(-beta (t - t_m))

p_i(c) is the share of i's events on trace c, so picking up a trace that is new
for i right after j worked on it is attributed to j. alpha_ij is the expected
number of i's trace sessions each session of j triggers: a continuous-time,
probabilistic version of van der Aalst's handover network, fitted by EM as in
the Hawkes literature (Xu et al. 2016 add sparsity priors to read it as Granger
causality).

Influence must beat the day-swap null (each agent's events, with their traces,
moved to another of its active days). Outputs, checked against the held-out
requests: the influence network, the branching ratio, a pair-day feature (the
expected number of one agent's sessions triggered by the other that day), and a
per-day share of sessions triggered by other agents, minus its null expectation.

Run: python -I -m swarmcollab.spread   (first slice only)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from . import labels as L
from . import metrics as M
from . import pairs as P

SPREAD = config.OUT / 'spread'
TRACE_GAP = 1800.0
BETA = 1.0 / 1800.0
WINDOW = 2 * 3600.0
MAX_CAND = 60
EM_ITERS = 100
NULL_FITS = 10


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def trace_sessions(tm, shared_reference=None):
    """One event per (agent, trace) session start, on traces touched by two or more agents."""
    e = pd.DataFrame({'agent': tm.agent.astype(str).values, 'tid': tm.tid.astype(str).values, 't': tm.t.values})
    reference = tm if shared_reference is None else shared_reference
    counts = reference.groupby('tid', observed=True).agent.nunique()
    allowed = set(counts[counts >= 2].index.astype(str))
    e = e[e.tid.isin(allowed)].sort_values(['agent', 'tid', 't'])
    if e.empty:
        return e.assign(day=pd.Series(dtype=str), hour=pd.Series(dtype=int))
    new = np.r_[True, (e.agent.values[1:] != e.agent.values[:-1]) | (e.tid.values[1:] != e.tid.values[:-1])
                | (np.diff(e.t.values) > TRACE_GAP)]
    e = e[new].reset_index(drop=True)
    e['day'] = pd.to_datetime(e.t, unit='s', utc=True).dt.tz_convert('America/Los_Angeles').dt.strftime('%Y-%m-%d')
    e['hour'] = pd.to_datetime(e.t, unit='s', utc=True).dt.tz_convert('America/Los_Angeles').dt.hour
    return e


class MarkedHawkes:
    def __init__(self, agents, beta=None, window=None):
        self.agents = list(agents)
        self.K = len(agents)
        self.beta = BETA if beta is None else beta
        self.window = WINDOW if window is None else window

    def prepare(self, e, learn=False):
        idx = {a: k for k, a in enumerate(self.agents)}
        self.unseen_agent_events = int((~e.agent.isin(idx)).sum())
        e = e[e.agent.isin(idx)].sort_values(['tid', 't']).reset_index(drop=True)
        if e.empty:
            raise ValueError('No events on the fitted agent support')
        self.e = e
        i = e.agent.map(idx).values
        c = pd.factorize(e.tid)[0]
        t = e.t.values
        key = c * 1e8 + (t - t.min())
        lo = np.searchsorted(key, key - self.window, side='left')
        k = np.arange(len(e))
        lo = np.maximum(lo, k - MAX_CAND)
        n = k - lo
        total = int(n.sum())
        tgt = np.repeat(k, n)
        src = np.arange(total) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
        dt = t[tgt] - t[src]
        keep = (dt > 0) & (c[tgt] == c[src])
        self.tgt, self.src = tgt[keep].astype(np.int64), src[keep].astype(np.int64)
        self.kern = self.beta * np.exp(-self.beta * dt[keep])
        self.i = i
        # Learn marks and hourly rhythm on the fit period, then freeze them.
        # A fixed pseudo-count gives new marks/hours finite predictive density.
        if learn:
            counts = e.groupby(['agent', 'tid']).size()
            self.mark_counts = counts.to_dict()
            self.mark_totals = e.groupby('agent').size().to_dict()
            self.mark_vocabulary = set(e.tid)
            self.mark_prior = 0.5
            hrs = np.bincount(e.hour.values, minlength=24).astype(float) + 0.5
            self.hour_profile = hrs / hrs.mean()
        denominator = np.array([self.mark_totals.get(a, 0) + self.mark_prior * (len(self.mark_vocabulary) + 1)
                                for a in e.agent])
        self.p = np.array([self.mark_counts.get((a, tid), 0) + self.mark_prior
                           for a, tid in zip(e.agent, e.tid)]) / denominator
        hrs = e.hour.values
        g = self.hour_profile
        self.g_at = g[hrs]
        days_active = e.groupby('agent').day.nunique()
        self.exposure = np.array([days_active.get(a, 0) * g.sum() * 3600.0 for a in self.agents])
        self.n_src = np.bincount(i, minlength=self.K).astype(float)
        return self

    def fit(self, e, iters=EM_ITERS):
        self.prepare(e, learn=True)
        self.fit_event_counts = self.n_src.copy()
        self.fit_days = sorted(self.e.day.unique())
        K = self.K
        mu = self.n_src / np.maximum(self.exposure, 1.0) * 0.5
        alpha = np.full((K, K), 0.01)
        code = self.i[self.tgt] * K + self.i[self.src]
        for _ in range(iters):
            trig = alpha[self.i[self.tgt], self.i[self.src]] * self.kern
            bg = mu[self.i] * self.g_at * self.p
            lam = np.maximum(bg + np.bincount(self.tgt, weights=trig, minlength=len(self.i)), 1e-300)
            r0 = bg / lam
            r = trig / lam[self.tgt]
            mu = np.bincount(self.i, weights=r0, minlength=K) / np.maximum(self.exposure, 1.0)
            alpha = np.bincount(code, weights=r, minlength=K * K).reshape(K, K) / np.maximum(self.n_src[None, :], 1.0)
        self.mu, self.alpha, self.r, self.r0 = mu, alpha, r, r0
        return self

    def score(self, e):
        """Attribute observed events with frozen fitted parameters, without EM/refitting."""
        self.prepare(e, learn=False)
        trig = self.alpha[self.i[self.tgt], self.i[self.src]] * self.kern
        bg = self.mu[self.i] * self.g_at * self.p
        lam = np.maximum(bg + np.bincount(self.tgt, weights=trig, minlength=len(self.i)), 1e-300)
        self.r0 = bg / lam
        self.r = trig / lam[self.tgt]
        return self

    def cross_excitation(self, source_counts=None):
        """Off-diagonal offspring mass: equal-agent mean and source-event-weighted mean."""
        off = ~np.eye(self.K, dtype=bool)
        per_source = (self.alpha * off).sum(axis=0)
        counts = self.fit_event_counts if source_counts is None else np.asarray(source_counts, dtype=float)
        return {'unweighted_per_agent': float(per_source.mean()),
                'source_event_weighted': float(np.dot(per_source, counts) / max(counts.sum(), 1.0))}

    def loglik(self):
        trig = self.alpha[self.i[self.tgt], self.i[self.src]] * self.kern
        lam = self.mu[self.i] * self.g_at * self.p + np.bincount(self.tgt, weights=trig, minlength=len(self.i))
        mass = 1 - np.exp(-self.beta * self.window)
        comp = (self.mu * self.exposure).sum() + (self.alpha.sum(0) * self.n_src).sum() * mass
        return float(np.log(np.maximum(lam, 1e-300)).sum() - comp)

    def branching_ratio(self):
        return float(np.max(np.abs(np.linalg.eigvals(self.alpha))))

    def per_day_other_share(self):
        other = self.i[self.tgt] != self.i[self.src]
        trig = np.bincount(self.tgt[other], weights=self.r[other], minlength=len(self.i))
        return pd.Series(trig).groupby(self.e.day.values).mean()

    def pair_day_triggers(self):
        """Expected number of sessions of b triggered by a, per (a, b, day), a != b."""
        other = self.i[self.tgt] != self.i[self.src]
        d = pd.DataFrame({'src': np.array(self.agents)[self.i[self.src[other]]],
                          'dst': np.array(self.agents)[self.i[self.tgt[other]]],
                          'day': self.e.day.values[self.tgt[other]], 'r': self.r[other]})
        return d.groupby(['src', 'dst', 'day']).r.sum().reset_index()


def day_swap(e, rng):
    """Permute complete agent-day blocks without preceding observed artifact availability."""
    out = []
    first_seen = e.groupby('tid').day.min().to_dict()
    for a, g in e.groupby('agent'):
        uniq = sorted(g.day.unique())
        available = {day: max(first_seen[tid] for tid in sub.tid.unique()) for day, sub in g.groupby('day')}
        target = list(uniq)
        for _ in range(4 * len(uniq)):
            i, j = rng.integers(len(uniq), size=2)
            if available[uniq[i]] <= target[j] and available[uniq[j]] <= target[i]:
                target[i], target[j] = target[j], target[i]
        off = {d: (pd.Timestamp(x) - pd.Timestamp(d)).total_seconds() for d, x in zip(uniq, target)}
        h = g.copy()
        h['t'] = h.t + h.day.map(off)
        h['day'] = h.day.map(dict(zip(uniq, target)))
        out.append(h)
    return pd.concat(out, ignore_index=True)


def main(retrospective=False):
    rng = np.random.default_rng(20261106)
    days = [d.isoformat() for d in pd.date_range(config.DEV_START, config.DEV_END).date]
    config.check_unsealed(days)
    m, _ = P.prepare_mentions(set(days), retrospective=retrospective)
    tm = m[m.type.isin(P.TRACE_TYPES)]
    reference = tm if retrospective else tm[tm.day.astype(str) <= config.TRAIN_END.isoformat()]
    e = trace_sessions(tm, shared_reference=reference)
    fit_e = e if retrospective else e[e.day <= config.TRAIN_END.isoformat()]
    agents = sorted(fit_e.agent.unique())
    log(f'{len(e)} trace sessions on shared traces, {e.tid.nunique()} traces, {len(agents)} agents')
    model = MarkedHawkes(agents).fit(fit_e)
    train_loglik = model.loglik()
    valid_e = e[e.day >= config.VALID_START.isoformat()]
    validation_loglik = model.score(valid_e).loglik() if not valid_e.empty else None
    model.score(e)
    A = model.alpha
    off = ~np.eye(len(agents), dtype=bool)
    log(f'fitted: {len(model.tgt)} candidate triggers, branching ratio {model.branching_ratio():.3f}, '
        f'cross-excitation per agent {float((A * off).sum(1).mean()):.3f}')
    nulls, null_days, null_br, null_weighted = [], [], [], []
    for s in range(NULL_FITS):
        nm = MarkedHawkes(agents).fit(day_swap(fit_e, rng))
        null_weighted.append(nm.cross_excitation()['source_event_weighted'])
        nm.score(e)
        nulls.append(nm.alpha)
        null_days.append(nm.per_day_other_share())
        null_br.append(nm.branching_ratio())
        log(f'null {s + 1}/{NULL_FITS}: branching ratio {null_br[-1]:.3f}, '
            f'cross-excitation per agent {float((nm.alpha * off).sum(1).mean()):.3f}')
    N = np.array(nulls)
    excess = A - N.mean(0)

    rows, _, _ = L.load_requests()
    label_start = config.DEV_START if retrospective else config.VALID_START
    dev = [r for r in rows if label_start <= r['day'] <= config.DEV_END]
    idx = {a: k for k, a in enumerate(agents)}
    asked = np.zeros_like(A, dtype=bool)   # asked[i, j]: j asked i
    for r in dev:
        if r['from'] in idx and r['to'] in idx and r['from'] != r['to']:
            asked[idx[r['to']], idx[r['from']]] = True
    und = asked | asked.T

    # pair-day feature against pair-day labels
    F = pd.read_csv(P.PAIRS / 'features_dev.csv.gz')
    pdt = model.pair_day_triggers()
    pdt = P.undirected(pdt.rename(columns={'src': 'a', 'dst': 'b'})).groupby(['a', 'b', 'day'], as_index=False).r.sum()
    F = F.merge(pdt.rename(columns={'r': 'hk_trig'}), on=['a', 'b', 'day'], how='left').fillna({'hk_trig': 0.0})
    tr = F.day.isin([d for d in days if d <= config.TRAIN_END.isoformat()]).values
    va = F.day.isin([d for d in days if d >= config.VALID_START.isoformat()]).values
    base = ['act_min', 'act_max'] + P.SETS['pipelines 1 + 2 (behaviour only)']
    models = {}
    for name, cols in (('pipelines 1 + 2', base), ('pipelines 1 + 2 + Hawkes triggers', base + ['hk_trig']),
                       ('activity + Hawkes triggers', ['act_min', 'act_max', 'hk_trig'])):
        X = np.log1p(F[cols].clip(lower=0).values)
        mdl = M.Logistic().fit(X[tr], F.y.values[tr])
        s = mdl.decision(X[va])
        models[name] = {'auc': round(M.auc(s, F.y.values[va]), 3),
                        'recall_at_10pct_fa': round(M.recall_at_fpr(s, F.y.values[va]), 3)}

    # per-day share triggered by others, against its null expectation
    real = model.per_day_other_share()
    nullmean = pd.concat(null_days, axis=1).mean(axis=1)
    W = pd.read_csv(config.OUT / 'regimes' / 'windows_dev.csv')
    W['hk_other_share'] = W.day.map(real)
    W['hk_other_share_null'] = W.day.map(nullmean)
    W['hk_excess'] = W.hk_other_share - W.hk_other_share_null
    report_windows = W if retrospective else W[W.day >= config.VALID_START.isoformat()]
    rho_ex, p_ex = stats.spearmanr(report_windows.hk_excess, report_windows.labelled_share)
    rho_raw, p_raw = stats.spearmanr(report_windows.hk_other_share, report_windows.labelled_share)
    rho_hc, p_hc = stats.spearmanr(report_windows.hk_excess, report_windows.active_pairs)
    res = {
        'evaluation_mode': 'retrospective' if retrospective else 'prospective',
        'fit_days': model.fit_days,
        'fit_sessions': len(fit_e),
        'training_loglik': train_loglik,
        'validation_loglik': validation_loglik,
        'likelihood_note': 'Conditional on observed active days; truncated kernel compensator is approximate.',
        'shared_trace_support': 'whole period' if retrospective else 'training shared traces only',
        'null_method': 'training agent-day block permutation constrained by first observed trace availability',
        'kernel_mean_delay_minutes': 1.0 / (model.beta * 60.0),
        'kernel_selection': 'fixed 30-minute baseline; separate kernel scan chooses on training likelihood only',
        'unseen_agent_events_excluded': model.unseen_agent_events,
        'request_network_and_daily_summary_scope': 'whole development slice' if retrospective else 'validation only',
        'sessions': len(e),
        'traces': int(e.tid.nunique()),
        'candidate_triggers': int(len(model.tgt)),
        'branching_ratio': round(model.branching_ratio(), 3),
        'branching_ratio_null': [round(float(np.mean(null_br)), 3), round(float(np.std(null_br)), 3)],
        'cross_excitation_per_agent': round(float((A * off).sum(1).mean()), 4),
        'cross_excitation_per_source_event': round(model.cross_excitation()['source_event_weighted'], 4),
        'cross_excitation_per_source_event_null': round(float(np.mean(null_weighted)), 4),
        'excitation_weighting': 'per_agent is unweighted; per_source_event uses training event counts',
        'cross_excitation_per_agent_null': round(float((N * off).sum(2).mean()), 4),
        'self_excitation_mean': round(float(np.diag(A).mean()), 4),
        'auc_excess_vs_any_request_pair': round(M.auc(excess[off], und[off]), 3),
        'auc_excess_asker_triggers_askee': round(M.auc(excess[off], asked[off]), 3),
        'auc_excess_askee_triggers_asker': round(M.auc(excess[off], asked.T[off]), 3),
        'share_of_agent_pairs_with_requests': round(float(und[off].mean()), 3),
        'pair_day_feature_single': M.summary(F.hk_trig.values, F.y.values),
        'pair_day_models_validation': models,
        'daily_spearman_excess_vs_labelled_share': [round(float(rho_ex), 3), float(p_ex)],
        'daily_spearman_raw_vs_labelled_share': [round(float(rho_raw), 3), float(p_raw)],
        'daily_spearman_excess_vs_active_pairs': [round(float(rho_hc), 3), float(p_hc)],
    }
    SPREAD.mkdir(parents=True, exist_ok=True)
    tag = 'retrospective_dev' if retrospective else 'dev'
    pd.DataFrame(A, index=agents, columns=agents).to_csv(SPREAD / f'alpha_{tag}.csv')
    pd.DataFrame(excess, index=agents, columns=agents).to_csv(SPREAD / f'alpha_excess_{tag}.csv')
    W.to_csv(SPREAD / f'days_{tag}.csv', index=False)
    F[['a', 'b', 'day', 'hk_trig']].to_csv(SPREAD / f'pair_day_triggers_{tag}.csv.gz', index=False)
    with open(SPREAD / f'summary_{tag}.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


def kernel_scan(retrospective=False):
    """Choose the delay on training likelihood; report frozen validation likelihood separately."""
    days = [d.isoformat() for d in pd.date_range(config.DEV_START, config.DEV_END).date]
    config.check_unsealed(days)
    m, _ = P.prepare_mentions(set(days), retrospective=retrospective)
    tm = m[m.type.isin(P.TRACE_TYPES)]
    reference = tm if retrospective else tm[tm.day.astype(str) <= config.TRAIN_END.isoformat()]
    e = trace_sessions(tm, shared_reference=reference)
    fit_e = e if retrospective else e[e.day <= config.TRAIN_END.isoformat()]
    valid_e = e[e.day >= config.VALID_START.isoformat()]
    agents = sorted(fit_e.agent.unique())
    off = ~np.eye(len(agents), dtype=bool)
    out = []
    for minutes in (2, 5, 10, 20, 30, 60, 120, 240):
        beta = 1.0 / (60.0 * minutes)
        mdl = MarkedHawkes(agents, beta=beta, window=4.0 / beta).fit(fit_e)
        out.append({'mean_delay_min': minutes, 'loglik': round(mdl.loglik(), 1),
                    'fit_scope': 'retrospective development' if retrospective else 'training only',
                    'cross_excitation_per_agent': round(float((mdl.alpha * off).sum(1).mean()), 3),
                    'cross_excitation_per_source_event': round(mdl.cross_excitation()['source_event_weighted'], 3),
                    'validation_loglik': round(mdl.score(valid_e).loglik(), 1) if not valid_e.empty else None})
        log(json.dumps(out[-1]))
    best = max(out, key=lambda r: r['loglik'])
    print(json.dumps({'best_mean_delay_min': best['mean_delay_min'], 'scan': out}, indent=1))
    SPREAD.mkdir(parents=True, exist_ok=True)
    tag = 'retrospective_dev' if retrospective else 'dev'
    with open(SPREAD / f'kernel_scan_{tag}.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel-scan', '--scan-beta', action='store_true')
    ap.add_argument('--retrospective', action='store_true', help='explicit exploratory fit on the whole development slice')
    args = ap.parse_args()
    kernel_scan(args.retrospective) if args.kernel_scan else main(args.retrospective)
