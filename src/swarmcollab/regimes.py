"""Stage D: collaboration regimes in every time window (Pacific workdays).

A pair-day is *flagged* when its detector score passes a threshold fixed without
labels: the 95th percentile of the same score computed between two agents on
different workdays, who could not have been working together in real time.

For each window the village is in a *collaborative regime* when its flagged
share exceeds whole-agent/day-block randomization (p < 0.01 after Bonferroni).
Blocks preserve all traces and counts within an agent-day. Training blocks
are permuted only onto days when their artifacts were already observed;
validation nulls sample from training blocks without fitting on validation.
Labels never enter the detector: they are used afterwards.

Per window: active pairs, flagged pairs and share, regime, the labelled share
(pairs with a request), and the within-window AUC of the score.
Across windows: rank correlation between flagged and labelled shares, and
agreement with a label-based regime (labelled share above its median).

Run: python -I -m swarmcollab.regimes   (first slice only)
"""
import json

import numpy as np
import pandas as pd
from scipy import sparse, stats

from . import config
from . import metrics as M
from . import pairs as P

REGIMES = config.OUT / 'regimes'
NULL_QUANTILE = 0.95
NULL_MAX_GAP_DAYS = 7
NULL_SAMPLES = 60000
BLOCK_NULL_SAMPLES = 4999


def day_vectors(tm, idf, agents):
    """Per day: the agents' IDF-weighted trace vectors, row-normalised."""
    aidx = {a: i for i, a in enumerate(agents)}
    g = tm.groupby(['day', 'agent', 'tid'], observed=True).size().reset_index(name='n')
    g['w'] = g.tid.map(idf).astype(float) * (1 + np.log(g.n))
    n_t = len(g.tid.cat.categories)
    out = {}
    for day, sub in g.groupby('day', observed=True):
        X = sparse.csr_matrix((sub.w.values, (sub.agent.map(aidx).values, sub.tid.cat.codes.values)),
                              shape=(len(agents), n_t))
        norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        out[str(day)] = (sparse.diags(inv) @ X).tocsr(), norms > 0
    return out


def cross_day_null(vecs, rng, n=NULL_SAMPLES, max_gap=NULL_MAX_GAP_DAYS):
    """Cosine between agent a on day d and agent b != a on day d' != d (|d - d'| <= max_gap)."""
    days = sorted(vecs)
    ords = np.array([pd.Timestamp(d).toordinal() for d in days])
    vals = []
    tries = 0
    while len(vals) < n and tries < n * 20:
        tries += 1
        i, j = rng.integers(len(days), size=2)
        if i == j or abs(ords[i] - ords[j]) > max_gap:
            continue
        Xi, ai = vecs[days[i]]
        Xj, aj = vecs[days[j]]
        ca, cb = np.flatnonzero(ai), np.flatnonzero(aj)
        if len(ca) == 0 or len(cb) == 0:
            continue
        a, b = rng.choice(ca), rng.choice(cb)
        if a == b:
            continue
        vals.append(float(Xi[a].multiply(Xj[b]).sum()))
    return np.array(vals)


class AgentDayBlockNull:
    """Training-only empirical block null, retaining within-day trace dependence.

    Each training permutation preserves each agent's active dates and the whole
    multiset of its trace vectors, hence overall artifact frequency. It does not
    preserve artifact frequency separately on every date. Validation draws are
    empirical samples of those training blocks, not observed validation vectors.
    """

    def __init__(self, vecs, agents, active_by_day, train_days, available_after=None):
        self.days = sorted(d for d in train_days if d in vecs)
        if not self.days:
            raise ValueError('The block null needs training days')
        self.agents = list(agents)
        self.agent_index = {a: i for i, a in enumerate(agents)}
        self.active = active_by_day
        self.available_after = available_after or {}
        self.width = len(agents)
        self.rows = {}
        for ai, agent in enumerate(agents):
            self.rows[agent] = [(di, di * self.width + ai) for di, day in enumerate(self.days)
                                if agent in active_by_day.get(day, set())]
        X = sparse.vstack([vecs[d][0] for d in self.days]).tocsr()
        self.cosines = (X @ X.T).toarray()

    def training_permutation(self, rng):
        """Return donor-row IDs for every training day/agent; -1 means inactive."""
        result = np.full((len(self.days), self.width), -1, dtype=int)
        for agent, rows in self.rows.items():
            ai = self.agent_index[agent]
            positions = [di for di, _ in rows]
            donors = [row for _, row in rows]
            for _ in range(4 * len(donors)):
                if len(donors) < 2:
                    break
                i, j = rng.integers(len(donors), size=2)
                donor_i = self.days[donors[i] // self.width]
                donor_j = self.days[donors[j] // self.width]
                limit_i = self.available_after.get((agent, donor_i), self.days[0])
                limit_j = self.available_after.get((agent, donor_j), self.days[0])
                if limit_i <= self.days[positions[j]] and limit_j <= self.days[positions[i]]:
                    donors[i], donors[j] = donors[j], donors[i]
            result[positions, ai] = donors
        return result

    def distributions(self, threshold, rng, n=BLOCK_NULL_SAMPLES):
        out = {d: np.zeros(n) for d in self.active}
        training_index = {day: i for i, day in enumerate(self.days)}
        covered = {d: all(self.rows.get(a) for a in active) for d, active in self.active.items()}
        for rep in range(n):
            permuted = self.training_permutation(rng)
            for day, active in self.active.items():
                active = sorted(active)
                if len(active) < 2 or not covered[day]:
                    continue
                if day in training_index:
                    ids = np.array([permuted[training_index[day], self.agent_index[a]] for a in active])
                else:
                    ids = np.array([self.rows[a][rng.integers(len(self.rows[a]))][1] for a in active])
                scores = self.cosines[np.ix_(ids, ids)]
                upper = np.triu_indices(len(ids), 1)
                out[day][rep] = np.mean(scores[upper] > threshold)
        return out, covered


def artifact_availability(tm):
    """Earliest observed availability bound for every complete agent-day block."""
    first = tm.groupby('tid', observed=True).day.min().astype(str).to_dict()
    frame = tm[['agent', 'day', 'tid']].copy()
    frame['available'] = frame.tid.map(first).astype(str)
    return frame.groupby(['agent', 'day'], observed=True).available.max().to_dict()


def window_table(F, score, thr, null_distributions, n_windows, covered=None):
    rows = []
    for day, g in F.groupby('day'):
        s = g[score].values
        y = g.y.values
        flagged = s > thr
        k, n = int(flagged.sum()), len(g)
        null = np.asarray(null_distributions.get(day, []))
        calibrated = bool(len(null)) and (covered or {}).get(day, True)
        p = float((1 + np.sum(null >= k / n)) / (1 + len(null))) if calibrated and n else float('nan')
        rows.append({
            'day': day,
            'active_pairs': n,
            'flagged': k,
            'flagged_share': k / n if n else 0.0,
            'p_value': p,
            'null_calibrated': calibrated,
            'null_flagged_share_mean': float(null.mean()) if calibrated else float('nan'),
            'null_samples': len(null),
            'collaborative': bool(p < 0.01 / n_windows),
            'labelled': int(y.sum()),
            'labelled_share': float(y.mean()) if n else 0.0,
            'auc_in_window': M.auc(s, y) if 0 < y.sum() < n else float('nan'),
            'flagged_precision': float(y[flagged].mean()) if k else float('nan'),
            'flagged_recall': float(flagged[y].mean()) if y.any() else float('nan'),
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(20261106)
    days = [d.isoformat() for d in pd.date_range(config.DEV_START, config.DEV_END).date]
    config.check_unsealed(days)
    F = pd.read_csv(P.PAIRS / 'features_dev.csv.gz')
    # rebuild the day vectors for the null
    act = P.load_activity(set(days))
    agents = sorted(act.agent.unique())
    m, _ = P.prepare_mentions(set(days))
    tm = m[m.type.isin(P.TRACE_TYPES)].copy()
    tm['tid'] = tm['tid'].astype(str).astype('category')
    train_days = {d for d in days if d <= config.TRAIN_END.isoformat()}
    training_agents = act.loc[act.day.isin(train_days), 'agent'].unique()
    idf = P.trace_weights(tm, training_agents, train_days)
    vecs = day_vectors(tm, idf, agents)
    train_vecs = {d: v for d, v in vecs.items() if d in train_days}
    null = cross_day_null(train_vecs, rng)
    thr = float(np.quantile(null, NULL_QUANTILE))
    null_rate = float((null > thr).mean())
    active_by_day = {str(day): set(g.agent[g.working > 0]) for day, g in act.groupby('day')}
    training_tm = tm[tm.day.astype(str).isin(train_days)].copy()
    training_tm['day'] = training_tm.day.astype(str)
    block_null = AgentDayBlockNull(train_vecs, agents, active_by_day, train_days,
                                  available_after=artifact_availability(training_tm))
    distributions, covered = block_null.distributions(thr, rng)
    W = window_table(F, 'p1_cos', thr, distributions, F.day.nunique(), covered=covered)
    rho, p_rho = stats.spearmanr(W.flagged_share, W.labelled_share)
    label_regime = W.labelled_share > W.labelled_share.median()
    agree = float((label_regime == (W.flagged_share > W.flagged_share.median())).mean())
    summary = {
        'calibration_scope': 'training only; validation blocks drawn only from training',
        'regime_test': 'empirical whole-agent/day trace-block null, Bonferroni familywise alpha 0.01',
        'null_preserves': 'agent active dates and complete within-day trace vectors; training permutations preserve overall artifact frequencies, not per-date artifact frequencies',
        'artifact_availability': 'training donor blocks cannot precede first observed availability of any contained trace',
        'block_null_samples_per_window': BLOCK_NULL_SAMPLES,
        'minimum_resolvable_p': 1.0 / (BLOCK_NULL_SAMPLES + 1),
        'score': 'p1_cos',
        'null_samples': len(null),
        'threshold': round(thr, 4),
        'null_rate': round(null_rate, 4),
        'same_day_rate': round(float((F.p1_cos > thr).mean()), 4),
        'windows': len(W),
        'collaborative_windows': int(W.collaborative.sum()),
        'spearman_flagged_vs_labelled_share': [round(float(rho), 3), float(p_rho)],
        'median_split_agreement': round(agree, 3),
        'median_auc_in_window': round(float(W.auc_in_window.median()), 3),
        'flagged_precision_overall': round(float(F.y[F.p1_cos > thr].mean()), 3),
        'flagged_recall_overall': round(float((F.p1_cos[F.y] > thr).mean()), 3),
        'base_rate': round(float(F.y.mean()), 3),
    }
    # regime-level check: are labelled shares higher on days called collaborative?
    on, off_ = W.labelled_share[W.collaborative], W.labelled_share[~W.collaborative]
    diff = float(on.mean() - off_.mean()) if len(off_) else float('nan')
    perm = []
    for _ in range(5000):
        lab = rng.permutation(W.collaborative.values)
        perm.append(W.labelled_share[lab].mean() - W.labelled_share[~lab].mean())
    summary['labelled_share_collaborative_vs_not'] = [round(float(on.mean()), 4), round(float(off_.mean()), 4) if len(off_) else None,
                                                      round(float((np.array(perm) >= diff).mean()), 4)]
    late = W.day >= '2026-07-13'
    r2, p2 = stats.spearmanr(W.flagged_share[late], W.labelled_share[late])
    summary['spearman_without_first_week'] = [round(float(r2), 3), float(p2)]
    # baselines: activity alone, per day
    daily_act = F.groupby('day').act_min.mean().reindex(W.day).values
    r3, p3 = stats.spearmanr(daily_act, W.labelled_share)
    summary['spearman_activity_vs_labelled_share'] = [round(float(r3), 3), float(p3)]
    # Prospective supervised index: fit on training only and report validation association.
    cols = ['act_min', 'act_max'] + P.SETS['pipelines 1 + 2 (behaviour only)']
    X = np.log1p(F[cols].clip(lower=0).values)
    first = (F.day <= config.TRAIN_END.isoformat()).values
    mdl = M.Logistic().fit(X[first], F.y.values[first])
    prob = 1 / (1 + np.exp(-mdl.decision(X)))
    W['prospective_index'] = pd.Series(prob, index=F.index).groupby(F.day).mean().reindex(W.day).values
    valid = W.day >= config.VALID_START.isoformat()
    r4, p4 = stats.spearmanr(W.prospective_index[valid], W.labelled_share[valid])
    summary['spearman_validation_index_vs_labelled_share'] = [round(float(r4), 3), float(p4)]
    summary['validation_windows'] = int(valid.sum())
    summary['validation_collaborative_windows'] = int(W.collaborative[valid].sum())
    # agent-day level: is the agent's flagged degree higher when it has request partners that day?
    flag = F[F.p1_cos > thr]
    deg = pd.concat([flag.groupby(['a', 'day']).size(), flag.groupby(['b', 'day']).size()]).groupby(level=[0, 1]).sum()
    lab = F[F.y]
    ldeg = pd.concat([lab.groupby(['a', 'day']).size(), lab.groupby(['b', 'day']).size()]).groupby(level=[0, 1]).sum()
    ad = pd.concat([F[['a', 'day']].rename(columns={'a': 'agent'}), F[['b', 'day']].rename(columns={'b': 'agent'})]).drop_duplicates()
    ad['deg'] = [deg.get((x, d), 0) for x, d in zip(ad.agent, ad.day)]
    ad['ldeg'] = [ldeg.get((x, d), 0) for x, d in zip(ad.agent, ad.day)]
    summary['agent_days'] = len(ad)
    summary['agent_day_auc_any_request'] = round(M.auc(ad.deg.values, ad.ldeg.values > 0), 3)
    summary['agent_day_spearman_degrees'] = round(float(stats.spearmanr(ad.deg, ad.ldeg)[0]), 3)
    REGIMES.mkdir(parents=True, exist_ok=True)
    ad.to_csv(REGIMES / 'agent_days_dev.csv', index=False)
    W.to_csv(REGIMES / 'windows_dev.csv', index=False)
    with open(REGIMES / 'summary_dev.json', 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(W[['day', 'active_pairs', 'flagged', 'flagged_share', 'collaborative', 'labelled', 'labelled_share',
             'auc_in_window']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
