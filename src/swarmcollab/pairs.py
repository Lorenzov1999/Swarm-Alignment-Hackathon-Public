"""Stage C: pair-day features, and how each one scores against the request labels.

Unit: the pair-day, two agents who both did some work (not only waiting) on the
same Pacific workday. A pair-day is labelled when the request records show one
of the two asking the other for something that day.

Feature groups
  activity  both agents' working steps: the baseline every detector must beat
  p1        Pacheco co-use: IDF-weighted cosine between the two agents' traces
            that day; synchrony: IDF sum over traces both touched within an hour
  p2        van der Aalst handovers on shared traces that day, by evidence:
            write then use, or read then read; proof of use: one agent uses an
            identifier (commit, issue, pipeline, project, doc) the other created
  names     each agent naming the other inside computer activity, by where the
            name appears, weighted by 1 / (number of agent names in that text)

By default only first-slice days are built and scored; --final builds the
sealed slice too (once, at the end).

Run: python -I -m swarmcollab.pairs [--final]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from . import config
from . import labels as L
from . import metrics as M

MENTIONS = config.OUT / 'mentions'
PAIRS = config.OUT / 'pairs'
TRACE_TYPES = ['repo', 'pages', 'page', 'commit', 'issue', 'pipeline', 'project', 'gdoc', 'site']
IDENT_TYPES = ['commit', 'issue', 'pipeline', 'project', 'gdoc']
NAME_SOURCES = ['comment', 'code', 'typed', 'gui', 'out']
SYNC_SECONDS = 3600


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- loading
def load_mentions(days):
    """All mentions on the given days. Malformed rows (an ID that held a decoded
    tab or newline) are skipped and counted."""
    frames = []
    keep = set(days)
    bad = 0
    for p in sorted(MENTIONS.glob('*.tsv.gz')):
        df = pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, na_values=[], quoting=3,
                         on_bad_lines='skip')
        n0 = len(df)
        df['t'] = pd.to_numeric(df.t, errors='coerce')
        df['step'] = pd.to_numeric(df.step, errors='coerce')
        df['k'] = pd.to_numeric(df.k, errors='coerce')
        df = df.dropna(subset=['t', 'step', 'k'])
        df = df[df.day.str.match(r'^\d{4}-\d{2}-\d{2}$')]
        bad += n0 - len(df)
        df = df[df.day.isin(keep)]
        df['step'] = df.step.astype(np.int32)
        df['k'] = df.k.astype(np.int16)
        frames.append(df)
    m = pd.concat(frames, ignore_index=True)
    for c in ('agent', 'type', 'op', 'src', 'day'):
        m[c] = m[c].astype('category')
    if bad:
        log(f'skipped {bad} malformed mention rows')
    return m


def resolve_repos(m, reference=None):
    """Map uncertain GitLab paths ('gitlab?:') to known projects, drop group pages,
    and map numeric project IDs to repositories (API output, then co-occurrence votes)."""
    m = m.copy().reset_index(drop=True)
    m['type'] = m['type'].astype(str)
    reference = m if reference is None else reference.copy()
    reference['type'] = reference['type'].astype(str)
    exact = set(reference.loc[(reference.type == 'repo') & reference.id.str.startswith('gitlab:'), 'id'].str[7:])
    # project IDs: exact pairs from API output, else a clear majority of co-occurrence votes
    mapping = {}
    for v in reference.loc[reference.type == 'pidmap', 'id'].unique():
        pid, path = v.split('|', 1)
        mapping.setdefault(pid, path)
        exact.add(path)
    votes = reference.loc[reference.type == 'pidvote', 'id'].str.split('|', n=1, expand=True)
    if len(votes):
        votes.columns = ['pid', 'repo']
        votes['repo'] = votes.repo.str.replace(r'^gitlab\??:', '', regex=True)
        tally = votes.groupby(['pid', 'repo']).size().rename('n').reset_index()
        tally['share'] = tally.n / tally.groupby('pid').n.transform('sum')
        best = tally.sort_values('n', ascending=False).drop_duplicates('pid')
        best = best[(best.n >= 2) & (best.share >= 0.6)]
        for pid, path in zip(best.pid, best.repo):
            mapping.setdefault(pid, path)
    m = m[~m.type.isin(['pidmap', 'pidvote'])].copy()

    # a path that prefixes another project is a group (GitLab paths are either groups or projects)
    groups = {k for k in exact if any(o.startswith(k + '/') for o in exact)}
    known = sorted(exact - groups)

    def resolve(path):
        if path in groups or any(k.startswith(path + '/') for k in known):
            return None                      # a group page, not a project
        if path in exact:
            return path
        best = None
        for k in known:
            if path.startswith(k + '/') and (best is None or len(k) > len(best)):
                best = k
        return best or path
    gl = (m.type == 'repo') & m.id.str.startswith('gitlab')
    # A newly observed explicit /-/ project route is an identity, not a learned
    # frequency. Do not collapse it into an older project's prefix. Ambiguous
    # paths still use only the frozen reference mapping.
    cache = {u: (u.split(':', 1)[1] if u.startswith('gitlab:') and u.split(':', 1)[1] not in exact
                  else resolve(u.split(':', 1)[1])) for u in m.loc[gl, 'id'].unique()}
    new = m.loc[gl, 'id'].map(cache)
    m.loc[gl, 'id'] = 'gitlab:' + new.fillna('')
    m = m.drop(index=new.index[new.isna()])
    n_groups = sum(v is None for v in cache.values())

    proj = (m.type == 'project') & m.id.isin(mapping)
    if proj.any():
        m.loc[proj, 'id'] = 'gitlab:' + m.loc[proj, 'id'].map(mapping)
        m.loc[proj, 'type'] = 'repo'
    iss = (m.type == 'issue') & m.id.str.startswith('pid:')
    if iss.any():
        parts = m.loc[iss, 'id'].str.extract(r'^pid:(\d+)([#!]\d+)$')
        mapped = parts[0].map(mapping)
        ok = mapped.notna()
        idx = parts.index[ok]
        m.loc[idx, 'id'] = mapped[ok] + parts.loc[ok, 1]
    m['type'] = m['type'].astype('category')
    m['tid'] = (m['type'].astype(str) + '|' + m['id']).astype('category')
    info = {'project_ids_mapped': len(mapping), 'gitlab_paths': len(cache), 'group_paths_dropped': n_groups,
            'known_projects': len(known)}
    return m, info


def prepare_mentions(days, retrospective=False):
    """Resolve identities using training observations only, unless explicitly exploratory."""
    m = load_mentions(days)
    reference = m if retrospective else m[m.day.astype(str) <= config.TRAIN_END.isoformat()]
    resolved, info = resolve_repos(m, reference=reference)
    info['mapping_fit'] = 'whole requested period (retrospective)' if retrospective else 'training only'
    return resolved, info


def trace_weights(tm, training_agents, train_days=None):
    """Frozen IDF with the training maximum rarity assigned to unseen traces.

    Validation frequency and newly observed agents never change fitted weights.
    New identifiers retain a fixed prior rather than being assigned validation IDF.
    """
    reference = tm if train_days is None else tm[tm.day.astype(str).isin(set(train_days))]
    n = max(len(set(training_agents)), 1)
    counts = reference.groupby('tid', observed=True).agent.nunique()
    learned = np.log(n / counts.clip(upper=n)).to_dict()
    return {tid: learned.get(tid, float(np.log(n))) for tid in tm.tid.astype(str).unique()}


def load_activity(days):
    rows = []
    for p in MENTIONS.glob('*.activity.json'):
        with open(p, encoding='utf-8') as f:
            a = json.load(f)
        for day, c in a['days'].items():
            if day in days:
                rows.append({'agent': a['agent'], 'day': day, **c})
    act = pd.DataFrame(rows).fillna(0)
    for c in ('steps', 'working', 'waiting', 'shell', 'gui', 'typed', 'trace_steps', 'other'):
        if c not in act:
            act[c] = 0
    return act


def universe(act):
    rows = []
    for day, g in act[act.working > 0].groupby('day'):
        ags = sorted(g.agent)
        for i in range(len(ags)):
            for j in range(i + 1, len(ags)):
                rows.append((ags[i], ags[j], day))
    return pd.DataFrame(rows, columns=['a', 'b', 'day'])


def undirected(df, a='a', b='b'):
    lo = np.where(df[a] < df[b], df[a], df[b])
    hi = np.where(df[a] < df[b], df[b], df[a])
    out = df.copy()
    out[a], out[b] = lo, hi
    return out


# ---------------------------------------------------------------- pipeline 1
def p1_features(tm, idf, agents):
    """Co-use cosine and one-hour synchrony per pair-day."""
    aidx = {a: i for i, a in enumerate(agents)}
    g = tm.groupby(['day', 'agent', 'tid'], observed=True).size().reset_index(name='n')
    g['w'] = g.tid.map(idf).astype(float).fillna(0.0) * (1 + np.log(g.n))
    out = []
    for day, sub in g.groupby('day', observed=True):
        rows = sub.agent.map(aidx).values
        X = sparse.csr_matrix((sub.w.values, (rows, sub.tid.cat.codes.values)),
                              shape=(len(agents), len(g.tid.cat.categories)))
        norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
        S = (X @ X.T).toarray()
        present = np.flatnonzero(norms > 0)
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                i, j = present[x], present[y]
                if S[i, j] > 0:
                    out.append((agents[i], agents[j], day, S[i, j] / (norms[i] * norms[j])))
    cos = pd.DataFrame(out, columns=['a', 'b', 'day', 'p1_cos'])

    # synchrony: both touched the trace within SYNC_SECONDS of each other
    k = pd.DataFrame({'day': tm.day.astype(str).values, 'tid': tm.tid.astype(str).values,
                      'agent': tm.agent.astype(str).values, 't': tm.t.values})
    nag = k.groupby(['day', 'tid']).agent.transform('nunique')
    k = k[nag >= 2].sort_values(['day', 'tid', 'agent', 't'])
    if k.empty:
        return undirected(cos), pd.DataFrame(columns=['a', 'b', 'day', 'p1_sync'])
    gkey = (k.day + '|' + k.tid).values
    agent = k.agent.values
    t = k.t.values
    starts = np.flatnonzero(np.r_[True, gkey[1:] != gkey[:-1]])
    ends = np.r_[starts[1:], len(k)]
    sync = []
    for s0, e0 in zip(starts, ends):
        ag = agent[s0:e0]
        cut = np.flatnonzero(np.r_[True, ag[1:] != ag[:-1]])
        bounds = list(zip(cut, np.r_[cut[1:], len(ag)]))
        day, tid = gkey[s0].split('|', 1)
        w = idf.get(tid, 0.0)
        for x in range(len(bounds)):
            a1 = t[s0 + bounds[x][0]: s0 + bounds[x][1]]
            for y in range(x + 1, len(bounds)):
                a2 = t[s0 + bounds[y][0]: s0 + bounds[y][1]]
                pos = np.searchsorted(a2, a1)
                d1 = np.abs(a2[np.minimum(pos, len(a2) - 1)] - a1).min()
                d2 = np.abs(a2[np.maximum(pos - 1, 0)] - a1).min()
                if min(d1, d2) <= SYNC_SECONDS:
                    sync.append((ag[bounds[x][0]], ag[bounds[y][0]], day, w))
    sync = pd.DataFrame(sync, columns=['a', 'b', 'day', 'p1_sync'])
    sync = undirected(sync).groupby(['a', 'b', 'day'], as_index=False).p1_sync.sum()
    return undirected(cos), sync


# ---------------------------------------------------------------- pipeline 2
def p2_features(tm, idf):
    """Same-day handovers on shared traces, and proof of use of created identifiers."""
    tm = tm.assign(write=tm.op.isin(['create', 'write']))
    per = tm.groupby(['day', 'tid', 'agent'], observed=True).agg(first=('t', 'min'), last=('t', 'max')).reset_index()
    fw = tm[tm.write].groupby(['day', 'tid', 'agent'], observed=True).t.min().rename('first_write').reset_index()
    per = per.merge(fw, on=['day', 'tid', 'agent'], how='left')
    n_ag = per.groupby(['day', 'tid'], observed=True).agent.transform('size')
    per = per[n_ag >= 2]
    per['day'] = per['day'].astype(str)
    per['tid'] = per['tid'].astype(str)
    per['agent'] = per['agent'].astype(str)
    x = per.merge(per, on=['day', 'tid'], suffixes=('_a', '_b'))
    x = x[x.agent_a != x.agent_b]
    x = x[x.first_a < x.last_b]                      # a touched it before b's last touch
    x['w'] = x.tid.map(idf).astype(float)
    x['kind'] = np.where(x.first_write_a < x.last_b, 'h_write', 'h_read')
    directed = x.groupby(['agent_a', 'agent_b', 'day', 'kind']).w.sum().unstack('kind', fill_value=0).reset_index()
    directed = directed.rename(columns={'agent_a': 'a', 'agent_b': 'b'})
    for c in ('h_write', 'h_read'):
        if c not in directed:
            directed[c] = 0.0
    hand = undirected(directed).groupby(['a', 'b', 'day'], as_index=False)[['h_write', 'h_read']].sum()

    # proof of use
    ident = tm[tm.type.isin(IDENT_TYPES)][['agent', 't', 'day', 'tid', 'type', 'op', 'src']].copy()
    for c in ('agent', 'tid', 'type', 'op', 'src', 'day'):
        ident[c] = ident[c].astype(str)
    ident = ident.sort_values('t')
    first = ident.drop_duplicates('tid')[['tid', 'agent', 't']].rename(columns={'agent': 'creator', 't': 't_c'})
    created = ident[ident.op == 'create'].drop_duplicates('tid')[['tid', 'agent', 't']].rename(columns={'agent': 'creator', 't': 't_c'})
    feats = []
    for label, cr in (('pou_created', created), ('pou_first', first)):
        u = ident.merge(cr, on='tid')
        u = u[(u.agent != u.creator) & (u.t > u.t_c)]
        u = u.drop_duplicates(['tid', 'agent', 'day'])
        u['grp'] = np.where(u.type == 'commit', '_commit', '_other')
        f = u.groupby(['creator', 'agent', 'day', 'grp']).size().unstack('grp', fill_value=0)
        f.columns = [label + c for c in f.columns]
        f = f.reset_index().rename(columns={'creator': 'a', 'agent': 'b'})
        feats.append(f)
    pou_directed = feats[0].merge(feats[1], on=['a', 'b', 'day'], how='outer').fillna(0)
    pou_cols = [c for c in pou_directed.columns if c.startswith('pou_')]
    pou = undirected(pou_directed).groupby(['a', 'b', 'day'], as_index=False)[pou_cols].sum()
    return hand, pou, directed, pou_directed


# ---------------------------------------------------------------- names
def name_features(m):
    nm = m[m.type == 'name'][['agent', 'id', 'day', 'src', 'k']].copy()
    nm['agent'] = nm.agent.astype(str)
    nm['day'] = nm.day.astype(str)
    nm['src'] = nm.src.astype(str)
    nm = nm[nm.agent != nm.id]
    nm['w'] = 1.0 / nm.k.clip(lower=1)
    d = nm.groupby(['agent', 'id', 'day', 'src']).w.sum().unstack('src', fill_value=0).reset_index()
    d = d.rename(columns={'agent': 'a', 'id': 'b'})
    for s in NAME_SOURCES:
        if s not in d:
            d[s] = 0.0
    d = d.rename(columns={s: 'name_' + s for s in NAME_SOURCES})
    cols = ['name_' + s for s in NAME_SOURCES]
    return undirected(d).groupby(['a', 'b', 'day'], as_index=False)[cols].sum(), d


# ---------------------------------------------------------------- assembly
def build(days, retrospective=False):
    t0 = time.time()
    act = load_activity(days)
    agents = sorted(act.agent.unique())
    U = universe(act)
    log(f'universe: {len(U)} pair-days over {U.day.nunique()} days, {len(agents)} agents')
    m, info = prepare_mentions(days, retrospective=retrospective)
    log(f'mentions: {len(m)}; {info}')
    tm = m[m.type.isin(TRACE_TYPES)].copy()
    tm['tid'] = tm['tid'].astype(str).astype('category')
    train_days = set(days) if retrospective else {d for d in days if d <= config.TRAIN_END.isoformat()}
    training_agents = act.loc[act.day.isin(train_days), 'agent'].unique()
    idf = trace_weights(tm, training_agents, train_days)
    log('p1 ...')
    cos, sync = p1_features(tm, idf, agents)
    log('p2 ...')
    hand, pou, hand_dir, pou_dir = p2_features(tm, idf)
    log('names ...')
    names, names_dir = name_features(m)
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
    log(f'features built in {time.time() - t0:.0f} s')
    extras = {'hand_directed': hand_dir, 'pou_directed': pou_dir, 'names_directed': names_dir,
              'activity': act, 'idf': idf, 'mentions': m, 'mapping_info': info,
              'feature_fit_days': sorted(train_days)}
    return F, extras


def add_labels(F, rows):
    pdays = L.pair_days(rows)
    F['y'] = [d in {x.isoformat() for x in pdays.get((a, b), ())} for a, b, d in zip(F.a, F.b, F.day)]
    return F


FEATURES = ['p1_cos', 'p1_sync', 'h_write', 'h_read', 'pou_created_commit', 'pou_created_other',
            'pou_first_commit', 'pou_first_other', 'name_comment', 'name_code', 'name_typed', 'name_gui', 'name_out']
SETS = {
    'activity only': [],
    'pipeline 1': ['p1_cos', 'p1_sync'],
    'pipeline 2': ['h_write', 'h_read', 'pou_created_commit', 'pou_created_other', 'pou_first_commit', 'pou_first_other'],
    'pipelines 1 + 2 (behaviour only)': ['p1_cos', 'p1_sync', 'h_write', 'h_read', 'pou_created_commit',
                                         'pou_created_other', 'pou_first_commit', 'pou_first_other'],
    '+ names in comments': ['name_comment'],
    '+ all names': ['name_comment', 'name_code', 'name_typed', 'name_gui', 'name_out'],
}


def model_features():
    """Yield cumulative feature sets without repeating a regularized column."""
    cols = []
    for name, extra in SETS.items():
        cols = list(dict.fromkeys(cols + extra if name.startswith('+') else extra))
        yield name, ['act_min', 'act_max'] + cols


def evaluate(F, train_days, valid_days):
    res = {'single': {}, 'models': {}}
    for c in FEATURES:
        if c in F:
            res['single'][c] = M.summary(F[c].values, F.y.values)
    res['single']['activity (act_min)'] = M.summary(F.act_min.values, F.y.values)
    tr = F.day.isin(train_days).values
    va = F.day.isin(valid_days).values
    for name, use in model_features():
        use = [c for c in use if c in F]
        X = np.log1p(F[use].clip(lower=0).values) if use else None
        model = M.Logistic(l2=1.0).fit(X[tr], F.y.values[tr])
        s = model.decision(X[va])
        res['models'][name] = {
            'auc': round(M.auc(s, F.y.values[va]), 3),
            'recall_at_10pct_fa': round(M.recall_at_fpr(s, F.y.values[va]), 3),
            'coefficients': {k: float(v) for k, v in model.coefficients(use).items()},
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true', help='also build the sealed slice (final evaluation only)')
    ap.add_argument('--retrospective', action='store_true', help='learn mappings and IDF on the whole requested period (exploratory)')
    args = ap.parse_args()
    days = []
    d = config.MAIN_START
    while d <= (config.MAIN_END if args.final else config.DEV_END):
        days.append(d.isoformat())
        d = d.fromordinal(d.toordinal() + 1)
    config.check_unsealed(days, final=args.final)
    F, extras = build(set(days), retrospective=args.retrospective)
    rows, _, _ = L.load_requests()
    F = add_labels(F, rows)
    PAIRS.mkdir(parents=True, exist_ok=True)
    tag = 'final' if args.final else ('retrospective_dev' if args.retrospective else 'dev')
    F.to_csv(PAIRS / f'features_{tag}.csv.gz', index=False)
    for k in ('hand_directed', 'pou_directed', 'names_directed'):
        extras[k].to_csv(PAIRS / f'{k}_{tag}.csv.gz', index=False)
    if args.final:
        return
    train = [x for x in days if x <= config.TRAIN_END.isoformat()]
    valid = [x for x in days if x >= config.VALID_START.isoformat()]
    log(f'pair-days {len(F)}, labelled {int(F.y.sum())} ({F.y.mean():.1%})')
    res = evaluate(F, train, valid)
    res['evaluation'] = {'mode': 'retrospective' if args.retrospective else 'prospective',
                         'feature_fit_days': extras['feature_fit_days'],
                         'mapping': extras['mapping_info'],
                         'single_feature_scope': 'descriptive whole development slice',
                         'model_scoring_scope': 'validation only'}
    with open(PAIRS / f'scores_{tag}.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('\nsingle features (whole first slice)')
    for k, v in res['single'].items():
        print(f'  {k:<22s} reach {v["reach"]:.3f}  lift {v["lift"]}  AUC {v["auc"]:.3f}  recall@10%FA {v["recall_at_10pct_fa"]:.3f}')
    print('\nlogistic models (train 6-31 Jul, validate 3-14 Aug)')
    for k, v in res['models'].items():
        print(f'  {k:<34s} AUC {v["auc"]:.3f}  recall@10%FA {v["recall_at_10pct_fa"]:.3f}')


if __name__ == '__main__':
    main()
