"""Stage F: descriptive request-label recall by observed trace visibility.

Visibility of an agent on a day = how many of its working steps that day name at
least one trace (as a count, and as a share). Computed without labels. A pair-day
is as visible as its less visible agent.

Pairs are split into thirds by visibility, then whole-day score recall is
summarized at a global 10% validation false-alarm threshold. This module does
not remove observations or run a controlled hiding experiment.

Run: python -I -m swarmcollab.visibility   (first slice only)
"""
import json

import numpy as np
import pandas as pd

from . import config
from . import metrics as M
from . import pairs as P

VIS = config.OUT / 'visibility'


def agent_visibility(act):
    tot = act.groupby('agent')[['working', 'trace_steps']].sum()
    return (tot.trace_steps / tot.working.clip(lower=1)).rename('visibility')


def pair_visibility(F, act):
    """Per pair-day: the less visible agent's trace steps that day, as a count and as a share of its working steps.

    Visibility varies a lot from day to day, so it is measured per agent-day; an
    agent-level average hides the effect."""
    ad = act.set_index(['agent', 'day'])
    share = ad.trace_steps / ad.working.clip(lower=1)
    count = ad.trace_steps
    ka, kb = list(zip(F.a, F.day)), list(zip(F.b, F.day))
    return (np.minimum(count.reindex(ka).values, count.reindex(kb).values),
            np.minimum(share.reindex(ka).values, share.reindex(kb).values))


def recall_by_visibility(F, score, vis, valid_mask):
    """Recall at one global threshold (10% false alarms on validation negatives), per visibility third."""
    y = F.y.values
    thr = np.quantile(score[valid_mask & ~y], 0.9)
    cuts = np.quantile(vis[valid_mask], [1 / 3, 2 / 3])
    grp = np.digitize(vis, cuts, right=True)
    out = []
    for g, name in enumerate(('lowest third', 'middle third', 'highest third')):
        sel = valid_mask & (grp == g)
        yy = y[sel]
        out.append({'group': name,
                    'range': [None if g == 0 else round(float(cuts[g - 1]), 3), None if g == 2 else round(float(cuts[g]), 3)],
                    'pair_days': int(sel.sum()), 'labelled': int(yy.sum()), 'base_rate': round(float(yy.mean()), 3),
                    'recall_at_10pct_fa': round(float((score[sel][yy] > thr).mean()), 3) if yy.any() else None,
                    'auc_within_group': round(M.auc(score[sel], yy), 3) if 0 < yy.sum() < len(yy) else None})
    return out


def main():
    days = [d.isoformat() for d in pd.date_range(config.DEV_START, config.DEV_END).date]
    config.check_unsealed(days)
    F = pd.read_csv(P.PAIRS / 'features_dev.csv.gz')
    act = P.load_activity(set(days))
    vis_count, vis_share = pair_visibility(F, act)
    train = [d for d in days if d <= config.TRAIN_END.isoformat()]
    valid = [d for d in days if d >= config.VALID_START.isoformat()]
    tr, va = F.day.isin(train).values, F.day.isin(valid).values
    res = {'agent_visibility_whole_slice': agent_visibility(act).round(3).sort_values().to_dict()}
    sets = {'activity only': [],
            'behaviour only': P.SETS['pipelines 1 + 2 (behaviour only)'],
            'with names': P.SETS['pipelines 1 + 2 (behaviour only)'] + P.SETS['+ all names']}
    for label, cols in sets.items():
        use = ['act_min', 'act_max'] + cols
        X = np.log1p(F[use].clip(lower=0).values)
        score = M.Logistic().fit(X[tr], F.y.values[tr]).decision(X)
        res[label] = {'by_trace_steps_that_day': recall_by_visibility(F, score, vis_count, va),
                      'by_share_of_steps_that_day': recall_by_visibility(F, score, vis_share, va)}
    VIS.mkdir(parents=True, exist_ok=True)
    with open(VIS / 'recall_by_visibility_dev.json', 'w') as f:
        json.dump(res, f, indent=1)
    for label in sets:
        for k, groups in res[label].items():
            print(f'{label:<15s} {k:<28s} ' + ' | '.join(f"{g['recall_at_10pct_fa']:.2f}" for g in groups))


if __name__ == '__main__':
    main()
