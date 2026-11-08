#!/usr/bin/env python3
"""Recall against the daily alert budget on the test period (17 Aug-15 Sep).

Reporting only: reads the saved frozen predictions and features, fits nothing on
the test period, and changes no detector, threshold or earlier output. Every
score ranks all test pair-days pooled over the observed days; flagging the top
k pair-days gives k / days alerts per day and a recall. Tied scores are broken
at random, so recall is the expected value (linear between tie blocks).

Scores: the generic detector (saved frozen model, `layer1_behavior`), shared
resources alone (raw rarity-weighted cosine, `p1_cos`), the frozen
activity-only model and July pair history (the same July-only score as
review_technique_transfer.py). The operating point is the generic detector's
saved alert flag, set in July at a 10% false-positive rate. Baselines are read
at the same number of alerts. Intervals resample whole days with the rules
fixed (1,000 draws, seed 20261106); they are descriptive.

Usage: python scripts/review_alert_budget.py [--output DIR]
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from swarmcollab import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
STUDY = OUT / 'generalization-20261106' / 'temporal'
KEYS = ['a', 'b', 'day']
DRAWS, SEED = 1000, 20261106
GRID_STEP = 0.5   # alerts per day
SCORES = {
    'generic_detector': 'Generic detector',
    'shared_resources': 'Shared resources alone',
    'pair_history': 'July pair history',
    'activity': 'Activity only',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(series):
    return series.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()


def pair_history(dev):
    """July request rate of each pair, shrunk toward July prevalence (alpha = 1)."""
    july = dev[dev.day <= '2026-07-31']
    prevalence = july.y.mean()
    grouped = july.assign(pair=july.a + '|' + july.b).groupby('pair').y.agg(['sum', 'count'])

    def score(frame):
        pair = frame.a + '|' + frame.b
        k = pair.map(grouped['sum']).fillna(0.0)
        n = pair.map(grouped['count']).fillna(0.0)
        return ((k + prevalence) / (n + 1.0)).to_numpy(float)
    return score


def budget_curve(score, y):
    """Cumulative alerts and true positives at the end of each tie block, best first."""
    order = np.argsort(-score, kind='mergesort')
    ordered, hits = score[order], y[order]
    ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    alerts = np.r_[0, ends + 1].astype(float)
    found = np.r_[0, np.cumsum(hits)[ends]].astype(float)
    return alerts, found


def flag_probability(score, k):
    """Probability that each row is among the top k when ties are broken at random."""
    order = np.sort(score)[::-1]
    cut = order[k - 1]
    above = score > cut
    tied = score == cut
    remaining = k - int(above.sum())
    return above.astype(float) + tied * (remaining / tied.sum())


def interval(days, values, rng):
    """Percentile interval of a ratio of sums over whole-day resamples."""
    unique = np.unique(days)
    index = {day: np.flatnonzero(days == day) for day in unique}
    numerator, denominator = values
    draws = []
    for _ in range(DRAWS):
        rows = np.concatenate([index[d] for d in rng.choice(unique, size=len(unique), replace=True)])
        if denominator[rows].sum() > 0:
            draws.append(numerator[rows].sum() / denominator[rows].sum())
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=OUT / 'alert-budget')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; refusing to overwrite a reporting run.')

    dev_path = OUT / 'research-layers' / 'features_dev.csv.gz'
    features_path = STUDY / 'features.csv.gz'
    predictions_path = STUDY / 'predictions.csv.gz'
    reference_path = OUT / 'technique-transfer' / 'summary.json'
    inputs = {str(p.relative_to(ROOT)): sha256(p)
              for p in (dev_path, features_path, predictions_path, reference_path)}

    dev = pd.read_csv(dev_path, usecols=KEYS + ['y'])
    dev['y'] = boolean(dev.y)
    saved = pd.read_csv(predictions_path)
    saved['y'] = boolean(saved.y)
    features = pd.read_csv(features_path, usecols=KEYS + ['p1_cos'])
    frame = saved.merge(features, on=KEYS, validate='one_to_one')
    frame['generic_detector'] = frame.layer1_behavior
    frame['shared_resources'] = frame.p1_cos
    frame['pair_history'] = pair_history(dev)(frame)
    y = frame.y.to_numpy(bool)
    days = frame.day.to_numpy()
    n_days, n_pos = len(np.unique(days)), int(y.sum())

    # The recomputed rankings must reproduce the saved Table 1 values.
    reference = json.loads(reference_path.read_text())
    expected = {'generic_detector': reference['combined_models']['layer1_behavior']['temporal']['auroc'],
                'shared_resources': reference['techniques']['co_use']['temporal']['auroc'],
                'pair_history': reference['other']['pair_history']['temporal']['auroc'],
                'activity': reference['combined_models']['activity']['temporal']['auroc']}
    aurocs = {name: metrics.auc(frame[name].to_numpy(float), y) for name in SCORES}
    for name, value in aurocs.items():
        if not np.isclose(value, expected[name], atol=1e-9):
            raise ValueError(f'{name} AUROC {value} differs from the saved {expected[name]}')

    flags = boolean(frame['layer1_behavior__alert'])
    k = int(flags.sum())
    rng = np.random.default_rng(SEED)
    operating = dict(
        rule='Generic detector alert flag saved with the frozen July threshold (10% July FPR target).',
        alerts=k, alerts_per_day=k / n_days, recall=float(flags[y].mean()),
        precision=float(y[flags].mean()), false_positive_rate=float(flags[~y].mean()),
        recall_interval_95=interval(days, (flags * y, y.astype(float)), rng),
        precision_interval_95=interval(days, (flags * y, flags.astype(float)), rng))

    grid = np.arange(0.0, len(frame) / n_days + GRID_STEP / 2, GRID_STEP)
    curves = pd.DataFrame({'alerts_per_day': grid,
                           'random': np.minimum(grid * n_days / len(frame), 1.0)})
    at_budget = {}
    for name, label in SCORES.items():
        score = frame[name].to_numpy(float)
        alerts, found = budget_curve(score, y)
        curves[name] = np.interp(grid * n_days, alerts, found) / n_pos
        if name == 'generic_detector':
            probability = flags.astype(float)
        else:
            probability = flag_probability(score, k)
        at_budget[name] = dict(
            label=label, auroc=aurocs[name],
            recall=float(probability[y].sum() / n_pos),
            precision=float(probability[y].sum() / probability.sum()),
            recall_interval_95=interval(days, (probability * y, y.astype(float)), rng))

    args.output.mkdir(parents=True)
    curves_path = args.output / 'curves.csv'
    curves.to_csv(curves_path, index=False, float_format='%.6f')
    summary = dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Reporting only. Saved frozen test-period predictions; no fitting or threshold change.',
        period=['2026-08-17', '2026-09-15'], pair_days=len(frame), request_pair_days=n_pos,
        observed_days=n_days, pairs_per_day=len(frame) / n_days, draws=DRAWS, seed=SEED,
        ties='Broken at random; recall is the expected value.',
        operating_point=operating, at_operating_budget=at_budget,
        inputs_sha256=inputs, code_sha256=sha256(__file__),
        outputs_sha256={'curves.csv': sha256(curves_path)})
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'operating_point': {k_: v for k_, v in operating.items() if k_ != 'rule'},
                      'at_operating_budget': {n: {'recall': round(v['recall'], 4),
                                                  'interval': [round(x, 4) for x in v['recall_interval_95']],
                                                  'precision': round(v['precision'], 4)}
                                              for n, v in at_budget.items()}}, indent=1))


if __name__ == '__main__':
    main()
