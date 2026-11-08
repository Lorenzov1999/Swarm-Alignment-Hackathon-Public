"""Reproduce the paper's single shared-resource score and its diagnostic checks.

Run: python3 scripts/check_shared_resources.py
Only saved July-derived parameters, test-period features, scores, and request
annotations are used. No raw logs are opened and no model is refitted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from swarmcollab import metrics
from swarmcollab.transfer_scoring import average_precision, score_frozen

KEYS = ['a', 'b', 'day']
MODELS = ('shared_resources', 'activity')
DRAWS, SEED = 1000, 20261106


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_frame(path, columns):
    frame = pd.read_csv(path, usecols=columns)
    if frame[KEYS].duplicated().any() or not (frame.a < frame.b).all():
        raise ValueError('Expected unique canonical agent-pair/day rows')
    if not frame.day.between('2026-08-17', '2026-09-15').all():
        raise ValueError('A saved row falls outside the authorized test window')
    return frame


def ranking_parts(score, y):
    _, group = np.unique(score, return_inverse=True)
    size = group.max() + 1
    def evaluate(weights):
        positive = np.bincount(group, weights=weights * y, minlength=size)
        negative = np.bincount(group, weights=weights * (~y), minlength=size)
        n1, n0 = positive.sum(), negative.sum()
        if not n1 or not n0:
            return np.array([np.nan, np.nan])
        auc = (positive * (np.cumsum(negative) - .5 * negative)).sum() / (n1 * n0)
        hits = np.cumsum(positive[::-1])
        total = np.cumsum((positive + negative)[::-1])
        precision = np.divide(hits, total, out=np.zeros_like(hits), where=total > 0)
        ap = (positive[::-1] * precision).sum() / n1
        return np.array([auc, ap])
    return evaluate


def interval(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError('No valid bootstrap draws')
    return [float(x) for x in np.percentile(finite, [2.5, 97.5])]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def operating(flags, y, weights):
    tp = (weights * flags * y).sum()
    alerts = (weights * flags).sum()
    return np.array([ratio(tp, (weights * y).sum()), ratio(tp, alerts),
                     ratio((weights * flags * (~y)).sum(), (weights * (~y)).sum())])


def top_k_probability(score, k):
    cutoff = np.sort(score)[-k]
    above, tied = score > cutoff, score == cutoff
    return above.astype(float) + tied * ((k - above.sum()) / tied.sum())


def grouped_comparisons(frame, columns, day_index):
    positive_rows, negative_rows, group_sizes, group_aucs = [], [], [], []
    for _, group in frame.groupby(columns, sort=True):
        positive = group.index[group.y].to_numpy()
        negative = group.index[~group.y].to_numpy()
        if not len(positive) or not len(negative):
            continue
        left, right = np.repeat(positive, len(negative)), np.tile(negative, len(positive))
        positive_rows.extend(left); negative_rows.extend(right)
        group_sizes.append(len(left))
        group_aucs.append([float(((frame[m].to_numpy()[left] > frame[m].to_numpy()[right]) +
                                  .5 * (frame[m].to_numpy()[left] == frame[m].to_numpy()[right])).mean())
                           for m in MODELS])
    left, right = np.array(positive_rows), np.array(negative_rows)
    wins = np.array([(frame[m].to_numpy()[left] > frame[m].to_numpy()[right]) +
                    .5 * (frame[m].to_numpy()[left] == frame[m].to_numpy()[right]) for m in MODELS])
    def evaluate(day_weights):
        # Resampled copies of a day are separate within-day groups. For a
        # within-pair contrast, all sampled positive/negative days are compared.
        weight = day_weights[day_index[left]]
        if columns != ['day']:
            weight = weight * day_weights[day_index[right]]
        return np.array([ratio((values * weight).sum(), weight.sum()) for values in wins])
    details = dict(eligible_groups=len(group_sizes), comparisons=len(left),
                   equal_group_auroc={m: float(np.mean(group_aucs, axis=0)[i]) for i, m in enumerate(MODELS)})
    return evaluate, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT,
                        help='Root containing saved outputs (default: this repository)')
    for name in ('features', 'predictions', 'parameters', 'annotations', 'annotation-manifest'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--skip-followthrough', action='store_true',
                        help='Omit the secondary annotation analysis when its private audit is unavailable')
    parser.add_argument('--mode', choices=['current-run', 'frozen-paper'], default='current-run',
                        help='Describe the supplied score provenance; input hashes are checked in both modes')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/current-paper/provenance/shared-resource-results.json')
    args = parser.parse_args(argv)
    relative = {
        'features': 'outputs/generalization-20261106/temporal/features.csv.gz',
        'predictions': 'outputs/generalization-20261106/temporal/predictions.csv.gz',
        'parameters': 'outputs/technique-transfer/summary.json',
        'annotations': 'outputs/claude-review-20261107/labels-visibility/request-positive-pair-days.csv.gz',
        'annotation_manifest': 'outputs/claude-review-20261107/labels-visibility/manifest.json'}
    paths = {name: getattr(args, name) or args.input_root / path for name, path in relative.items()
             if not (args.skip_followthrough and name.startswith('annotation'))}
    reference = json.loads(paths['parameters'].read_text())
    for name in ('features', 'predictions'):
        if sha(paths[name]) != reference['inputs_sha256'][relative[name]]:
            raise ValueError('Saved test input differs from the original technique evaluation')
    if not args.skip_followthrough:
        annotation_manifest = json.loads(paths['annotation_manifest'].read_text())
        if sha(paths['annotations']) != annotation_manifest['outputs_sha256']['request-positive-pair-days.csv.gz']:
            raise ValueError('Follow-through annotations differ from the saved audit')
    features = checked_frame(paths['features'], KEYS + ['p1_cos', 'act_min', 'act_max'])
    saved = checked_frame(paths['predictions'], KEYS + ['y', 'activity', 'activity__alert'])
    models = {'shared_resources': reference['fitted']['co_use'], 'activity': reference['fitted']['activity']}
    scored = score_frozen(features, {'models': models}, period='temporal')
    frame = scored.merge(saved, on=KEYS, validate='one_to_one', suffixes=('', '_saved'))
    frame = frame.merge(features[KEYS + ['p1_cos']], on=KEYS, validate='one_to_one')
    if len(frame) != len(features) or len(frame) != len(saved):
        raise ValueError('Incomplete score/feature join')
    if not np.allclose(frame.activity, frame.activity_saved, rtol=0, atol=1e-12):
        raise ValueError('Activity score changed')
    if not np.array_equal(frame.activity__alert, frame.activity__alert_saved):
        raise ValueError('Activity alert threshold changed')
    if frame.y.dtype != bool:
        raise ValueError('Expected binary request labels')
    y = frame.y.to_numpy(bool)
    days, day_index = np.unique(frame.day, return_inverse=True)
    rng = np.random.default_rng(SEED)
    weights = np.array([np.bincount(rng.choice(len(days), size=len(days), replace=True), minlength=len(days))
                        for _ in range(DRAWS)])
    ones = np.ones(len(frame))
    rank = {m: ranking_parts(frame[m].to_numpy(), y) for m in MODELS}
    primary = {m: rank[m](ones) for m in MODELS}
    primary_draws = {m: np.array([rank[m](w[day_index]) for w in weights]) for m in MODELS}
    for m, original in [('shared_resources', 'co_use'), ('activity', 'activity')]:
        if not np.isclose(primary[m][0], reference['techniques'][original]['temporal']['auroc'], atol=1e-12):
            raise ValueError('Primary AUROC differs from the earlier evaluation')
        if not np.isclose(primary[m][1], average_precision(frame[m].to_numpy(), y), atol=1e-12):
            raise ValueError('Average precision implementation disagrees')
    if models['shared_resources']['fitted']['w'][1] <= 0:
        raise ValueError('Single-resource score no longer preserves increasing similarity')
    if not np.array_equal(np.unique(frame.p1_cos, return_inverse=True)[1],
                          np.unique(frame.shared_resources, return_inverse=True)[1]):
        raise ValueError('Score transformation changes ranking or ties')
    flags = frame.shared_resources__alert.to_numpy(bool)
    k = int(flags.sum())
    probabilities = {m: top_k_probability(frame[m].to_numpy(), k) for m in MODELS}
    if not np.array_equal(probabilities['shared_resources'], flags.astype(float)):
        raise ValueError('Top-budget ranking disagrees with the July threshold')
    op = operating(flags, y, ones)
    op_draws = np.array([operating(flags, y, w[day_index]) for w in weights])
    within = {}
    for name, columns in [('within_pair', ['a', 'b']), ('within_day', ['day'])]:
        compare, detail = grouped_comparisons(frame, columns, day_index)
        values = compare(np.ones(len(days)))
        draws = np.array([compare(w) for w in weights])
        detail['models'] = {m: dict(auroc=float(values[i]), interval_95=interval(draws[:, i])) for i, m in enumerate(MODELS)}
        detail['difference'] = dict(auroc=float(values[0] - values[1]), interval_95=interval(draws[:, 0] - draws[:, 1]))
        within[name] = detail
    if not args.skip_followthrough:
        annotation = checked_frame(paths['annotations'], KEYS + ['y', 'requests', 'followed_group'])
        if len(annotation) != int(y.sum()) or not annotation.y.all():
            raise ValueError('Unexpected request-positive annotation population')
        joined = annotation.merge(frame.assign(row=np.arange(len(frame)))[KEYS + ['row']], on=KEYS, validate='one_to_one')
        if len(joined) != len(annotation):
            raise ValueError('Missing annotated pair-day')
        single = joined[joined.requests == 1]
        if not set(single.followed_group) <= {'yes', 'no', 'ambiguous'}:
            raise ValueError('Unexpected follow-through annotation')
        selected = single[single.followed_group.isin(['yes', 'no'])]
        indices = selected.row.to_numpy(int)
        followed = (selected.followed_group == 'yes').to_numpy(bool)
        counts = Counter(single.followed_group)
        if counts != Counter(yes=157, no=65, ambiguous=33):
            raise ValueError('Single-request population differs from the earlier audit')
        follow_ranks = {m: ranking_parts(frame[m].to_numpy()[indices], followed) for m in MODELS}
        follow_values = {m: follow_ranks[m](np.ones(len(indices)))[0] for m in MODELS}
        follow_draws = {m: np.array([follow_ranks[m](w[day_index[indices]])[0] for w in weights]) for m in MODELS}
    fitted = models['shared_resources']['fitted']
    raw_threshold = np.expm1(fitted['mu'][0] + fitted['sd'][0] *
                            (models['shared_resources']['threshold'] - fitted['w'][0]) / fitted['w'][1])
    report = {
        'reproduction_mode': args.mode,
        'score_version': ('Original pre-correction frozen paper scores' if args.mode == 'frozen-paper'
                          else 'Supplied run; not asserted to reproduce the original frozen paper scores'),
        'parameter_training_period': ['2026-07-06', '2026-07-31'],
        'scope': 'Post-hoc diagnostics for the saved July-derived single shared-resource score on 17 August–15 September. No fitting, raw-log inspection, or threshold changes.',
        'period': ['2026-08-17', '2026-09-15'], 'timezone': 'America/Los_Angeles',
        'pair_days': len(frame), 'request_pair_days': int(y.sum()), 'observed_days': len(days),
        'pairs_per_day': len(frame) / len(days), 'request_prevalence': float(y.mean()),
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'whole observed day', 'interval': '2.5/97.5 percentiles; descriptive, fitted scores held fixed',
                      'within_pair': 'Day multiplicities weight both members of each positive/negative comparison.',
                      'within_day': 'Each resampled day copy contributes its original within-day comparisons once.',
                      'followthrough': 'The same 22-day resamples, retaining only single-request yes/no rows.',
                      'same_budget': 'Fixed original top-k inclusion probabilities, with random tie breaking; each bootstrap repeats whole-day review workloads, not a new top-k selection.'},
        'model_checks': {'features': ['p1_cos'], 'positive_coefficient': fitted['w'][1],
                         'raw_and_fitted_score_ranks_and_ties_identical': True,
                         'activity_scores_and_alerts_match_saved_predictions': True,
                         'july_score_threshold': models['shared_resources']['threshold'],
                         'equivalent_cosine_threshold': float(raw_threshold)},
        'primary': {m: {'auroc': float(primary[m][0]), 'auroc_interval_95': interval(primary_draws[m][:, 0]),
                        'average_precision': float(primary[m][1]), 'average_precision_interval_95': interval(primary_draws[m][:, 1])} for m in MODELS},
        'primary_difference': {'auroc': float(primary[MODELS[0]][0] - primary[MODELS[1]][0]),
                               'interval_95': interval(primary_draws[MODELS[0]][:, 0] - primary_draws[MODELS[1]][:, 0])},
        **within,
        **({'followthrough': {'request_pair_days': len(annotation), 'single_request_pair_days': len(single),
                          'compared_pair_days': len(indices), 'yes': counts['yes'], 'no': counts['no'], 'ambiguous_excluded': counts['ambiguous'],
                          'models': {m: {'auroc': float(follow_values[m]), 'interval_95': interval(follow_draws[m]),
                                         'alerts_yes': int(frame[m + '__alert'].to_numpy()[indices][followed].sum()),
                                         'alerts_no': int(frame[m + '__alert'].to_numpy()[indices][~followed].sum()),
                                         'alert_share_yes': float(frame[m + '__alert'].to_numpy()[indices][followed].mean()),
                                         'alert_share_no': float(frame[m + '__alert'].to_numpy()[indices][~followed].mean())} for m in MODELS},
                          'difference': {'auroc': float(follow_values[MODELS[0]] - follow_values[MODELS[1]]),
                                         'interval_95': interval(follow_draws[MODELS[0]] - follow_draws[MODELS[1]])}}} if not args.skip_followthrough else
           {'followthrough': {'status': 'omitted; supporting annotations not supplied'}}),
        'operating_point': {'rule': 'Saved single-feature July threshold targeting 10% FPR on training negatives; strict greater-than, no target recalibration.',
                            'alerts': k, 'alerts_per_day': k / len(days), 'true_positives': int((flags & y).sum()),
                            'false_positives': int((flags & ~y).sum()), 'false_negatives': int((~flags & y).sum()), 'true_negatives': int((~flags & ~y).sum()),
                            **{name: float(op[i]) for i, name in enumerate(('recall', 'precision', 'false_positive_rate'))},
                            **{name + '_interval_95': interval(op_draws[:, i]) for i, name in enumerate(('recall', 'precision', 'false_positive_rate'))}},
        'at_operating_budget': {},
        'inputs_sha256': {name: sha(p) for name, p in paths.items()},
        'script_sha256': sha(Path(__file__)),
        'interpretation': 'Recorded requests and website follow-through annotations are the reference labels; the diagnostics do not independently establish executed cooperation. Within-pair and within-day comparisons hold pair or day identity fixed, not activity counts.'}
    budget_draws = {}
    for m in MODELS:
        values = operating(probabilities[m], y, ones)
        draws = np.array([operating(probabilities[m], y, w[day_index]) for w in weights])
        report['at_operating_budget'][m] = {'recall': float(values[0]), 'precision': float(values[1]), 'recall_interval_95': interval(draws[:, 0])}
        budget_draws[m] = draws[:, 0]
    report['at_operating_budget']['difference'] = {
        'recall': report['at_operating_budget'][MODELS[0]]['recall'] - report['at_operating_budget'][MODELS[1]]['recall'],
        'interval_95': interval(budget_draws[MODELS[0]] - budget_draws[MODELS[1]])}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('primary', 'within_pair', 'within_day', 'followthrough', 'operating_point', 'at_operating_budget')}, indent=2))


if __name__ == '__main__':
    main()
