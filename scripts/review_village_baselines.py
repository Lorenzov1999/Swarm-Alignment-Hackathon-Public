"""Reporting-only checks of saved Village baselines after Claude's review.

No raw trace loading, model refitting, threshold changes, or reserved-period
access. The July request-history score is an explicitly post-hoc diagnostic.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from swarmcollab.metrics import auc
from swarmcollab.transfer_scoring import average_precision


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/generalization-20261106'
PERIODS = {'development_validation': ('2026-08-03', '2026-08-14'),
           'temporal': ('2026-08-17', '2026-09-15'),
           'shared_goals': ('2026-06-01', '2026-07-05')}
MODELS = ('activity', 'historical_behavior', 'general_existing_provenance',
          'layer1_behavior', 'layer1_plus_layer2_behavior',
          'layer1_with_attribution', 'layer1_plus_layer2_with_attribution')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def boolean(value):
    if value in ('True', '1', '1.0', True):
        return True
    if value in ('False', '0', '0.0', False):
        return False
    raise ValueError('Expected an explicit binary label')


def canonical(row):
    a, b = row['a'], row['b']
    if not isinstance(a, str) or not isinstance(b, str) or not a or not b or a == b:
        raise ValueError('Expected two distinct nonempty actor identifiers')
    return tuple(sorted((a, b)))


def validate_rows(rows, lower, upper):
    seen = set()
    if not rows:
        raise ValueError('Empty pair-day table')
    for row in rows:
        day = row['day']
        parsed = dt.date.fromisoformat(day)
        if parsed.isoformat() != day or not lower <= day <= upper:
            raise ValueError('Pair-day outside authorized period')
        key = canonical(row) + (day,)
        if key in seen:
            raise ValueError('Duplicate undirected pair-day')
        seen.add(key)
        boolean(row['y'])


def load(path):
    with gzip.open(path, 'rt', newline='') as stream:
        return list(csv.DictReader(stream))


def fit_pair_history(training):
    validate_rows(training, '2026-07-06', '2026-07-31')
    active, positive = Counter(), Counter()
    for row in training:
        pair = canonical(row)
        active[pair] += 1
        positive[pair] += boolean(row['y'])
    return dict(active=active, positive=positive,
                prevalence=sum(positive.values()) / sum(active.values()),
                rows=len(training), positives=sum(positive.values()))


def history_scores(rows, fitted, alpha=1.):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('Smoothing must be finite and nonnegative')
    # Evaluation labels are intentionally never read here.
    values = []
    for row in rows:
        pair = canonical(row)
        n, k = fitted['active'][pair], fitted['positive'][pair]
        values.append((k + alpha * fitted['prevalence']) / (n + alpha)
                      if n + alpha else fitted['prevalence'])
    return np.array(values)


def within_pair(rows, scores):
    """Compare only a positive day and a negative day of the SAME actor pair.

    The weighted estimator weights each pair by its positive-negative day
    comparisons. Macro AUROC weights each classifiable pair equally. Neither
    estimator pools residual scores across different pairs.
    """
    score = np.asarray(scores, float)
    if score.shape != (len(rows),) or not np.isfinite(score).all():
        raise ValueError('Need aligned finite scores')
    days = sorted({row['day'] for row in rows})
    day_index = {day: i for i, day in enumerate(days)}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[canonical(row)].append(i)
    numerator = np.zeros((len(days), len(days)))
    denominator = np.zeros_like(numerator)
    details = []
    for pair, indexes in sorted(groups.items()):
        pos = [i for i in indexes if boolean(rows[i]['y'])]
        neg = [i for i in indexes if not boolean(rows[i]['y'])]
        if not pos or not neg:
            continue
        wins = 0.
        for i in pos:
            for j in neg:
                value = float(score[i] > score[j]) + .5 * float(score[i] == score[j])
                x, y = day_index[rows[i]['day']], day_index[rows[j]['day']]
                numerator[x, y] += value
                denominator[x, y] += 1
                wins += value
        details.append(dict(a=pair[0], b=pair[1], rows=len(indexes), positives=len(pos),
                            negatives=len(neg), comparisons=len(pos) * len(neg),
                            concordance=wins / (len(pos) * len(neg))))
    total = float(denominator.sum())
    report = dict(eligible_pairs=len(details), eligible_pair_days=sum(d['rows'] for d in details),
                  positive_days=sum(d['positives'] for d in details),
                  positive_negative_comparisons=int(total),
                  comparison_weighted_auc=float(numerator.sum() / total) if total else None,
                  macro_pair_auc=float(np.mean([d['concordance'] for d in details])) if details else None,
                  pair_results=details,
                  interpretation='Request-positive versus request-negative days within the same pair only; '
                                 'descriptive temporal discrimination, not event ordering, causal influence or deployment latency.')
    return report, days, numerator, denominator


def centered_pooled_auc(rows, scores):
    """Reproduce the reviewer diagnostic, retaining its nonconditional meaning."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[canonical(row)].append(i)
    values, targets = [], []
    for indexes in groups.values():
        labels = [boolean(rows[i]['y']) for i in indexes]
        if not any(labels) or all(labels):
            continue
        block = np.asarray(scores)[indexes]
        values.extend(block - block.mean())
        targets.extend(labels)
    return float(auc(values, targets)) if values else None


def zscore(values):
    values = np.asarray(values, float)
    sd = values.std()
    return (values - values.mean()) / sd if sd else np.zeros_like(values)


def interval(draws):
    return dict(valid_draws=len(draws), interval_95=np.quantile(draws, [.025, .975]).tolist() if draws else None)


def evaluate(rows, fitted, repetitions=1000, seed=20261107):
    y = np.array([boolean(row['y']) for row in rows])
    values = {name: np.array([float(row[name]) for row in rows]) for name in MODELS}
    values['july_pair_history'] = history_scores(rows, fitted)
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Saved predictions must be finite')
    point = {name: dict(auc=float(auc(score, y)), average_precision=float(average_precision(score, y)))
             for name, score in values.items()}
    within, days, numerator, denominator = within_pair(rows, values['layer1_behavior'])
    blocks = [np.array([i for i, row in enumerate(rows) if row['day'] == day]) for day in days]
    contrasts = [('activity', 'layer1_behavior'), ('july_pair_history', 'layer1_behavior'),
                 ('historical_behavior', 'layer1_behavior'), ('historical_behavior', 'general_existing_provenance'),
                 ('layer1_behavior', 'layer1_plus_layer2_behavior')]
    draws = {a + '__to__' + b: {'auc': [], 'average_precision': []} for a, b in contrasts}
    within_draws = []
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        choice = rng.integers(0, len(days), len(days))
        indexes = np.concatenate([blocks[i] for i in choice])
        target = y[indexes]
        if target.any() and not target.all():
            metrics = {name: (auc(score[indexes], target), average_precision(score[indexes], target))
                       for name, score in values.items() if name in {x for pair in contrasts for x in pair}}
            for a, b in contrasts:
                for k, field in enumerate(('auc', 'average_precision')):
                    draws[a + '__to__' + b][field].append(float(metrics[b][k] - metrics[a][k]))
        weights = np.bincount(choice, minlength=len(days))
        support = float(weights @ denominator @ weights)
        if support:
            within_draws.append(float(weights @ numerator @ weights / support))
    comparisons = {}
    for a, b in contrasts:
        comparisons[a + '__to__' + b] = {field: dict(estimate=point[b][field] - point[a][field],
            **interval(draws[a + '__to__' + b][field])) for field in ('auc', 'average_precision')}
    within['day_block_interval'] = interval(within_draws)
    sensitivity = {str(alpha): float(auc(history_scores(rows, fitted, alpha), y)) for alpha in (0, 1, 10, 100)}
    naive = zscore(values['july_pair_history']) + zscore(values['layer1_behavior'])
    seen = sum(fitted['active'][canonical(row)] > 0 for row in rows)
    return dict(rows=len(rows), positives=int(y.sum()), days=len(days), july_active_pair_rows=seen,
                metrics=point, comparisons=comparisons, within_pair=within,
                july_history_smoothing_auc_sensitivity=sensitivity,
                reviewer_centered_pooled_auc=centered_pooled_auc(rows, values['layer1_behavior']),
                posthoc_target_standardized_sum=dict(auc=float(auc(naive, y)),
                    average_precision=float(average_precision(naive, y)),
                    interpretation='Reviewer diagnostic only: equal weight sum with each score standardized on the target '
                    'period. No direct target-label fitting, but target covariate calibration and post-hoc selection; '
                    'not a frozen July model or a validated improvement.'),
                bootstrap=dict(repetitions=repetitions, seed=seed,
                    interpretation='Paired whole-day resampling with fixed predictions and July prior. Descriptive intervals '
                    'omit fitting and post-hoc selection uncertainty and do not remove across-day/goal/actor dependence.'))


def run(output, repetitions=1000):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a prior diagnostic report')
    freeze_path = STUDY / 'freeze-manifest.json'
    frozen = json.loads(freeze_path.read_text())
    village_lock_path = STUDY / 'village-results-lock.json'
    village = json.loads(village_lock_path.read_text())
    inputs = {str(freeze_path.relative_to(ROOT)): sha(freeze_path),
              str(village_lock_path.relative_to(ROOT)): sha(village_lock_path)}
    def original(relative):
        path = STUDY / 'frozen' / relative
        if sha(path) != frozen['files'][relative]['sha256']:
            raise ValueError('Frozen training artifact hash mismatch: ' + relative)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return path
    def target(relative):
        path = STUDY / relative
        if sha(path) != village['files_sha256'][relative]:
            raise ValueError('Frozen target artifact hash mismatch: ' + relative)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return path
    features = load(original('outputs/pairs/features_dev.csv.gz'))
    validate_rows(features, '2026-07-06', '2026-08-14')
    training = [row for row in features if row['day'] <= '2026-07-31']
    fitted = fit_pair_history(training)
    model_spec = json.loads(original('outputs/research-layers/scores.json').read_text())
    paths = {'development_validation': original('outputs/research-layers/predictions_dev.csv.gz'),
             'temporal': target('temporal/predictions.csv.gz'),
             'shared_goals': target('shared_goals/predictions.csv.gz')}
    report = dict(schema='posthoc-village-baseline-review-v1', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  scope='Existing frozen pair-day tables only; no raw trace or reserved-period access.',
                  training=dict(start='2026-07-06', end='2026-07-31', rows=fitted['rows'], positives=fitted['positives'],
                                active_pairs=len(fitted['active']), prevalence=fitted['prevalence']),
                  baseline_definitions={name: dict(features=model_spec['models'][name]['features'],
                        status='Already July-fitted and frozen before target scoring; re-reported here.') for name in MODELS},
                  pair_history_definition=dict(formula='(July positive pair-days + alpha * July global prevalence) / (July active pair-days + alpha)',
                    alpha=1., unseen_pair='July global prevalence', status='Post-hoc diagnostic selected after reviewer inspected target results; no target labels in fitting.',
                    june='Uses later July pair labels, so this is a backward-transfer/later-history diagnostic, not a historically available June monitor.'),
                  periods={})
    for period, path in paths.items():
        rows = load(path)
        validate_rows(rows, *PERIODS[period])
        report['periods'][period] = evaluate(rows, fitted, repetitions)
    report.update(input_sha256=inputs, code_sha256={name: sha(ROOT / name) for name in
        ['scripts/review_village_baselines.py', 'src/swarmcollab/metrics.py', 'src/swarmcollab/transfer_scoring.py']},
        models_refitted=False, frozen_predictions_changed=False, true_cooperation_labels=False,
        limitations=['Within-pair concordance is not pooled centered-score AUC and is not evidence of within-day onset.',
                     'An unsupervised-looking score sum chosen after outcomes is exploratory, not confirmatory.',
                     'Historical behavior is the frozen corrected/refitted feature comparator, not an untouched original implementation.'])
    for relative, digest in inputs.items():
        if sha(ROOT / relative) != digest:
            raise ValueError('Input changed during reporting')
    output.mkdir(parents=True)
    save(output / 'summary.json', report)
    rows = []
    for period, value in report['periods'].items():
        for model, metric in value['metrics'].items():
            rows.append(dict(period=period, model=model, **metric, rows=value['rows'], positives=value['positives']))
    with (output / 'baseline-comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save(output / 'manifest.json', dict(created_utc=report['created_utc'], inputs_sha256=inputs,
         outputs_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file()},
         code_sha256=report['code_sha256']))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/claude-review-20261107/baselines')
    parser.add_argument('--repetitions', type=int, default=1000)
    args = parser.parse_args()
    if args.repetitions < 0:
        parser.error('repetitions must be nonnegative')
    result = run(args.output, args.repetitions)
    print(json.dumps({name: value['metrics'] for name, value in result['periods'].items()}, indent=2))


if __name__ == '__main__':
    main()
