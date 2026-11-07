"""Run frozen secondary transfer analyses without modifying primary outputs.

Examples (set BLAS threads to one in the invoking environment):
  python -I -m swarmcollab.secondary_run --stage fit --study outputs/generalization-20261106
  python -I -m swarmcollab.secondary_run --stage apply --period temporal --study outputs/generalization-20261106
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, metrics, pairs
from . import secondary_transfer as secondary
from .transfer_features import _resolve_repos, _verify_historical
from .transfer_scoring import average_precision, _summary


PERIODS = {'temporal': ('2026-08-17', '2026-09-15'),
           'shared_goals': ('2026-06-01', '2026-07-05')}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def _mention_directory(directory):
    previous = pairs.MENTIONS
    pairs.MENTIONS = Path(directory)
    try:
        yield
    finally:
        pairs.MENTIONS = previous


def _load_inputs(directory, start, end, historical):
    days = {day.strftime('%Y-%m-%d') for day in pd.date_range(start, end)}
    with _mention_directory(directory):
        raw = pairs.load_mentions(days)
        activity = pairs.load_activity(days)
    reference = pd.DataFrame(historical['mapping_reference'], columns=['type', 'id'])
    resolved, information = _resolve_repos(raw, reference=reference)
    return resolved, activity, information


def _historical(study):
    path = study / 'historical-freeze.json'
    historical = json.loads(path.read_text())
    _verify_historical(historical)
    if (historical['train_start'], historical['train_end']) != (secondary.FIT_START, secondary.FIT_END):
        raise ValueError('Historical mapping was not frozen on the declared July interval')
    return historical


def fit_stage(study, *, training_mentions=None, plan_only=False, null_fits=10):
    study = Path(study)
    output = study / 'secondary-frozen.json'
    if output.exists() and not plan_only:
        raise ValueError('Secondary calibration already exists; refusing to replace the frozen bundle')
    historical = _historical(study)
    source = Path(training_mentions) if training_mentions is not None else config.OUT / 'mentions'
    mentions, activity, mapping = _load_inputs(source, secondary.FIT_START, secondary.FIT_END, historical)
    plan = secondary.secondary_plan(mentions, activity, null_fits=null_fits)
    plan.update(historical_sha256=historical['sha256'], mapping=mapping,
                orchestration_sha256=_sha(__file__), training_mentions_directory=str(source))
    _save(study / 'secondary-training-plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)
    if plan_only:
        return plan
    alpha_path = study / 'frozen/outputs/spread/alpha_dev.csv'
    alpha = pd.read_csv(alpha_path, index_col=0)
    bundle = secondary.fit_secondary(mentions, activity, reference_alpha=alpha, null_fits=null_fits)
    _save(output, bundle)
    record = dict(bundle_sha256=bundle['sha256'], bundle_file_sha256=_sha(output),
                  historical_sha256=historical['sha256'], original_alpha_sha256=_sha(alpha_path),
                  alpha_parity=bundle['spread']['original_alpha_comparison'],
                  fit_scope=bundle['fit_scope'], fit_environment=bundle['fit_environment'],
                  orchestration_sha256=_sha(__file__), null_models=len(bundle['spread']['null_models']),
                  elapsed_seconds=bundle['elapsed_seconds'])
    _save(study / 'secondary-fit-record.json', record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def _numeric(value):
    return float(value) if np.isfinite(value) else None


def _rank_metrics(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    if not len(scores):
        return dict(rows=0, positives=0, auc=None, average_precision=None)
    return dict(rows=len(scores), positives=int(labels.sum()),
                auc=_numeric(metrics.auc(scores, labels)), average_precision=average_precision(scores, labels))


def evaluate_locked_predictions(output, predictions_path):
    """Read labels only after verifying the previously written prediction lock."""
    output = Path(output)
    lock_path = output / 'secondary-prediction-lock.json'
    if not lock_path.is_file():
        raise ValueError('Secondary predictions must be locked before label evaluation')
    lock = json.loads(lock_path.read_text())
    for name, expected in lock['prediction_files'].items():
        if Path(name).name != name or _sha(output / name) != expected:
            raise ValueError('Secondary prediction lock mismatch')
    # Request labels are loaded only here, after the independent prediction files.
    labels = pd.read_csv(predictions_path, usecols=['a', 'b', 'day', 'y'])
    secondary._dates(labels, *PERIODS[lock['period']])
    if labels.duplicated(['a', 'b', 'day']).any() or not labels.y.isin([False, True]).all():
        raise ValueError('Malformed target request labels')
    scores = pd.read_csv(output / 'secondary-pair-features.csv.gz')
    scored = scores.merge(labels, on=['a', 'b', 'day'], how='outer', validate='one_to_one', indicator=True)
    if not scored._merge.eq('both').all():
        raise ValueError('Request labels and secondary prediction universes differ')
    scored = scored.drop(columns='_merge')
    result = dict(target='Recorded request pair-days; not verified consumption',
                  prediction_lock_sha256=_sha(lock_path), labels_file_sha256=_sha(predictions_path),
                  regimes=_summary(scored.p1_cos, scored.regime_flag, scored.y),
                  marked_hawkes_all_pairs=_rank_metrics(scored.hk_trig, scored.y),
                  marked_hawkes_supported_actor_pairs=_rank_metrics(
                      scored.loc[scored.hk_supported_pair, 'hk_trig'],
                      scored.loc[scored.hk_supported_pair, 'y']))
    legacy = pd.read_csv(output / 'secondary-legacy-network.csv.gz')
    legacy['a'] = np.minimum(legacy.source, legacy.target)
    legacy['b'] = np.maximum(legacy.source, legacy.target)
    legacy = legacy.merge(labels, on=['a', 'b', 'day'], how='left', validate='many_to_one')
    legacy = legacy[legacy.supported]
    network = legacy.groupby(['source', 'target'], as_index=False).agg(
        alpha=('alpha', 'first'), excess=('excess', 'first'), y=('y', 'max'))
    result['legacy_whole_development_network'] = dict(
        fit_scope=dict(start='2026-07-06', end='2026-08-14'),
        alpha_pair_day=_rank_metrics(legacy.alpha, legacy.y),
        excess_pair_day=_rank_metrics(legacy.excess, legacy.y),
        alpha_any_request_pair=_rank_metrics(network.alpha, network.y),
        excess_any_request_pair=_rank_metrics(network.excess, network.y),
        interpretation='Static directed scores on eligible, supported target pairs; distinct from July-only models.')
    daily = pd.read_csv(output / 'secondary-daily.csv')
    fractions = scored.groupby('day').y.mean()
    daily['request_pair_fraction'] = daily.day.map(fractions)
    correlations = {}
    for name in ('flagged_share', 'hk_other_share', 'hk_excess'):
        if name not in daily:
            continue
        complete = daily[[name, 'request_pair_fraction']].dropna()
        correlations[name] = dict(days=len(complete), spearman=None)
        if len(complete) >= 3 and complete[name].nunique() > 1 and complete.request_pair_fraction.nunique() > 1:
            correlations[name]['spearman'] = _numeric(stats.spearmanr(complete[name], complete.request_pair_fraction).statistic)
    result['daily_associations_descriptive'] = correlations
    _save(output / 'secondary-label-evaluation.json', result)
    return result


def apply_stage(study, period, *, evaluate_labels=True):
    if period not in PERIODS:
        raise ValueError('Unknown authorized target period')
    study = Path(study)
    output = study / period
    if (output / 'secondary-prediction-lock.json').exists():
        raise ValueError('Secondary target predictions are already locked; refusing to overwrite')
    historical = _historical(study)
    bundle = json.loads((study / 'secondary-frozen.json').read_text())
    start, end = PERIODS[period]
    mentions, activity, mapping = _load_inputs(output / 'mentions', start, end, historical)
    universe = pd.read_csv(output / 'features.csv.gz', usecols=['a', 'b', 'day'])
    summary, pair_features, daily = secondary.apply_secondary(
        bundle, mentions, activity, start=start, end=end, target_pair_days=universe)
    legacy_root = study / 'frozen/outputs/hawkes'
    legacy_summary, legacy = secondary.score_legacy_network(
        pd.read_csv(legacy_root / 'alpha_dev.csv', index_col=0),
        pd.read_csv(legacy_root / 'alpha_excess_dev.csv', index_col=0),
        universe, start=start, end=end)
    outputs = {'secondary-pair-features.csv.gz': pair_features,
               'secondary-daily.csv': daily,
               'secondary-legacy-network.csv.gz': legacy}
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    lock = dict(period=period, created_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                frozen_bundle_sha256=bundle['sha256'], historical_sha256=historical['sha256'],
                prediction_files={name: _sha(output / name) for name in outputs},
                orchestration_sha256=_sha(__file__),
                target_labels_accessed=False)
    _save(output / 'secondary-prediction-lock.json', lock)
    summary.update(mapping=mapping, legacy=legacy_summary, prediction_lock_sha256=_sha(output / 'secondary-prediction-lock.json'))
    _save(output / 'secondary-summary.json', summary)
    label_path = output / 'predictions.csv.gz'
    if evaluate_labels and label_path.is_file():
        evaluation = evaluate_locked_predictions(output, label_path)
        summary['label_evaluation_file'] = 'secondary-label-evaluation.json'
        summary['request_evaluation'] = evaluation
        _save(output / 'secondary-summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=config.OUT / 'generalization-20261106')
    parser.add_argument('--stage', required=True, choices=['plan', 'fit', 'apply', 'evaluate'])
    parser.add_argument('--period', choices=sorted(PERIODS))
    parser.add_argument('--training-mentions', type=Path)
    parser.add_argument('--null-fits', type=int, default=10)
    parser.add_argument('--without-label-evaluation', action='store_true')
    args = parser.parse_args()
    if args.stage in {'fit', 'plan'}:
        fit_stage(args.study, training_mentions=args.training_mentions,
                  plan_only=args.stage == 'plan', null_fits=args.null_fits)
    elif not args.period:
        parser.error('--period is required for target application/evaluation')
    elif args.stage == 'apply':
        apply_stage(args.study, args.period, evaluate_labels=not args.without_label_evaluation)
    else:
        result = evaluate_locked_predictions(args.study / args.period,
                                             args.study / args.period / 'predictions.csv.gz')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
