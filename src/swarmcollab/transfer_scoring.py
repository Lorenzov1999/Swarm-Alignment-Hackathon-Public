"""Apply the existing layer models to explicitly authorized transfer periods.

No fitting, calibration, data loading, or network access occurs here. Labels are
optional during prediction and are consulted only by the evaluation functions.
The September--October reserve is deliberately outside all permitted periods.
"""
import datetime as dt
import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import metrics


PERIODS = {
    'development_validation': ('2026-08-03', '2026-08-14'),
    'temporal': ('2026-08-17', '2026-09-15'),
    'shared_goals': ('2026-06-01', '2026-07-05'),
}
KEYS = ['a', 'b', 'day']
RESERVED_COLUMNS = set(KEYS + ['y', 'agent_stratum', 'novel_agent_count'])
STRATA = ('both_known', 'one_unseen', 'both_unseen')


def _validate_rows(frame, period, require_labels=False):
    if period not in PERIODS:
        raise ValueError('Unknown authorized transfer period: ' + str(period))
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or frame.empty:
        raise ValueError('Expected a nonempty DataFrame with unique columns')
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError('Missing pair-day fields: ' + ', '.join(sorted(missing)))
    for column in KEYS:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError('Pair-day identifiers and dates must be nonempty strings')
    if (frame.a == frame.b).any():
        raise ValueError('Self-pairs are not valid evaluation units')
    canonical = pd.DataFrame({'a': np.minimum(frame.a, frame.b),
                              'b': np.maximum(frame.a, frame.b), 'day': frame.day})
    if canonical.duplicated().any():
        raise ValueError('Duplicate undirected pair-day keys')
    lower, upper = PERIODS[period]
    for day in frame.day.unique():
        try:
            parsed = dt.date.fromisoformat(day)
        except ValueError as error:
            raise ValueError('Invalid ISO pair-day date: ' + day) from error
        if parsed.isoformat() != day or not lower <= day <= upper:
            raise ValueError('Pair-day date outside authorized period: ' + day)
    if require_labels:
        if 'y' not in frame or frame.y.isna().any() or not frame.y.isin([False, True]).all():
            raise ValueError('Evaluation requires complete boolean or numeric zero/one labels')


def _models(saved_report, model_names=None):
    if not isinstance(saved_report, Mapping) or not isinstance(saved_report.get('models'), Mapping):
        raise ValueError('Expected a saved layer-scoring report with models')
    models = saved_report['models']
    names = list(models) if model_names is None else list(model_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError('At least one nonempty model name is required')
    if len(set(names)) != len(names) or set(names) - set(models):
        raise ValueError('Duplicate or unknown model names')
    occupied = set(RESERVED_COLUMNS)
    result = {}
    for name in names:
        for column in (name, name + '__alert'):
            if column in occupied:
                raise ValueError('Model names collide with prediction columns')
            occupied.add(column)
        model = models[name]
        if not isinstance(model, Mapping):
            raise ValueError('Invalid saved model: ' + name)
        requested, features = model.get('requested_features'), model.get('features')
        if any(not isinstance(columns, list) for columns in (requested, features)):
            raise ValueError('Saved model requires ordered feature lists: ' + name)
        for columns in (requested, features):
            if (any(not isinstance(column, str) or not column for column in columns)
                    or len(set(columns)) != len(columns) or set(columns) & RESERVED_COLUMNS):
                raise ValueError('Invalid saved feature names: ' + name)
        if features != [column for column in requested if column in set(features)]:
            raise ValueError('Retained features must preserve the requested feature order: ' + name)
        transformations = model.get('transformations', {})
        if (not isinstance(transformations, Mapping)
                or set(transformations) != set(requested)
                or any(value not in ('log1p', 'already_log1p') for value in transformations.values())):
            raise ValueError('Missing or unsupported saved transformations: ' + name)
        fitted = model.get('fitted', {})
        if (not isinstance(fitted, Mapping) or fitted.get('feature_names') != features
                or fitted.get('intercept_index') != 0):
            raise ValueError('Saved fitted feature order or intercept is inconsistent: ' + name)
        try:
            mu, sd, weights = [np.asarray(fitted[key], dtype=float) for key in ('mu', 'sd', 'w')]
            threshold = float(model['threshold'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Invalid frozen model parameters: ' + name) from error
        if (mu.shape != (len(features),) or sd.shape != mu.shape
                or weights.shape != (len(features) + 1,)
                or not all(np.isfinite(array).all() for array in (mu, sd, weights))
                or np.any(sd <= 0) or not np.isfinite(threshold)):
            raise ValueError('Invalid frozen model parameter dimensions or values: ' + name)
        result[name] = (model, features, mu, sd, weights, threshold)
    return result


def _strata(frame, known_agents):
    if isinstance(known_agents, str):
        raise ValueError('known_agents must be a collection of identifiers, not one string')
    known = set(known_agents)
    if any(not isinstance(actor, str) or not actor for actor in known):
        raise ValueError('Known agent identifiers must be nonempty strings')
    unseen = (~frame.a.isin(known)).astype(int) + (~frame.b.isin(known)).astype(int)
    return unseen.map(dict(enumerate(STRATA))), unseen


def score_frozen(frame, saved_report, *, period, model_names=None, known_agents=None):
    """Return aligned decisions and strict frozen-threshold alerts; never fit.

    All requested feature columns, including training-constant columns, must be
    finite and nonnegative. Only the saved retained features enter decisions.
    Existing labels are copied unchanged without influencing scores. A caller
    can omit labels entirely and join them only after predictions are frozen.
    """
    _validate_rows(frame, period)
    models = _models(saved_report, model_names)
    output = frame[KEYS + (['y'] if 'y' in frame else [])].copy().reset_index(drop=True)
    for name, (model, features, mu, sd, weights, threshold) in models.items():
        missing = set(model['requested_features']) - set(frame.columns)
        if missing:
            raise ValueError('Missing frozen model features: ' + ', '.join(sorted(missing)))
        transformed = {}
        for column in model['requested_features']:
            try:
                values = frame[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError('Feature must be numeric: ' + column) from error
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Feature must be finite and nonnegative: ' + column)
            transformed[column] = (values if model['transformations'][column] == 'already_log1p'
                                   else np.log1p(values))
        values = (np.column_stack([transformed[column] for column in features])
                  if features else np.empty((len(frame), 0)))
        score = weights[0] + ((values - mu) / sd) @ weights[1:]
        if not np.isfinite(score).all():
            raise ValueError('Frozen model produced nonfinite scores: ' + name)
        output[name] = score
        output[name + '__alert'] = score > threshold
    if known_agents is not None:
        strata, unseen = _strata(output, known_agents)
        output['agent_stratum'], output['novel_agent_count'] = strata, unseen
    return output


def average_precision(score, labels):
    """Noninterpolated average precision; tied scores enter as one threshold.

    This is sum(delta recall * precision), not trapezoidal PR area. It is
    undefined without positives and equals prevalence for constant scores.
    """
    score, labels = np.asarray(score, dtype=float), np.asarray(labels)
    if (score.ndim != 1 or labels.shape != score.shape or not np.isfinite(score).all()
            or not np.isin(labels, [False, True]).all()):
        raise ValueError('Average precision requires finite scores and aligned binary labels')
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-score, kind='stable')
    sorted_scores, sorted_labels = score[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(score) - 1]
    tp = np.cumsum(sorted_labels)[ends]
    precision, recall = tp / (ends + 1), tp / positives
    return float(np.sum(np.diff(np.r_[0., recall]) * precision))


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _summary(scores, alerts, labels=None):
    scores, alerts = np.asarray(scores, dtype=float), np.asarray(alerts, dtype=bool)
    result = dict(rows=len(scores), alerts=int(alerts.sum()),
                  alert_rate=float(alerts.mean()) if len(alerts) else None)
    if labels is None:
        return result
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    tp, fp = int((labels & alerts).sum()), int((~labels & alerts).sum())
    result.update(positives=positives, negatives=negatives,
                  positive_prevalence=positives / len(labels) if len(labels) else None,
                  true_positives=tp, false_positives=fp,
                  false_negatives=positives - tp, true_negatives=negatives - fp,
                  recall=tp / positives if positives else None,
                  fpr=fp / negatives if negatives else None,
                  precision=tp / (tp + fp) if tp + fp else None,
                  auc=_number(metrics.auc(scores, labels)),
                  average_precision=average_precision(scores, labels))
    return result


def _prediction_arrays(predictions, names):
    result = {}
    for name in names:
        if name not in predictions or name + '__alert' not in predictions:
            raise ValueError('Missing scores or alerts for model: ' + str(name))
        try:
            scores = predictions[name].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('Prediction scores must be numeric: ' + name) from error
        alerts = predictions[name + '__alert']
        if (not np.isfinite(scores).all() or alerts.isna().any()
                or not alerts.isin([False, True]).all()):
            raise ValueError('Predictions need finite scores and binary alerts: ' + name)
        result[name] = (scores, alerts.to_numpy(dtype=bool))
    return result


def evaluate_predictions(predictions, saved_report, *, period, model_names=None, known_agents=None):
    """Summarize frozen predictions, optionally against separately supplied y.

    The evaluator checks that alerts still use each saved threshold. Neither
    target-label operating-point selection nor target-dependent fitting occurs.
    """
    _validate_rows(predictions, period, require_labels='y' in predictions)
    models = _models(saved_report, model_names)
    arrays = _prediction_arrays(predictions, models)
    labels = predictions.y.to_numpy(dtype=bool) if 'y' in predictions else None
    days = sorted(predictions.day.unique())
    report = dict(period=period, authorized_start=PERIODS[period][0],
                  authorized_end=PERIODS[period][1], observed_days=days,
                  rows=len(predictions), labelled=labels is not None, models={},
                  target='Undirected request pair-days; requests are not verified consumption.',
                  interpretation='Frozen-model transfer; no target fitting or threshold calibration. '
                                 'Whole-day features remain retrospective.',
                  average_precision_definition='Noninterpolated sum(delta recall * precision), '
                                               'with tied scores grouped at one threshold.')
    for name, (model, _, _, _, _, threshold) in models.items():
        scores, alerts = arrays[name]
        if not np.array_equal(alerts, scores > threshold):
            raise ValueError('Prediction alerts do not match the saved threshold: ' + name)
        summary = _summary(scores, alerts, labels)
        summary['alerts_per_observed_day'] = int(alerts.sum()) / len(days)
        frozen = {key: model[key] for key in ('requested_features', 'features', 'transformations',
                                             'fitted', 'threshold')}
        report['models'][name] = dict(
            **summary, threshold=threshold,
            frozen_model_sha256=hashlib.sha256(json.dumps(frozen, sort_keys=True,
                                                        allow_nan=False).encode()).hexdigest())
    if known_agents is not None:
        strata, _ = _strata(predictions, known_agents)
        report['agent_strata'] = {}
        for stratum in STRATA:
            mask = strata.eq(stratum).to_numpy()
            report['agent_strata'][stratum] = dict(rows=int(mask.sum()), models={})
            for name, (scores, alerts) in arrays.items():
                report['agent_strata'][stratum]['models'][name] = _summary(
                    scores[mask], alerts[mask], labels[mask] if labels is not None else None)
    return report


def paired_block_comparison(predictions, baseline_name, candidate_name, *, period,
                            block_column='day', repetitions=500, seed=20261106):
    """Resample complete days or supplied episodes, paired across both models.

    Existing prediction scores/alerts stay fixed. Each block is sampled with
    replacement, preserving all its pair-days and their shared actors. Intervals
    are descriptive and conditional on the fitted models and selected rules.
    """
    _validate_rows(predictions, period, require_labels=True)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 0:
        raise ValueError('repetitions must be a nonnegative integer')
    if (block_column not in predictions
            or not predictions[block_column].map(lambda v: isinstance(v, str) and bool(v)).all()):
        raise ValueError('Bootstrap blocks require a complete string identifier column')
    arrays = _prediction_arrays(predictions, [baseline_name, candidate_name])
    labels = predictions.y.to_numpy(dtype=bool)
    base_score, base_alert = arrays[baseline_name]
    candidate_score, candidate_alert = arrays[candidate_name]
    block_names = sorted(predictions[block_column].unique())
    blocks = [np.flatnonzero(predictions[block_column].to_numpy() == name) for name in block_names]
    fields = {'delta_auc': 'auc', 'delta_average_precision': 'average_precision',
              'delta_recall_at_fixed_threshold': 'recall',
              'delta_fpr_at_fixed_threshold': 'fpr',
              'delta_precision_at_fixed_threshold': 'precision',
              'delta_alert_rate': 'alert_rate'}

    def deltas(index):
        baseline = _summary(base_score[index], base_alert[index], labels[index])
        candidate = _summary(candidate_score[index], candidate_alert[index], labels[index])
        return {key: (candidate[field] - baseline[field]
                      if candidate[field] is not None and baseline[field] is not None else None)
                for key, field in fields.items()}

    estimates = deltas(np.arange(len(predictions)))
    draws = {key: [] for key in fields}
    rng = np.random.default_rng(seed)
    actual_repetitions = repetitions if len(blocks) >= 2 else 0
    for _ in range(actual_repetitions):
        index = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        for key, value in deltas(index).items():
            if value is not None:
                draws[key].append(value)
    report = dict(baseline=baseline_name, candidate=candidate_name, period=period,
                  block_column=block_column, block_ids=block_names, blocks=len(blocks),
                  repetitions=repetitions, seed=seed,
                  insufficient_blocks_for_uncertainty=len(blocks) < 2,
                  interpretation='Descriptive paired whole-block bootstrap with saved predictions '
                                 'and thresholds fixed. Dependence across blocks can remain; '
                                 'intervals omit fitting and rule-selection uncertainty and are '
                                 'not causal or independent-sample significance tests.')
    for key in fields:
        report[key] = dict(estimate=estimates[key], valid_draws=len(draws[key]),
                           skipped_undefined=actual_repetitions - len(draws[key]),
                           skipped_insufficient_blocks=repetitions - actual_repetitions,
                           interval_95=np.quantile(draws[key], [.025, .975]).tolist()
                           if draws[key] else None)
    return report
