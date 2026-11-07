"""Frozen training/validation scoring for the two research layers.

This scores retrospective pair-days, not streaming alerts. No extraction, label
loading, network access, or sealed-slice override is performed here.
"""
import datetime as dt
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import config, metrics


KEYS = ['a', 'b', 'day', 'y']
PRETRANSFORMED = {'act_min', 'act_max'}
TARGET_FPR = 0.10


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _validate_rows(frame, validation_only=False):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('Expected a DataFrame with unique column names')
    missing = set(KEYS) - set(frame.columns)
    if missing:
        raise ValueError('Missing pair-day fields: ' + ', '.join(sorted(missing)))
    if frame.empty:
        raise ValueError('Cannot score an empty pair-day table')
    for column in ('a', 'b', 'day'):
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError('Pair-day identifiers and dates must be nonempty strings')
    if (frame.a == frame.b).any():
        raise ValueError('Self-pairs are not valid evaluation units')
    # Treat opposite orientations as the same undirected request-label unit.
    canonical = pd.DataFrame({'a': np.minimum(frame.a, frame.b),
                              'b': np.maximum(frame.a, frame.b), 'day': frame.day})
    if canonical.duplicated().any():
        raise ValueError('Duplicate undirected pair-day keys')
    lower = config.VALID_START if validation_only else config.DEV_START
    for day in frame.day.unique():
        try:
            parsed = dt.date.fromisoformat(day)
        except ValueError as error:
            raise ValueError('Invalid ISO pair-day date: ' + day) from error
        if parsed.isoformat() != day or not lower <= parsed <= config.DEV_END:
            raise ValueError('Pair-day date outside the permitted scoring period: ' + day)
    if frame.y.isna().any() or not frame.y.isin([False, True]).all():
        raise ValueError('Labels must be booleans or numeric zero/one')


def _alert_metrics(y, alert):
    y, alert = np.asarray(y, dtype=bool), np.asarray(alert, dtype=bool)
    tp = int(np.sum(y & alert))
    fp = int(np.sum(~y & alert))
    positives, negatives = int(y.sum()), int((~y).sum())
    return dict(rows=len(y), positives=positives, negatives=negatives,
                alerts=int(alert.sum()), true_positives=tp, false_positives=fp,
                false_negatives=positives - tp, true_negatives=negatives - fp,
                recall=tp / positives if positives else None,
                fpr=fp / negatives if negatives else None,
                precision=tp / (tp + fp) if tp + fp else None,
                alert_rate=float(alert.mean()) if len(alert) else None)


def training_threshold(scores, y, target_fpr=TARGET_FPR):
    """Largest empirical alert budget at or below target, with strict > ties.

    The selected order statistic permits at most floor(target_fpr * n_negative)
    training negatives to alert. Interpolated quantiles can exceed the requested
    budget on small samples; this definition avoids that ambiguity.
    """
    if not 0 <= target_fpr < 1:
        raise ValueError('target_fpr must be in [0, 1)')
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=bool)
    if scores.ndim != 1 or y.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Threshold calibration needs aligned, finite scores')
    negatives = np.sort(scores[~y])
    if not len(negatives):
        raise ValueError('Threshold calibration requires training negatives')
    allowed = math.floor(target_fpr * len(negatives))
    return float(negatives[len(negatives) - allowed - 1])


def evaluate_layers(frame, feature_sets):
    """Return a JSON-safe report and validation-only prediction DataFrame.

    ``feature_sets`` maps a model name to its ordered feature-column list. The
    prediction columns are a, b, day, y, plus each model name (decision score)
    and ``<model name>__alert`` (score strictly above its training threshold).
    Already logged act_min/act_max are left unchanged; other counts use log1p.
    Constant columns are selected using training rows only and removed before
    fitting. An empty remaining feature list fits an intercept-only model.
    """
    _validate_rows(frame)
    if not isinstance(feature_sets, Mapping) or not feature_sets:
        raise ValueError('feature_sets must be a nonempty model-name mapping')
    prediction_names = set(KEYS)
    all_columns = []
    for name, columns in feature_sets.items():
        if not isinstance(name, str) or not name or not isinstance(columns, (list, tuple)):
            raise ValueError('Each model needs a nonempty name and a feature list')
        for column in (name, name + '__alert'):
            if column in prediction_names:
                raise ValueError('Model names collide with prediction columns')
            prediction_names.add(column)
        if any(not isinstance(column, str) or not column for column in columns):
            raise ValueError('Feature names must be nonempty strings')
        if len(columns) != len(set(columns)):
            raise ValueError('Duplicate features in model: ' + name)
        if set(columns) & set(KEYS):
            raise ValueError('Identifiers and target labels cannot be features')
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError('Missing features: ' + ', '.join(sorted(missing)))
        all_columns.extend(columns)
    all_columns = list(dict.fromkeys(all_columns))
    transformed = {}
    for column in all_columns:
        try:
            values = frame[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('Feature must be numeric: ' + column) from error
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Feature must be finite and nonnegative: ' + column)
        transformed[column] = values if column in PRETRANSFORMED else np.log1p(values)

    train = frame.day.to_numpy() <= config.TRAIN_END.isoformat()
    valid = frame.day.to_numpy() >= config.VALID_START.isoformat()
    y = frame.y.to_numpy(dtype=bool)
    if not train.any() or not valid.any():
        raise ValueError('Both training and validation rows are required')
    if not y[train].any() or y[train].all():
        raise ValueError('Training requires both positive and negative labels')
    predictions = frame.loc[valid, KEYS].copy().reset_index(drop=True)
    predictions['y'] = predictions.y.astype(bool)
    report = dict(
        scope=dict(start=config.DEV_START.isoformat(), end=config.DEV_END.isoformat(),
                   train_end=config.TRAIN_END.isoformat(), valid_start=config.VALID_START.isoformat(),
                   timezone='America/Los_Angeles'),
        train_rows=int(train.sum()), validation_rows=int(valid.sum()),
        rows_between_splits=int((~(train | valid)).sum()),
        train_positive_rows=int(y[train].sum()), validation_positive_rows=int(y[valid].sum()),
        train_days=sorted(frame.loc[train, 'day'].unique().tolist()),
        validation_days=sorted(frame.loc[valid, 'day'].unique().tolist()),
        target='Undirected request pair-days; requests are not verified consumption.',
        interpretation='Retrospective development validation, already inspected during research. '
                       'Frozen training calibration does not make whole-day features an online monitor.',
        models={})
    for name, requested in feature_sets.items():
        retained = [column for column in requested if np.ptp(transformed[column][train]) > 0]
        dropped = [column for column in requested if column not in retained]
        values = (np.column_stack([transformed[column] for column in retained])
                  if retained else np.empty((len(frame), 0)))
        fitted = metrics.Logistic(l2=1.0).fit(values[train], y[train])
        train_score, valid_score = fitted.decision(values[train]), fitted.decision(values[valid])
        if not np.isfinite(train_score).all() or not np.isfinite(valid_score).all():
            raise ValueError('Model returned nonfinite scores: ' + name)
        threshold = training_threshold(train_score, y[train])
        train_alert, valid_alert = train_score > threshold, valid_score > threshold
        training_metrics = _alert_metrics(y[train], train_alert)
        validation_metrics = _alert_metrics(y[valid], valid_alert)
        validation_metrics['alerts_per_day'] = int(valid_alert.sum()) / len(report['validation_days'])
        report['models'][name] = dict(
            requested_features=list(requested), features=retained, dropped_constant_features=dropped,
            transformations={column: ('already_log1p' if column in PRETRANSFORMED else 'log1p')
                             for column in requested},
            auc=_number(metrics.auc(valid_score, y[valid])),
            descriptive_recall_at_10pct_fpr=_number(metrics.recall_at_fpr(valid_score, y[valid])),
            descriptive_recall_interpretation='ROC summary using a validation-negative quantile; '
                                               'not the deployed training threshold.',
            threshold=threshold, threshold_target_fpr=TARGET_FPR,
            threshold_fit='Training negatives only; empirical order statistic; strict score > threshold.',
            training_at_train_threshold=training_metrics,
            validation_at_train_threshold=validation_metrics,
            coefficients={column: float(value) for column, value in zip(retained, fitted.w[1:])},
            fitted=dict(feature_names=retained, mu=fitted.mu.tolist(), sd=fitted.sd.tolist(),
                        w=fitted.w.tolist(), intercept_index=0, l2=1.0))
        predictions[name] = valid_score
        predictions[name + '__alert'] = valid_alert
    return report, predictions


def paired_day_comparison(predictions, baseline_name, candidate_name, repetitions=500, seed=20261106):
    """Descriptive paired bootstrap: resample entire validation days together.

    Models and training thresholds remain fixed. Shared agents and workflows can
    induce dependence across days; these intervals are not independent-sample
    significance tests and do not include rule-selection or model-fit uncertainty.
    """
    _validate_rows(predictions, validation_only=True)
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 0:
        raise ValueError('repetitions must be a nonnegative integer')
    for name in (baseline_name, candidate_name):
        if name not in predictions or name + '__alert' not in predictions:
            raise ValueError('Missing model scores or fixed alerts: ' + str(name))
        try:
            scores = predictions[name].to_numpy(dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError('Bootstrap scores must be numeric') from error
        alerts = predictions[name + '__alert']
        if not np.isfinite(scores).all() or alerts.isna().any() or not alerts.isin([False, True]).all():
            raise ValueError('Bootstrap requires finite scores and boolean alerts')
    y = predictions.y.to_numpy(dtype=bool)
    base_score = predictions[baseline_name].to_numpy(dtype=float)
    new_score = predictions[candidate_name].to_numpy(dtype=float)
    base_alert = predictions[baseline_name + '__alert'].to_numpy(dtype=bool)
    new_alert = predictions[candidate_name + '__alert'].to_numpy(dtype=bool)
    days = sorted(predictions.day.unique())
    blocks = [np.flatnonzero(predictions.day.to_numpy() == day) for day in days]
    rng = np.random.default_rng(seed)
    auc_deltas, recall_deltas = [], []
    skipped_auc, skipped_recall = 0, 0
    for _ in range(repetitions):
        sample = np.concatenate([blocks[index] for index in rng.integers(0, len(blocks), len(blocks))])
        target = y[sample]
        if target.any() and not target.all():
            auc_deltas.append(metrics.auc(new_score[sample], target) - metrics.auc(base_score[sample], target))
        else:
            skipped_auc += 1
        if target.any():
            positive = sample[target]
            recall_deltas.append(float(new_alert[positive].mean() - base_alert[positive].mean()))
        else:
            skipped_recall += 1

    def interval(estimate, draws, skipped):
        return dict(estimate=_number(estimate), valid_draws=len(draws), skipped_single_class=skipped,
                    interval_95=np.quantile(draws, [0.025, 0.975]).tolist() if draws else None)

    recall_delta = float(new_alert[y].mean() - base_alert[y].mean()) if y.any() else float('nan')
    return dict(
        baseline=baseline_name, candidate=candidate_name, repetitions=repetitions, seed=seed,
        days=days, blocks=len(days),
        delta_auc=interval(metrics.auc(new_score, y) - metrics.auc(base_score, y), auc_deltas, skipped_auc),
        delta_recall_at_fixed_threshold=interval(recall_delta, recall_deltas, skipped_recall),
        interpretation='Descriptive paired bootstrap over whole validation days, with fitted models '
                       'and training thresholds fixed. Shared agents create dependence across days; '
                       'intervals omit fitting and rule-selection uncertainty and are not causal or '
                       'independent-sample significance tests.')
