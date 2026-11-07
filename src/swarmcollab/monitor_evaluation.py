"""Chronological monitoring replay with training-only model calibration.

Signal counts are rebuilt from time-stamped evidence, never from whole-day
feature tables. Request labels are offline pair-day evaluation targets. They are
not verified interaction times and do not enter the replay feature state.
"""
import collections
import copy
import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import config, metrics
from .layer_scoring import training_threshold
from .provenance_common import PACIFIC, timestamp

UTC = dt.timezone.utc
ACTIVITY = ('activity_min', 'activity_max')
SCHEMA_VERSION = 1


def _when(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError('Nonfinite evidence timestamp')
        result = dt.datetime.fromtimestamp(float(value), UTC)
    elif isinstance(value, dt.datetime):
        result = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    else:
        result = timestamp(str(value))
    day = result.astimezone(PACIFIC).date()
    if not config.DEV_START <= day <= config.DEV_END:
        raise ValueError('Evidence is outside the authorized development period')
    return result


def _name(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(field + ' must be a nonempty string')
    return value


def _support(value):
    if isinstance(value, Mapping):
        actor, turn = value.get('actor'), value.get('turn_id', value.get('turn'))
        when = value.get('time', value.get('t'))
    elif isinstance(value, (tuple, list)) and len(value) in (2, 3):
        actor, turn = value[:2]
        when = value[2] if len(value) == 3 else None
    else:
        raise ValueError('Evidence turns require actor and turn_id, not bare turn IDs')
    out = dict(actor=_name(actor, 'support actor'), turn_id=_name(str(turn) if turn is not None else None, 'support turn'))
    if when is not None:
        out['time'] = _when(when).timestamp()
    return out


def normalize_signals(signals):
    """Return chronological deduplicated signals and availability diagnostics.

    Each signal requires source_actor, target_actor, kind, and time (or
    available_time). Its effective time is no earlier than any supplied support
    timestamp. ``evidence_turns`` uses {actor, turn_id, time?} records. This
    function never establishes provenance authority or successful consumption.
    """
    found = {}
    diagnostics = collections.Counter()
    for original in signals:
        when = _when(original.get('available_time', original.get('time', original.get('target_time'))))
        source = _name(original.get('source_actor'), 'source_actor')
        target = _name(original.get('target_actor'), 'target_actor')
        if source == target:
            diagnostics['self_signals_ignored'] += 1
            continue
        kind = _name(original.get('kind', original.get('pattern', original.get('family'))), 'signal kind')
        if kind in ACTIVITY:
            raise ValueError('Signal kind collides with an activity feature')
        weight = float(original.get('weight', 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Signal weights must be finite and nonnegative')
        support = [_support(s) for s in original.get('evidence_turns', [])]
        for actor, key, time_key in ((source, 'source_turn', 'source_time'), (target, 'target_turn', 'target_time')):
            if original.get(key) is not None:
                support.append(_support(dict(actor=actor, turn_id=original[key], time=original.get(time_key))))
        by_turn = {}
        for record in support:
            key = (record['actor'], record['turn_id'])
            previous = by_turn.get(key)
            if previous is None or record.get('time', -math.inf) > previous.get('time', -math.inf):
                by_turn[key] = record
        support = [by_turn[key] for key in sorted(by_turn)]
        available = max([when.timestamp()] + [s['time'] for s in support if 'time' in s])
        for time_key in ('source_time', 'target_time'):
            if original.get(time_key) is not None:
                available = max(available, _when(original[time_key]).timestamp())
        if available > when.timestamp():
            diagnostics['availability_delayed_to_support'] += 1
        actual = _when(available)
        day = actual.astimezone(PACIFIC).date().isoformat()
        if original.get('day') is not None and str(original['day']) != day:
            diagnostics['day_corrected_to_available_time'] += 1
        normalized = dict(source_actor=source, target_actor=target, time=available,
                          day=day, kind=kind, weight=weight, evidence_turns=support)
        if original.get('entity') is not None:
            normalized['entity'] = str(original['entity'])
        identity = original.get('signal_id', original.get('match_id', original.get('pattern_id')))
        if identity is None:
            identity = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:24]
        identity = str(identity)
        if identity in found:
            if found[identity] != dict(normalized, signal_id=identity):
                raise ValueError('Conflicting duplicate signal ID: ' + identity)
            diagnostics['duplicates_removed'] += 1
        else:
            found[identity] = dict(normalized, signal_id=identity)
    result = sorted(found.values(), key=lambda x: (x['time'], x['signal_id']))
    diagnostics['signals'] = len(result)
    diagnostics['signals_without_turn_support'] = sum(not s['evidence_turns'] for s in result)
    diagnostics['untimed_support_references'] = sum('time' not in r for s in result for r in s['evidence_turns'])
    return result, dict(diagnostics)


def _turn_events(events):
    turns = {}
    for event in events:
        when = _when(event.get('time', event.get('t')))
        actor = _name(event.get('actor'), 'actor')
        turn = _name(str(event.get('turn_id', event.get('i'))) if event.get('turn_id', event.get('i')) is not None else None, 'turn_id')
        key = (actor, turn)
        observed = (when.timestamp(), when.astimezone(PACIFIC).date().isoformat())
        if key in turns and turns[key] != observed:
            raise ValueError('One actor/turn has conflicting observation times')
        turns[key] = observed
    return turns


def _pair_table(pair_days):
    frame = pair_days.copy() if isinstance(pair_days, pd.DataFrame) else pd.DataFrame(pair_days)
    if not frame.columns.is_unique or not set(('a', 'b', 'day', 'y')).issubset(frame) or frame.empty:
        raise ValueError('Nonempty pair_days requires a,b,day,y')
    frame = frame[['a', 'b', 'day', 'y']].copy()
    for column in ('a', 'b', 'day'):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError('Pair identifiers/dates must be nonempty strings')
    if (frame.a == frame.b).any() or frame.y.isna().any() or not frame.y.isin([True, False]).all():
        raise ValueError('Invalid pair-day actors or labels')
    a, b = np.minimum(frame.a, frame.b), np.maximum(frame.a, frame.b)
    frame['a'], frame['b'] = a, b
    for value in frame.day.unique():
        day = dt.date.fromisoformat(value)
        if day.isoformat() != value or not config.DEV_START <= day <= config.DEV_END:
            raise ValueError('Pair-day outside authorized development scope')
    if frame.duplicated(['a', 'b', 'day']).any():
        raise ValueError('Duplicate undirected pair-day')
    frame['y'] = frame.y.astype(bool)
    return frame.sort_values(['day', 'a', 'b']).reset_index(drop=True)


def _requests(request_times):
    result = collections.defaultdict(list)
    if request_times is None:
        return result
    if isinstance(request_times, Mapping):
        entries = []
        for key, values in request_times.items():
            if not isinstance(values, (list, tuple)):
                values = [values]
            entries.extend(dict(a=key[0], b=key[1], time=t) for t in values)
    else:
        entries = request_times
    for item in entries:
        when = _when(item.get('time', item.get('utc', item.get('timestamp'))))
        a = _name(item.get('a', item.get('from')), 'request actor')
        b = _name(item.get('b', item.get('to')), 'request actor')
        if a == b:
            continue
        a, b = sorted((a, b))
        day = when.astimezone(PACIFIC).date().isoformat()
        result[(a, b, day)].append(when.timestamp())
    return {key: sorted(set(values)) for key, values in result.items()}


def _snapshot_frame(signals, pairs, activity_events, kinds, snapshot_seconds):
    if not isinstance(snapshot_seconds, int) or snapshot_seconds <= 0 or 86400 % snapshot_seconds:
        raise ValueError('snapshot_seconds must be a positive divisor of 86400')
    if activity_events is None:
        activity = []
        for signal in signals:
            timed = [s for s in signal['evidence_turns'] if 'time' in s]
            if timed:
                activity.extend(timed)
            else:
                activity.extend(dict(actor=a, turn_id='signal:' + signal['signal_id'], time=signal['time'])
                                for a in (signal['source_actor'], signal['target_actor']))
        activity_source = 'Inferred from signal support/endpoints; observed-active hours are only a lower-bound proxy.'
    else:
        activity = activity_events
        activity_source = 'Supplied observed working turns, deduplicated by actor/turn.'
    turns = _turn_events(activity)
    times = collections.defaultdict(list)
    hours = collections.defaultdict(set)
    for (actor, _), (value, day) in turns.items():
        times[(day, actor)].append(value)
        hours[day].add((actor, int(value // 3600)))
    times = {key: np.sort(values) for key, values in times.items()}
    grouped = collections.defaultdict(list)
    ignored_kind = 0
    for signal in signals:
        if signal['kind'] in kinds:
            grouped[signal['day']].append(signal)
        else:
            ignored_kind += 1
    kind_index = {kind: i for i, kind in enumerate(kinds)}
    columns = ['signal__' + kind for kind in kinds]
    frames = []
    outside_pairs = 0
    for day, table in pairs.groupby('day', sort=True):
        table = table.reset_index(drop=True)
        lookup = {(a, b): i for i, (a, b) in enumerate(zip(table.a, table.b))}
        start = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), PACIFIC)
        close = (start + dt.timedelta(days=1)).timestamp() - .000001
        checkpoints = list(np.arange(start.timestamp() + snapshot_seconds, close, snapshot_seconds)) + [close]
        counts = np.zeros((len(table), len(kinds)), dtype=float)
        rows = []
        for signal in grouped[day]:
            pair = tuple(sorted((signal['source_actor'], signal['target_actor'])))
            if pair not in lookup:
                outside_pairs += 1
                continue
            rows.append((signal['time'], lookup[pair], kind_index[signal['kind']], signal['weight']))
        rows.sort()
        at = 0
        first = {}
        for actor in set(table.a) | set(table.b):
            values = times.get((day, actor), np.empty(0))
            first[actor] = values[0] if len(values) else math.inf
        available = np.maximum(table.a.map(first).to_numpy(), table.b.map(first).to_numpy())
        for cut in checkpoints:
            end = at
            while end < len(rows) and rows[end][0] <= cut:
                end += 1
            if end > at:
                chunk = rows[at:end]
                np.add.at(counts, ([r[1] for r in chunk], [r[2] for r in chunk]), [r[3] for r in chunk])
            at = end
            eligible = available <= cut
            is_close = cut == close
            keep = np.ones(len(table), dtype=bool) if is_close else eligible
            if not keep.any():
                continue
            part = table.loc[keep].copy()
            part['time'] = cut
            part['eligible'] = eligible[keep]
            part['is_close'] = is_close
            observed = {actor: int(np.searchsorted(times.get((day, actor), np.empty(0)), cut, side='right'))
                        for actor in first}
            aa, bb = part.a.map(observed).to_numpy(), part.b.map(observed).to_numpy()
            part['activity_min'], part['activity_max'] = np.minimum(aa, bb), np.maximum(aa, bb)
            for i, column in enumerate(columns):
                part[column] = counts[keep, i]
            frames.append(part)
    snapshots = pd.concat(frames, ignore_index=True).sort_values(['time', 'a', 'b']).reset_index(drop=True)
    return snapshots, dict(activity_source=activity_source, activity_turns=len(turns),
                           observed_active_agent_hours_by_day={day: len(value) for day, value in sorted(hours.items())},
                           unseen_kind_signals_ignored=ignored_kind, signals_outside_pair_universe=outside_pairs)


def _finite(value):
    return float(value) if math.isfinite(float(value)) else None


def _operating_metrics(table, max_score, threshold):
    y = table.y.to_numpy(bool)
    flagged = max_score > threshold
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    return dict(pair_days=len(table), positives=int(y.sum()), negatives=int((~y).sum()),
                alerts=int(flagged.sum()), true_positives=tp, false_positives=fp,
                recall=tp / int(y.sum()) if y.any() else None,
                fpr=fp / int((~y).sum()) if (~y).any() else None,
                precision=tp / int(flagged.sum()) if flagged.any() else None,
                max_prefix_auc=_finite(metrics.auc(max_score, y)))


def replay(signals, pair_days, request_times=None, activity_events=None,
           feature_sets=None, fitted=None, snapshot_seconds=3600):
    """Fit or replay daily risk scores using only currently available signals.

    Returns ``(report, snapshots, first_alerts)``. ``report['fitted']`` can be
    passed back to replay a saved model with its original vocabulary and
    threshold. Feature sets map model names to signal kinds and optional
    activity_min/activity_max. By default one model uses all training kinds and
    both activity counts. Fit uses training close-of-day rows; thresholding uses
    maximum eligible prefix scores per training negative pair-day. No-observation
    pair-days remain in evaluation denominators and cannot raise an alert.
    """
    normalized, normalization = normalize_signals(signals)
    pairs = _pair_table(pair_days)
    requests = _requests(request_times)
    train_pair = pairs.day <= str(config.TRAIN_END)
    valid_pair = pairs.day >= str(config.VALID_START)
    if not valid_pair.any():
        raise ValueError('Replay requires validation pair-days')
    if fitted is None:
        if not train_pair.any() or not pairs.loc[train_pair, 'y'].any() or pairs.loc[train_pair, 'y'].all():
            raise ValueError('Fitting requires training pair-days with both classes')
        kinds = sorted({s['kind'] for s in normalized if s['day'] <= str(config.TRAIN_END)})
        if feature_sets is None:
            feature_sets = {'all': list(ACTIVITY) + kinds}
        if not isinstance(feature_sets, Mapping) or not feature_sets:
            raise ValueError('Feature sets must map model names to feature lists')
        state = dict(version=SCHEMA_VERSION, train_end=str(config.TRAIN_END),
                     valid_start=str(config.VALID_START), snapshot_seconds=snapshot_seconds,
                     kinds=kinds, models={})
    else:
        state = copy.deepcopy(fitted)
        if (state.get('version') != SCHEMA_VERSION or state.get('train_end') != str(config.TRAIN_END)
                or state.get('valid_start') != str(config.VALID_START)
                or state.get('snapshot_seconds') != snapshot_seconds):
            raise ValueError('Frozen monitor state has an incompatible schema/split/checkpoint schedule')
        kinds = state['kinds']
        if feature_sets is not None:
            raise ValueError('Frozen replay uses its saved feature sets')
        feature_sets = {name: value['requested_features'] for name, value in state['models'].items()}
    snapshots, observation = _snapshot_frame(normalized, pairs, activity_events, kinds, snapshot_seconds)
    report = dict(scope=dict(start=str(config.DEV_START), end=str(config.DEV_END), train_end=str(config.TRAIN_END),
                             valid_start=str(config.VALID_START), timezone='America/Los_Angeles'),
                  mode='frozen model replay' if fitted is not None else 'fit on training, replay validation',
                  snapshot_seconds=snapshot_seconds, snapshots=len(snapshots), normalization=normalization,
                  observation=observation, models={},
                  interpretation='Chronological prefix replay of retrospective pair-day risk models. Request timestamps '
                                 'are evaluation proxies, not verified interaction onset. Vocabulary/model/threshold '
                                 'are frozen before validation; thresholds target training daily alert probability, '
                                 'not independent per-checkpoint false alarms.')
    labels = {(a, b, day): bool(y) for a, b, day, y in pairs.itertuples(index=False, name=None)}
    report['request_timestamp_diagnostics'] = dict(
        request_pairs_outside_evaluation_universe=sum(key not in labels for key in requests),
        request_timestamps_on_negative_pairs=sum(key in labels and not labels[key] for key in requests))
    alerts = []
    train_rows = (snapshots.day <= str(config.TRAIN_END)) & snapshots.is_close
    valid_rows = snapshots.day >= str(config.VALID_START)
    lookup = {(a, b, day): i for i, (a, b, day) in enumerate(pairs[['a', 'b', 'day']].itertuples(index=False, name=None))}
    row_pair = np.array([lookup[key] for key in snapshots[['a', 'b', 'day']].itertuples(index=False, name=None)])
    validation = pairs[valid_pair].reset_index(drop=True)
    valid_days = sorted(validation.day.unique())
    observed_hours = sum(observation['observed_active_agent_hours_by_day'].get(day, 0) for day in valid_days)
    for name, requested in feature_sets.items():
        if not isinstance(name, str) or not name or not isinstance(requested, (list, tuple)) or len(requested) != len(set(requested)):
            raise ValueError('Models require a unique feature list and nonempty name')
        if any(not isinstance(value, str) for value in requested):
            raise ValueError('Feature names must be strings')
        columns = {feature: feature if feature in ACTIVITY else 'signal__' + feature for feature in requested}
        available = [f for f in requested if f in ACTIVITY or f in kinds]
        if fitted is None:
            retained = [f for f in available if np.ptp(snapshots.loc[train_rows, columns[f]].to_numpy(float)) > 0]
            values = np.column_stack([np.log1p(snapshots[columns[f]].to_numpy(float)) for f in retained]) if retained else np.empty((len(snapshots), 0))
            model = metrics.Logistic().fit(values[train_rows], snapshots.loc[train_rows, 'y'].to_numpy(bool))
            score = model.decision(values)
            params = dict(requested_features=list(requested), features=retained,
                          unavailable_training_features=[f for f in requested if f not in available],
                          dropped_constant_features=[f for f in available if f not in retained],
                          mu=model.mu.tolist(), sd=model.sd.tolist(), w=model.w.tolist(),
                          transformation='log1p each raw cumulative count', fit='training close-of-day rows')
        else:
            params = state['models'][name]
            retained = params['features']
            values = np.column_stack([np.log1p(snapshots[columns[f]].to_numpy(float)) for f in retained]) if retained else np.empty((len(snapshots), 0))
            score = np.asarray(params['w'])[0] + ((values - params['mu']) / params['sd']) @ np.asarray(params['w'])[1:]
        if not np.isfinite(score).all():
            raise ValueError('Replay model produced nonfinite scores')
        # Fixed finite sentinel: validation scores must not alter calibration for
        # training days without observed activity. These rows can never alert.
        maxima = np.full(len(pairs), -1e300)
        eligible = snapshots.eligible.to_numpy(bool)
        np.maximum.at(maxima, row_pair[eligible], score[eligible])
        close_scores = np.full(len(pairs), -1e300)
        eligible_close = eligible & snapshots.is_close.to_numpy(bool)
        close_scores[row_pair[eligible_close]] = score[eligible_close]
        if fitted is None:
            threshold = training_threshold(maxima[train_pair], pairs.loc[train_pair, 'y'].to_numpy(bool))
            params['threshold'] = threshold
            params['threshold_fit'] = 'Training-negative maximum eligible prefix score per pair-day; strict >; daily 10% budget.'
            params['retrospective_threshold'] = training_threshold(
                close_scores[train_pair], pairs.loc[train_pair, 'y'].to_numpy(bool))
            state['models'][name] = params
        else:
            threshold = params['threshold']
        snapshots['score__' + name] = score
        snapshots['alert__' + name] = eligible & (score > threshold)
        valid_max = maxima[valid_pair]
        result = _operating_metrics(validation, valid_max, threshold)
        result.update(threshold=threshold, alerts_per_day=result['alerts'] / len(valid_days),
                      observed_active_agent_hours=observed_hours,
                      alerts_per_observed_active_agent_hour=result['alerts'] / observed_hours if observed_hours else None,
                      false_alerts_per_observed_active_agent_hour=result['false_positives'] / observed_hours if observed_hours else None,
                      training=_operating_metrics(pairs[train_pair], maxima[train_pair], threshold) if train_pair.any() else None)
        retrospective = _operating_metrics(validation, close_scores[valid_pair], params['retrospective_threshold'])
        retrospective['close_of_day_auc'] = retrospective.pop('max_prefix_auc')
        retrospective.update(threshold=params['retrospective_threshold'],
                             interpretation='Same fitted model and pair-day universe, using close-of-day scores only; '
                                            'threshold calibrated separately on training negative close-of-day scores.')
        result['retrospective_comparison'] = retrospective
        selected = snapshots[valid_rows & snapshots['alert__' + name]].sort_values(['time', 'a', 'b']).drop_duplicates(['a', 'b', 'day'])
        delays, leads, unknown_time = [], [], 0
        for _, row in selected.iterrows():
            key = (row.a, row.b, row.day)
            request = requests.get(key, [])
            relation, delta = 'request_negative_pair_day', None
            if row.y and request:
                delta = row.time - request[0]
                relation = 'pre_request' if delta < 0 else 'at_or_after_request'
                (leads if delta < 0 else delays).append(abs(delta))
            elif row.y:
                relation = 'positive_pair_day_without_request_timestamp'
                unknown_time += 1
            alerts.append(dict(model=name, a=row.a, b=row.b, day=row.day, time=row.time,
                               y=bool(row.y), score=float(row['score__' + name]),
                               relation_to_first_request=relation, seconds_from_first_request=delta))
        def distribution(values):
            return dict(count=len(values), median_seconds=float(np.median(values)) if values else None,
                        p90_seconds=float(np.quantile(values, .9)) if values else None)
        positive_keys = [(a, b, day) for a, b, day, y in validation.itertuples(index=False, name=None) if y]
        result['timing'] = dict(pre_request_alerts=distribution(leads),
                                at_or_after_request_alerts=distribution(delays),
                                alerted_positive_pairs_without_request_time=unknown_time,
                                positive_pairs_without_request_time=sum(key not in requests for key in positive_keys),
                                interpretation='First alert per pair-day relative to first recorded request. Pre-request alerts '
                                               'are reported as leads, not counted as negative delays or proof of prediction. '
                                               'Missed positives are not assigned zero delay.')
        report['models'][name] = result
    report['fitted'] = state
    alert_frame = pd.DataFrame(alerts, columns=['model', 'a', 'b', 'day', 'time', 'y', 'score',
                                               'relation_to_first_request', 'seconds_from_first_request'])
    return report, snapshots, alert_frame
