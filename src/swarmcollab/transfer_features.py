"""Apply existing feature definitions using explicit, frozen training knowledge.

This module neither reads raw logs nor labels, changes global scope guards, nor
fits on transfer rows. Callers supply already scoped observations. The historical
feature functions are reused directly; the trace aggregator mirrors the existing
definition but takes its family vocabulary and weights from the saved manifest.
"""
import collections
import datetime as dt
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd

from . import pairs
from .provenance_links import directed_features
from .research_layers import KEYS, READS, WRITES, any_near, feature_prefix


TRACE_COLUMNS = ['actor', 'day', 'time', 'turn', 'layer', 'family', 'entity',
                 'operation', 'status', 'verbal', 'related_actor', 'weight']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _bounds(start, end):
    first, last = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
    if first > last:
        raise ValueError('Transfer start must not follow end')
    return first.isoformat(), last.isoformat()


def _check_days(frame, start, end, name):
    first, last = _bounds(start, end)
    if 'day' not in frame:
        raise ValueError(name + ' requires day')
    for value in frame.day.unique():
        if not isinstance(value, str):
            raise ValueError(name + ' requires ISO date strings')
        if dt.date.fromisoformat(value).isoformat() != value or not first <= value <= last:
            raise ValueError(name + ' contains out-of-period observations: ' + value)


def _check_universe(universe, start, end):
    if not set(KEYS).issubset(universe):
        raise ValueError('Pair universe requires a, b, day')
    _check_days(universe, start, end, 'Pair universe')
    if universe[KEYS].duplicated().any() or (universe.a >= universe.b).any():
        raise ValueError('Pair universe must contain unique, canonically ordered non-self pairs')


def _resolve_repos(mentions, reference):
    """Reuse the resolver, including a pandas empty-series compatibility case.

    The legacy resolver adds a string prefix to an empty dictionary-mapped float
    Series when no GitLab repository rows exist. One temporary identity row
    avoids that dtype error; it never enters fitting, events, or feature counts.
    Its identity cannot change the separately supplied frozen reference.
    """
    has_gitlab = ((mentions.type.astype(str) == 'repo') & mentions.id.str.startswith('gitlab')).any()
    if has_gitlab:
        return pairs.resolve_repos(mentions, reference=reference)
    marker = len(mentions)
    padded = pd.concat([mentions, pd.DataFrame([{
        'type': 'repo', 'id': 'gitlab:transfer-empty-frame-compatibility-8bb2ff98',
    }])], ignore_index=True)
    resolved, information = pairs.resolve_repos(padded, reference=reference)
    resolved = resolved[resolved.index != marker].reset_index(drop=True)
    information['gitlab_paths'] -= 1
    return resolved, information


def freeze_historical(training_mentions, training_actors,
                      train_start='2026-07-06', train_end='2026-07-31'):
    """Reconstruct the original mapping/IDF fit once, from training-only mentions.

    The compact mapping reference preserves ordering and every project-vote row
    because the existing resolver counts those votes. Actual future observations
    must never be passed here. The signed date bounds reject accidental extras.
    """
    _check_days(training_mentions, train_start, train_end, 'Training mentions')
    required = {'agent', 'type', 'id', 'day'}
    if not required.issubset(training_mentions):
        raise ValueError('Training mention fields are incomplete')
    actors = sorted(set(training_actors))
    if not actors:
        raise ValueError('At least one training actor is required')
    raw = training_mentions.copy()
    raw['type'] = raw.type.astype(str)
    ordinary = raw[raw.type.isin(['repo', 'pidmap'])][['type', 'id']].drop_duplicates()
    votes = raw[raw.type == 'pidvote'][['type', 'id']]
    reference = pd.concat([ordinary, votes], ignore_index=True)
    resolved, mapping = _resolve_repos(raw, reference=reference)
    traces = resolved[resolved.type.isin(pairs.TRACE_TYPES)]
    idf = pairs.trace_weights(traces, actors, set(raw.day))
    frozen = dict(version=1, train_start=str(train_start), train_end=str(train_end),
                  training_actors=actors, training_mention_rows=len(raw),
                  mapping_reference=reference.to_dict('records'), mapping_diagnostics=mapping,
                  fitted_idf={str(k): float(v) for k, v in idf.items()},
                  unseen_idf=float(np.log(len(actors))))
    frozen['sha256'] = fingerprint(frozen)
    return frozen


def _verify_historical(frozen):
    if frozen.get('version') != 1:
        raise ValueError('Unknown historical freeze format')
    payload = {k: v for k, v in frozen.items() if k != 'sha256'}
    if fingerprint(payload) != frozen.get('sha256'):
        raise ValueError('Historical freeze fingerprint mismatch')


def historical_features(mentions, activity, frozen, target_start, target_end,
                        history_mentions=None):
    """Use existing pair functions with a frozen mapping and rarity vocabulary.

    Optional history supplies earlier observations to identifier handoffs only;
    it cannot change IDF/mappings and must precede every scoring date. The caller
    decides the authorized warm-up interval. June transfer must not receive July
    observations. Names and same-day co-use use only target-period observations.
    """
    _verify_historical(frozen)
    first, last = _bounds(target_start, target_end)
    _check_days(mentions, first, last, 'Target mentions')
    _check_days(activity, first, last, 'Target activity')
    if not {'agent', 'working'}.issubset(activity):
        raise ValueError('Activity requires agent and working counts')
    if activity[['agent', 'day']].duplicated().any():
        raise ValueError('Duplicate actor/day activity')
    work = activity.working.to_numpy(float)
    if not np.isfinite(work).all() or (work < 0).any():
        raise ValueError('Working counts must be finite and nonnegative')
    universe = pairs.universe(activity)
    _check_universe(universe, first, last)
    reference = pd.DataFrame(frozen['mapping_reference'], columns=['type', 'id'])
    resolved, mapping = _resolve_repos(mentions, reference=reference)
    traces = resolved[resolved.type.isin(pairs.TRACE_TYPES)].copy()
    traces['tid'] = traces.tid.astype(str).astype('category')
    historical = None
    if history_mentions is not None and len(history_mentions):
        for value in history_mentions.day.unique():
            if dt.date.fromisoformat(value).isoformat() != value or value >= first:
                raise ValueError('Handoff history must strictly precede the scoring period')
        historical, _ = _resolve_repos(history_mentions, reference=reference)
        historical = historical[historical.type.isin(pairs.TRACE_TYPES)].copy()
    all_traces = pd.concat([historical, traces], ignore_index=True) if historical is not None else traces
    idf = {str(tid): frozen['fitted_idf'].get(str(tid), frozen['unseen_idf'])
           for tid in all_traces.tid.unique()}
    frame = universe.copy()
    parts = []
    if len(traces):
        parts.extend(pairs.p1_features(traces, idf, sorted(activity.agent.unique())))
    if len(all_traces):
        hand, pou, _, _ = pairs.p2_features(all_traces, idf)
        parts.extend([hand[hand.day.between(first, last)], pou[pou.day.between(first, last)]])
    if len(resolved):
        parts.append(pairs.name_features(resolved)[0])
    for part in parts:
        part = part.copy()
        part['day'] = part.day.astype(str)
        frame = frame.merge(part, on=KEYS, how='left', validate='one_to_one')
    for column in pairs.FEATURES:
        if column not in frame:
            frame[column] = 0.0
    for column in frame.columns.difference(KEYS):
        frame[column] = pd.to_numeric(frame[column]).fillna(0.0)
    working = activity.set_index(['agent', 'day']).working
    a = working.reindex(list(zip(frame.a, frame.day))).to_numpy(float)
    b = working.reindex(list(zip(frame.b, frame.day))).to_numpy(float)
    frame['act_min'] = np.log1p(np.minimum(a, b))
    frame['act_max'] = np.log1p(np.maximum(a, b))
    return frame, dict(frozen_sha256=frozen['sha256'], mapping=mapping,
                       target_mention_rows=len(mentions), target_trace_rows=len(traces),
                       history_trace_rows=0 if historical is None else len(historical),
                       unseen_trace_entities=sum(str(x) not in frozen['fitted_idf']
                                                 for x in traces.tid.unique()),
                       unseen_idf=frozen['unseen_idf'], pair_days=len(frame))


def trace_features(rows, universe, family_manifest, training_actors,
                   target_start, target_end):
    """Aggregate target traces using exactly the saved training families/IDF.

    Same feature definitions as research_layers.build_trace_features; there is
    no date-based training selector and no fit operation on supplied rows.
    """
    _check_universe(universe, target_start, target_end)
    _check_days(rows, target_start, target_end, 'Target trace rows')
    if not set(TRACE_COLUMNS).issubset(rows):
        raise ValueError('Trace row fields are incomplete')
    if not set(rows.layer).issubset({1, 2}):
        raise ValueError('Unknown research layer')
    actors = sorted(set(training_actors))
    if not actors:
        raise ValueError('Frozen training actor roster is required')
    default = math.log(len(actors))
    features = universe[KEYS].copy()
    known_pairs = set(map(tuple, features.itertuples(index=False, name=None)))
    families = family_manifest['families']
    family_keys = {(int(v['layer']), v['family'], bool(v['verbal'])) for v in families.values()}
    report = dict(frozen_manifest_sha256=fingerprint(family_manifest), families={},
                  training_actors_sha256=fingerprint(actors), unseen_idf=default,
                  unseen_family_events=sum((int(a), b, bool(c)) not in family_keys
                                           for a, b, c in zip(rows.layer, rows.family, rows.verbal)))
    for prefix, frozen in families.items():
        layer, family, verbal = int(frozen['layer']), frozen['family'], bool(frozen['verbal'])
        expected = feature_prefix(layer, family) + ('_verbal' if verbal else '')
        if prefix != expected:
            raise ValueError('Frozen trace family prefix mismatch')
        sub = rows[(rows.layer == layer) & (rows.family == family) & (rows.verbal == verbal)].copy()
        aggregated = collections.defaultdict(collections.Counter)
        if verbal:
            attributed = sub[(sub.related_actor != '') & (sub.related_actor != sub.actor)]
            attributed = attributed.sort_values('weight').drop_duplicates(
                ['actor', 'turn', 'related_actor'], keep='last')
            for row in attributed.itertuples():
                key = (*sorted((row.actor, row.related_actor)), row.day)
                if key in known_pairs:
                    aggregated[key]['mentions'] += row.weight
            suffixes = ['mentions']
        else:
            touches = sub.drop_duplicates(['actor', 'turn', 'entity'])
            fitted_weights = frozen['fitted_idf']
            if any(not math.isfinite(v) or v < 0 for v in fitted_weights.values()):
                raise ValueError('Frozen IDF values must be finite and nonnegative')
            counts = touches.groupby(['day', 'actor', 'entity']).size()
            magnitudes, norms = {}, collections.Counter()
            for (day, actor, entity), count in counts.items():
                weight = fitted_weights.get(entity, default) * (1 + math.log(count))
                magnitudes[(day, actor, entity)] = weight
                norms[(day, actor)] += weight * weight
            # Singleton object/day groups cannot make a pair. Retain them in
            # actor norms above and in attributed-owner counts below, but avoid
            # constructing empty pair comparisons for them.
            shared = touches[touches.groupby(['day', 'entity']).actor.transform('nunique') >= 2]
            shared_keys = pd.MultiIndex.from_frame(shared[['day', 'entity']].drop_duplicates())
            operation_keys = pd.MultiIndex.from_frame(sub[['day', 'entity']])
            operations = sub[(sub.status != 'failed') & operation_keys.isin(shared_keys)]
            first_writes, last_reads = {}, {}
            for key, group in operations.groupby(['day', 'entity', 'actor']):
                writes, reads = group[group.operation.isin(WRITES)].time, group[group.operation.isin(READS)].time
                if len(writes):
                    first_writes[key] = float(writes.min())
                if len(reads):
                    last_reads[key] = float(reads.max())
            for (day, entity), group in shared.groupby(['day', 'entity'], sort=False):
                by_actor = {actor: g.time.to_numpy() for actor, g in group.groupby('actor', sort=True)}
                weight = fitted_weights.get(entity, default)
                for a, b in itertools.combinations(sorted(by_actor), 2):
                    key = (a, b, day)
                    if key not in known_pairs:
                        continue
                    value = aggregated[key]
                    value['shared'] += 1
                    value['idf_overlap'] += weight
                    value['cosine'] += magnitudes[(day, a, entity)] * magnitudes[(day, b, entity)]
                    if any_near(by_actor[a], by_actor[b]):
                        value['sync'] += weight
                    if any(first_writes.get((day, entity, x), math.inf) <
                           last_reads.get((day, entity, y), -math.inf) for x, y in ((a, b), (b, a))):
                        value['ordered_candidate'] += weight
            for (a, b, day), values in aggregated.items():
                denominator = math.sqrt(norms[(day, a)] * norms[(day, b)])
                values['cosine'] = values['cosine'] / denominator if denominator else 0.0
            attributed = sub[(sub.related_actor != '') & (sub.related_actor != sub.actor)]
            attributed = attributed.drop_duplicates(['actor', 'turn', 'entity', 'related_actor'])
            for row in attributed.itertuples():
                key = (*sorted((row.actor, row.related_actor)), row.day)
                if key in known_pairs:
                    aggregated[key]['related_actor_candidate'] += 1
            suffixes = ['shared', 'idf_overlap', 'cosine', 'sync', 'ordered_candidate', 'related_actor_candidate']
        columns = [prefix + '_' + suffix for suffix in suffixes]
        if columns != frozen['features']:
            raise ValueError('Frozen family columns disagree with current definition')
        part = pd.DataFrame([dict(zip(KEYS, key), **{prefix + '_' + s: values[s] for s in suffixes})
                             for key, values in aggregated.items()], columns=KEYS + columns)
        features = features.merge(part, on=KEYS, how='left', validate='one_to_one')
        report['families'][prefix] = dict(layer=layer, family=family, verbal=verbal,
            events=len(sub), unique_turns=int(sub[['actor', 'turn']].drop_duplicates().shape[0]),
            entities=int(sub.entity.nunique()), nonzero_pair_days=len(aggregated), features=columns,
            fitted_idf_sha256=fingerprint(frozen['fitted_idf']))
    for column in features.columns.difference(KEYS):
        features[column] = pd.to_numeric(features[column]).fillna(0.0)
    return features, report


def provenance_features(universe, chains, requested_columns, target_start, target_end):
    """Aggregate existing directed chain features, selecting only target dates."""
    _check_universe(universe, target_start, target_end)
    first, last = _bounds(target_start, target_end)
    columns = list(dict.fromkeys(requested_columns))
    if any(not c.startswith('prov_') for c in columns):
        raise ValueError('Provenance columns must use the prov_ namespace')
    for chain in chains:
        day = chain['day']
        if dt.date.fromisoformat(day).isoformat() != day or day > last:
            raise ValueError('Provenance contains a malformed date or future scoring evidence')
    selected = [chain for chain in chains if first <= chain['day'] <= last]
    directed = pd.DataFrame(directed_features(selected))
    frame = universe[KEYS].copy()
    observed_columns = []
    if len(directed):
        directed['a'] = directed[['source', 'target']].min(axis=1)
        directed['b'] = directed[['source', 'target']].max(axis=1)
        observed_columns = [c for c in directed if c.startswith('prov_')]
        part = directed.groupby(KEYS)[observed_columns].sum().reset_index()
        frame = frame.merge(part, on=KEYS, how='left', validate='one_to_one')
    for column in columns:
        if column not in frame:
            frame[column] = 0.0
    frame = frame[KEYS + columns].fillna(0.0)
    return frame, dict(target_chains=len(selected),
                       chain_levels=dict(collections.Counter(str(c['level']) for c in selected)),
                       unmapped_feature_columns=sorted(set(observed_columns) - set(columns)))


def combine_features(frames, requested_columns):
    """Join already assembled feature families, never silently inventing columns."""
    if not frames:
        raise ValueError('At least one feature frame is required')
    combined = frames[0].copy()
    for frame in frames[1:]:
        overlap = (set(combined) & set(frame)) - set(KEYS)
        if overlap:
            raise ValueError('Overlapping feature columns: ' + ', '.join(sorted(overlap)))
        if set(map(tuple, combined[KEYS].itertuples(index=False, name=None))) != set(
                map(tuple, frame[KEYS].itertuples(index=False, name=None))):
            raise ValueError('Feature frames have different pair universes')
        combined = combined.merge(frame, on=KEYS, how='left', validate='one_to_one')
    requested = list(dict.fromkeys(requested_columns))
    missing = set(requested) - set(combined)
    if missing:
        raise ValueError('Required frozen features not assembled: ' + ', '.join(sorted(missing)))
    values = combined[requested].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Assembled features must be finite and nonnegative')
    return combined[KEYS + requested]
