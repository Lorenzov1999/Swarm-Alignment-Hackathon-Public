"""Frozen out-of-period application of the existing spread and regime methods.

Inputs are already resolved mention/activity tables supplied by the caller.
This module never reads data, request labels, or external reports. All fitting
is restricted to July 6--31; scoring only accepts the two authorized transfer
windows. It leaves the original pipeline and its historical outputs unchanged.
"""
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from . import pairs as P
from . import regimes as R
from . import spread as S


FIT_START, FIT_END = '2026-07-06', '2026-07-31'
TARGET_WINDOWS = {('2026-08-17', '2026-09-15'), ('2026-06-01', '2026-07-05')}
SCHEMA = 'secondary-transfer-v1'
DEFAULT_SEED = 20261106
MAX_EVENTS = 250_000
MAX_PAIRS = 5_000_000
MAX_DONOR_ROWS = 2_000


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _environment():
    packages = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy')}
    sources = {}
    for name, filename in [('secondary_transfer', __file__), ('pairs', P.__file__),
                           ('regimes', R.__file__), ('spread', S.__file__)]:
        sources[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return dict(python_version=platform.python_version(), python_build=sys.version,
                packages=packages, packages_sha256=_hash(packages), source_sha256=sources)


def _frame_hash(frame, columns):
    data = frame[columns].copy()
    data = data.sort_values(columns, kind='stable').reset_index(drop=True)
    return hashlib.sha256(data.to_csv(index=False, float_format='%.17g').encode()).hexdigest()


def _dates(frame, start, end):
    for value in frame.day.unique():
        if not isinstance(value, str):
            raise ValueError('Dates must be canonical ISO strings')
        day = dt.date.fromisoformat(value)
        if day.isoformat() != value or not start <= value <= end:
            raise ValueError('Input date outside declared period: ' + value)


def _inputs(mentions, activity, start, end):
    for frame, columns in ((mentions, {'agent', 'day', 'type', 'tid', 't'}),
                           (activity, {'agent', 'day', 'working'})):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError('Expected a DataFrame with unique columns')
        if not columns.issubset(frame.columns):
            raise ValueError('Missing columns: ' + ', '.join(sorted(columns - set(frame.columns))))
        _dates(frame, start, end)
        if not frame.agent.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError('Agent identifiers must be nonempty strings')
    act = activity.copy()
    if act.duplicated(['agent', 'day']).any():
        raise ValueError('Duplicate activity agent-day rows')
    working = pd.to_numeric(act.working, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(working).all() or (working < 0).any():
        raise ValueError('Working counts must be finite and nonnegative')
    act['working'] = working
    tm = mentions[mentions.type.isin(P.TRACE_TYPES)].copy()
    if not tm.tid.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError('Trace identities must be nonempty strings')
    tm['t'] = pd.to_numeric(tm.t, errors='raise').astype(float)
    if not np.isfinite(tm.t).all():
        raise ValueError('Event times must be finite')
    actual_days = pd.to_datetime(tm.t, unit='s', utc=True).dt.tz_convert(
        'America/Los_Angeles').dt.strftime('%Y-%m-%d')
    if not np.array_equal(actual_days.to_numpy(), tm.day.to_numpy()):
        raise ValueError('Event time disagrees with Pacific day')
    observed = set(zip(act.agent, act.day))
    if any(key not in observed for key in zip(tm.agent, tm.day)):
        raise ValueError('Mention actor/day missing from activity table')
    for column in ('agent', 'day', 'type'):
        tm[column] = tm[column].astype(str)
    tm['tid'] = tm.tid.astype(str).astype('category')
    return tm, act


def _candidate_count(events, window=S.WINDOW):
    total = 0
    for _, group in events.groupby('tid', observed=True, sort=False):
        values = np.sort(group.t.to_numpy(dtype=float))
        before = np.arange(len(values)) - np.searchsorted(values, values - window, side='left')
        total += int(np.minimum(before, S.MAX_CAND).sum())
    return total


def _budget(events, max_events, max_pairs):
    if len(events) > max_events:
        raise ValueError(f'Trace sessions exceed declared compute cap: {len(events)} > {max_events}')
    candidates = _candidate_count(events)
    if candidates > max_pairs:
        raise ValueError(f'Candidate pairs exceed declared compute cap: {candidates} > {max_pairs}')
    return candidates


def secondary_plan(training_mentions, training_activity, null_fits=0):
    """Cheap sizing pass; no EM fitting, null sampling, or target data access."""
    tm, act = _inputs(training_mentions, training_activity, FIT_START, FIT_END)
    events = S.trace_sessions(tm, shared_reference=tm)
    vector_rows = len(act.day.unique()) * len(act.agent.unique())
    candidates = _candidate_count(events)
    return dict(training_mentions=len(tm), training_sessions=len(events),
                candidate_pairs_upper_bound=candidates, em_fits=1 + null_fits,
                em_iterations_per_fit=S.EM_ITERS,
                approximate_candidate_array_bytes=64 * candidates,
                regime_cosine_matrix_bytes=8 * vector_rows * vector_rows,
                training_days=int(act.day.nunique()), training_actors=int(act.agent.nunique()),
                note='Memory counts describe principal arrays, not total process peak RSS.')


def _pack_model(model):
    return dict(agents=list(model.agents), beta=float(model.beta), window=float(model.window),
                mu=model.mu.tolist(), alpha=model.alpha.tolist(),
                hour_profile=model.hour_profile.tolist(), mark_prior=float(model.mark_prior),
                mark_counts=[[str(a), str(tid), int(n)] for (a, tid), n
                             in sorted(model.mark_counts.items())],
                mark_totals={str(a): int(n) for a, n in model.mark_totals.items()},
                mark_vocabulary=sorted(model.mark_vocabulary),
                fit_event_counts=model.fit_event_counts.tolist(), fit_days=list(model.fit_days))


def _unpack_model(state):
    model = S.MarkedHawkes(state['agents'], beta=state['beta'], window=state['window'])
    for name in ('mu', 'alpha', 'hour_profile', 'fit_event_counts'):
        setattr(model, name, np.array(state[name], dtype=float))
    model.mark_prior = state['mark_prior']
    model.mark_counts = {(a, tid): n for a, tid, n in state['mark_counts']}
    model.mark_totals = dict(state['mark_totals'])
    model.mark_vocabulary = set(state['mark_vocabulary'])
    model.fit_days = list(state['fit_days'])
    return model


def fit_secondary(training_mentions, training_activity, *, reference_alpha=None,
                  null_fits=0, cross_day_samples=R.NULL_SAMPLES, seed=DEFAULT_SEED,
                  max_events=MAX_EVENTS, max_pairs=MAX_PAIRS):
    """Reconstruct original July calibration and return a JSON-safe full bundle.

    ``reference_alpha`` is the original spread alpha DataFrame, with actor names
    as rows and columns. A mismatch raises before a bundle can be used on target
    data. Ten null fits reproduce the original spread control; zero deliberately
    omits that control rather than fabricating an excess estimate.
    """
    if not isinstance(null_fits, int) or not 0 <= null_fits <= S.NULL_FITS:
        raise ValueError('null_fits must be an integer from zero through ten')
    if not isinstance(cross_day_samples, int) or cross_day_samples < 1:
        raise ValueError('cross_day_samples must be a positive integer')
    started = time.monotonic()
    tm, act = _inputs(training_mentions, training_activity, FIT_START, FIT_END)
    if tm.empty or act.empty:
        raise ValueError('Calibration needs nonempty July training observations')
    training_actors = sorted(act.agent.unique())
    train_days = sorted(act.day.unique())
    if len(training_actors) * len(train_days) > MAX_DONOR_ROWS:
        raise ValueError('Training donor matrix exceeds the declared compute cap')
    idf = P.trace_weights(tm, training_actors, set(train_days))
    vecs = R.day_vectors(tm, idf, training_actors)
    rng = np.random.default_rng(seed)
    sampled = R.cross_day_null(vecs, rng, n=cross_day_samples)
    if not len(sampled):
        raise ValueError('No eligible training cross-day null comparisons')
    threshold = float(np.quantile(sampled, R.NULL_QUANTILE))
    active = {day: set(group.agent[group.working > 0]) for day, group in act.groupby('day')}
    donors = R.AgentDayBlockNull(vecs, training_actors, active, train_days,
                                available_after=R.artifact_availability(tm))
    events = S.trace_sessions(tm, shared_reference=tm)
    candidates = _budget(events, max_events, max_pairs)
    model_state, null_states, comparison = None, [], None
    if not events.empty:
        model = S.MarkedHawkes(sorted(events.agent.unique())).fit(events)
        model_state = _pack_model(model)
        if reference_alpha is not None:
            if set(reference_alpha.index) != set(model.agents) or set(reference_alpha.columns) != set(model.agents):
                raise ValueError('Reconstructed spread support differs from saved alpha')
            expected = reference_alpha.loc[model.agents, model.agents].to_numpy(dtype=float)
            match = bool(np.allclose(expected, model.alpha, rtol=1e-8, atol=1e-10))
            comparison = dict(matches=match, max_absolute_difference=float(np.max(np.abs(expected - model.alpha))),
                              reference_sha256=_frame_hash(reference_alpha.rename_axis('agent').reset_index(),
                                                           ['agent'] + list(reference_alpha.columns)))
            if not match:
                raise ValueError('Reconstructed spread alpha differs from original saved fit: ' + str(comparison))
        null_rng = np.random.default_rng(seed)  # Original spread.main uses a fresh generator.
        for _ in range(null_fits):
            shuffled = S.day_swap(events, null_rng)
            _budget(shuffled, max_events, max_pairs)
            null_states.append(_pack_model(S.MarkedHawkes(model.agents).fit(shuffled)))
    elif reference_alpha is not None:
        raise ValueError('Cannot reconstruct saved alpha without shared training sessions')
    bundle = dict(schema=SCHEMA, fit_environment=_environment(),
                  fit_scope=dict(start=FIT_START, end=FIT_END, timezone='America/Los_Angeles'),
                  input_sha256=dict(mentions=_frame_hash(tm, ['agent', 'day', 'type', 'tid', 't']),
                                    activity=_frame_hash(act, ['agent', 'day', 'working'])),
                  parameters=dict(seed=seed, trace_gap=S.TRACE_GAP, max_candidates=S.MAX_CAND,
                                  em_iterations=S.EM_ITERS, beta=S.BETA, window=S.WINDOW,
                                  max_events=max_events, max_pairs=max_pairs,
                                  max_donor_rows=MAX_DONOR_ROWS),
                  spread=dict(model=model_state, null_models=null_states,
                              original_alpha_comparison=comparison, training_sessions=len(events),
                              candidate_pairs_upper_bound=candidates,
                              support='Only traces shared by at least two July training actors; fitted actors only.'),
                  regimes=dict(agents=training_actors, days=donors.days,
                               fitted_idf={str(k): float(v) for k, v in idf.items()}, unseen_idf=math.log(len(training_actors)),
                               threshold=threshold, quantile=R.NULL_QUANTILE,
                               cross_day_samples=len(sampled), null_scores_sha256=_hash(sampled.tolist()),
                               max_gap_days=R.NULL_MAX_GAP_DAYS,
                               donor_rows={a: [int(row) for _, row in rows] for a, rows in donors.rows.items()},
                               donor_cosines=donors.cosines.tolist(),
                               block_availability=[[str(a), str(day), str(limit)]
                                                   for (a, day), limit in sorted(donors.available_after.items())]),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    bundle['sha256'] = _hash(bundle)
    return bundle


def _check_bundle(bundle):
    if bundle.get('schema') != SCHEMA or bundle.get('fit_scope') != dict(
            start=FIT_START, end=FIT_END, timezone='America/Los_Angeles'):
        raise ValueError('Unexpected frozen secondary bundle')
    if bundle.get('sha256') != _hash({k: v for k, v in bundle.items() if k != 'sha256'}):
        raise ValueError('Frozen secondary bundle hash mismatch')
    expected = dict(trace_gap=S.TRACE_GAP, max_candidates=S.MAX_CAND, em_iterations=S.EM_ITERS,
                    beta=S.BETA, window=S.WINDOW)
    if any(bundle['parameters'].get(k) != value for k, value in expected.items()):
        raise ValueError('Temporal method constants changed after calibration')
    if bundle.get('fit_environment', {}).get('source_sha256') != _environment()['source_sha256']:
        raise ValueError('Temporal source code changed after calibration')


def _pair_universe(activity, supplied):
    expected = P.universe(activity).sort_values(['day', 'a', 'b']).reset_index(drop=True)
    if supplied is None:
        return expected
    supplied = supplied[['a', 'b', 'day']].copy().sort_values(['day', 'a', 'b']).reset_index(drop=True)
    if not supplied.equals(expected):
        raise ValueError('Supplied pair-days do not match eligible working-agent universe')
    return expected


def _null_share(cosines, donor_rows, actors, threshold, samples, rng):
    if len(actors) < 2 or any(not donor_rows.get(a) for a in actors):
        return None
    ids = [rng.choice(donor_rows[a], size=samples) for a in actors]
    flags = np.zeros(samples, dtype=float)
    pairs = 0
    for i in range(len(actors)):
        for j in range(i + 1, len(actors)):
            flags += cosines[ids[i], ids[j]] > threshold
            pairs += 1
    return flags / pairs


def _available_donor_rows(reference, target_day):
    """Keep complete training blocks whose artifacts were observable by query day."""
    availability = {(actor, day): limit for actor, day, limit in reference['block_availability']}
    width = len(reference['agents'])
    result = {}
    for actor, rows in reference['donor_rows'].items():
        eligible = []
        for row in rows:
            donor_day = reference['days'][row // width]
            # A working block with no represented artifact is still an observed
            # July training block, never a historically available June sample.
            earliest = availability.get((actor, donor_day), reference['days'][0])
            if earliest <= target_day:
                eligible.append(row)
        result[actor] = eligible
    return result


def apply_secondary(bundle, target_mentions, target_activity, *, start, end,
                    target_pair_days=None, block_samples=R.BLOCK_NULL_SAMPLES, seed=DEFAULT_SEED):
    """Return ``(summary, pair_features, daily_table)`` without any target fitting.

    Target windows start cold. No observations outside the declared window are
    used as history. Regime nulls draw only July donor blocks whose artifacts
    were available by the target date. June retains retrospective threshold
    shares, but July donors cannot supply valid June null calibration.
    """
    _check_bundle(bundle)
    if (start, end) not in TARGET_WINDOWS:
        raise ValueError('Window is not authorized for this transfer study')
    if not isinstance(block_samples, int) or block_samples < 1:
        raise ValueError('block_samples must be a positive integer')
    started = time.monotonic()
    tm, act = _inputs(target_mentions, target_activity, start, end)
    pairs = _pair_universe(act, target_pair_days)
    days = sorted(act.loc[act.working > 0, 'day'].unique())
    reference = bundle['regimes']
    weights = {tid: reference['fitted_idf'].get(tid, reference['unseen_idf'])
               for tid in tm.tid.astype(str).unique()}
    actors = sorted(act.agent.unique())
    idx = {a: i for i, a in enumerate(actors)}
    vecs = R.day_vectors(tm, weights, actors) if not tm.empty else {}
    cosine_by_day = {day: (matrix @ matrix.T).toarray() for day, (matrix, _) in vecs.items()}
    pairs['p1_cos'] = [float(cosine_by_day[day][idx[a], idx[b]]) if day in cosine_by_day else 0.
                       for a, b, day in pairs[['a', 'b', 'day']].itertuples(index=False, name=None)]
    pairs['regime_flag'] = pairs.p1_cos > reference['threshold']
    donor_rows = reference['donor_rows']
    cosines = np.asarray(reference['donor_cosines'])
    rng = np.random.default_rng(seed)
    daily_rows = []
    for day in days:
        active = sorted(act.loc[(act.day == day) & (act.working > 0), 'agent'])
        frame = pairs[pairs.day == day]
        known = [a for a in active if donor_rows.get(a)]
        known_frame = frame[frame.a.isin(known) & frame.b.isin(known)]
        available_rows = _available_donor_rows(reference, day)
        available = [a for a in known if available_rows.get(a)]
        draws = _null_share(cosines, available_rows, known, reference['threshold'], block_samples, rng)
        known_share = float(known_frame.regime_flag.mean()) if len(known_frame) else None
        p = float((1 + np.sum(draws >= known_share)) / (1 + len(draws))) if draws is not None else None
        covered = len(known) == len(active) and p is not None
        daily_rows.append(dict(day=day, active_agents=len(active), active_pairs=len(frame),
                               flagged=int(frame.regime_flag.sum()),
                               flagged_share=float(frame.regime_flag.mean()) if len(frame) else None,
                               null_calibrated=covered, p_value=p if covered else None,
                               collaborative=bool(p < .01 / max(len(days), 1)) if covered else None,
                               known_training_agents=len(known), available_donor_agents=len(available),
                               known_agent_pairs=len(known_frame), known_agent_flagged_share=known_share,
                               known_agent_null_calibrated=p is not None, known_agent_p_value=p,
                               known_agent_collaborative=bool(p < .01 / max(len(days), 1)) if p is not None else None,
                               null_flagged_share_mean=float(draws.mean()) if draws is not None else None))
    daily = pd.DataFrame(daily_rows)
    state = bundle['spread']['model']
    pairs['hk_trig'] = 0.
    pairs['hk_supported_pair'] = False
    spread_report = dict(available=False, reason='No shared July training sessions')
    if state is not None:
        support = pd.DataFrame([(a, tid) for a, tid, _ in state['mark_counts']], columns=['agent', 'tid'])
        events = S.trace_sessions(tm, shared_reference=support)
        supported = events[events.agent.isin(state['agents'])]
        _budget(supported, bundle['parameters']['max_events'], bundle['parameters']['max_pairs'])
        pairs['hk_supported_pair'] = pairs.a.isin(state['agents']) & pairs.b.isin(state['agents'])
        spread_report = dict(available=bool(len(supported)), sessions_on_training_shared_traces=len(events),
                             supported_sessions=len(supported), unseen_agent_sessions=len(events) - len(supported),
                             fitted_agents=len(state['agents']), observed_fitted_agents=int(supported.agent.nunique()),
                             target_traces=int(tm.tid.nunique()),
                             target_traces_in_training_support=int(tm.tid.isin(state['mark_vocabulary']).groupby(tm.tid, observed=True).any().sum()),
                             null_models=len(bundle['spread']['null_models']))
        if len(supported):
            model = _unpack_model(state).score(events)
            triggers = model.pair_day_triggers().rename(columns={'src': 'a', 'dst': 'b', 'r': 'hk_trig'})
            triggers = P.undirected(triggers).groupby(['a', 'b', 'day'], as_index=False).hk_trig.sum()
            pairs = pairs.drop(columns='hk_trig').merge(triggers, on=['a', 'b', 'day'], how='left')
            pairs['hk_trig'] = pairs.hk_trig.fillna(0.)
            if not daily.empty:
                daily['hk_other_share'] = daily.day.map(model.per_day_other_share())
                daily['hk_supported_sessions'] = daily.day.map(model.e.groupby('day').size()).fillna(0).astype(int)
            null_daily = [_unpack_model(s).score(events).per_day_other_share()
                          for s in bundle['spread']['null_models']]
            if null_daily and not daily.empty:
                null_mean = pd.concat(null_daily, axis=1).mean(axis=1)
                daily['hk_other_share_null'] = daily.day.map(null_mean)
                daily['hk_excess'] = daily.hk_other_share - daily.hk_other_share_null
            other = model.i[model.tgt] != model.i[model.src]
            other_probability = np.bincount(model.tgt[other], weights=model.r[other], minlength=len(model.i))
            spread_report.update(conditional_loglik=model.loglik(),
                                 conditional_loglik_per_session=model.loglik() / len(supported),
                                 candidate_pairs=len(model.tgt), frozen_branching_ratio=model.branching_ratio(),
                                 frozen_cross_excitation=model.cross_excitation(),
                                 mean_other_share=float(other_probability.mean()))
    summary = dict(scope=dict(start=start, end=end, timezone='America/Los_Angeles'),
                   scoring_environment=_environment(),
                   frozen_bundle_sha256=bundle['sha256'], target_fitting=False,
                   input_sha256=dict(mentions=_frame_hash(tm, ['agent', 'day', 'type', 'tid', 't']),
                                     activity=_frame_hash(act, ['agent', 'day', 'working'])),
                   target_mentions=len(tm), target_actors=len(actors), working_steps=float(act.working.sum()),
                   pair_days=len(pairs), target_days=len(days),
                   regimes=dict(threshold=reference['threshold'], block_samples=block_samples,
                                fully_calibrated_days=int(daily.null_calibrated.sum()) if len(daily) else 0,
                                null_source='July training blocks with artifacts available by each target day',
                                donor_availability_enforced=True,
                                interpretation='Fixed-threshold shares are descriptive; backward June comparison is retrospective '
                                               'and has no null calibration from future July donor blocks.'),
                   spread=spread_report, history='Cold start at the target window boundary; no extra-window observations.',
                   interpretation='Secondary model-based temporal associations, not causal interaction probabilities. '
                                  'Branching ratio and fitted excitation are fixed model summaries, not target changes. '
                                  'Likelihood uses the existing approximate conditional compensator.',
                   elapsed_seconds=round(time.monotonic() - started, 3))
    _hash(summary)  # Reject nonfinite JSON results before publishing a summary.
    return summary, pairs, daily


def score_legacy_network(alpha, excess, pair_days, *, start, end):
    """Apply saved legacy agent-Hawkes networks, with their original fit scope.

    Matrix rows are recipients and columns are sources. Output is directed and
    repeats fixed network scores on eligible target pair-days. It does not claim
    new event-level attribution or silently refit the whole-development model
    on July training. Unknown actors have missing scores, never fabricated zero
    influence. Target labels are neither read nor returned.
    """
    if (start, end) not in TARGET_WINDOWS:
        raise ValueError('Window is not authorized for this transfer study')
    for matrix in (alpha, excess):
        if (not isinstance(matrix, pd.DataFrame) or not matrix.columns.is_unique
                or not matrix.index.is_unique or set(matrix.index) != set(matrix.columns)
                or not np.isfinite(matrix.to_numpy(dtype=float)).all()):
            raise ValueError('Expected a finite square actor-indexed network')
    if set(alpha.index) != set(excess.index):
        raise ValueError('Legacy alpha and excess actor supports differ')
    if not {'a', 'b', 'day'}.issubset(pair_days.columns):
        raise ValueError('Missing target pair-day keys')
    _dates(pair_days, start, end)
    frame = pair_days[['a', 'b', 'day']].copy()
    if ((frame.a >= frame.b).any() or frame.duplicated().any()
            or not frame.a.map(lambda x: isinstance(x, str) and bool(x)).all()
            or not frame.b.map(lambda x: isinstance(x, str) and bool(x)).all()):
        raise ValueError('Expected unique canonical undirected pair-days')
    rows = []
    known = set(alpha.index)
    for a, b, day in frame.itertuples(index=False, name=None):
        supported = a in known and b in known
        for source, target in ((a, b), (b, a)):
            rows.append(dict(source=source, target=target, day=day, supported=supported,
                             alpha=float(alpha.loc[target, source]) if supported else None,
                             excess=float(excess.loc[target, source]) if supported else None))
    result = pd.DataFrame(rows, columns=['source', 'target', 'day', 'supported', 'alpha', 'excess'])
    summary = dict(method='Legacy agent-Hawkes static-network transfer',
                   original_fit_scope=dict(start='2026-07-06', end='2026-08-14'),
                   target_scope=dict(start=start, end=end), target_fitting=False,
                   directed_pair_days=len(result),
                   supported_directed_pair_days=int(result.supported.sum()) if len(result) else 0,
                   alpha_sha256=_frame_hash(alpha.rename_axis('agent').reset_index(), ['agent'] + list(alpha.columns)),
                   excess_sha256=_frame_hash(excess.rename_axis('agent').reset_index(), ['agent'] + list(excess.columns)),
                   interpretation='Historical whole-development fit, separate from July-only primary models. '
                                  'Static directed network scores; no target event-level attribution or causal claim.')
    return summary, result
