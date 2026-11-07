"""Predeclared call-index windows on already prepared, outcome-blind arrays.

This module has no file readers, schema adapter, detector, or outcome labels.
Cuts are boundaries: ``cut`` is the index of the first post-anchor call. Calls
at [cut-window, cut) are before; calls at [cut, cut+window) are after. Runs are
never concatenated. Reference distributions condition on these observed arrays;
they do not estimate a counterfactual message effect or independent-call error.
"""
from collections.abc import Mapping

import numpy as np


PRIMARY_WINDOW = 100
SENSITIVITY_WINDOWS = (50, 200)
WINDOWS = (PRIMARY_WINDOW,) + SENSITIVITY_WINDOWS
PLACEBO_DRAWS = 500
SEED = 20261106
ANCHOR_KINDS = ('external_message', 'task_start', 'session_start')
FALLBACK_STRIDE = 100
FALLBACK_TOP_K = 5
FALLBACK_EXCLUSION_DISTANCE = 200


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return int(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(name + ' must be a nonempty string')
    return value


def _prepare_runs(runs):
    if not isinstance(runs, (list, tuple)) or not runs:
        raise ValueError('At least one explicitly bounded run is required')
    prepared, family_names, count_support = {}, None, None
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError('Each run must be a mapping')
        run_id = _identifier(run.get('run_id'), 'run_id')
        if run_id in prepared:
            raise ValueError('Duplicate run_id: ' + run_id)
        families = run.get('families')
        if not isinstance(families, Mapping) or not families:
            raise ValueError('Each run must declare nonempty families')
        names = sorted(_identifier(name, 'family') for name in families)
        counts_present = {name: 'counts' in families[name] for name in names
                          if isinstance(families[name], Mapping)}
        if family_names is None:
            family_names, count_support = names, counts_present
        if names != family_names or counts_present != count_support:
            raise ValueError('Every run must declare identical families and count availability')
        arrays, prefixes, lengths = {}, {}, set()
        for family in names:
            measurements = families[family]
            if not isinstance(measurements, Mapping) or 'binary' not in measurements:
                raise ValueError('Each family requires a binary call-participation array')
            if set(measurements) - {'binary', 'counts'}:
                raise ValueError('Unknown family measurement; expected binary or counts')
            arrays[family], prefixes[family] = {}, {}
            for measure, values in measurements.items():
                try:
                    array = np.asarray(values, dtype=float)
                except (TypeError, ValueError) as error:
                    raise ValueError('Call measurements must be numeric arrays') from error
                if (array.ndim != 1 or not np.isfinite(array).all()
                        or (array < 0).any() or (array != np.floor(array)).any()):
                    raise ValueError('Call measurements must be finite nonnegative integer arrays')
                if measure == 'binary' and not np.isin(array, [0, 1]).all():
                    raise ValueError('Binary participation arrays must contain only zero or one')
                # Prefix sums stay exactly representable for these count arrays.
                if array.sum() > 2 ** 53:
                    raise ValueError('Call counts exceed exact prefix-sum numeric capacity')
                lengths.add(len(array))
                arrays[family][measure] = array.copy()
                prefixes[family][measure] = np.r_[0., np.cumsum(array)]
        if len(lengths) != 1:
            raise ValueError('All family arrays within a run must have equal length')
        n_calls = next(iter(lengths))
        if 'n_calls' in run and _integer(run['n_calls'], 'n_calls') != n_calls:
            raise ValueError('Declared n_calls differs from the family-array length')
        prepared[run_id] = dict(run_id=run_id, n_calls=n_calls, arrays=arrays, prefixes=prefixes)
    return {name: prepared[name] for name in sorted(prepared)}, family_names


def _prepare_anchors(anchors, runs):
    if anchors is None:
        return []
    if not isinstance(anchors, (list, tuple)):
        raise ValueError('anchors must be a sequence of metadata-selected boundaries')
    result, seen = [], set()
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise ValueError('Each anchor must be a mapping')
        run_id = _identifier(anchor.get('run_id'), 'anchor run_id')
        anchor_id = _identifier(anchor.get('anchor_id'), 'anchor_id')
        kind = anchor.get('kind')
        if run_id not in runs or kind not in ANCHOR_KINDS:
            raise ValueError('Anchor refers to an unknown run or unsupported metadata kind')
        cut = _integer(anchor.get('cut'), 'anchor cut')
        if cut > runs[run_id]['n_calls']:
            raise ValueError('Anchor boundary lies outside its recorded run')
        if (run_id, anchor_id) in seen:
            raise ValueError('Duplicate anchor ID within a run')
        seen.add((run_id, anchor_id))
        result.append(dict(run_id=run_id, anchor_id=anchor_id, kind=kind, cut=cut))
    return sorted(result, key=lambda item: (ANCHOR_KINDS.index(item['kind']),
                                          item['run_id'], item['cut'], item['anchor_id']))


def _eligible(run, cut, window):
    return window <= cut <= run['n_calls'] - window


def _delta_count(prefix, cuts, window):
    return (prefix[cuts + window] - prefix[cuts]) - (prefix[cuts] - prefix[cuts - window])


def _window(run, cut, window, families=None):
    available_before, available_after = cut, run['n_calls'] - cut
    record = dict(window_calls_per_side=window, available_before_calls=available_before,
                  available_after_calls=available_after, full_symmetric_window=_eligible(run, cut, window))
    if not record['full_symmetric_window']:
        record.update(status='excluded_insufficient_calls', before_denominator=None, after_denominator=None,
                      missing_before_calls=max(window-available_before, 0),
                      missing_after_calls=max(window-available_after, 0), families={})
        return record
    record.update(status='available', before_index_start=cut-window, before_index_end_exclusive=cut,
                  after_index_start=cut, after_index_end_exclusive=cut+window,
                  before_denominator=window, after_denominator=window, families={})
    for family in families or run['prefixes']:
        record['families'][family] = {}
        for measure, prefix in run['prefixes'][family].items():
            before, after = int(prefix[cut]-prefix[cut-window]), int(prefix[cut+window]-prefix[cut])
            value = dict(before_sum=before, after_sum=after, difference_sum=after-before,
                         before_per_1000_calls=1000*before/window,
                         after_per_1000_calls=1000*after/window,
                         change_per_1000_calls=1000*(after-before)/window)
            if measure == 'binary':
                value.update(before_fraction=before/window, after_fraction=after/window,
                             change_fraction=(after-before)/window)
            record['families'][family][measure] = value
    return record


def summarize_cut(run, cut, window=PRIMARY_WINDOW):
    """Report one boundary without partial windows or cross-run observations."""
    prepared, _ = _prepare_runs([run])
    selected = next(iter(prepared.values()))
    cut, window = _integer(cut, 'cut'), _integer(window, 'window', 1)
    if cut > selected['n_calls']:
        raise ValueError('Cut boundary lies outside its recorded run')
    return _window(selected, cut, window)


def _anchor_reference(runs, anchors, families, kind, window):
    selected = [item for item in anchors if item['kind'] == kind]
    eligible = [item for item in selected if _eligible(runs[item['run_id']], item['cut'], window)]
    counts = {run_id: sum(item['run_id'] == run_id for item in eligible) for run_id in runs}
    record = dict(kind=kind, window_calls_per_side=window, metadata_anchors=len(selected),
                  eligible_anchors=len(eligible), excluded_anchors=len(selected)-len(eligible),
                  eligible_anchors_per_run=counts, draws=PLACEBO_DRAWS, seed=SEED,
                  status='available' if eligible else ('no_metadata_anchors' if not selected else 'insufficient_data'),
                  interpretation='Conditional within-run uniform-cutpoint reference; not a causal test.',
                  multiplicity='Maximum absolute change across eligible anchors within each family and measure. '
                               'No correction across families, measures, anchor kinds, or sensitivity windows.',
                  families={})
    # Reinitialize for each kind/window to make results independent of which
    # other metadata kinds are present. Draw cuts once and reuse across families.
    rng = np.random.default_rng(SEED)
    placebo = {run_id: rng.integers(window, run['n_calls']-window+1,
                                   size=(PLACEBO_DRAWS, counts[run_id]))
               for run_id, run in runs.items() if counts[run_id]}
    for family in families:
        record['families'][family] = {}
        for measure in next(iter(runs.values()))['prefixes'][family]:
            if not eligible:
                record['families'][family][measure] = dict(observed_max_absolute_change_per_1000_calls=None,
                    placebo_max_absolute_changes_per_1000_calls=[], reference_tail_probability=None,
                    null_at_least_observed=None)
                continue
            observed = max(abs(_delta_count(runs[item['run_id']]['prefixes'][family][measure],
                                             item['cut'], window)) for item in eligible)
            maxima = np.zeros(PLACEBO_DRAWS)
            for run_id, cuts in placebo.items():
                values = np.abs(_delta_count(runs[run_id]['prefixes'][family][measure], cuts, window))
                maxima = np.maximum(maxima, values.max(axis=1))
            at_least = int((maxima >= observed).sum())
            record['families'][family][measure] = dict(
                observed_max_absolute_change_per_1000_calls=float(1000*observed/window),
                placebo_max_absolute_changes_per_1000_calls=(1000*maxima/window).tolist(),
                null_at_least_observed=at_least,
                reference_tail_probability=(1+at_least)/(PLACEBO_DRAWS+1))
    return record


def _fallback(runs, anchors, families):
    if any(anchor['kind'] == 'external_message' for anchor in anchors):
        return dict(triggered=False, status='external_message_metadata_present', families={})
    candidates = {run_id: np.arange(PRIMARY_WINDOW, run['n_calls']-PRIMARY_WINDOW+1,
                                   FALLBACK_STRIDE, dtype=int) for run_id, run in runs.items()}
    record = dict(triggered=True, status='available' if any(len(cuts) for cuts in candidates.values()) else 'insufficient_data',
                  reason='No external-message metadata anchors were supplied.',
                  candidate_cuts_per_run={run_id: cuts.tolist() for run_id, cuts in candidates.items()},
                  selection='Top five absolute primary-window changes per family and measure; '
                            'ties use earliest call index, then lexicographic run ID. Greedily exclude '
                            'same-run cuts at distance <= 200 calls from a selected cut.',
                  interpretation='Descriptive selected maxima, not external-message interventions. '
                                 'No p-values or placebo tail probabilities are attached.', families={})
    for family in families:
        record['families'][family] = {}
        for measure in next(iter(runs.values()))['prefixes'][family]:
            ranked = []
            for run_id, cuts in candidates.items():
                changes = _delta_count(runs[run_id]['prefixes'][family][measure], cuts, PRIMARY_WINDOW)
                ranked.extend((abs(float(change)), int(cut), run_id) for cut, change in zip(cuts, changes))
            ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
            chosen = []
            for _, cut, run_id in ranked:
                if any(item['run_id'] == run_id and abs(cut-item['cut']) <= FALLBACK_EXCLUSION_DISTANCE
                       for item in chosen):
                    continue
                chosen.append(dict(rank=len(chosen)+1, run_id=run_id, cut=cut, selected_measure=measure,
                                   windows={str(window): _window(runs[run_id], cut, window, [family])
                                            for window in WINDOWS}))
                if len(chosen) == FALLBACK_TOP_K:
                    break
            record['families'][family][measure] = chosen
    return record


def analyze_windows(runs, anchors=None):
    """Analyze normalized per-call evidence arrays using the locked protocol.

    ``runs`` is a list of ``{'run_id': str, 'families': {name:
    {'binary': sequence, 'counts': optional sequence}}, 'n_calls': optional int}``.
    Every run declares identical families and optional count availability. An
    empty recorded run is supported. Binary and count arrays are separate
    measurements and may peak at different cuts. Counts need not equal binary
    participation indicators (one call may participate in several artifacts).

    ``anchors`` is a list of ``{'run_id': str, 'anchor_id': str, 'kind':
    'external_message'|'task_start'|'session_start', 'cut': int}``. Selection
    must be frozen from metadata before arrays are evaluated. Existing external
    anchors without enough surrounding calls do not trigger fallback scanning.
    """
    prepared, families = _prepare_runs(runs)
    selected = _prepare_anchors(anchors, prepared)
    return dict(schema_version=1,
        configuration=dict(primary_window_calls_per_side=PRIMARY_WINDOW,
            sensitivity_window_calls_per_side=list(SENSITIVITY_WINDOWS),
            placebo_draws=PLACEBO_DRAWS, seed=SEED, placebo_sampling='Uniform eligible integer cutpoints with replacement, matched eligible anchor count per run.',
            placebo_seed_scope='Reset seed for each metadata kind and window; same cuts reused across families and measures.',
            tail_probability='(1 + number of placebo maxima >= observed maximum) / 501',
            fallback_stride_calls=FALLBACK_STRIDE, fallback_top_k_per_family_and_measure=FALLBACK_TOP_K,
            fallback_exclusion='Same run and absolute cut distance <= 200 calls.',
            cut_definition='Zero-based index of first post-anchor call; before [cut-window, cut), after [cut, cut+window).'),
        runs=[dict(run_id=run_id, n_calls=run['n_calls']) for run_id, run in prepared.items()],
        declared_families=families,
        count_measures={family: 'counts' in next(iter(prepared.values()))['prefixes'][family] for family in families},
        anchor_counts={kind: sum(item['kind'] == kind for item in selected) for kind in ANCHOR_KINDS},
        anchors=[dict(**item, windows={str(window): _window(prepared[item['run_id']], item['cut'], window)
                                      for window in WINDOWS}) for item in selected],
        anchor_references={kind: {str(window): _anchor_reference(prepared, selected, families, kind, window)
                                  for window in WINDOWS} for kind in ANCHOR_KINDS},
        fallback=_fallback(prepared, selected, families),
        limitations=['Call-index windows are not elapsed-time windows.',
                     'Array preparation must separately report tool coverage, observation order, and actor-boundary availability.',
                     'No run-level independence is assumed and no independent-call confidence interval is estimated.',
                     'Conditional cutpoint references and selected changes do not establish causal effects or cross-agent identity.'])
