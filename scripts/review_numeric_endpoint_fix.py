#!/usr/bin/env python3
"""Audit newly admitted numeric URL endpoints without replacing published scores.

The primary comparison preserves July resource weights and the published strict
cosine threshold. A separate optional variant fits weights for the new resources
and its threshold on July only. Neither variant reruns combined detectors.

Run --plan first. June compact steps can be supplied with --shared-steps; the
ordinary steps cache supplies development, and --temporal-steps supplies the
Pacific-bounded test cache. Output directories must be new.
Only records admitted by generalization.cached_records' timestamp prefilter are
decoded. Existing raw mentions are resolved with the frozen July mapping, and
their cosine scores must reproduce every saved pair before any additions run.
The pre-fix parser, mentions, scores and July freeze are required inputs; a fresh
run of the current parser does not reconstruct that historical baseline. Use
--study and --input-cache to select archived inputs. --published-results can
optionally cross-check the threshold against a separately exported paper result.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import gzip
import importlib.util
import ipaddress
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from swarmcollab import config, labels, metrics, pairs, traces  # noqa: E402
from swarmcollab.generalization import cached_records, digest  # noqa: E402
from swarmcollab.layer_scoring import training_threshold  # noqa: E402
from swarmcollab.transfer_features import _resolve_repos, _verify_historical  # noqa: E402

STUDY = ROOT / 'outputs/generalization-20261106'
CACHE = ROOT / 'outputs/claude-review-20261107/regimes/inputs'
KEYS = ['a', 'b', 'day']
COUNT_KEYS = ['day', 'agent', 'tid']
THRESHOLD = 0.017487554042194993
PERIODS = {
    'earlier_shared_goals': ('2026-06-01', '2026-07-05'),
    'training': ('2026-07-06', '2026-07-31'),
    'validation': ('2026-08-03', '2026-08-14'),
    'test': ('2026-08-17', '2026-09-15'),
}
SNAPSHOTS = ['2026-06-13', '2026-06-22', '2026-06-26', '2026-08-18']
REPLAY_SCOPES = {
    'shared_goals': ('2026-06-01', '2026-07-05'),
    'development': ('2026-07-06', '2026-08-14'),
    'temporal': ('2026-08-17', '2026-09-15'),
}
ACTIVITY_FIELDS = ['steps', 'working', 'waiting', 'shell', 'gui', 'typed', 'other']
NUMERIC_CANDIDATE = re.compile(r'https?://(?:[^/\s@]+@)?(?:\d|\[)')


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def numeric_entity(entity):
    """New endpoint identifiers must be URLs with actual numeric hosts."""
    try:
        host = urlsplit(entity).hostname
        address = ipaddress.ip_address(host or '')
    except ValueError:
        return False
    effective = getattr(address, 'ipv4_mapped', None) or address
    return not effective.is_loopback and not effective.is_unspecified


def frozen_parser(path=None):
    path = Path(path) if path is not None else STUDY / 'frozen/src/swarmcollab/traces.py'
    if not path.is_file():
        raise ValueError('Missing pre-fix parser archive: ' + str(path))
    spec = importlib.util.spec_from_file_location('swarmcollab._numeric_audit_frozen_traces', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Numeric URL changes must not change stateful command/cwd/name logic.
    def parser_ast(filename):
        tree = ast.parse(Path(filename).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.ClassDef)
                             and n.name == 'AgentParser'), include_attributes=False)
    if parser_ast(path) != parser_ast(traces.__file__):
        raise ValueError('AgentParser changed beyond URL parsing; targeted delta proof no longer applies')
    if module.parse_url('http://10.108.0.39:6080/vnc.html'):
        raise ValueError('Frozen reference already admits numeric endpoints')
    added = traces.parse_url('http://10.108.0.39:6080/vnc.html')
    if not added or any(t not in ('site', 'page') or not numeric_entity(i) for t, i in added):
        raise ValueError('Numeric-endpoint parser fix is not ready or has an unexpected schema')
    return module, path


@contextmanager
def mention_directory(directory):
    previous = pairs.MENTIONS
    pairs.MENTIONS = directory
    try:
        yield
    finally:
        pairs.MENTIONS = previous


def cosine(counts, universe, idf, unseen):
    """Only p1_cos: the original IDF * (1 + log count) sparse dot product."""
    scores = pd.Series(0.0, index=universe.index)
    for day, target in universe.groupby('day', sort=True):
        group = counts[counts.day == day].copy()
        if group.empty:
            continue
        agents = sorted(set(target.a) | set(target.b))
        index = {a: i for i, a in enumerate(agents)}
        if not set(group.agent).issubset(index):
            raise ValueError('Resource mention belongs to an inactive agent')
        columns, entities = pd.factorize(group.tid, sort=True)
        weights = group.tid.map(idf).fillna(unseen).to_numpy(float)
        weights *= 1 + np.log(group.n.to_numpy(float))
        matrix = sparse.csr_matrix((weights, (group.agent.map(index).to_numpy(), columns)),
                                   shape=(len(agents), len(entities)))
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        products = (matrix @ matrix.T).toarray()
        ia, ib = target.a.map(index).to_numpy(), target.b.map(index).to_numpy()
        denominator = norms[ia] * norms[ib]
        scores.loc[target.index] = np.divide(products[ia, ib], denominator,
                                            out=np.zeros(len(target)), where=denominator > 0)
    return scores.to_numpy(float)


def augment_cosine(original, additions, universe, idf, unseen):
    """Add disjoint resource coordinates while preserving published tie values.

The old numerator is the published cosine times its two original norms. This
avoids attributing CSV/reconstruction roundoff to the parser change, and leaves
every unaffected profile pair byte-for-byte equal to its published score.
"""
    base = original.copy()
    base['square'] = (base.tid.map(idf).fillna(unseen).to_numpy(float)
                      * (1 + np.log(base.n.to_numpy(float)))) ** 2
    old_squares = base.groupby(['day', 'agent']).square.sum()
    scores = universe.published.to_numpy(float).copy()
    for day, target in universe.groupby('day', sort=True):
        added = additions[additions.day == day]
        if added.empty:
            continue
        agents = sorted(set(target.a) | set(target.b))
        index = {a: i for i, a in enumerate(agents)}
        columns, entities = pd.factorize(added.tid, sort=True)
        weights = added.tid.map(idf).fillna(unseen).to_numpy(float)
        weights *= 1 + np.log(added.n.to_numpy(float))
        matrix = sparse.csr_matrix((weights, (added.agent.map(index).to_numpy(), columns)),
                                   shape=(len(agents), len(entities)))
        extra_square = np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()
        old_square = np.array([old_squares.get((day, agent), 0.0) for agent in agents])
        products = (matrix @ matrix.T).toarray()
        ia, ib = target.a.map(index).to_numpy(), target.b.map(index).to_numpy()
        old_product = np.sqrt(old_square[ia]) * np.sqrt(old_square[ib])
        new_product = np.sqrt(old_square[ia] + extra_square[ia]) * np.sqrt(old_square[ib] + extra_square[ib])
        numerator = target.published.to_numpy() * old_product + products[ia, ib]
        updated = np.divide(numerator, new_product, out=np.zeros(len(target)), where=new_product > 0)
        affected = (extra_square[ia] > 0) | (extra_square[ib] > 0)
        scores[target.index[affected]] = updated[affected]
    return scores


def baseline_inputs(include_june, historical, study=STUDY, cache=CACHE):
    sources = [dict(name='development', start='2026-07-06', end='2026-08-14',
                    mentions=ROOT / 'outputs/mentions',
                    features=ROOT / 'outputs/research-layers/features_dev.csv.gz'),
               dict(name='test', start='2026-08-17', end='2026-09-15',
                    mentions=cache / 'temporal/mentions', features=study / 'temporal/features.csv.gz')]
    if include_june:
        sources.insert(0, dict(name='earlier_shared_goals', start='2026-06-01', end='2026-07-05',
                               mentions=cache / 'shared_goals/mentions',
                               features=study / 'shared_goals/features.csv.gz'))
    reference = pd.DataFrame(historical['mapping_reference'], columns=['type', 'id'])
    tables, counts, activities, provenance, parity = [], [], [], {}, []
    for source in sources:
        days = set(pd.date_range(source['start'], source['end']).strftime('%Y-%m-%d'))
        with mention_directory(source['mentions']):
            raw = pairs.load_mentions(days)
            activity = pairs.load_activity(days)
        resolved, _ = _resolve_repos(raw, reference=reference)
        represented = resolved[resolved.type.isin(pairs.TRACE_TYPES)].copy()
        represented['tid'] = represented.tid.astype(str)
        counted = represented.groupby(COUNT_KEYS, observed=True).size().reset_index(name='n')
        for column in COUNT_KEYS:
            counted[column] = counted[column].astype(str)
        frame = pd.read_csv(source['features'], usecols=KEYS + ['p1_cos'])
        if not frame.day.isin(days).all() or frame.duplicated(KEYS).any():
            raise ValueError('Saved features contain duplicate or out-of-scope rows')
        expected = pairs.universe(activity).sort_values(KEYS).reset_index(drop=True)
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        if not frame[KEYS].equals(expected):
            raise ValueError('Complete working-agent pair universe differs from saved features')
        rebuilt = cosine(counted, frame, historical['fitted_idf'], historical['unseen_idf'])
        error = float(np.max(np.abs(rebuilt - frame.p1_cos.to_numpy())))
        if error > 1e-12:
            raise ValueError(f'Published cosine reconstruction failed for {source["name"]}: {error}')
        frame['baseline_reconstructed'] = rebuilt
        parity.append(dict(source=source['name'], pairs=len(frame), raw_mentions=len(raw),
                           trace_mentions=len(represented), max_absolute_cosine_error=error))
        for path in sorted(source['mentions'].glob('*')):
            if path.is_file():
                provenance[str(path)] = digest(path)
        provenance[str(source['features'])] = digest(source['features'])
        tables.append(frame.rename(columns={'p1_cos': 'published'}))
        counts.append(counted)
        activities.append(activity)
        print('baseline verified:', source['name'], len(frame), 'pairs; error', error, flush=True)
    frame = pd.concat(tables, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    if frame.duplicated(KEYS).any():
        raise ValueError('Baseline periods overlap')
    return frame, pd.concat(counts, ignore_index=True), pd.concat(activities), provenance, parity


def relevant_text(record):
    kind = record['k']
    if kind == 'shell':
        return (record.get('x') or '') + '\n' + (record.get('o') or '') + '\n' + (record.get('e') or '')
    if kind == 'type':
        return record.get('x') or ''
    if kind not in traces.WAITING and kind not in traces.GUI_KINDS:
        return record.get('o') or ''
    return ''


def feed(parser, module, record, ordinal, when, day):
    parser.rows.clear()
    kind, t = record['k'], when.timestamp()
    if kind == 'shell':
        parser.shell(t, day, ordinal, record['s'], record.get('x') or '',
                     (record.get('o') or '') + ('\n' + record['e'] if record.get('e') else ''))
    elif kind == 'type':
        parser.typed(t, day, ordinal, record.get('x') or '')
    else:
        for typ, entity in parser.urls(record.get('o') or ''):
            parser.emit(t, day, ordinal, typ, entity, 'seen', 'out')
    return Counter(parser.rows)


def record_activity(counter, kind):
    counter['steps'] += 1
    if kind in traces.WAITING:
        counter['waiting'] += 1
        return
    counter['working'] += 1
    if kind == 'shell':
        counter['shell'] += 1
    elif kind == 'type':
        counter['gui'] += 1
        counter['typed'] += 1
    elif kind in traces.GUI_KINDS:
        counter['gui'] += 1
    else:
        counter['other'] += 1


def reconcile_activity(observed, expected, start, end):
    """Fail on missing, duplicate, or differently classified cached steps."""
    expected = expected[expected.day.between(start, end)]
    if expected.duplicated(['agent', 'day']).any():
        raise ValueError('Saved activity has duplicate agent/day rows')
    target = {(row.agent, row.day): {field: int(getattr(row, field)) for field in ACTIVITY_FIELDS}
              for row in expected.itertuples(index=False)}
    differences, rows = [], []
    for agent, day in sorted(set(target) | set(observed)):
        actual, wanted = observed.get((agent, day), {}), target.get((agent, day), {})
        for field in ACTIVITY_FIELDS:
            a, b = actual.get(field, 0), wanted.get(field, 0)
            if a != b:
                differences.append(dict(agent=agent, day=day, field=field, observed=a, expected=b))
        rows.append(dict(agent=agent, day=day, **{field: actual.get(field, 0) for field in ACTIVITY_FIELDS}))
    if differences:
        raise ValueError('Compact-step activity differs from saved baseline: ' + json.dumps(differences[:20]))
    return rows


def replay(directory, name, old, roster, output, allowed_agent_days, activity):
    """Full parser row difference on candidate steps; no replay of other payloads."""
    if name not in REPLAY_SCOPES:
        raise ValueError('Unregistered compact-step replay scope')
    start, end = REPLAY_SCOPES[name]
    meta_path = directory / 'summary.json'
    metadata = json.loads(meta_path.read_text())
    input_hashes = {str(meta_path): digest(meta_path)}
    parity_path = directory.parent / 'cache-parity.json'
    if parity_path.is_file():
        input_hashes[str(parity_path)] = digest(parity_path)
    aggregates, coverage, stats = Counter(), Counter(), []
    observed_activity = {}
    with gzip.open(output / f'{name}-new-mentions.jsonl.gz', 'wt') as stream:
        for actor, info in sorted(metadata['agents'].items()):
            path = directory / info['file']
            input_hashes[str(path)] = digest(path)
            before, after = old.AgentParser(actor, roster), traces.AgentParser(actor, roster)
            scanned = candidates = changed = additions = 0
            for ordinal, record in enumerate(cached_records(path, start, end)):
                scanned += 1
                when = config.parse_utc(record['t'])
                day = when.astimezone(labels.PACIFIC).date().isoformat()
                record_activity(observed_activity.setdefault((actor, day), Counter()), record['k'])
                text = relevant_text(record)
                if not NUMERIC_CANDIDATE.search(text):
                    continue
                candidates += 1
                new_entities = set()
                for url in traces.URL_RE.findall(text):
                    delta = Counter(traces.parse_url(url)) - Counter(old.parse_url(url))
                    for (typ, entity), count in delta.items():
                        if typ not in ('site', 'page') or not numeric_entity(entity):
                            raise ValueError('URL delta is not exclusively newly admitted numeric endpoints')
                        new_entities.add((typ, entity))
                if (actor, day) not in allowed_agent_days:
                    raise ValueError(f'Eligible numeric URL action not in existing working universe: {actor}, {day}')
                a = feed(before, old, record, ordinal, when, day)
                b = feed(after, traces, record, ordinal, when, day)
                if a - b:
                    raise ValueError('Current parser removed or changed old mentions; additive replay is invalid')
                extra = b - a
                if not extra:
                    continue
                changed += 1
                coverage[(actor, day)] += 1
                for row, multiplicity in extra.items():
                    t, _, step, typ, entity, op, src, k = row
                    if (typ, entity) not in new_entities:
                        raise ValueError('Full AgentParser difference is not attributable to the URL difference')
                    aggregates[(day, actor, typ + '|' + entity)] += multiplicity
                    additions += multiplicity
                    stream.write(json.dumps(dict(agent=actor, day=day, t=t, step=step, type=typ,
                                                 id=entity, op=op, src=src, k=k, count=multiplicity,
                                                 record_id=record['i'], session_id=record.get('s'),
                                                 source_cache=str(path)), ensure_ascii=False) + '\n')
            stats.append(dict(agent=actor, scope=name, eligible_steps=scanned, candidate_steps=candidates,
                              changed_steps=changed, new_mentions=additions))
            print('replay:', name, actor, scanned, 'steps,', additions, 'new mentions', flush=True)
    reconciled = reconcile_activity(observed_activity, activity, start, end)
    pd.DataFrame(reconciled).to_csv(output / f'{name}-activity-parity.csv', index=False)
    return aggregates, coverage, stats, input_hashes


def scoped_labels(start, end):
    if (start, end) not in set(PERIODS.values()):
        raise ValueError('Labels may only be loaded for explicitly registered periods')
    rows, count, orphan = labels.load_requests(start=start, end=end)
    if any(not start <= str(row['day']) <= end for row in rows):
        raise ValueError('Label scope escape')
    return rows, dict(start=start, end=end, request_instances=count, orphan_instances=orphan)


def performance(frame, column, threshold):
    score, y = frame[column].to_numpy(float), frame.y.to_numpy(bool)
    alert = score > threshold
    tp, fp = int((alert & y).sum()), int((alert & ~y).sum())
    return dict(pair_days=len(frame), positive_pair_days=int(y.sum()), observed_days=frame.day.nunique(),
                auroc=metrics.auc(score, y) if y.any() and not y.all() else None,
                threshold=threshold, alerts=int(alert.sum()), alerts_per_day=float(alert.sum()/frame.day.nunique()),
                true_positives=tp, false_positives=fp, false_negatives=int((~alert & y).sum()),
                true_negatives=int((~alert & ~y).sum()), recall=tp/int(y.sum()) if y.any() else None,
                precision=tp/int(alert.sum()) if alert.any() else None,
                false_positive_rate=fp/int((~y).sum()) if (~y).any() else None)


def snapshot(frame, column, threshold):
    names = sorted(set(frame.a) | set(frame.b))
    edges = [dict(a=r.a, b=r.b, score=float(getattr(r, column))) for r in
             frame[frame[column] > threshold].itertuples(index=False)]
    degree = Counter(a for e in edges for a in (e['a'], e['b']))
    return dict(nodes=names, edges=edges, isolates=[a for a in names if not degree[a]],
                degrees={a: degree[a] for a in names})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/numeric-endpoint-fix-20261108')
    ap.add_argument('--shared-steps', type=Path,
                    default=ROOT / 'outputs/numeric-endpoint-fix-inputs/shared_goals/steps')
    ap.add_argument('--main-steps', type=Path, default=ROOT / 'outputs/steps')
    ap.add_argument('--temporal-steps', type=Path,
                    default=ROOT / 'outputs/numeric-endpoint-fix-inputs/temporal/steps')
    ap.add_argument('--study', type=Path, default=STUDY,
                    help='Historical study directory containing frozen/src, historical-freeze.json and saved features')
    ap.add_argument('--input-cache', type=Path, default=CACHE,
                    help='Archived pre-fix mentions under shared_goals/mentions and temporal/mentions')
    ap.add_argument('--published-results', type=Path,
                    help='Optional result JSON with model_checks.equivalent_cosine_threshold to cross-check')
    ap.add_argument('--skip-shared-goals', action='store_true')
    ap.add_argument('--no-refit', action='store_true', help='Omit the separately reported July-only recalibration')
    ap.add_argument('--plan', action='store_true', help='Inspect input availability without replay or writes')
    args = ap.parse_args()
    scopes = [(args.main_steps, 'development'), (args.temporal_steps, 'temporal')]
    if not args.skip_shared_goals:
        scopes.insert(0, (args.shared_steps, 'shared_goals'))
    frozen_path = args.study / 'frozen/src/swarmcollab/traces.py'
    freeze_path = args.study / 'historical-freeze.json'
    if args.plan:
        print(json.dumps(dict(output=str(args.output), output_exists=args.output.exists(),
                              replays=[dict(directory=str(p), scope=name,
                                            start=REPLAY_SCOPES[name][0], end=REPLAY_SCOPES[name][1],
                                            summary_exists=(p/'summary.json').is_file(),
                                            cache_files=len(list(p.glob('*.jsonl.gz')))) for p, name in scopes],
                              historical_inputs={str(p): p.exists() for p in [
                                  frozen_path, freeze_path, ROOT / 'outputs/mentions',
                                  ROOT / 'outputs/research-layers/features_dev.csv.gz',
                                  args.input_cache / 'temporal/mentions', args.study / 'temporal/features.csv.gz',
                                  *([] if args.skip_shared_goals else [args.input_cache / 'shared_goals/mentions',
                                                                      args.study / 'shared_goals/features.csv.gz'])]},
                              historical_input_requirement='Pre-fix parser, mentions, scores and July freeze; current parser outputs are not a substitute.',
                              published_results_cross_check=str(args.published_results) if args.published_results else None,
                              primary_threshold=THRESHOLD, july_refit=not args.no_refit), indent=2))
        return
    if args.output.exists():
        raise ValueError('Audit output already exists; refusing to overwrite any previous run')
    for directory, _ in scopes:
        if not (directory / 'summary.json').is_file():
            raise ValueError('Compact step cache is incomplete: ' + str(directory))
    started = time.monotonic()
    old, frozen_path = frozen_parser(frozen_path)
    historical = json.loads(freeze_path.read_text())
    _verify_historical(historical)
    if (historical['train_start'], historical['train_end']) != PERIODS['training']:
        raise ValueError('Historical weights were not fitted on the registered July period')
    if args.published_results is not None:
        if json.loads(args.published_results.read_text())['model_checks']['equivalent_cosine_threshold'] != THRESHOLD:
            raise ValueError('Published fixed threshold differs from audit definition')
    frame, original_counts, activity, inputs, parity = baseline_inputs(not args.skip_shared_goals, historical,
                                                                     args.study, args.input_cache)
    args.output.mkdir(parents=True)
    (args.output/'audit-runner.py').write_bytes(Path(__file__).read_bytes())
    code_paths = [Path(__file__), Path(traces.__file__), frozen_path, Path(pairs.__file__),
                  ROOT/'src/swarmcollab/generalization.py', ROOT/'src/swarmcollab/transfer_features.py',
                  ROOT/'src/swarmcollab/layer_scoring.py', ROOT/'src/swarmcollab/agent_traces.py',
                  ROOT/'src/swarmcollab/provenance_common.py', Path(config.__file__),
                  Path(labels.__file__), Path(metrics.__file__),
                  ROOT/'tests/test_review_numeric_endpoint_fix.py']
    code_hashes = {str(p): digest(p) for p in code_paths}
    inputs[str(freeze_path)] = digest(freeze_path)
    if args.published_results is not None:
        inputs[str(args.published_results)] = digest(args.published_results)
    roster = sorted(set(activity.agent))
    allowed = set(zip(activity.loc[activity.working > 0, 'agent'], activity.loc[activity.working > 0, 'day']))
    additions, coverage, stats = Counter(), Counter(), []
    for directory, name in scopes:
        delta, covered, agent_stats, hashes = replay(directory, name, old, roster, args.output, allowed, activity)
        additions.update(delta); coverage.update(covered); stats.extend(agent_stats); inputs.update(hashes)
    delta_counts = pd.DataFrame([(*key, n) for key, n in sorted(additions.items())], columns=COUNT_KEYS + ['n'])
    if delta_counts.empty:
        raise ValueError('Parser fix admitted no new mentions; inspect the replay before interpreting results')
    if set(delta_counts.tid) & set(original_counts.tid):
        raise ValueError('New numeric identifiers collide with existing resource identities')
    all_counts = pd.concat([original_counts, delta_counts], ignore_index=True)
    frame['fixed'] = augment_cosine(original_counts, delta_counts, frame,
                                   historical['fitted_idf'], historical['unseen_idf'])
    full_rebuild = cosine(all_counts, frame, historical['fitted_idf'], historical['unseen_idf'])
    delta_equivalence_error = float(np.max(np.abs(full_rebuild-frame.fixed.to_numpy())))
    if delta_equivalence_error > 1e-12:
        raise ValueError('Incremental cosine differs from full augmented-count reconstruction')
    thresholds = {'published': THRESHOLD, 'fixed': THRESHOLD}
    refit_details = None
    if not args.no_refit:
        july = delta_counts[delta_counts.day.between(*PERIODS['training'])]
        counts = july.groupby('tid').agent.nunique()
        n_agents = len(historical['training_actors'])
        if not set(july.agent).issubset(historical['training_actors']):
            raise ValueError('July additions include agents outside the July fitting universe')
        refit_idf = dict(historical['fitted_idf'])
        refit_idf.update({tid: float(np.log(n_agents/count)) for tid, count in counts.items()})
        frame['july_refit'] = augment_cosine(original_counts, delta_counts, frame,
                                            refit_idf, historical['unseen_idf'])
        rebuilt_refit = cosine(all_counts, frame, refit_idf, historical['unseen_idf'])
        if np.max(np.abs(rebuilt_refit-frame.july_refit.to_numpy())) > 1e-12:
            raise ValueError('Incremental July-refit cosine differs from full reconstruction')
        refit_details = dict(training_agents=n_agents, new_july_resource_agents=counts.to_dict(),
                             new_july_idf={tid: refit_idf[tid] for tid in counts.index},
                             unchanged_original_idf_entries=len(historical['fitted_idf']))
    # Lock all numerical scores before consulting any request labels.
    score_path = args.output / 'scores-before-labels.csv.gz'
    frame.to_csv(score_path, index=False)
    lock = dict(scores_sha256=digest(score_path), scope='No request labels loaded before this score lock.',
                source_sha256=code_hashes, input_sha256=inputs, baseline_parity=parity)
    save(args.output/'prediction-lock.json', lock)
    label_scopes = []
    if refit_details is not None:
        july_rows, scope = scoped_labels(*PERIODS['training']); label_scopes.append(scope)
        training = pairs.add_labels(frame[frame.day.between(*PERIODS['training'])].copy(), july_rows)
        thresholds['july_refit'] = float(training_threshold(training.july_refit.to_numpy(), training.y.to_numpy()))
        refit_details['threshold'] = thresholds['july_refit']
        refit_details['threshold_rule'] = 'July negatives only; empirical order statistic, strict score > threshold, target FPR 0.10.'
        save(args.output/'july-refit-calibration.json', refit_details)
    if digest(score_path) != lock['scores_sha256']:
        raise ValueError('Scores changed after label lock')
    reports = {}
    labelled = []
    for period, bounds in PERIODS.items():
        part = frame[frame.day.between(*bounds)].copy()
        if part.empty:
            continue
        rows, scope = scoped_labels(*bounds); label_scopes.append(scope)
        part = pairs.add_labels(part, rows)
        reports[period] = {variant: performance(part, variant, cut) for variant, cut in thresholds.items()}
        reports[period]['reconstruction_only_auroc'] = metrics.auc(part.baseline_reconstructed.to_numpy(),
                                                                 part.y.to_numpy())
        reports[period]['changed_pairs'] = int((np.abs(part.fixed - part.published) > 1e-12).sum())
        reports[period]['new_fixed_alerts'] = int(((part.fixed > THRESHOLD) & ~(part.published > THRESHOLD)).sum())
        reports[period]['removed_fixed_alerts'] = int((~(part.fixed > THRESHOLD) & (part.published > THRESHOLD)).sum())
        labelled.append(part)
    evaluated = pd.concat(labelled).sort_values(KEYS)
    evaluated.to_csv(args.output/'evaluated-pairs.csv.gz', index=False)
    graphs = {day: {v: snapshot(evaluated[evaluated.day == day], v, t) for v, t in thresholds.items()}
              for day in SNAPSHOTS if (evaluated.day == day).any()}
    save(args.output/'snapshots.json', graphs)
    delta_counts.to_csv(args.output/'new-resource-counts.csv.gz', index=False)
    pd.DataFrame(stats).to_csv(args.output/'replay-counts.csv', index=False)
    pd.DataFrame([dict(agent=a, day=d, steps_with_added_resources=n)
                  for (a, d), n in sorted(coverage.items())]).to_csv(args.output/'affected-agent-days.csv', index=False)
    near_boundary = evaluated[(evaluated.published-THRESHOLD).abs() < 1e-12]
    near_boundary.to_csv(args.output/'published-threshold-near-ties.csv', index=False)
    for path, expected in code_hashes.items():
        if digest(path) != expected:
            raise ValueError('Source code changed during the audit: ' + path)
    result = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  scope='Additive numeric-endpoint parser audit; all published outputs preserved. No combined methods rerun.',
                  interpretation='Post-hoc diagnostic. Fixed variant isolates extraction under the original weights/threshold; July-refit variant recalibrates only on July. June uses later July information retrospectively.',
                  primary_threshold=THRESHOLD, historical_semantic_sha256=historical['sha256'],
                  published_results_cross_check=str(args.published_results) if args.published_results else None,
                  periods=reports, replay_steps=sum(s['eligible_steps'] for s in stats),
                  changed_steps=sum(s['changed_steps'] for s in stats), new_mentions=sum(additions.values()),
                  new_resource_identifiers=delta_counts.tid.nunique(), source_sha256=code_hashes,
                  input_sha256=inputs, label_source_sha256=digest(config.LABELS),
                  label_scopes=label_scopes, prediction_lock_sha256=digest(args.output/'prediction-lock.json'),
                  targeted_replay_equivalence='AgentParser class AST unchanged; full current-minus-frozen row multiset checked on every numeric-URL candidate. No removals, only new site/page identifiers from the URL parser delta. These identifiers cannot modify cwd/repo identity state.',
                  score_update='Disjoint-coordinate norm/dot update anchored to published cosine; unchanged profiles retain published values. Full augmented-count reconstruction cross-checked separately.',
                  augmented_cosine_equivalence_max_error=delta_equivalence_error,
                  activity_parity='Every replayed agent/Pacific-day exactly matches the saved baseline steps, working, waiting, shell, gui, typed, and other counts.',
                  trace_coverage_note='Steps with added resources may already contain other resources; this is not a count of newly trace-bearing steps.',
                  baseline_parity=parity, refit=refit_details,
                  elapsed_seconds=time.monotonic()-started)
    save(args.output/'summary.json', result)
    print(json.dumps({'periods': reports, 'new_mentions': result['new_mentions'],
                      'changed_steps': result['changed_steps'], 'output': str(args.output)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
