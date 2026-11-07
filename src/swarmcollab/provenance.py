"""Extract and evaluate versioned interaction evidence from frozen first-slice logs.

Run ``python -I -m swarmcollab.provenance`` after ``swarmcollab.pairs``.
No network access, external data, sealed-slice override, or new dependencies.
"""
import argparse
import collections
import concurrent.futures
import csv
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, labels, metrics
from .provenance_common import in_scope, iter_scoped_steps, timestamp, PACIFIC
from .provenance_links import build_links, directed_features, training_common_content
from .provenance_controls import direction_diagnostic, time_shift_control

OUT = config.OUT / 'provenance'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def extract_agent(item):
    from .provenance_git import GitExtractor
    from .provenance_structured import StructuredExtractor
    actor, filename = item
    git, structured = GitExtractor(actor), StructuredExtractor(actor)
    path = OUT / 'events' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    counts, steps = collections.Counter(), 0
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for record in iter_scoped_steps(config.STEPS / filename):
            steps += 1
            events = git.extract(record)
            structured.set_context(repo=git.current_repo(record.get('s', '')))
            events += structured.extract(record)
            for event in {e['event_id']: e for e in events}.values():
                counts[(event['family'], event['operation'], event['status'])] += 1
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    return dict(actor=actor, file=filename, steps=steps, events=sum(counts.values()),
                by_kind={'|'.join(k): v for k, v in sorted(counts.items())})


def load_events():
    summary = json.loads((OUT / 'extraction.json').read_text())
    events = []
    for entry in summary['agents']:
        with gzip.open(OUT / 'events' / entry['file'], 'rt', encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if not in_scope(event['time']) or timestamp(event['time']).astimezone(PACIFIC).date().isoformat() != event['day']:
                    raise ValueError('Cached provenance event has out-of-scope or inconsistent date')
                events.append(event)
    return events


def export_audit(chains, size, seed=20261106):
    """Label-blind stratified sample; annotation is separate from model inference."""
    groups = collections.defaultdict(list)
    for chain in chains:
        groups[(chain['level'], chain['family'])].append(chain)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected = []
    while keys and len(selected) < size:
        next_keys = []
        for key in keys:
            if len(selected) == size:
                break
            chain = groups[key].pop()
            selected.append(dict(chain, annotation='', reviewer='', rationale=''))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys
    with open(OUT / 'audit_sample.jsonl', 'w', encoding='utf-8') as stream:
        for chain in selected:
            stream.write(json.dumps(chain) + '\n')
    return dict(sample_size=len(selected), seed=seed,
                design='Round-robin sample by evidence level and family; no request labels used.',
                instructions='Inspect source, target and relation turn spans. Annotate confirmed_consumption, observation_only, failed_attempt, common_source, wrong_actor, or ambiguous. Record reviewer and rationale. Unannotated rows are never counted as negatives.')


def score_annotations(path, chains):
    if path is None:
        return {'annotated': 0, 'precision': None, 'reason': 'No independently supplied chain annotations; request AUC is not consumption precision.'}
    lookup = {chain['chain_id']: chain for chain in chains}
    groups, seen = collections.defaultdict(collections.Counter), set()
    allowed = {'confirmed_consumption', 'observation_only', 'failed_attempt', 'common_source', 'wrong_actor', 'ambiguous'}
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            annotation = row.get('annotation')
            if not annotation:
                continue
            if row['chain_id'] not in lookup or row['chain_id'] in seen:
                raise ValueError('Unknown or duplicate annotated chain')
            if annotation not in allowed or not row.get('reviewer') or not row.get('rationale'):
                raise ValueError('Annotations require a known label, reviewer and rationale')
            seen.add(row['chain_id'])
            chain = lookup[row['chain_id']]
            groups[str(chain['level'])][annotation] += 1
    result = {}
    for level, counts in groups.items():
        resolved = sum(counts.values()) - counts['ambiguous']
        result[level] = {'counts': dict(counts), 'precision_on_resolved_sample': counts['confirmed_consumption'] / resolved if resolved else None}
    return {'annotated': len(seen), 'by_level': result,
            'interpretation': 'Descriptive precision in the annotated sample, not population precision or independent human ground truth unless the recorded reviewers establish that.'}


def evaluate(chains):
    baseline = config.OUT / 'pairs' / 'features_dev.csv.gz'
    if not baseline.exists():
        return {'available': False, 'reason': 'Run swarmcollab.pairs first.'}
    F = pd.read_csv(baseline)
    # Guard stale or manually substituted baseline files too.
    if not F.day.between(config.DEV_START.isoformat(), config.DEV_END.isoformat()).all():
        raise ValueError('Baseline contains days outside the authorized development slice')
    directed = pd.DataFrame(directed_features(chains))
    levels = ['prov_level_' + str(i) for i in range(1, 5)]
    if len(directed):
        directed.to_csv(OUT / 'directed_features.csv.gz', index=False)
        directed['a'] = directed[['source', 'target']].min(axis=1)
        directed['b'] = directed[['source', 'target']].max(axis=1)
        columns = [c for c in directed if c.startswith('prov_')]
        undirected = directed.groupby(['a', 'b', 'day'])[columns].sum().reset_index()
        F = F.merge(undirected, on=['a', 'b', 'day'], how='left')
    for column in levels + ['prov_cross_artifact', 'prov_strict_identity']:
        if column not in F:
            F[column] = 0.0
    F = F.fillna(0)
    tr = F.day <= config.TRAIN_END.isoformat()
    va = F.day >= config.VALID_START.isoformat()
    # Select family vocabulary from training only; the evidence levels remain fixed.
    families = [c for c in F if c.startswith('prov_confirmed_') and F.loc[tr, c].sum() > 0]
    behaviour = ['p1_cos', 'p1_sync', 'h_write', 'h_read', 'pou_created_commit', 'pou_created_other', 'pou_first_commit', 'pou_first_other']
    base = ['act_min', 'act_max']
    sets = {'activity only': base, 'behaviour baseline': base + behaviour,
            '+ shared exposure': base + behaviour + levels[:1],
            '+ ordered/version candidates': base + behaviour + levels[:2],
            '+ stricter identity consumption': base + behaviour + levels[:2] + ['prov_strict_identity'],
            '+ confirmed consumption/downstream': base + behaviour + levels,
            '+ artifact families': base + behaviour + levels + families + ['prov_cross_artifact']}
    verbal = [c for c in F if c.startswith('prov_verbal_') and F.loc[tr, c].sum() > 0]
    sets['+ separate review channel'] = sets['+ artifact families'] + verbal
    scores = {}
    y = F.y.astype(bool).to_numpy()
    for name, columns in sets.items():
        columns = list(dict.fromkeys(columns))
        values = np.log1p(F[columns].clip(lower=0).to_numpy(float))
        model = metrics.Logistic().fit(values[tr], y[tr])
        score = model.decision(values[va])
        scores[name] = dict(auc=metrics.auc(score, y[va]), recall_at_10pct_fa=metrics.recall_at_fpr(score, y[va]),
                            features=columns, coefficients={k: float(v) for k, v in model.coefficients(columns).items()})
    F.to_csv(OUT / 'features_dev.csv.gz', index=False)
    requests, _, _ = labels.load_requests()
    forward = labels.directed_days([r for r in requests if config.DEV_START <= r['day'] <= config.DEV_END])
    observed, reversed_ = 0, 0
    for chain in chains:
        if chain['level'] >= 3:
            observed += config.as_date(chain['day']) in forward.get((chain['source_actor'], chain['target_actor']), ())
            reversed_ += config.as_date(chain['day']) in forward.get((chain['target_actor'], chain['source_actor']), ())
    return dict(available=True, models=scores, train_rows=int(tr.sum()), validation_rows=int(va.sum()),
                target='Request pair-day labels; not verified consumption labels. Development validation has been inspected during research and is not a sealed test.',
                directed_request_alignment={'same_direction_chain_count': observed, 'reversed_direction_chain_count': reversed_,
                                            'interpretation': 'Descriptive control only: fulfillment often runs opposite to request direction.'})


def actor_block_null(events, common, repetitions, seed=20261106):
    """Permute whole actor/day blocks while preserving object/time lifecycles.

    No event, version, timestamp, operation, outcome or within-day block is moved.
    Labels are not consulted. This tests sensitivity to actor identity continuity,
    not causal effects; the aggregate count can be unchanged under a permutation.
    """
    rng = random.Random(seed)
    actors = collections.defaultdict(set)
    for event in events:
        actors[event['day']].add(event['actor'])
    counts = []
    for rep in range(repetitions):
        mapping = {}
        for day, values in sorted(actors.items()):
            original = sorted(values)
            permuted = original[:]
            rng.shuffle(permuted)
            mapping.update({(day, a): b for a, b in zip(original, permuted)})
        shuffled = [dict(e, actor=mapping[(e['day'], e['actor'])]) for e in events]
        _, diagnostics = build_links(shuffled, common, include_exposure=False)
        counts.append(sum(diagnostics.get('level_' + str(i), 0) for i in (3, 4)))
        print('provenance actor-block null', rep + 1, '/', repetitions, flush=True)
    return dict(repetitions=repetitions, seed=seed, confirmed_chain_counts=counts,
                mean=float(np.mean(counts)) if counts else None,
                design='Independent permutations of whole actor/day blocks; timestamps, versions, internal correlations and observed object availability unchanged. Descriptive sensitivity, not independent pair trials.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-extract', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--null-reps', type=int, default=10)
    parser.add_argument('--audit-size', type=int, default=80)
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.null_reps < 0 or args.audit_size < 0:
        parser.error('workers must be 1..4; repetitions and audit size must be nonnegative')
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_extract:
        metadata = json.loads((config.STEPS / 'summary.json').read_text())
        items = sorted((name, info['file']) for name, info in metadata['agents'].items())
        summaries = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for summary in pool.map(extract_agent, items):
                summaries.append(summary)
                print(summary['actor'], summary['steps'], 'scoped steps,', summary['events'], 'events', flush=True)
        write_json(OUT / 'extraction.json', {'agents': summaries, 'scope': 'original main village; July 6-August 14 Pacific only'})
    events = load_events()
    common = training_common_content(events, config.TRAIN_END.isoformat())
    chains, diagnostics = build_links(events, common)
    with gzip.open(OUT / 'chains.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for chain in chains:
            stream.write(json.dumps(chain) + '\n')
    with gzip.open(OUT / 'relations.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for event in events:
            for relation in event['relations']:
                stream.write(json.dumps(dict(relation, evidence_event=event['event_id'], observed_time=event['time'])) + '\n')
    report = dict(scope={'start': str(config.DEV_START), 'end': str(config.DEV_END), 'timezone': 'America/Los_Angeles'},
                  diagnostics=diagnostics, common_content_exclusions=len(common),
                  event_families=dict(collections.Counter(e['family'] for e in events)),
                  event_statuses=dict(collections.Counter(e['status'] for e in events)),
                  audit=export_audit(chains, args.audit_size), annotations=score_annotations(args.annotations, chains),
                  evaluation=evaluate(chains), actor_block_control=actor_block_null(events, common, args.null_reps),
                  time_shift_control=time_shift_control(events, common_content=common, repetitions=min(args.null_reps, 5)),
                  direction_diagnostic=direction_diagnostic(chains),
                  elapsed_seconds=round(time.monotonic() - start, 3))
    write_json(OUT / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
