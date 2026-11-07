"""Compare general agent-work traces with AI Village-specific traces.

The AI researcher develops these deterministic rules; no runtime model is used.
Run after pairs and provenance. Only the authorized development cache is read.
Outputs are separate from the historical pipeline, under outputs/research-layers.
"""
import argparse
import collections
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .provenance_common import command_segments, in_scope, iter_scoped_steps, make_event, timestamp, PACIFIC

KEYS = ['a', 'b', 'day']
BEHAVIOR = ['p1_cos', 'p1_sync', 'h_write', 'h_read', 'pou_created_commit',
            'pou_created_other', 'pou_first_commit', 'pou_first_other']
ACTIVITY = ['act_min', 'act_max']
READS = {'read', 'execute', 'acquire', 'consume', 'observe', 'mention', 'seen'}
WRITES = {'write', 'create', 'produce', 'transform', 'incorporate'}


def is_history_dump(record):
    """Shared measurement mask for known Village communication-history wrappers.

    Inspect direct executed operands, not incidental names in source or output.
    This dataset exclusion applies equally to both new layers, not as a detector
    feature. The historical precomputed baseline is retained unchanged.
    """
    if not in_scope(record['t']) or record.get('k') != 'shell':
        return False
    wrappers = {'pre_send_chat.sh', 'session_start.sh', 'session_end.sh'}
    for argv in command_segments(record.get('x') or ''):
        head = argv[0].rsplit('/', 1)[-1]
        if head in wrappers:
            return True
        if head in ('bash', 'sh', 'source', '.') and len(argv) > 1 and argv[1].rsplit('/', 1)[-1] in wrappers:
            return True
        if head in ('cat', 'head', 'tail', 'less', 'more') and any(
                value.rsplit('/', 1)[-1] == 'public_comms.md' for value in argv[1:] if not value.startswith('-')):
            return True
    return False


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class RosterMentions:
    """Exact supplied names are a general verbal signal, with no village aliases."""
    def __init__(self, actor, roster):
        self.actor = actor
        self.patterns = [(name, re.compile(r'(?<!\w)' + re.escape(name) + r'(?![._-]?\d)(?!\w)', re.I))
                         for name in sorted(set(roster), key=lambda x: (-len(x), x)) if name != actor]
        self.by_first = collections.defaultdict(list)
        for name, pattern in self.patterns:
            self.by_first[name.split()[0].lower()].append((name, pattern))
        first = sorted(self.by_first, key=lambda x: (-len(x), x))
        self.trigger = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(x) for x in first) + ')', re.I) if first else None

    def extract(self, record):
        if not in_scope(record['t']):
            return []
        if record.get('k') in ('shell', 'type'):
            sources = [('cmd' if record['k'] == 'shell' else 'typed', record.get('x') or '')]
        else:
            sources = []
        sources.append(('out', record.get('o') or ''))
        result = []
        for source, text in sources:
            found = {}
            if self.trigger is not None:
                for start in self.trigger.finditer(text):
                    for name, pattern in self.by_first[start.group().lower()]:
                        match = pattern.match(text, start.start())
                        if match:
                            found.setdefault(name, match)
                            break
            for name, match in sorted(found.items()):
                result.append(make_event(self.actor, record, 'roster_name', 'mention', 'unknown',
                    'actor:' + name, research_layer=1, rule_id='exact_supplied_actor_name',
                    source=source, verbal_channel=True, related_actor=name, mention_weight=1 / len(found),
                    relation_authority=False, evidence=[{'field': source, 'start': match.start(),
                                                       'end': match.end(), 'text': match.group()}]))
        return result


def training_records(items):
    for actor, filename in items:
        for record in iter_scoped_steps(config.STEPS / filename):
            if timestamp(record['t']).astimezone(PACIFIC).date() <= config.TRAIN_END:
                yield actor, record


def extract_agent(task):
    from .agent_traces import AgentTraceExtractor
    from .village_traces import VillageTraceExtractor
    actor, filename, roster, rules, output = task
    general = AgentTraceExtractor(actor)
    village = VillageTraceExtractor(actor, rules) if rules is not None else None
    names = RosterMentions(actor, roster)
    path = Path(output) / 'events' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    counts, coverage, sample = collections.Counter(), collections.Counter(), {}
    steps, excluded = 0, 0
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for record in iter_scoped_steps(config.STEPS / filename):
            steps += 1
            if is_history_dump(record):
                excluded += 1
                continue
            events = general.extract(record) + names.extract(record)
            if village is not None:
                events += village.extract(record)
            unique = {event['event_id']: event for event in events}
            for event in unique.values():
                key = str(event['research_layer']) + '|' + event['family']
                counts[key] += 1
                sample.setdefault(key, event)
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            for layer in {event['research_layer'] for event in unique.values()}:
                coverage[str(layer)] += 1
    return dict(actor=actor, file=filename, steps=steps, counts=dict(counts),
                covered_steps=dict(coverage), excluded_history_steps=excluded, samples=list(sample.values()))


def load_trace_rows(output, manifest):
    """Keep small scoring fields; full evidence remains in compressed event files."""
    rows = []
    for item in manifest['agents']:
        with gzip.open(output / 'events' / item['file'], 'rt', encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                if not in_scope(event['time']):
                    raise ValueError('Out-of-scope trace event')
                when = timestamp(event['time'])
                if when.astimezone(PACIFIC).date().isoformat() != event['day']:
                    raise ValueError('Trace day disagrees with timestamp')
                rows.append((event['actor'], event['day'], when.timestamp(), event['turn_id'],
                    int(event['research_layer']), event['family'], event['entity'],
                    event['operation'], event['status'], bool(event.get('verbal_channel', False)),
                    event.get('related_actor', ''), float(event.get('mention_weight', 1.0))))
    return pd.DataFrame(rows, columns=['actor', 'day', 'time', 'turn', 'layer', 'family',
                                      'entity', 'operation', 'status', 'verbal', 'related_actor', 'weight'])


def feature_prefix(layer, family):
    # Hash suffix prevents accidental collisions between normalized family names.
    clean = re.sub('[^a-z0-9]+', '_', family.lower()).strip('_')
    return 'l%d_%s_%s' % (layer, clean, hashlib.sha256(family.encode()).hexdigest()[:6])


def any_near(first, second, seconds=3600):
    first, second = np.sort(first), np.sort(second)
    at = np.searchsorted(second, first)
    after = np.abs(second[np.minimum(at, len(second) - 1)] - first)
    before = np.abs(second[np.maximum(at - 1, 0)] - first)
    return bool(np.minimum(before, after).min() <= seconds)


def build_trace_features(rows, universe, training_actors):
    """Layer/family ablations, frozen rarity, and explicit weak ordered candidates.

    Counts deduplicate each actor/turn/object, including command-output echoes.
    Evidence stays retrospective within a day; none of these features confirms use.
    """
    features = universe[KEYS].copy()
    if rows.empty:
        return features, {'families': {}, 'interpretation': 'No extracted traces.'}
    if not rows.day.between(str(config.DEV_START), str(config.DEV_END)).all():
        raise ValueError('Out-of-scope feature records')
    if not set(rows.layer).issubset({1, 2}):
        raise ValueError('Unknown research layer')
    fitted = rows[rows.day <= str(config.TRAIN_END)]
    family_keys = sorted(set(zip(fitted.layer, fitted.family, fitted.verbal)))
    known_pairs = set(map(tuple, features[KEYS].itertuples(index=False, name=None)))
    n_actors = max(len(set(training_actors)), 1)
    report = {'families': {}, 'unseen_family_events': 0,
              'idf_fit': 'training actors and training observations only',
              'interpretation': 'Shared/ordered trace candidates, not confirmed consumption; same-day retrospective features.'}
    family_set = set(family_keys)
    report['unseen_family_events'] = sum((int(a), b, bool(c)) not in family_set
                                        for a, b, c in zip(rows.layer, rows.family, rows.verbal))
    for layer, family, verbal in family_keys:
        sub = rows[(rows.layer == layer) & (rows.family == family) & (rows.verbal == verbal)].copy()
        prefix = feature_prefix(layer, family) + ('_verbal' if verbal else '')
        aggregated = collections.defaultdict(collections.Counter)
        if verbal:
            attributed = sub[(sub.related_actor != '') & (sub.related_actor != sub.actor)]
            attributed = attributed.sort_values('weight').drop_duplicates(['actor', 'turn', 'related_actor'], keep='last')
            for row in attributed.itertuples():
                a, b = sorted((row.actor, row.related_actor))
                key = (a, b, row.day)
                if key in known_pairs:
                    aggregated[key]['mentions'] += row.weight
            suffixes = ['mentions']
            fitted_weights = {}
        else:
            # One mention per object/turn for co-use; preserve operations separately.
            touches = sub.drop_duplicates(['actor', 'turn', 'entity'])
            train = touches[touches.day <= str(config.TRAIN_END)]
            popularity = train.groupby('entity').actor.nunique()
            fitted_weights = {key: float(math.log(n_actors / min(int(count), n_actors)))
                              for key, count in popularity.items()}
            default = math.log(n_actors)
            counts = touches.groupby(['day', 'actor', 'entity']).size()
            magnitudes, norms = {}, collections.Counter()
            for (day, actor, entity), count in counts.items():
                weight = fitted_weights.get(entity, default) * (1 + math.log(count))
                magnitudes[(day, actor, entity)] = weight
                norms[(day, actor)] += weight * weight
            # Use raw operations for candidates but explicitly exclude observed failures.
            operations = sub[sub.status != 'failed'].groupby(['day', 'entity', 'actor'])
            first_writes, last_reads = {}, {}
            for key, group in operations:
                writes = group[group.operation.isin(WRITES)].time
                reads = group[group.operation.isin(READS)].time
                if len(writes):
                    first_writes[key] = float(writes.min())
                if len(reads):
                    last_reads[key] = float(reads.max())
            for (day, entity), group in touches.groupby(['day', 'entity'], sort=False):
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
                    ordered = any(first_writes.get((day, entity, x), math.inf) <
                                  last_reads.get((day, entity, y), -math.inf) for x, y in ((a, b), (b, a)))
                    if ordered:
                        value['ordered_candidate'] += weight
            for (a, b, day), values in aggregated.items():
                denom = math.sqrt(norms[(day, a)] * norms[(day, b)])
                values['cosine'] = values['cosine'] / denom if denom else 0.0
            # A Village directory can identify whose project was visited even
            # when that owner did not touch the project that day. Ownership is
            # a candidate relation, never confirmation of a handoff.
            attributed = sub[(sub.related_actor != '') & (sub.related_actor != sub.actor)]
            attributed = attributed.drop_duplicates(['actor', 'turn', 'entity', 'related_actor'])
            for row in attributed.itertuples():
                a, b = sorted((row.actor, row.related_actor))
                key = (a, b, row.day)
                if key in known_pairs:
                    aggregated[key]['related_actor_candidate'] += 1
            suffixes = ['shared', 'idf_overlap', 'cosine', 'sync', 'ordered_candidate', 'related_actor_candidate']
        columns = [prefix + '_' + suffix for suffix in suffixes]
        part = pd.DataFrame([dict(zip(KEYS, key), **{prefix + '_' + s: values[s] for s in suffixes})
                             for key, values in aggregated.items()], columns=KEYS + columns)
        features = features.merge(part, on=KEYS, how='left')
        report['families'][prefix] = dict(layer=int(layer), family=family, verbal=bool(verbal),
            events=len(sub), unique_turns=int(sub[['actor', 'turn']].drop_duplicates().shape[0]),
            entities=int(sub.entity.nunique()), nonzero_pair_days=len(aggregated), features=columns,
            fitted_idf=fitted_weights)
    for column in features.columns.difference(KEYS):
        features[column] = pd.to_numeric(features[column]).fillna(0.0)
    return features, report


def model_sets(frame, family_report):
    train = frame.day <= str(config.TRAIN_END)
    provenance = [c for c in frame if c.startswith('prov_') and not c.startswith('prov_verbal_')
                  and frame.loc[train, c].sum() > 0]
    verbal_provenance = [c for c in frame if c.startswith('prov_verbal_') and frame.loc[train, c].sum() > 0]
    general, village, general_verbal, village_verbal = [], [], [], []
    for family in family_report['families'].values():
        target = (general_verbal if family['verbal'] else general) if family['layer'] == 1 else (
                  village_verbal if family['verbal'] else village)
        target.extend(family['features'])
    legacy = ACTIVITY + BEHAVIOR
    full = legacy + provenance + general
    sets = {'activity': ACTIVITY, 'historical_behavior': legacy,
            'general_existing_provenance': legacy + provenance,
            'layer1_behavior': full,
            'layer1_plus_layer2_behavior': full + village,
            'layer1_with_attribution': full + general_verbal + verbal_provenance,
            'layer1_plus_layer2_with_attribution': full + general_verbal + verbal_provenance + village + village_verbal}
    old_names = [c for c in frame if c.startswith('name_')]
    if old_names:
        sets['historical_with_village_names'] = legacy + old_names
    return {name: list(dict.fromkeys(columns)) for name, columns in sets.items()}


def source_digest():
    names = ('agent_traces.py', 'village_traces.py', 'research_layers.py', 'provenance_common.py',
             'layer_scoring.py', 'config.py', 'metrics.py')
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}


def main():
    from .layer_scoring import evaluate_layers, paired_day_comparison
    from .village_traces import fit_village_rules
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-extract', action='store_true')
    parser.add_argument('--general-only', action='store_true')
    parser.add_argument('--output', type=Path, default=config.OUT / 'research-layers')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error('workers must be 1..4')
    started = time.monotonic()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = config.OUT / 'provenance' / 'features_dev.csv.gz'
    baseline = pd.read_csv(baseline_path)
    if not baseline.day.between(str(config.DEV_START), str(config.DEV_END)).all():
        raise ValueError('Historical feature cache contains unauthorized dates')
    if baseline.duplicated(KEYS).any():
        raise ValueError('Historical feature cache contains duplicate pair-days')
    training = baseline[baseline.day <= str(config.TRAIN_END)]
    roster = sorted(set(training.a) | set(training.b))
    metadata = json.loads((config.STEPS / 'summary.json').read_text())
    items = sorted((name, info['file']) for name, info in metadata['agents'].items())
    sources = source_digest()
    cache_inputs = {filename: {'size': (config.STEPS / filename).stat().st_size,
                               'mtime_ns': (config.STEPS / filename).stat().st_mtime_ns}
                    for _, filename in items}
    if not args.skip_extract:
        rules = None if args.general_only else fit_village_rules(training_records(items), actor_names=roster)
        save_json(output / 'village_rules.json', rules)
        tasks = [(actor, filename, roster, rules, str(output)) for actor, filename in items]
        summaries = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(extract_agent, task) for task in tasks]
            for future in concurrent.futures.as_completed(pending):
                result = future.result()
                summaries.append(result)
                print(result['actor'], result['steps'], 'scoped steps;', sum(result['counts'].values()), 'trace records', flush=True)
        summaries.sort(key=lambda item: (item['actor'], item['file']))
        if source_digest() != sources:
            raise ValueError('Sources changed during extraction; rerun before using these traces')
        manifest = dict(agents=summaries, sources=sources, general_only=args.general_only,
                        fit_end=str(config.TRAIN_END), roster=roster, cache_inputs=cache_inputs)
        save_json(output / 'extraction.json', manifest)
    else:
        manifest = json.loads((output / 'extraction.json').read_text())
        if (manifest['sources'] != sources or manifest['general_only'] != args.general_only
                or manifest.get('cache_inputs') != cache_inputs or manifest['roster'] != roster):
            raise ValueError('Cached traces were generated with different source/configuration; rerun extraction')
    rows = load_trace_rows(output, manifest)
    print('Building features for', len(rows), 'trace records', flush=True)
    new, families = build_trace_features(rows, baseline, roster)
    frame = baseline.merge(new, on=KEYS, how='left').fillna(0)
    sets = model_sets(frame, families)
    scores, predictions = evaluate_layers(frame, sets)
    scores['paired_comparisons'] = {
        name: paired_day_comparison(predictions, before, after)
        for name, before, after in (
            ('general_trace_increment', 'general_existing_provenance', 'layer1_behavior'),
            ('village_behavior_increment', 'layer1_behavior', 'layer1_plus_layer2_behavior'),
            ('village_increment_with_attribution', 'layer1_with_attribution', 'layer1_plus_layer2_with_attribution'))}
    frame.to_csv(output / 'features_dev.csv.gz', index=False)
    predictions.to_csv(output / 'predictions_dev.csv.gz', index=False)
    save_json(output / 'feature_manifest.json', families)
    save_json(output / 'scores.json', scores)
    report = dict(scope={'start': str(config.DEV_START), 'end': str(config.DEV_END),
                         'train_end': str(config.TRAIN_END), 'validation_start': str(config.VALID_START),
                         'timezone': 'America/Los_Angeles'},
        general_only=args.general_only, steps=sum(a['steps'] for a in manifest['agents']),
        excluded_history_steps=sum(a['excluded_history_steps'] for a in manifest['agents']),
        trace_records=len(rows), feature_sets=sets, elapsed_seconds=round(time.monotonic() - started, 3),
        interpretation='Deterministic rules developed by the AI researcher. Retrospective pair-day analysis on inspected development data; no runtime AI, sealed evaluation, cross-swarm transfer, or online-monitor claim.',
        measurement='Both new layers exclude recognized executed communication-history wrappers. Historical precomputed features are preserved and may retain embedded communication text in computer outputs; behavior-only denotes feature selection, not a certified communication-free corpus.',
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(), sources=sources)
    save_json(output / 'summary.json', report)
    print('Saved research-layer comparison to', output, flush=True)


if __name__ == '__main__':
    main()
