"""Orchestrate the explicitly authorized frozen Village transfer studies.

Original extractors are unchanged. Separate processes temporarily select an
exact observation window; all fitting is confined to reconstructed July inputs.
"""
import argparse
import collections
import concurrent.futures
from contextlib import contextmanager
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time

import pandas as pd

from . import config, pairs, traces
from .provenance_common import PACIFIC, timestamp
from .transfer_scoring import PERIODS

STUDY = config.OUT / 'generalization-20261106'
FROZEN = STUDY / 'frozen'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_original_sources():
    manifest = json.loads((STUDY / 'freeze-manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if relative.startswith('src/'):
            if digest(config.ROOT / relative) != expected['sha256']:
                raise ValueError('Original detector source changed since freeze: ' + relative)
    return manifest


@contextmanager
def observation_scope(start, end):
    """Narrow parser scope only; never alter TRAIN_END or original global files."""
    start, end = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
    allowed = {(dt.date(2026, 7, 6), dt.date(2026, 9, 15)),
               (dt.date(2026, 8, 17), dt.date(2026, 9, 15)),
               (dt.date(2026, 6, 1), dt.date(2026, 7, 5))}
    if (start, end) not in allowed:
        raise ValueError('Unregistered observation window')
    previous = config.DEV_START, config.DEV_END
    config.DEV_START, config.DEV_END = start, end
    try:
        yield
    finally:
        config.DEV_START, config.DEV_END = previous


def days_between(start, end):
    first, last = dt.date.fromisoformat(str(start)), dt.date.fromisoformat(str(end))
    return [(first + dt.timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def read_events(directory, start, end):
    result = []
    for path in sorted(directory.glob('*.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                if start <= event['day'] <= end:
                    result.append(event)
    return result


def freeze_training():
    from .transfer_features import freeze_historical
    from .provenance_links import training_common_content
    verify_original_sources()
    destination = STUDY / 'historical-freeze.json'
    companions = [STUDY / 'common-content-freeze.json', STUDY / 'training-reconstruction-inputs.json']
    if destination.exists() and all(path.exists() for path in companions):
        return json.loads(destination.read_text())
    days = days_between('2026-07-06', '2026-07-31')
    mentions = pairs.load_mentions(days)
    activity = pairs.load_activity(days)
    frozen = freeze_historical(mentions, activity.agent.unique())
    if destination.exists() and json.loads(destination.read_text()) != frozen:
        raise ValueError('Existing training freeze differs from deterministic reconstruction')
    save(destination, frozen)
    events = read_events(config.OUT / 'provenance/events', '2026-07-06', '2026-07-31')
    save(STUDY / 'common-content-freeze.json',
         {'fit_period': ['2026-07-06', '2026-07-31'],
          'keys': sorted(training_common_content(events, '2026-07-31'))})
    inputs = list((config.OUT / 'mentions').glob('*'))
    inputs += list((config.OUT / 'provenance/events').glob('*.jsonl.gz'))
    save(STUDY / 'training-reconstruction-inputs.json',
         {str(p.relative_to(config.ROOT)): digest(p) for p in sorted(inputs) if p.is_file()})
    print('Frozen historical identities/rarity and common-content exclusions', flush=True)
    return frozen


def cached_records(path, start, end):
    """Compact cache field t precedes action/output; gate before parsing those."""
    if path is None or not path.exists():
        return
    import re
    date_key = re.compile(rb'"t"\s*:\s*"([^"\\]+)"')
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            match = date_key.search(line)
            if match is None:
                raise ValueError('Cached record has no unambiguous timestamp')
            day = timestamp(match[1].decode()).astimezone(PACIFIC).date().isoformat()
            if not start <= day <= end:
                continue
            record = json.loads(line)
            if timestamp(record['t']).astimezone(PACIFIC).date().isoformat() != day:
                raise ValueError('Timestamp prefilter disagrees with parsed record')
            yield record


def parse_agent(task):
    from .agent_traces import AgentTraceExtractor
    from .provenance_git import GitExtractor
    from .provenance_structured import StructuredExtractor
    from .research_layers import RosterMentions, is_history_dump
    from .village_traces import VillageTraceExtractor
    actor, filename, period, legacy_roster, roster, rules, warm_file = task
    first, last = PERIODS[period]
    destination = STUDY / period
    # Constructor validates the original training interval; do not rewrite rules.
    village = VillageTraceExtractor(actor, rules)
    general, names = AgentTraceExtractor(actor), RosterMentions(actor, roster)
    git, structured = GitExtractor(actor), StructuredExtractor(actor)
    legacy = traces.AgentParser(actor, legacy_roster)
    activity = collections.defaultdict(collections.Counter)
    counts, covered, provenance_counts = collections.Counter(), collections.Counter(), collections.Counter()
    steps = excluded = warm_steps = 0
    output_paths = [destination / 'mentions' / filename.replace('.jsonl.gz', '.tsv.gz'),
                    destination / 'events' / filename,
                    destination / 'provenance-events' / filename]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    def legacy_step(record, index, keep, stream):
        when = timestamp(record['t']); day = when.astimezone(PACIFIC).date().isoformat()
        kind = record['k']; counter = activity[day] if keep else collections.Counter()
        counter['steps'] += 1
        if kind in traces.WAITING:
            counter['waiting'] += 1
            if kind == 'restart':
                legacy.cwd.pop(record['s'], None)
            return
        counter['working'] += 1
        n = 0
        if kind == 'shell':
            counter['shell'] += 1
            n = legacy.shell(when.timestamp(), day, index, record['s'], record.get('x') or '',
                             (record.get('o') or '') + ('\n' + record['e'] if record.get('e') else ''))
        elif kind == 'type':
            counter['gui'] += 1; counter['typed'] += 1
            n = legacy.typed(when.timestamp(), day, index, record.get('x') or '')
        elif kind in traces.GUI_KINDS:
            counter['gui'] += 1
            if kind == 'get_pixel_coords_of_element':
                legacy.names(when.timestamp(), day, index, record.get('x') or '', 'gui')
        else:
            counter['other'] += 1
            for typ, entity in legacy.urls(record.get('o') or ''):
                legacy.emit(when.timestamp(), day, index, typ, entity, 'seen', 'out'); n += 1
        if n:
            counter['trace_steps'] += 1
        if keep:
            for row in legacy.rows:
                stream.write(actor + '\t' + '\t'.join(str(x) for x in row) + '\n')
        legacy.rows.clear()

    def provenance_step(record):
        events = git.extract(record)
        structured.set_context(repo=git.current_repo(record.get('s', '')))
        events += structured.extract(record)
        return {e['event_id']: e for e in events}.values()

    start = '2026-07-06' if period == 'temporal' else first
    with observation_scope(start, last), \
            gzip.open(output_paths[0], 'wt', compresslevel=4) as mentions, \
            gzip.open(output_paths[1], 'wt', compresslevel=4) as new_events, \
            gzip.open(output_paths[2], 'wt', compresslevel=4) as provenance:
        mentions.write('agent\tt\tday\tstep\ttype\tid\top\tsrc\tk\n')
        if warm_file:
            for record in cached_records(Path(warm_file), '2026-07-06', '2026-08-14'):
                legacy_step(record, warm_steps, False, mentions)
                provenance_step(record)
                if not is_history_dump(record):
                    general.extract(record)
                warm_steps += 1
        for record in cached_records(destination / 'steps' / filename, first, last):
            legacy_step(record, warm_steps + steps, True, mentions)
            for event in provenance_step(record):
                provenance.write(json.dumps(event, ensure_ascii=False) + '\n')
                provenance_counts['|'.join([event['family'], event['operation'], event['status']])] += 1
            steps += 1
            if is_history_dump(record):
                excluded += 1
                continue
            events = general.extract(record) + names.extract(record) + village.extract(record)
            for event in {e['event_id']: e for e in events}.values():
                new_events.write(json.dumps(event, ensure_ascii=False) + '\n')
                counts[str(event['research_layer']) + '|' + event['family']] += 1
            for layer in {e['research_layer'] for e in events}:
                covered[str(layer)] += 1
    save(destination / 'mentions' / filename.replace('.jsonl.gz', '.activity.json'),
         {'agent': actor, 'days': dict(activity), 'clone_roots': len(legacy.dir_repo)})
    return dict(actor=actor, file=filename, steps=steps, warm_steps=warm_steps,
                counts=dict(counts), covered_steps=dict(covered),
                excluded_history_steps=excluded, provenance_counts=dict(provenance_counts))


def parse_period(period, workers=2):
    verify_original_sources()
    destination = STUDY / period
    metadata = json.loads((destination / 'steps/summary.json').read_text())
    old = json.loads((config.STEPS / 'summary.json').read_text())
    extraction = json.loads((FROZEN / 'outputs/research-layers/extraction.json').read_text())
    rules = json.loads((FROZEN / 'outputs/research-layers/village_rules.json').read_text())
    sources = {name: digest(config.ROOT / 'src/swarmcollab' / name) for name in (
        'generalization.py', 'extract.py', 'traces.py', 'agent_traces.py',
        'provenance_common.py', 'provenance_git.py', 'provenance_structured.py',
        'research_layers.py', 'village_traces.py')}
    inputs = {info['file']: digest(destination / 'steps' / info['file'])
              for info in metadata['agents'].values()}
    warm_inputs = {actor: digest(config.STEPS / old['agents'][actor]['file'])
                   for actor in metadata['agents'] if actor in old['agents']} if period == 'temporal' else {}
    signature = dict(sources=sources, target_steps=inputs, warm_steps=warm_inputs,
                     target_summary=digest(destination / 'steps/summary.json'),
                     original_summary=digest(config.STEPS / 'summary.json'),
                     rules=digest(FROZEN / 'outputs/research-layers/village_rules.json'),
                     roster=extraction['roster'])
    if (destination / 'extraction.json').exists():
        if json.loads((destination / 'extraction.json').read_text()).get('signature') != signature:
            raise ValueError('Completed extraction has different frozen inputs or source; inspect before rerunning')
        print('Using verified completed frozen extraction', period, flush=True)
        return
    tasks = []
    for actor, info in sorted(metadata['agents'].items()):
        previous = old['agents'].get(actor)
        warm = str(config.STEPS / previous['file']) if period == 'temporal' and previous else None
        tasks.append((actor, info['file'], period, sorted(old['agents']), extraction['roster'], rules, warm))
    start = time.monotonic(); results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(parse_agent, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); results.append(result)
            print(period, result['actor'], result['steps'], 'target steps;',
                  result['warm_steps'], 'warm steps', flush=True)
    results.sort(key=lambda row: row['actor'])
    save(destination / 'extraction.json', dict(agents=results, period=period, signature=signature,
         legacy_roster=sorted(old['agents']), roster=extraction['roster'],
         elapsed_seconds=time.monotonic() - start))


def evaluate_period(period):
    from . import labels
    from .provenance_links import build_links
    from .research_layers import load_trace_rows
    from .transfer_features import historical_features, trace_features, provenance_features, combine_features
    from .transfer_scoring import score_frozen, evaluate_predictions, paired_block_comparison
    verify_original_sources()
    started = time.monotonic(); first, last = PERIODS[period]
    destination = STUDY / period
    fitted = json.loads((FROZEN / 'outputs/research-layers/scores.json').read_text())
    family_manifest = json.loads((FROZEN / 'outputs/research-layers/feature_manifest.json').read_text())
    historical = json.loads((STUDY / 'historical-freeze.json').read_text())
    metadata = json.loads((destination / 'extraction.json').read_text())
    requested = list(dict.fromkeys(c for model in fitted['models'].values() for c in model['requested_features']))
    old_mentions = pairs.MENTIONS
    try:
        pairs.MENTIONS = destination / 'mentions'
        target_mentions = pairs.load_mentions(days_between(first, last))
        activity = pairs.load_activity(days_between(first, last))
    finally:
        pairs.MENTIONS = old_mentions
    history = pairs.load_mentions(days_between('2026-07-06', '2026-08-14')) if period == 'temporal' else None
    base, historical_report = historical_features(target_mentions, activity, historical, first, last, history)
    del target_mentions, history
    print('Historical features complete:', len(base), 'pair-days', flush=True)
    with observation_scope(first, last):
        rows = load_trace_rows(destination, metadata)
    new, trace_report = trace_features(rows, base, family_manifest, metadata['roster'], first, last)
    del rows
    print('Frozen general/Village trace features complete', flush=True)
    events = read_events(destination / 'provenance-events', first, last)
    if period == 'temporal':
        events += read_events(config.OUT / 'provenance/events', '2026-07-06', '2026-08-14')
    common = json.loads((STUDY / 'common-content-freeze.json').read_text())['keys']
    chains, link_diagnostics = build_links(events, common)
    chains = [c for c in chains if first <= c['day'] <= last]
    with gzip.open(destination / 'chains.jsonl.gz', 'wt') as stream:
        for chain in chains:
            stream.write(json.dumps(chain) + '\n')
    prov, provenance_report = provenance_features(base, chains, [c for c in requested if c.startswith('prov_')], first, last)
    frame = combine_features([base, new, prov], requested)
    # Predict and save before reading target labels.
    predictions = score_frozen(frame, fitted, period=period, known_agents=historical['training_actors'])
    frame.to_csv(destination / 'features.csv.gz', index=False)
    predictions.to_csv(destination / 'predictions-before-labels.csv.gz', index=False)
    prediction_hash = digest(destination / 'predictions-before-labels.csv.gz')
    request_rows, request_count, orphan_count = labels.load_requests(start=first, end=last)
    labelled = pairs.add_labels(frame[['a', 'b', 'day']].copy(), request_rows)
    predictions = predictions.merge(labelled, on=['a', 'b', 'day'], validate='one_to_one')
    report = evaluate_predictions(predictions, fitted, period=period, known_agents=historical['training_actors'])
    report['paired_comparisons'] = {
        'village_behavior_increment': paired_block_comparison(predictions, 'layer1_behavior',
            'layer1_plus_layer2_behavior', period=period),
        'village_attribution_increment': paired_block_comparison(predictions, 'layer1_with_attribution',
            'layer1_plus_layer2_with_attribution', period=period)}
    working = int(activity.working.sum())
    report['measurement'] = dict(working_steps=working, eligible_agents=int(activity.loc[activity.working > 0, 'agent'].nunique()),
        cached_steps=sum(x['steps'] for x in metadata['agents']),
        covered_steps_by_layer={str(layer):sum(x['covered_steps'].get(str(layer), 0) for x in metadata['agents']) for layer in [1,2]},
        excluded_history_steps=sum(x['excluded_history_steps'] for x in metadata['agents']),
        historical=historical_report, trace_families=trace_report, provenance=provenance_report,
        identity_confidence=dict(collections.Counter(c['identity_confidence'] for c in chains if c['level'] >= 3)),
        chains_per_1000_working_steps={str(level):1000 * sum(c['level']==level for c in chains)/working if working else None for level in [1,2,3,4]},
        link_diagnostics_including_warm_history=link_diagnostics)
    report['labels'] = dict(requests=request_count, recipient_rows=len(request_rows), orphans=orphan_count,
                            target='Request existence only; not verified consumption.')
    report['prediction_before_labels_sha256'] = prediction_hash
    report['resources'] = dict(evaluation_seconds=time.monotonic()-started,
                              max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['interpretation'] = ('Frozen temporal transfer; earlier pre-sprint exploration touched this period.' if period=='temporal'
        else 'Backward transfer to earlier goals; no future producer state; July-derived Village rules retain their availability guards.')
    predictions.to_csv(destination / 'predictions.csv.gz', index=False)
    activity.to_csv(destination / 'activity.csv.gz', index=False)
    save(destination / 'feature-report.json', {'historical':historical_report,'traces':trace_report,'provenance':provenance_report})
    save(destination / 'scores.json', report)
    print('Saved frozen transfer evaluation:', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['freeze','parse','evaluate'], required=True)
    parser.add_argument('--period', choices=['temporal','shared_goals'])
    parser.add_argument('--workers', type=int, choices=[1,2,3,4], default=2)
    args = parser.parse_args()
    if args.stage=='freeze':
        freeze_training()
    elif not args.period:
        parser.error('--period is required')
    elif args.stage=='parse':
        parse_period(args.period, args.workers)
    else:
        evaluate_period(args.period)


if __name__ == '__main__':
    main()
