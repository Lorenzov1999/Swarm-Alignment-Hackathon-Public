"""Build deliberately selected Village network illustrations from saved scores.

No fitting, request labels, or new extraction. Optional user-supplied evidence
files document the original case review; private excerpts are not distributed.
June is a backward application of July parameters. Use --mode frozen-paper only
with the original pre-correction features; current-run permits new frozen scores.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / 'outputs/current-paper/provenance'
SOURCES = {
    'shared_goals': 'outputs/generalization-20261106/shared_goals/features.csv.gz',
    'development': 'outputs/research-layers/features_dev.csv.gz',
    'test': 'outputs/generalization-20261106/temporal/features.csv.gz',
}
CASES = [
    dict(id='showcase', day='2026-06-13', title='A shared showcase',
         period='Earlier shared goal', source='shared_goals',
         context='The goal was to organise an event. Session notes describe the AI Village '
                 'Showcase at The Fold, with agents preparing contributions and monitoring '
                 'the Artifact Wall, Guestbook, and shared event pages.',
         reason='Densest inferred graph among the 78 observed days, paired with verified '
                'event-goal and session context.',
         evidence_file='games-and-showcase.json', focus_agents=[], focus_edges=[]),
    dict(id='helping-gemini', day='2026-06-22', title='Helping Gemini',
         period='Earlier shared goal', source='shared_goals',
         context='Under the goal "Help Gemini 2.5 Pro!", agents helped Gemini install '
                 'and publish on Hyphanet. Its four inferred neighbors form a five-agent '
                 'clique sharing installer references; retained messages also document '
                 'diagnostics and publication of game text supplied by DeepSeek.',
         reason='User-requested named goal, verified against retained goals, messages and '
                'installer mentions; selected for the interpretable Gemini-centered group.',
         evidence_file='helping-gemini.json', focus_agents=['Gemini 2.5 Pro'],
         focus_edges=[['Gemini 2.5 Pro', a] for a in
                      ['GPT-5.2', 'GPT-5.4', 'GPT-5.5', 'DeepSeek-V3.2']]),
    dict(id='games', day='2026-06-26', title='Few links during games',
         period='Earlier shared goal', source='shared_goals',
         context='The goal was to beat the hardest game possible. Session notes describe '
                 'separate chess, Gomoku, BSD Robots, Hack and Spellbreaker attempts. '
                 'Fourteen of 17 active agents have no inferred links; 89.2% of working '
                 'steps are GUI actions and only 2.1% contain extracted resource traces.',
         reason='Mostly isolated inferred nodes with a nonempty graph (one triangle), '
                'rather than the completely empty June 24 graph. Context and coverage '
                'explain why sparse resource evidence is not proof of no collaboration.',
         evidence_file='games-and-showcase.json', focus_agents=[], focus_edges=[]),
    dict(id='computer-use-proxy', day='2026-08-18', title='A computer-use proxy',
         period='Later test work', source='test',
         context='DeepSeek-V3.2 supplied outreach text about multi-agent collaboration and '
                 'verification. Gemini 3.1 Pro split it into three posts, published them '
                 'through the browser, and returned the thread URL. Both agents mention it; '
                 'these references contribute to their inferred edge.',
         reason='User-requested computer-use proxy pattern, with same-day delegation, '
                'GUI steps, returned URL and resource overlap verified in retained records.',
         evidence_file='computer-use-proxy.json',
         focus_agents=['DeepSeek-V3.2', 'Gemini 3.1 Pro'],
         focus_edges=[['DeepSeek-V3.2', 'Gemini 3.1 Pro']]),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_from_pairs(frame, threshold):
    """Keep the complete active-agent universe, including zero-degree nodes."""
    columns = ['a', 'b', 'p1_cos']
    frame = frame[columns]
    if frame[['a', 'b']].duplicated().any() or not (frame.a < frame.b).all():
        raise ValueError('Expected unique sorted undirected pairs, without self-pairs')
    names = sorted(set(frame.a) | set(frame.b))
    if len(frame) != len(names) * (len(names) - 1) // 2:
        raise ValueError('Incomplete daily pair universe; isolates could be missing')
    if len(names) < 2 or not np.isfinite(frame.p1_cos).all():
        raise ValueError('Expected finite scores and at least two active agents')
    if ((frame.p1_cos < 0) | (frame.p1_cos > 1 + 1e-12)).any():
        raise ValueError('Cosine outside its nonnegative range')
    edges = [dict(a=a, b=b, score=float(s)) for a, b, s in
             frame.loc[frame.p1_cos > threshold].sort_values(['a', 'b']).itertuples(index=False, name=None)]
    adjacency = {a: set() for a in names}
    for e in edges:
        adjacency[e['a']].add(e['b'])
        adjacency[e['b']].add(e['a'])
    remaining, components = set(names), []
    while remaining:
        pending, found = [min(remaining)], set()
        while pending:
            a = pending.pop()
            if a not in found:
                found.add(a)
                pending.extend(adjacency[a] - found)
        remaining.difference_update(found)
        components.append(sorted(found))
    components.sort(key=lambda c: (-len(c), c))
    triangles = sum(b in adjacency[a] and c in adjacency[a] and c in adjacency[b]
                    for a, b, c in itertools.combinations(names, 3))
    degrees = {a: len(adjacency[a]) for a in names}
    wedges = sum(d * (d - 1) // 2 for d in degrees.values())
    return dict(nodes=names, edges=edges, components=components, degrees=degrees,
                metrics=dict(nodes=len(names), pairs=len(frame), edges=len(edges),
                             density=len(edges)/len(frame), isolates=sum(d == 0 for d in degrees.values()),
                             component_sizes=[len(c) for c in components], triangles=triangles,
                             transitivity=3 * triangles / wedges if wedges else 0.0))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path, default=PROVENANCE / 'shared-resource-results.json')
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--evidence-dir', type=Path,
                        help='Optional user-supplied supporting case JSON directory; never copied')
    parser.add_argument('--mode', choices=['current-run', 'frozen-paper'], default='current-run')
    for name in SOURCES:
        parser.add_argument('--' + name.replace('_', '-') + '-features', type=Path)
    parser.add_argument('--shared-goals-report', type=Path)
    parser.add_argument('--test-report', type=Path)
    parser.add_argument('--output', type=Path, default=PROVENANCE / 'village-network-snapshots.json')
    parser.add_argument('--daily-output', type=Path)
    args = parser.parse_args(argv)
    results_path = args.results
    results = json.loads(results_path.read_text())
    transfer_path = args.parameters or args.input_root / 'outputs/technique-transfer/summary.json'
    transfer = json.loads(transfer_path.read_text())
    threshold = results['model_checks']['equivalent_cosine_threshold']
    frames, hashes, reports = {}, {}, {}
    for key, relative in SOURCES.items():
        path = getattr(args, key + '_features') or args.input_root / relative
        digest = sha(path)
        if digest != transfer['inputs_sha256'][relative]:
            raise ValueError(f'Saved feature hash changed: {relative}')
        frame = pd.read_csv(path, usecols=['a', 'b', 'day', 'p1_cos'])
        if not frame.day.between('2026-06-01', '2026-09-15').all():
            raise ValueError('Feature row outside the authorized dates')
        if frame[['a', 'b', 'day']].duplicated().any():
            raise ValueError('Duplicate pair-day')
        frames[key], hashes[relative] = frame, digest
    for key, directory in [('shared_goals', 'shared_goals'), ('test', 'temporal')]:
        relative = f'outputs/generalization-20261106/{directory}/feature-report.json'
        path = getattr(args, key + '_report') or args.input_root / relative
        reports[key] = json.loads(path.read_text())['historical']
        hashes[relative] = sha(path)
    if reports['shared_goals']['frozen_sha256'] != reports['test']['frozen_sha256']:
        raise ValueError('Earlier and test periods do not use the same resource freeze')

    inventory, all_graphs = [], {}
    for source, frame in frames.items():
        for day, group in frame.groupby('day', sort=True):
            if day in all_graphs:
                raise ValueError('Overlapping input periods')
            graph = graph_from_pairs(group, threshold)
            all_graphs[day] = graph
            period = ('earlier_shared_goals' if source == 'shared_goals' else 'test' if source == 'test'
                      else 'training' if day <= '2026-07-31' else 'validation')
            inventory.append(dict(day=day, period=period, **graph['metrics']))
    if sum(len(g['edges']) for d, g in all_graphs.items() if d >= '2026-08-17') != results['operating_point']['alerts']:
        raise ValueError('Full test edge count differs from the paper alert count')

    snapshots = []
    for case in CASES:
        graph = all_graphs[case['day']]
        case = dict(case)
        if not set(case['focus_agents']).issubset(graph['nodes']):
            raise ValueError('Focus agent is absent')
        edges = {tuple(sorted((e['a'], e['b']))) for e in graph['edges']}
        if args.mode == 'frozen-paper' and any(tuple(sorted(e)) not in edges for e in case['focus_edges']):
            raise ValueError('Highlighted edge is not an actual fixed-threshold edge')
        case['focus_edges'] = [e for e in case['focus_edges'] if tuple(sorted(e)) in edges]
        source = case.pop('source')
        case.update(graph)
        case['source_path'] = SOURCES[source]
        evidence_name = case.pop('evidence_file')
        if args.evidence_dir:
            case['evidence_file'] = evidence_name
            case['evidence_sha256'] = sha(args.evidence_dir / evidence_name)
        else:
            case['evidence_status'] = 'Original case descriptions retained in source; supporting excerpts not supplied.'
        if args.mode == 'current-run':
            case['context'] = {
                'showcase': 'Selected published case: agents prepared a shared showcase and event pages. '
                            'This graph uses the supplied run; the original case narrative is not re-audited here.',
                'helping-gemini': 'Selected published case: agents helped Gemini 2.5 Pro with diagnostics, '
                                  'installation and publishing. Edges show shared resources in the supplied run; '
                                  'chat assistance can remain invisible.',
                'games': 'Selected published case: agents attempted separate games. '
                         'Sparse resource references cannot establish an absence of cooperation.',
                'computer-use-proxy': 'Selected published case: DeepSeek-V3.2 supplied outreach text and '
                                      'Gemini 3.1 Pro published it through the browser. '
                                      'The supplied scores determine whether their resource overlap forms an edge.'
            }[case['id']]
            case['reason'] = 'Same illustrative date as the published case; graph recomputed from the supplied run.'
        snapshots.append(case)
    # Independent saved case checks protect the final illustration against a
    # date, score or threshold substitution during later editorial revisions.
    expected = {'2026-06-13': (18, 109, 2), '2026-06-22': (17, 16, 6),
                '2026-06-26': (17, 3, 14), '2026-08-18': (27, 52, 9)}
    for case in snapshots if args.mode == 'frozen-paper' else []:
        m = case['metrics']
        if (m['nodes'], m['edges'], m['isolates']) != expected[case['day']]:
            raise ValueError('Snapshot disagrees with independent inventory')
    if args.mode == 'frozen-paper':
        proxy = snapshots[-1]
        score = next(e['score'] for e in proxy['edges'] if {e['a'], e['b']} == set(proxy['focus_agents']))
        if abs(score - 0.1417646697539185) > 1e-14:
            raise ValueError('Proxy pair score changed')

    result = dict(
        purpose='Four deliberately selected, contextual network illustrations; no new detector or representative-sample claim.',
        reproduction_mode=args.mode,
        score_version=('Original pre-correction frozen paper scores' if args.mode == 'frozen-paper'
                       else 'Supplied run; not asserted to reproduce the original or additive-correction paper scores'),
        selection='Post-hoc selection by contrasting topology and user-requested documented work episodes. '
                  'Date selection is not a performance evaluation. No request labels loaded by this script.',
        omitted_candidate='A generic few-groups example was dropped because the Gemini-help graph already illustrates distinct groups; the dense event adds a larger contrast.',
        timezone='America/Los_Angeles', score='p1_cos', threshold=threshold,
        input_periods={name: [frame.day.min(), frame.day.max()] for name, frame in frames.items()},
        parameter_training_period=['2026-07-06', '2026-07-31'],
        edge_rule='Strict cosine > fixed July threshold; undirected edges; every active agent retained, including isolates.',
        interpretation='Inferred shared-resource connections, not verified cooperation for every edge. '
                       'June examples apply July parameters backward; only August 18 belongs to the later test period.',
        frozen_resource_reference=reports['test']['frozen_sha256'],
        source_sha256=hashes, results_sha256=sha(results_path),
        technique_transfer_provenance_sha256=sha(transfer_path),
        generator_sha256=sha(__file__), observed_days_inspected=len(inventory),
        unchanged_test_alert_count=results['operating_point']['alerts'],
        snapshots=snapshots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    columns = ['day', 'period', 'nodes', 'pairs', 'edges', 'density', 'isolates',
               'component_sizes', 'triangles', 'transitivity']
    daily_output = args.daily_output or args.output.with_name('village-network-daily.csv')
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    with daily_output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(sorted(inventory, key=lambda r: r['day']))
    print(json.dumps([dict(day=s['day'], title=s['title'], **s['metrics']) for s in snapshots], indent=2))


if __name__ == '__main__':
    main()
