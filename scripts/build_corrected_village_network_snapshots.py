"""Build the paper's network views from the audited numeric-IP correction.

Uses retained scored pairs, original case selection and the unchanged July
threshold. No raw-log access, fitting, new selection or request-label analysis.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

from build_village_network_snapshots import graph_from_pairs


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / 'outputs/current-paper/provenance'
AUDIT = ROOT / 'outputs/numeric-endpoint-fix-20261108'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=PROVENANCE / 'village-network-snapshots-original.json')
    parser.add_argument('--audit', type=Path, default=AUDIT)
    parser.add_argument('--output', type=Path, default=PROVENANCE / 'village-network-snapshots-ip-corrected.json')
    args = parser.parse_args(argv)
    original = args.original
    summary = json.loads((args.audit / 'summary.json').read_text())
    # Fresh audits lock scores before reading labels. Older paper bundles also
    # support a file-hash manifest around the identical evaluated score columns.
    scores = args.audit / 'scores-before-labels.csv.gz'
    if scores.exists():
        lock_path = args.audit / 'prediction-lock.json'
        if sha(lock_path) != summary['prediction_lock_sha256']:
            raise ValueError('Correction prediction lock changed')
        expected_hash = json.loads(lock_path.read_text())['scores_sha256']
    else:
        scores = args.audit / 'evaluated-pairs.csv.gz'
        expected_hash = json.loads((args.audit / 'manifest.json').read_text())['files_sha256'][scores.name]
    if sha(scores) != expected_hash:
        raise ValueError('Corrected score input changed')
    data = copy.deepcopy(json.loads(original.read_text()))
    if data.get('reproduction_mode', 'frozen-paper') != 'frozen-paper':
        raise ValueError('Additive correction requires original frozen-paper snapshots, not a fresh extractor run')
    if abs(data['threshold'] - summary['primary_threshold']) > 1e-14:
        raise ValueError('Correction and original snapshot thresholds differ')
    frame = pd.read_csv(scores, usecols=['a', 'b', 'day', 'fixed'],
                        float_precision='round_trip')
    expected = {'2026-06-13': (18, 109, 2), '2026-06-22': (17, 27, 5),
                '2026-06-26': (17, 3, 14), '2026-08-18': (27, 52, 9)}
    for case in data['snapshots']:
        part = frame.loc[frame.day == case['day']].rename(columns={'fixed': 'p1_cos'})
        graph = graph_from_pairs(part, data['threshold'])
        if graph['nodes'] != case['nodes']:
            raise ValueError('Correction changed the displayed agent universe')
        counts = tuple(graph['metrics'][k] for k in ['nodes', 'edges', 'isolates'])
        if counts != expected[case['day']]:
            raise ValueError('Graph differs from the reviewed correction audit')
        case.update(graph)
        case['source_path'] = scores.name
        if case['id'] == 'helping-gemini':
            case['context'] = (
                'Under the goal "Help Gemini 2.5 Pro!", agents helped Gemini install '
                'and publish on Hyphanet. Gemini and its four neighbors share installer '
                'references. Six helpers also name the same remote-desktop endpoint, '
                'connecting another part of the support network. Gemini itself never '
                'names that endpoint in its retained work steps.')
    data['purpose'] = 'Main-paper network views including the audited numeric-IP correction.'
    data['reproduction_mode'] = 'additive-ip-correction-to-original-freeze'
    data['score_version'] = 'Original July weights and threshold; additive numeric-IP references only'
    data['score'] = 'p1_cos with additive numeric-IP resources, original July weights and threshold'
    data['generator_sha256'] = sha(Path(__file__))
    data['original_snapshot_sha256'] = sha(original)
    data['correction'] = {'scores': scores.name, 'scores_sha256': sha(scores),
                          'summary': 'summary.json',
                          'summary_sha256': sha(args.audit / 'summary.json'), 'variant': 'fixed'}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
