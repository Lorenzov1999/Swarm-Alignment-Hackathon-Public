"""Reproduce the activity baseline for the paper's follow-through comparison.

Run: python3 scripts/check_followthrough_control.py
Reads saved August 17–September 15 scores and annotations only; fits nothing.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs/claude-review-20261107/followthrough-control.json'
MODELS = ('layer1_behavior', 'activity')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concordance(rows, model):
    positive = [float(r[model]) for r in rows if r['followed_group'] == 'yes']
    negative = [float(r[model]) for r in rows if r['followed_group'] == 'no']
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    comparisons = len(positive) * len(negative)
    return dict(auroc=(wins + 0.5 * ties) / comparisons,
                positive_negative_comparisons=comparisons,
                strict_wins=wins, tied_comparisons=ties)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-dir', type=Path, default=ROOT,
                        help='repository containing the saved outputs (default: this repository)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    research = args.research_dir.resolve()
    folder = research / 'outputs/claude-review-20261107/labels-visibility'
    source = folder / 'request-positive-pair-days.csv.gz'
    manifest_path = folder / 'manifest.json'
    sensitivity_path = folder / 'single-request-sensitivity.json'
    manifest = json.loads(manifest_path.read_text())
    sensitivity = json.loads(sensitivity_path.read_text())
    if sha(source) != manifest['outputs_sha256'][source.name]:
        raise ValueError('Saved follow-through table differs from its manifest')
    for path in (source, manifest_path):
        if sha(path) != sensitivity['input_sha256'][str(path.relative_to(research))]:
            raise ValueError('Original single-request sensitivity used different inputs')

    with gzip.open(source, 'rt') as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        key = (row['a'], row['b'], row['day'])
        if key in seen or row['a'] >= row['b']:
            raise ValueError('Pair-days must be unique canonical agent pairs')
        seen.add(key)
        if not '2026-08-17' <= row['day'] <= '2026-09-15':
            raise ValueError('Saved pair-day falls outside the test period')
        if row['y'] != 'True' or not all(math.isfinite(float(row[m])) for m in MODELS):
            raise ValueError('Expected request-positive rows with finite frozen scores')

    single = [r for r in rows if int(r['requests']) == 1]
    if not all(r['followed_group'] in ('yes', 'no', 'ambiguous') for r in single):
        raise ValueError('Unexpected single-request follow-through annotation')
    selected = [r for r in single if r['followed_group'] in ('yes', 'no')]
    counts = dict(yes=sum(r['followed_group'] == 'yes' for r in selected),
                  no=sum(r['followed_group'] == 'no' for r in selected),
                  ambiguous=len(single) - len(selected))
    original = sensitivity['followthrough']['clean_contrast']
    if (counts['yes'], counts['no'], counts['ambiguous']) != (
            original['only_yes_pair_days'], original['only_no_pair_days'],
            original['excluded_mixed_or_ambiguous_pair_days']):
        raise ValueError('Population differs from the reported single-request contrast')
    results = {model: concordance(selected, model) for model in MODELS}
    if not math.isclose(results['layer1_behavior']['auroc'],
                        original['models']['layer1_behavior']['auroc'],
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError('Detector AUROC differs from the reported contrast')

    report = dict(
        scope='Post-hoc comparison of saved frozen scores on single-request pair-days; '
              'no raw traces, new labels, refitting, thresholds, or reserved data.',
        period=['2026-08-17', '2026-09-15'],
        inputs_sha256={str(p.relative_to(research)): sha(p)
                       for p in (source, manifest_path, sensitivity_path)},
        script_sha256=sha(Path(__file__)),
        original_request_pair_days=len(rows), single_request_pair_days=len(single),
        compared_pair_days=len(selected), annotations=counts,
        definitions={
            'population': 'Exactly one recorded request; site follow-through is yes or no. '
                          'Ambiguous annotations are excluded.',
            'auroc': 'All yes/no pair-day comparisons, with half credit for equal scores.',
            'activity': 'Saved July-fitted activity score from the pair minimum and maximum '
                        'working-step counts, applied without refitting.',
            'interpretation': 'Association with site-reported follow-through. This compares '
                              'a specific activity baseline; no significance claim or '
                              'independent verification of completed work is added.'},
        models=results)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(annotations=counts, models=results), indent=2))


if __name__ == '__main__':
    main()
