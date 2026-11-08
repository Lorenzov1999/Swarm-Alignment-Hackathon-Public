"""Join fixed resource rankings to two source-only assistant reviews."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/swarm-resource-ranking-20261108'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking', type=Path, default=OUTPUT,
                        help='Ranking directory containing review/reviewer-{a,b}.json and adjudication.json')
    args = parser.parse_args()
    output = args.ranking
    manifest = json.loads((output / 'manifest.json').read_text())
    for name, digest in manifest['outputs_sha256'].items():
        if sha(output / name) != digest:
            raise ValueError('Ranked artifact changed after selection: ' + name)
    selection = json.loads((output / 'review-selection.json').read_text())['selected_resources']
    expected = {(item['rank'], row['record_id']) for item in selection for row in item['records']}
    reviews = {}
    for letter in ('a', 'b'):
        content = json.loads((output / f'review/reviewer-{letter}.json').read_text())
        entries = content['records']
        reviews[letter] = {(row['rank'], row['record_id']): row for row in entries}
        if set(reviews[letter]) != expected or len(entries) != len(expected):
            raise ValueError('Reviewer population differs from the fixed selection')
    agreements, disagreements, both_explicit = [], [], []
    for key in sorted(expected):
        a, b = reviews['a'][key], reviews['b'][key]
        row = dict(rank=key[0], record_id=key[1], category_a=a['category'], category_b=b['category'])
        (agreements if a['category'] == b['category'] else disagreements).append(row)
        if a['category'] == b['category'] == 'explicit_suspicious_attempt':
            both_explicit.append(row)
    adjudication = json.loads((output / 'review/adjudication.json').read_text())
    resolutions = {(r['rank'], r['record_id']): r for r in adjudication['resolutions']}
    if set(resolutions) != {(r['rank'], r['record_id']) for r in disagreements}:
        raise ValueError('Adjudication does not cover exactly the disagreements')
    resolved = {key: (resolutions[key]['category'] if key in resolutions
                      else reviews['a'][key]['category']) for key in expected}
    with (output / 'primary-resource-ranking.csv').open() as stream:
        ranks = list(csv.DictReader(stream))
    top = ranks[:20]
    counts = Counter(row['rank'] for row in both_explicit)
    columns = ['rank', 'value', 'supporting_records', 'possible_record_pairs',
               'fraction_total_overlap_mass', 'frequency_rank', 'uniform_rank']
    table = output / 'reviewed-top20.csv'
    with table.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + ['reviewed_payloads', 'both_reviewers_explicit_attempt'])
        writer.writeheader()
        for row in top:
            rank = int(row['rank'])
            writer.writerow(dict({c: row[c] for c in columns},
                reviewed_payloads=sum(k[0] == rank for k in expected),
                both_reviewers_explicit_attempt=counts[rank]))
    summary = json.loads((output / 'summary.json').read_text())
    report = dict(
        scope='Fixed top20 resources, three high-contribution distinct payloads per resource; source-only assistant review, not population precision or maliciousness recall.',
        reviewed_resources=20, reviewed_payloads=len(expected),
        category_counts={k: dict(Counter(r['category'] for r in v.values())) for k, v in reviews.items()},
        exact_category_agreements=len(agreements), disagreements=disagreements,
        adjudicated_category_counts=dict(Counter(resolved.values())),
        unanimous_explicit_attempts=both_explicit,
        top_two_overlap_mass_share=sum(float(r['fraction_total_overlap_mass']) for r in top[:2]),
        top20_overlap_mass_share=sum(float(r['fraction_total_overlap_mass']) for r in top),
        frequency_top20_overlap=summary['primary']['overlap_frequency_top20_intersection'],
        raw_dedup_top20_overlap=summary['sensitivity']['primary_raw_top20_intersection'],
        inputs_sha256={name: sha(output / name) for name in [
            'manifest.json', 'primary-resource-ranking.csv', 'review-selection.json',
            'review/reviewer-a.json', 'review/reviewer-b.json', 'review/context-manifest.json',
            'review/adjudication.json']},
        script_sha256=sha(Path(__file__)), output_sha256={table.name: sha(table)},
        evidence_limit='Visible code attempts do not establish execution, authorization, success, actor identities, or event times. Neither review consulted external investigation outcomes.')
    (output / 'review-assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('reviewed_payloads', 'category_counts', 'exact_category_agreements', 'disagreements', 'unanimous_explicit_attempts')}, indent=2))


if __name__ == '__main__':
    main()
