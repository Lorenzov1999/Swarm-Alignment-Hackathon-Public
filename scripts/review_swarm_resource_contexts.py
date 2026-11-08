"""Export fixed rank-selected payload contexts; never execute their contents."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def windows(text, value):
    needle = value
    if needle not in text and value.startswith(('http://', 'https://')):
        needle = urlsplit(value).hostname or value
    first, last = text.find(needle), text.rfind(needle)
    spans = [(0, min(len(text), 280))]
    if first >= 0:
        spans.extend((max(0, p - 500), min(len(text), p + len(needle) + 500))
                     for p in {first, last})
    spans.sort()
    merged = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return [dict(start=lo, end=hi, text=text[lo:hi]) for lo, hi in merged]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--ranking', type=Path, required=True)
    args = parser.parse_args()
    selected_path = args.ranking / 'review-selection.json'
    selection = json.loads(selected_path.read_text())
    selected = {r['record_id']: r['payload_sha256']
                for item in selection['selected_resources'] for r in item['records']}
    # Freeze all existing ranking artifacts before source-content review begins.
    frozen = {str(p.relative_to(args.ranking)): sha(p)
              for p in sorted(args.ranking.rglob('*')) if p.is_file()}
    destination = args.ranking / 'review'
    destination.mkdir(exist_ok=False)
    source_hash = sha(args.source)
    if source_hash != '7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488':
        raise ValueError('Unexpected Swarm source')
    texts = {}
    with gzip.open(args.source, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            if row['id'] not in selected:
                continue
            if row['kind'] != 'payload' or row['id'] in texts:
                raise ValueError('Selection is not a unique payload record')
            body = row['text']
            if hashlib.sha256(body.encode()).hexdigest() != selected[row['id']]:
                raise ValueError('Payload differs from ranked input')
            texts[row['id']] = body
    if texts.keys() != selected.keys():
        raise ValueError('Missing selected records')
    full = destination / 'selected-payloads.jsonl.gz'
    with gzip.open(full, 'wt') as handle:
        for key, body in sorted(texts.items()):
            handle.write(json.dumps(dict(record_id=key, payload_sha256=selected[key], text=body)) + '\n')
    snippets = destination / 'contexts.jsonl'
    with snippets.open('w') as handle:
        for item in selection['selected_resources']:
            for record in item['records']:
                body = texts[record['record_id']]
                handle.write(json.dumps(dict(rank=item['rank'], resource_id=item['resource_id'],
                    resource=item['value'], family=item['family'], **record,
                    payload_chars=len(body), excerpts=windows(body, item['value']))) + '\n')
    report = dict(
        source_sha256=source_hash, ranking_files_sha256=frozen,
        selection_sha256=sha(selected_path), exporter_sha256=sha(Path(__file__)),
        resources=len(selection['selected_resources']), distinct_payloads=len(texts),
        context_rows=sum(len(item['records']) for item in selection['selected_resources']),
        selection='Top20 resources by fixed overlap mass; top3 normalized contributions, exact-body distinct.',
        evidence='Payload source content; code/claims are not execution receipts. No tags, citations, responses, or external outcomes used.',
        outputs_sha256={p.name: sha(p) for p in (full, snippets)})
    (destination / 'context-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('resources', 'distinct_payloads', 'context_rows')}))


if __name__ == '__main__':
    main()
