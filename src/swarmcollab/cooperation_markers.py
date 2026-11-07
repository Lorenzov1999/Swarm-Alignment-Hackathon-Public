"""Literal cooperation-related wording, not a classifier of executed cooperation.

This exploratory census is reviewed separately from observation-time evidence.
No embedded programs are executed and no absence of a marker is a negative
cooperation label. Administrative exact-text hashing is only duplicate accounting.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA256 = '7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488'
PATTERNS = {
    'teammate': r'\bteammates?\b',
    'other_agents': r'\bother\s+agents?\b',
    'cooperation_word': r'\b(?:cooperat(?:e|es|ed|ing|ion|ive)|collaborat(?:e|es|ed|ing|ion|ive))\b',
    'coordination_word': r'\b(?:coordination|coordinate|coordinated|coordinating)\b',
    'swarm': r'\bswarm\b',
    'handoff': r'\bhandoff\b',
    'shared_for_agents': r'\b(?:share|shared|sharing)\b[^\r\n]{0,80}\b(?:agents?|peers?|teammates?)\b',
    'multiagent': r'\bmulti[- ]agent\b',
}
COMPILED = {name: re.compile(pattern, re.I) for name, pattern in PATTERNS.items()}


def markers(text):
    """Return literal spans for review; their meaning is deliberately unresolved."""
    return [dict(pattern=name, start=m.start(), end=m.end(), literal=m.group())
            for name, regex in COMPILED.items() for m in regex.finditer(text)]


def census(records):
    counts, seen, bodies, hits = Counter(), set(), set(), []
    for record in records:
        if record['id'] in seen:
            raise ValueError('Duplicate source record ID')
        seen.add(record['id'])
        counts[record['kind']] += 1
        if record['kind'] != 'payload':
            continue
        text = record['text']
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        bodies.add(digest)
        found = markers(text)
        if found:
            hits.append(dict(record_id=record['id'], text_sha256=digest, matches=found,
                             excerpts=[text[max(0,m['start']-100):m['end']+200] for m in found]))
    return dict(records_by_kind=dict(counts), payload_records=counts['payload'],
                exact_unique_payloads=len(bodies), lexical_candidate_payloads=len(hits),
                lexical_candidate_unique_texts=len({r['text_sha256'] for r in hits}),
                pattern_record_counts=dict(Counter(name for row in hits
                    for name in {m['pattern'] for m in row['matches']})),
                candidates=hits,
                interpretation='Literal wording census requiring separate contextual review. Not action counts, observed cooperation, recall or prevalence. Unmarked records have unknown cooperation status.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a marker census')
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError('Different logical input source')
    with gzip.open(args.source, 'rt', encoding='utf-8') as stream:
        result = census(json.loads(line) for line in stream)
    result.update(source_sha256=source_hash, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  patterns=PATTERNS, temporal_signal_inspection=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['payload_records','exact_unique_payloads','lexical_candidate_payloads','pattern_record_counts']}))


if __name__ == '__main__':
    main()
