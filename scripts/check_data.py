#!/usr/bin/env python3
"""Check that the input data are in place and match the analysed files.

Compares each input's size and SHA-256 with data/manifest.json, which records
the exact files behind the reported results. Differences are reported, not
fixed. The AI Village dataset and the AI Digest request chart are updated
upstream, so a newer download can differ from the analysed one even when the
analysed dates are unchanged. The Swarm Traces export must match exactly: the
Swarm stages refuse any other file.

Usage: python scripts/check_data.py [--no-hash] [--strict]
Exit status is 1 if a required file is missing (or, with --strict, differs).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'data/ai-village/main-village/computer_use_turns.jsonl.gz': 'Village pipeline',
    'data/ai-village/main-village/computer_use_sessions.jsonl.gz': 'Village pipeline',
    'data/ai-village/main-village/agents.jsonl.gz': 'Village pipeline',
    'data/labels/ai-village-chart-props.json': 'request labels',
    'data/swarm-traces/redacted.jsonl.gz': 'Swarm Traces stages',
}
OPTIONAL = {
    'data/ai-village/main-village/village_goals.jsonl.gz': 'goal timeline table only',
}
NOTES = {
    'data/labels/ai-village-chart-props.json':
        'a newer request-chart snapshot works for the main pipeline; run the label '
        'diagnostic with --allow-label-snapshot-mismatch',
    'data/swarm-traces/redacted.jsonl.gz':
        'the Swarm stages will refuse this file',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_files():
    manifest = json.loads((ROOT / 'data' / 'manifest.json').read_text())
    return {item['target']: item for dataset in manifest['datasets'] for item in dataset['files']}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-hash', action='store_true', help='compare sizes only (fast)')
    parser.add_argument('--strict', action='store_true', help='fail when a file differs')
    args = parser.parse_args()
    expected = expected_files()
    missing, differing = [], []
    for relative, role in list(REQUIRED.items()) + list(OPTIONAL.items()):
        path = ROOT / relative
        required = relative in REQUIRED
        if not path.is_file():
            status = 'MISSING' if required else 'missing (optional)'
            if required:
                missing.append(relative)
        else:
            reference = expected.get(relative)
            same_size = reference is not None and path.stat().st_size == reference['bytes']
            if args.no_hash:
                same = same_size
            else:
                same = same_size and sha256(path) == reference['sha256']
            status = 'matches analysed file' if same else 'DIFFERS from analysed file'
            if not same:
                differing.append(relative)
                if relative in NOTES:
                    status += f' ({NOTES[relative]})'
        print(f'{status:<28} {relative}  [{role}]')
    if missing:
        print('\nMissing required inputs. See README.md, "Get the data".')
    if differing and not missing:
        print('\nSome inputs differ from the analysed files; results may differ slightly.')
    sys.exit(1 if missing or (args.strict and differing) else 0)


if __name__ == '__main__':
    main()
