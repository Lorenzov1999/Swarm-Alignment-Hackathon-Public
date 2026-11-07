#!/usr/bin/env python3
"""Record SHA-256 hashes of the completed Village transfer results.

Writes outputs/generalization-20261106/village-results-lock.json with the hash
of every file under temporal/, shared_goals/ and report/. The later review
scripts read this lock and refuse to run if a locked result has changed since.
This recreates the lock made in the original study once the Village evaluation
was complete.

Usage: python scripts/lock_village_results.py
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs' / 'generalization-20261106'
LOCKED_DIRECTORIES = ('temporal', 'shared_goals', 'report')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    lock_path = STUDY / 'village-results-lock.json'
    if lock_path.exists():
        raise SystemExit(f'{lock_path} already exists; refusing to overwrite the lock.')
    missing = [name for name in LOCKED_DIRECTORIES if not (STUDY / name).is_dir()]
    if missing:
        raise SystemExit('Missing result directories (run the transfer and report stages): '
                         + ', '.join(missing))
    paths = sorted(path for name in LOCKED_DIRECTORIES
                   for path in (STUDY / name).rglob('*') if path.is_file())
    lock = {
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'phase': 'Completed Village evaluation, before the follow-up and review diagnostics',
        'files_sha256': {str(path.relative_to(STUDY)): sha256(path) for path in paths},
    }
    lock_path.write_text(json.dumps(lock, indent=2) + '\n')
    print(f'Locked {len(paths)} Village result files in {lock_path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
