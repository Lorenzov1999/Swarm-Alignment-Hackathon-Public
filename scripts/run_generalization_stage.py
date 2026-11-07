#!/usr/bin/env python3
"""Run one predeclared Village transfer stage and retain a resumable status."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--period', choices=['temporal', 'shared_goals'], required=True)
parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
study = root / 'outputs/generalization-20261106'
target = study / args.period
target.mkdir(parents=True, exist_ok=True)
status_path = target / 'run-status.json'
now = lambda: dt.datetime.now(dt.timezone.utc).isoformat()
state = dict(period=args.period, started_utc=now(), status='running', stages=[],
             python=sys.version, pid=os.getpid())


def save():
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(status_path)


commands = [
    ('extract', [sys.executable, '-m', 'swarmcollab.transfer_extract', '--period', args.period,
                 '--output', str(target / 'steps'), '--freeze-manifest', str(study / 'freeze-manifest.json'), '--resume']),
    ('parse', [sys.executable, '-m', 'swarmcollab.generalization', '--period', args.period,
               '--stage', 'parse', '--workers', str(args.workers)]),
    ('evaluate', [sys.executable, '-m', 'swarmcollab.generalization', '--period', args.period,
                  '--stage', 'evaluate']),
]
environment = dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONUNBUFFERED='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0')
save()
for name, command in commands:
    entry = dict(name=name, started_utc=now(), command=command)
    state['stages'].append(entry); state['current_stage'] = name; save()
    print(now(), name, flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=root, env=environment)
    entry.update(returncode=result.returncode, elapsed_seconds=time.monotonic()-started, ended_utc=now())
    if result.returncode:
        state.update(status='failed', ended_utc=now()); save()
        raise SystemExit(result.returncode)
    save()
state.update(status='complete', ended_utc=now())
state['scores_sha256'] = hashlib.sha256((target / 'scores.json').read_bytes()).hexdigest()
save()
print(now(), 'Completed', args.period, flush=True)
