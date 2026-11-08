#!/usr/bin/env python3
"""Run the analysis from the raw data to the reported results.

Stages run in this order. A step is skipped when its output already exists, so
an interrupted run resumes when the same command is run again.

  data         check the input files against data/manifest.json
  development  6 July-14 August pipeline: working steps, resource traces,
               pair-day features, regimes, Hawkes models, provenance, layers
  freeze       freeze the code and development outputs, then fit the frozen
               July state used by the transfer and secondary analyses
  transfer     apply the frozen models to 17 Aug-15 Sep and to 1 Jun-5 Jul
  report       comparison tables, goal-episode summaries, results lock
  followup     alert workload and goal-episode follow-up
  diagnostics  baselines, visibility, follow-through, coverage, Figure 4
  swarm        Swarm Traces static analysis and censuses

Examples:
  python scripts/reproduce.py                    # everything
  python scripts/reproduce.py --stages swarm     # selected stages only
  python scripts/reproduce.py --from transfer    # one stage and all later ones
  python scripts/reproduce.py --dry-run          # print the commands

Logs go to outputs/logs/. Steps that need human or assistant annotations
(provenance chain audits, Swarm contextual reviews) are not run; see README.md.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
STUDY = OUT / 'generalization-20261106'
FOLLOWUP = OUT / 'cooperation-followup-20261106'
REVIEW = OUT / 'claude-review-20261107'
CONTENT = OUT / 'swarm-improvement-20261107'
SWARM_SOURCE = ROOT / 'data' / 'swarm-traces' / 'redacted.jsonl.gz'
SWARM_SHA256 = '7b66ab21674de52fcd3f557652f68b1801170c998e2f266862124e6edf283488'
GOAL_CLASSIFICATION = ROOT / 'config' / 'goal-regime-classification.json'
GOAL_SHA256 = '48c8961ccc0350781957fb0b4ea2e553db82d42d7dd4057eebee472db9cc753b'
STAGES = ('data', 'development', 'freeze', 'transfer', 'report', 'followup',
          'diagnostics', 'swarm')


@dataclass
class Step:
    name: str
    action: Union[list, Callable[[], None]]
    done: Optional[Callable[[], bool]] = None


def exists(path):
    return lambda: Path(path).exists()


def module(name, *args):
    return [sys.executable, '-m', 'swarmcollab.' + name, *map(str, args)]


def script(name, *args):
    return [sys.executable, str(ROOT / 'scripts' / name), *map(str, args)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def traces_done():
    summary = OUT / 'steps' / 'summary.json'
    if not summary.exists():
        return False
    agents = json.loads(summary.read_text())['agents']
    return len(list((OUT / 'mentions').glob('*.activity.json'))) >= len(agents)


def install_goal_classification():
    """Place the frozen goal classification where the study expects it.

    The goal categories were assigned from the goal descriptions before any
    detector result was seen; later stages check this exact file by hash.
    """
    if sha256(GOAL_CLASSIFICATION) != GOAL_SHA256:
        raise SystemExit(f'{GOAL_CLASSIFICATION} does not match its recorded SHA-256.')
    STUDY.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GOAL_CLASSIFICATION, STUDY / 'goal-regime-classification.json')


def build_stages(options):
    workers = str(options.workers)
    census = FOLLOWUP / 'swarm' / 'operation-census-v2'
    javascript = [] if options.javascript else ['--without-javascript']
    label_flag = ['--allow-label-snapshot-mismatch'] if options.allow_label_snapshot_mismatch else []
    return {
        'data': [
            Step('check input files', script('check_data.py', *(['--no-hash'] if options.quick_data_check else []))),
        ],
        'development': [
            Step('extract working steps (chat and history excluded)', module('extract'),
                 exists(OUT / 'steps' / 'summary.json')),
            Step('parse resource traces', module('traces'), traces_done),
            Step('pair-day features and scores', module('pairs'), exists(OUT / 'pairs' / 'scores_dev.json')),
            Step('daily co-use regimes', module('regimes'), exists(OUT / 'regimes' / 'summary_dev.json')),
            Step('agent-level Hawkes model', module('hawkes'), exists(OUT / 'hawkes' / 'summary_dev.json')),
            Step('artifact-mediated Hawkes model', module('spread'), exists(OUT / 'spread' / 'summary_dev.json')),
            Step('versioned provenance', module('provenance', '--workers', workers, '--null-reps', 5,
                                                '--audit-size', 80),
                 exists(OUT / 'provenance' / 'summary.json')),
            Step('portable and Village-specific layers', module('research_layers', '--workers', workers),
                 exists(OUT / 'research-layers' / 'scores.json')),
        ],
        'freeze': [
            Step('install frozen goal classification', install_goal_classification,
                 exists(STUDY / 'goal-regime-classification.json')),
            Step('freeze code and development outputs', script('freeze_development.py'),
                 exists(STUDY / 'freeze-manifest.json')),
            Step('frozen July identities, rarity and common content',
                 module('generalization', '--stage', 'freeze'),
                 exists(STUDY / 'training-reconstruction-inputs.json')),
            Step('fit secondary July models', module('secondary_run', '--stage', 'fit', '--study', STUDY),
                 exists(STUDY / 'secondary-fit-record.json')),
        ],
        'transfer': [
            Step('transfer to 17 Aug-15 Sep', script('run_generalization_stage.py', '--period', 'temporal',
                                                     '--workers', workers),
                 exists(STUDY / 'temporal' / 'scores.json')),
            Step('transfer to 1 Jun-5 Jul', script('run_generalization_stage.py', '--period', 'shared_goals',
                                                   '--workers', workers),
                 exists(STUDY / 'shared_goals' / 'scores.json')),
            Step('secondary analyses, 17 Aug-15 Sep',
                 module('secondary_run', '--stage', 'apply', '--period', 'temporal', '--study', STUDY),
                 exists(STUDY / 'temporal' / 'secondary-summary.json')),
            Step('secondary analyses, 1 Jun-5 Jul',
                 module('secondary_run', '--stage', 'apply', '--period', 'shared_goals', '--study', STUDY),
                 exists(STUDY / 'shared_goals' / 'secondary-summary.json')),
        ],
        'report': [
            Step('comparison tables', module('transfer_report', '--study', STUDY, '--goal-sha256', GOAL_SHA256),
                 exists(STUDY / 'report' / 'primary-comparison.csv')),
            Step('provenance by goal episode', script('summarize_goal_evidence.py', '--study', STUDY),
                 exists(STUDY / 'report' / 'goal-provenance.csv')),
            Step('lock Village results', script('lock_village_results.py'),
                 exists(STUDY / 'village-results-lock.json')),
        ],
        'followup': [
            Step('workload and goal-episode follow-up',
                 module('cooperation_followup', '--study', STUDY, '--output', FOLLOWUP / 'village'),
                 exists(FOLLOWUP / 'village' / 'manifest.json')),
        ],
        'diagnostics': [
            Step('baselines and within-pair comparison',
                 script('review_village_baselines.py', '--output', REVIEW / 'baselines'),
                 exists(REVIEW / 'baselines' / 'summary.json')),
            Step('visibility and follow-through',
                 script('review_label_visibility.py', '--output', REVIEW / 'labels-visibility', *label_flag),
                 exists(REVIEW / 'labels-visibility' / 'summary.json')),
            Step('single-request follow-through',
                 script('review_followthrough_sensitivity.py', '--folder', REVIEW / 'labels-visibility'),
                 exists(REVIEW / 'labels-visibility' / 'single-request-sensitivity.json')),
            Step('goal-episode coverage and sharing',
                 script('review_regime_robustness.py', '--study', STUDY, '--inputs', STUDY,
                        '--output', REVIEW / 'regimes' / 'report'),
                 exists(REVIEW / 'regimes' / 'report' / 'daily.csv')),
            Step('activity comparisons within pairs and days', script('check_activity_concordance.py'),
                 exists(REVIEW / 'activity-concordance-review.json')),
            Step('follow-through activity control', script('check_followthrough_control.py'),
                 exists(REVIEW / 'followthrough-control.json')),
            Step('each technique alone (Table 1)', script('review_technique_transfer.py'),
                 exists(OUT / 'technique-transfer' / 'table.csv')),
            Step('recall against the alert budget', script('review_alert_budget.py'),
                 exists(OUT / 'alert-budget' / 'summary.json')),
            Step('Figure 3', script('plot_alert_budget.py', '--source', OUT / 'alert-budget',
                                    '--output-dir', OUT / 'figures'),
                 exists(OUT / 'figures' / 'alert-budget.pdf')),
            Step('July signal composition', script('review_signal_composition.py'),
                 exists(OUT / 'signal-composition' / 'summary.json')),
            Step('Figure 4', script('review_goal_coverage.py', '--daily',
                                    REVIEW / 'regimes' / 'report' / 'daily.csv', '--output-dir', OUT / 'figures'),
                 exists(OUT / 'figures' / 'goal-co-use-robustness.pdf')),
        ],
        'swarm': [
            Step('static reference analysis',
                 module('swarm_static', '--config', ROOT / 'config' / 'swarm-schema-adapter.json',
                        '--source', SWARM_SOURCE, '--output', STUDY / 'swarm' / 'static'),
                 exists(STUDY / 'swarm' / 'static' / 'summary.json')),
            Step('cooperation wording census',
                 module('cooperation_markers', '--source', SWARM_SOURCE,
                        '--output', FOLLOWUP / 'swarm' / 'marker-census.json'),
                 exists(FOLLOWUP / 'swarm' / 'marker-census.json')),
            Step('request-expression census',
                 script('swarm_operation_census.py', '--source', SWARM_SOURCE, '--source-sha256', SWARM_SHA256,
                        '--output', census, *javascript),
                 exists(census / 'summary.json')),
            Step('embedded date/time inventory',
                 script('swarm_timestamp_profile.py', '--source', SWARM_SOURCE,
                        '--output', FOLLOWUP / 'swarm' / 'time-audit' / 'profile'),
                 exists(FOLLOWUP / 'swarm' / 'time-audit' / 'profile' / 'summary.json')),
            Step('artifact-text census',
                 script('run_swarm_content.py', '--source', SWARM_SOURCE,
                        '--operations', census / 'payload-operation-coverage.csv.gz',
                        '--operation-manifest', census / 'manifest.json',
                        '--development-candidates', FOLLOWUP / 'swarm' / 'marker-census.json',
                        '--output', CONTENT / 'census'),
                 exists(CONTENT / 'census' / 'summary.json')),
            Step('artifact-text census diagnostics',
                 script('summarize_swarm_content_diagnostics.py', '--census', CONTENT / 'census',
                        '--output', CONTENT / 'census-diagnostics.json'),
                 exists(CONTENT / 'census-diagnostics.json')),
        ],
    }


def environment():
    env = dict(os.environ)
    existing = env.get('PYTHONPATH')
    env['PYTHONPATH'] = str(ROOT / 'src') + (os.pathsep + existing if existing else '')
    env.update(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def show(command):
    return ' '.join(str(Path(part).relative_to(ROOT)) if str(part).startswith(str(ROOT)) else str(part)
                    for part in command)


def run_step(stage, index, step, dry_run):
    label = f'[{stage}] {step.name}'
    if step.done is not None and step.done():
        print(f'{label}: already done, skipping')
        return
    if dry_run:
        print(f'{label}:\n    ' + (show(step.action) if isinstance(step.action, list) else
                                    f'(internal) {step.action.__doc__.splitlines()[0]}'))
        return
    print(f'{label} ...', flush=True)
    started = time.monotonic()
    if callable(step.action):
        step.action()
    else:
        logs = OUT / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        slug = ''.join(c if c.isalnum() else '-' for c in step.name.lower()).strip('-')
        log = logs / f'{stage}-{index:02d}-{slug}.log'
        with log.open('w') as stream:
            stream.write('$ ' + show(step.action) + '\n')
            stream.flush()
            result = subprocess.run(step.action, cwd=ROOT, env=environment(), stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            tail = log.read_text().splitlines()[-25:]
            print('\n'.join(['    ' + line for line in tail]))
            raise SystemExit(f'{label} failed (exit {result.returncode}); full log: {log.relative_to(ROOT)}')
    print(f'{label}: done in {time.monotonic() - started:.0f} s')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', help='comma-separated stages to run (default: all)')
    parser.add_argument('--from', dest='first', choices=STAGES, help='run this stage and all later ones')
    parser.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2,
                        help='parallel workers for the heavier steps (default 2)')
    parser.add_argument('--allow-label-snapshot-mismatch', action='store_true',
                        help='let the label diagnostic use a request snapshot newer than the analysed one')
    parser.add_argument('--without-javascript', dest='javascript', action='store_false', default=None,
                        help='skip JavaScript parsing in the request-expression census')
    parser.add_argument('--quick-data-check', action='store_true', help='compare data file sizes only')
    parser.add_argument('--dry-run', action='store_true', help='print the commands without running them')
    options = parser.parse_args()
    if options.javascript is None:
        options.javascript = shutil.which('node') is not None
        if not options.javascript:
            print('Node.js not found: the request-expression census will skip JavaScript, '
                  'so its coverage will be lower than reported.')
    if options.stages and options.first:
        parser.error('use either --stages or --from')
    if options.stages:
        selected = [name.strip() for name in options.stages.split(',') if name.strip()]
        unknown = sorted(set(selected) - set(STAGES))
        if unknown:
            parser.error('unknown stage(s): ' + ', '.join(unknown))
    elif options.first:
        selected = list(STAGES[STAGES.index(options.first):])
    else:
        selected = list(STAGES)
    stages = build_stages(options)
    for stage in STAGES:
        if stage in selected:
            for index, step in enumerate(stages[stage], 1):
                run_step(stage, index, step, options.dry_run)
    if not options.dry_run:
        print('Finished. See README.md, "Where the results are", for the output of each paper result.')


if __name__ == '__main__':
    main()
