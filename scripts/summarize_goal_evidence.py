#!/usr/bin/env python3
"""Summarize frozen June provenance by the predeclared goal-day assignment.

This is descriptive reporting, not detector fitting or a goal-effect estimate.
Run with the repository venv from the repository root.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

GOAL_SHA256 = '48c8961ccc0350781957fb0b4ea2e553db82d42d7dd4057eebee472db9cc753b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(study):
    inputs = {name: study / name for name in (
        'goal-regime-classification.json', 'shared_goals/chains.jsonl.gz',
        'shared_goals/activity.csv.gz', 'shared_goals/scores.json')}
    assert sha(inputs['goal-regime-classification.json']) == GOAL_SHA256
    classification = json.loads(inputs['goal-regime-classification.json'].read_text())
    assignments = pd.DataFrame(classification['day_assignments'])
    assert not assignments.day.duplicated().any()
    with gzip.open(inputs['shared_goals/chains.jsonl.gz'], 'rt') as stream:
        chains = pd.DataFrame(json.loads(line) for line in stream)
    activity = pd.read_csv(inputs['shared_goals/activity.csv.gz'])
    assert not activity.duplicated(['agent', 'day']).any()
    assert set(chains.day) <= set(assignments.day)
    assert set(activity.day) <= set(assignments.day)
    assert chains.chain_id.is_unique and chains.level.isin([1, 2, 3, 4]).all()
    original = json.loads(inputs['shared_goals/scores.json'].read_text())
    assert len(chains) == original['measurement']['provenance']['target_chains']
    assert int(activity.working.sum()) == original['measurement']['working_steps']
    descriptions = {row['goal_id']: row['description'] for row in classification['intervals']}
    outputs = []
    for sensitivity in ('all_days', 'exclude_transition_days'):
        selected = assignments if sensitivity == 'all_days' else assignments[~assignments.transition_day]
        for grouping, column in (('goal', 'primary_goal_id'), ('category', 'primary_category')):
            for key, group in selected.groupby(column, sort=True):
                days = set(group.day)
                observations = chains[chains.day.isin(days)]
                steps = activity[activity.day.isin(days)]
                working = int(steps.working.sum())
                strong = observations[observations.level >= 3]
                row = dict(sensitivity=sensitivity, grouping=grouping, group=key,
                           description=descriptions.get(key, key),
                           category=group.primary_category.iloc[0],
                           assigned_calendar_days=len(days), observed_days=int(steps.day.nunique()),
                           working_steps=working, total_chains=len(observations),
                           strong_chains=len(strong), strong_target_turns=int(strong.target_turn.nunique()),
                           strong_directed_pair_days=len(strong[['source_actor', 'target_actor', 'day']].drop_duplicates()))
                for level in (1, 2, 3, 4):
                    count = int(observations.level.eq(level).sum())
                    row[f'level{level}_chains'] = count
                    row[f'level{level}_per_1000_working_steps'] = 1000 * count / working if working else None
                row['strong_per_1000_working_steps'] = 1000 * len(strong) / working if working else None
                outputs.append(row)
    result = pd.DataFrame(outputs)
    for grouping in ('goal', 'category'):
        selected = result[(result.grouping == grouping) & (result.sensitivity == 'all_days')]
        assert int(selected.total_chains.sum()) == len(chains)
        assert int(selected.working_steps.sum()) == int(activity.working.sum())
    destination = study / 'report'
    destination.mkdir(exist_ok=True)
    result.to_csv(destination / 'goal-provenance.csv', index=False)
    record = dict(inputs_sha256={name: sha(path) for name, path in inputs.items()},
                  reporting_source_sha256=sha(Path(__file__)),
                  outputs_sha256={'goal-provenance.csv': sha(destination / 'goal-provenance.csv')},
                  interpretation='Descriptive counts of frozen evidence chains. Chains can share turns, actors, and episodes; counts are not independent interactions. Goal-day assignment uses predeclared scheduled-clock overlap. No goal-effect estimate or consumption precision is inferred.')
    (destination / 'goal-provenance-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(result[(result.grouping == 'category') & (result.sensitivity == 'all_days')].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('outputs/generalization-20261106'))
    summarize(parser.parse_args().study)
