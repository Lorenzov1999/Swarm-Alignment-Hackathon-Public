#!/usr/bin/env python3
"""Rank recorded requests with each borrowed technique on its own.

Reporting only; no detector, threshold or earlier output changes. Each
technique's features are fitted alone on 6-31 July with the recipe used for the
paper's models (layer_scoring.evaluate_layers), then applied unchanged to
3-14 August, 17 August-15 September and 1 June-5 July
(transfer_scoring.score_frozen). A single-feature technique ranks pair-days
exactly as its raw feature does, so its ranking needs no fitted weights.

Also reported, from saved outputs: the combined frozen models, the frozen Hawkes
trigger counts, and the July pair-history baseline. Intervals resample whole
days with the models fixed (1,000 draws, seed 20261106); they are descriptive.

Usage: python scripts/review_technique_transfer.py [--output DIR]
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from swarmcollab import metrics
from swarmcollab.layer_scoring import evaluate_layers
from swarmcollab.transfer_scoring import score_frozen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
STUDY = OUT / 'generalization-20261106'
PERIODS = ('temporal', 'shared_goals')
KEYS = ['a', 'b', 'day']
DRAWS, SEED = 1000, 20261106

# label, source of the idea, feature selector over the shared feature columns
TECHNIQUES = {
    'activity': ('Activity only (control)', 'control',
                 lambda cols: ['act_min', 'act_max']),
    'co_use': ('Shared resources, rarity-weighted', 'Pacheco et al.',
               lambda cols: ['p1_cos']),
    'synchrony': ('Same-hour activity on shared resources', 'SynchroTrap',
                  lambda cols: ['p1_sync']),
    'handovers': ('Handovers (read or write after another agent)', 'van der Aalst et al.',
                  lambda cols: ['h_write', 'h_read']),
    'identifier_reuse': ("Later use of another agent's identifiers", 'van der Aalst et al.',
                         lambda cols: ['pou_created_commit', 'pou_created_other',
                                       'pou_first_commit', 'pou_first_other']),
    'provenance': ('Version provenance, levels 1-4', 'this work',
                   lambda cols: ['prov_level_1', 'prov_level_2', 'prov_level_3',
                                 'prov_level_4', 'prov_confirmed_git']),
    'generic_traces': ('Exact URLs and file paths', 'this work',
                       lambda cols: [c for c in cols if c.startswith(('l1_url_', 'l1_file_', 'l1_content_'))]),
    'village_rules': ('Village-specific rules', 'this work',
                      lambda cols: [c for c in cols if c.startswith(('l2_village_project_access_',
                                                                      'l2_village_shared_board_'))]),
    'names': ('Agent names in tool traces', 'this work',
              lambda cols: ['l1_roster_name_85f1ed_verbal_mentions']),
}
COMBINED = {
    'activity': 'Activity only (frozen model)',
    'historical_behavior': 'Earlier behavioral set',
    'layer1_behavior': 'Portable behavior (L1)',
    'layer1_plus_layer2_behavior': 'Behavior + Village rules (L1+2)',
    'layer1_with_attribution': 'L1 with agent names',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boolean(series):
    return series.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()


def interval(frame, score_column, rng_seed=SEED):
    """Percentile interval of AUROC over whole-day resamples."""
    days = frame.day.to_numpy()
    unique = np.unique(days)
    index = {day: np.flatnonzero(days == day) for day in unique}
    rng = np.random.default_rng(rng_seed)
    score, y = frame[score_column].to_numpy(float), frame.y.to_numpy(bool)
    values = []
    for _ in range(DRAWS):
        rows = np.concatenate([index[d] for d in rng.choice(unique, size=len(unique), replace=True)])
        if y[rows].any() and not y[rows].all():
            values.append(metrics.auc(score[rows], y[rows]))
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def describe(frame, score_column):
    return dict(auroc=metrics.auc(frame[score_column].to_numpy(float), frame.y.to_numpy(bool)),
                interval_95=interval(frame, score_column))


def pair_history(dev):
    july = dev[dev.day <= '2026-07-31']
    prevalence = july.y.mean()
    grouped = july.assign(pair=july.a + '|' + july.b).groupby('pair').y.agg(['sum', 'count'])
    def score(frame):
        pair = frame.a + '|' + frame.b
        k = pair.map(grouped['sum']).fillna(0.0)
        n = pair.map(grouped['count']).fillna(0.0)
        return ((k + prevalence) / (n + 1.0)).to_numpy(float)
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=OUT / 'technique-transfer')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'{args.output} exists; refusing to overwrite a reporting run.')

    dev_path = OUT / 'research-layers' / 'features_dev.csv.gz'
    dev = pd.read_csv(dev_path)
    dev['y'] = boolean(dev.y)
    feature_sets = {name: select(list(dev.columns)) for name, (_, _, select) in TECHNIQUES.items()}
    report, dev_predictions = evaluate_layers(dev, feature_sets)
    history = pair_history(dev)

    inputs = {str(dev_path.relative_to(ROOT)): sha256(dev_path)}
    rows = {name: dict(technique=label, source=source, features=len(feature_sets[name]),
                       retained_features=len(report['models'][name]['features']),
                       development=describe(dev_predictions, name))
            for name, (label, source, _) in TECHNIQUES.items()}
    combined = {name: dict(model=label) for name, label in COMBINED.items()}
    extra = {'pair_history': dict(model='July pair history'),
             'hawkes': dict(model='Hawkes trigger counts (Xu et al.), frozen July fit')}

    valid = dev_predictions[KEYS + ['y']].copy()
    valid['pair_history'] = history(valid)
    extra['pair_history']['development'] = describe(valid, 'pair_history')
    triggers_path = OUT / 'spread' / 'pair_day_triggers_dev.csv.gz'
    triggers = pd.read_csv(triggers_path)
    inputs[str(triggers_path.relative_to(ROOT))] = sha256(triggers_path)
    dev_hawkes = valid.merge(triggers, on=KEYS, how='left').fillna({'hk_trig': 0.0})
    extra['hawkes']['development'] = describe(dev_hawkes, 'hk_trig')

    for period in PERIODS:
        features_path = STUDY / period / 'features.csv.gz'
        predictions_path = STUDY / period / 'predictions.csv.gz'
        secondary_path = STUDY / period / 'secondary-pair-features.csv.gz'
        for path in (features_path, predictions_path, secondary_path):
            inputs[str(path.relative_to(ROOT))] = sha256(path)
        features = pd.read_csv(features_path)
        saved = pd.read_csv(predictions_path)
        saved['y'] = boolean(saved.y)
        scored = score_frozen(features, report, period=period).merge(saved[KEYS + ['y']], on=KEYS,
                                                                     validate='one_to_one')
        for name in TECHNIQUES:
            rows[name][period] = describe(scored, name)
            positives = scored[scored.y]
            nonzero = features.merge(positives[KEYS], on=KEYS)[feature_sets[name]].to_numpy(float)
            rows[name][period]['fires_on_positive_share'] = float((nonzero > 0).any(axis=1).mean())
        for name in COMBINED:
            combined[name][period] = describe(saved, name)
        saved['pair_history'] = history(saved)
        extra['pair_history'][period] = describe(saved, 'pair_history')
        secondary = saved[KEYS + ['y']].merge(pd.read_csv(secondary_path)[KEYS + ['hk_trig']], on=KEYS,
                                              how='left').fillna({'hk_trig': 0.0})
        extra['hawkes'][period] = describe(secondary, 'hk_trig')

    args.output.mkdir(parents=True)
    summary = dict(
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Reporting only. Techniques fitted alone on 6-31 July and frozen; no target refitting.',
        draws=DRAWS, seed=SEED, techniques=rows, combined_models=combined, other=extra,
        fitted=report['models'], inputs_sha256=inputs, code_sha256=sha256(__file__))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    table = []
    for group, entries in (('technique', rows), ('combined', combined), ('other', extra)):
        for name, entry in entries.items():
            line = dict(group=group, name=name, label=entry.get('technique') or entry.get('model'),
                        source=entry.get('source', ''))
            for period in ('development',) + PERIODS:
                if period in entry:
                    line[f'{period}_auroc'] = entry[period]['auroc']
                    line[f'{period}_low'], line[f'{period}_high'] = entry[period]['interval_95']
            table.append(line)
    pd.DataFrame(table).to_csv(args.output / 'table.csv', index=False)
    print(pd.DataFrame(table)[['group', 'label', 'development_auroc', 'temporal_auroc', 'temporal_low',
                               'temporal_high', 'shared_goals_auroc']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
