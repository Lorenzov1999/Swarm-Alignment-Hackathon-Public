"""Report saved frozen-transfer results without opening source observations.

Only score reports, prediction tables, activity aggregates, and locked goal
assignments are read. No model fitting, threshold selection, new labels, or
causal-effect estimation occurs. Missing measurements remain explicitly absent.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .transfer_scoring import PERIODS, evaluate_predictions


PERIOD_ORDER = ['development_validation', 'temporal', 'shared_goals']
PERIOD_LABELS = {'development_validation': 'Dev. validation\n3–14 Aug',
                 'temporal': 'Temporal transfer\n17 Aug–15 Sep',
                 'shared_goals': 'Earlier goals\n1 Jun–5 Jul'}
PLOT_MODELS = {'layer1_behavior': 'Layer 1',
               'layer1_plus_layer2_behavior': 'Layers 1 + 2',
               'layer1_with_attribution': 'Layer 1 + names',
               'layer1_plus_layer2_with_attribution': 'Layers 1 + 2 + names'}
METRICS = ['auc', 'average_precision', 'recall', 'fpr', 'precision', 'alert_rate']
COUNTS = ['rows', 'positives', 'negatives', 'alerts', 'true_positives', 'false_positives',
          'false_negatives', 'true_negatives']
GOAL_CATEGORIES = ['collaborative', 'competitive', 'individual', 'mixed']
NOTES = [
    'Recorded requests are proxy labels, not verified consumption or harmful intent.',
    'All model parameters and alert thresholds remain frozen; no target-dependent operating point is selected.',
    'Development validation was inspected during research; earlier exploratory summaries also touched the temporal period.',
    'Earlier-goal results are backward transfer. Differences in agents, tasks, tools and logging can confound goal comparisons.',
    'Goal categories describe the wording of assigned goals; mixed means unspecified coordination prescription.',
    'Goal-day assignment uses locked scheduled-clock overlap, not activity or detector results. Transition days are separately excluded as sensitivity analysis.',
    'Average precision is noninterpolated with tied scores grouped; its chance-level reference depends on request prevalence.',
    'Figures show point estimates only. No uncertainty intervals are invented or inferred from independent tool-call assumptions.',
    'Missing results and zero-denominator rates are null, never fabricated zero measurements.',
    'Where Layer 2 has zero trace coverage, differences between separately fitted Layer 1 and Layer 1+2 models do not establish a benefit from Village workflow matches.',
]


def _finite(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Expected a numeric saved measurement') from error
    return result if math.isfinite(result) else None


def _ratio(numerator, denominator, scale=1.0):
    numerator, denominator = _finite(numerator), _finite(denominator)
    if numerator is None or denominator is None or denominator <= 0:
        return None
    if numerator < 0:
        raise ValueError('Count numerator cannot be negative')
    return scale * numerator / denominator


def normalize_model_metrics(model):
    """Unify historical nested validation metrics and flat transfer metrics.

    ROC AUC and average precision are never inferred from confusion counts.
    ``descriptive_recall_at_10pct_fpr`` is intentionally not a fallback: it uses
    target labels to choose a threshold and differs from the frozen operating
    point represented in these tables.
    """
    if not isinstance(model, dict):
        raise ValueError('A saved model result must be an object')
    nested = model.get('validation_at_train_threshold', {})
    if not isinstance(nested, dict):
        raise ValueError('Invalid historical validation metrics')
    values = {**nested, **{key: value for key, value in model.items()
                           if not isinstance(value, (dict, list))}}
    result = {key: _finite(values.get(key)) for key in COUNTS + METRICS + ['threshold']}
    for key in COUNTS:
        if result[key] is not None:
            if result[key] < 0 or result[key] != int(result[key]):
                raise ValueError('Saved counts must be nonnegative integers: ' + key)
            result[key] = int(result[key])
    for total, left, right in [('rows', 'positives', 'negatives'),
                               ('alerts', 'true_positives', 'false_positives'),
                               ('positives', 'true_positives', 'false_negatives'),
                               ('negatives', 'false_positives', 'true_negatives')]:
        if all(result[key] is not None for key in (total, left, right)):
            if result[total] != result[left] + result[right]:
                raise ValueError('Inconsistent saved confusion counts: ' + total)
    fallback = {
        'recall': _ratio(result['true_positives'], result['positives']),
        'fpr': _ratio(result['false_positives'], result['negatives']),
        'precision': _ratio(result['true_positives'], result['alerts']),
        'alert_rate': _ratio(result['alerts'], result['rows']),
    }
    for key, value in fallback.items():
        if result[key] is None:
            result[key] = value
    result['positive_prevalence'] = _ratio(result['positives'], result['rows'])
    result['alerts_per_observed_day'] = _finite(values.get('alerts_per_observed_day', values.get('alerts_per_day')))
    for key in METRICS + ['positive_prevalence']:
        if result[key] is not None and not 0 <= result[key] <= 1:
            raise ValueError('Saved rate is outside [0, 1]: ' + key)
    return result


def comparison_table(reports, model_names):
    rows = []
    for period in PERIOD_ORDER:
        report = reports.get(period)
        layer2_covered = _finite((report or {}).get('measurement', {}).get('covered_steps_by_layer', {}).get('2'))
        for model in model_names:
            present = report is not None and model in report.get('models', {})
            metrics = normalize_model_metrics(report['models'][model]) if present else normalize_model_metrics({})
            rows.append(dict(period=period, start=PERIODS[period][0], end=PERIODS[period][1],
                             model=model, status='available' if present else 'missing',
                             layer2_covered_steps=layer2_covered,
                             layer2_no_trace_matches=layer2_covered == 0 if layer2_covered is not None else None,
                             **metrics))
    return pd.DataFrame(rows)


def _activity_counts(activity, days=None):
    if activity is None:
        return {'working_steps': None, 'active_agent_days': None, 'distinct_active_agents': None,
                'active_days': None}
    if not {'agent', 'day', 'working'}.issubset(activity) or activity.duplicated(['agent', 'day']).any():
        raise ValueError('Activity must have unique agent/day rows and working counts')
    frame = activity if days is None else activity[activity.day.isin(days)]
    values = pd.to_numeric(frame.working, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Working counts must be finite and nonnegative')
    active = frame.loc[values > 0]
    return dict(working_steps=float(values.sum()), active_agent_days=len(active),
                distinct_active_agents=int(active.agent.nunique()), active_days=int(active.day.nunique()))


def _check_activity_support(activity, predictions):
    if activity is None or predictions.empty:
        return
    _activity_counts(activity)
    active = activity.loc[pd.to_numeric(activity.working, errors='raise') > 0]
    supported = set(zip(active.agent, active.day))
    required = set(zip(predictions.a, predictions.day)) | set(zip(predictions.b, predictions.day))
    if required - supported:
        raise ValueError('Activity does not cover every scored actor/day with positive work')


def coverage_table(reports, activities=None):
    activities = activities or {}
    rows = []
    for period in PERIOD_ORDER:
        report = reports.get(period)
        measurement = (report or {}).get('measurement', {})
        if not isinstance(measurement, dict):
            measurement = {}
        activity = _activity_counts(activities.get(period))
        if activity['working_steps'] is None:
            activity['working_steps'] = _finite(measurement.get('working_steps'))
        cached = _finite(measurement.get('cached_steps'))
        excluded = _finite(measurement.get('excluded_history_steps'))
        eligible = cached - excluded if cached is not None and excluded is not None else None
        if eligible is not None and eligible < 0:
            raise ValueError('Excluded history steps exceed cached steps')
        status = 'missing' if report is None else ('available' if cached is not None else 'missing_measurement')
        row = dict(period=period, status=status,
                   cached_steps=cached, excluded_history_steps=excluded,
                   steps_after_history_exclusion=eligible, **activity)
        coverage = measurement.get('covered_steps_by_layer', {})
        for layer in (1, 2):
            count = _finite(coverage.get(str(layer)))
            if count is not None and (count < 0 or (eligible is not None and count > eligible)):
                raise ValueError('Covered step count is incompatible with its denominator')
            row[f'layer{layer}_covered_steps'] = count
            row[f'layer{layer}_coverage_per_cached_step'] = _ratio(count, cached)
            row[f'layer{layer}_coverage_after_history_exclusion'] = _ratio(count, eligible)
        provenance = measurement.get('provenance', {})
        levels = provenance.get('chain_levels', {})
        rates = measurement.get('chains_per_1000_working_steps', {})
        for level in range(1, 5):
            count = _finite(levels.get(str(level)))
            # A reported total plus a complete counter implies absent levels are
            # observed zeros. Absence of the provenance report itself does not.
            if count is None and 'target_chains' in provenance and isinstance(levels, dict):
                count = 0
            row[f'level{level}_chains'] = count
            value = _finite(rates.get(str(level)))
            row[f'level{level}_chains_per_1000_working_steps'] = (
                value if value is not None else _ratio(count, activity['working_steps'], 1000))
        rows.append(row)
    return pd.DataFrame(rows)


def novel_agent_table(reports, model_names):
    rows = []
    for period in PERIOD_ORDER:
        groups = (reports.get(period) or {}).get('agent_strata', {})
        for stratum in ('both_known', 'one_unseen', 'both_unseen'):
            group = groups.get(stratum)
            for model in model_names:
                found = group is not None and model in group.get('models', {})
                values = normalize_model_metrics(group['models'][model] if found else {})
                status = 'missing' if not found else ('empty_stratum' if values['rows'] == 0 else 'available')
                rows.append(dict(period=period, stratum=stratum, model=model, status=status, **values))
    return pd.DataFrame(rows)


def _goal_assignments(classification, required_days):
    if classification.get('period') != list(PERIODS['shared_goals']):
        raise ValueError('Goal assignments do not describe the authorized shared-goals window')
    assignments = pd.DataFrame(classification.get('day_assignments', []))
    required = {'day', 'primary_goal_id', 'primary_category', 'transition_day'}
    if not required.issubset(assignments) or assignments.day.duplicated().any():
        raise ValueError('Goal assignments require one complete mapping per day')
    if (assignments[list(required)].isna().any().any()
            or not assignments.transition_day.isin([True, False]).all()):
        raise ValueError('Goal assignments contain missing values or invalid transition flags')
    lo, hi = PERIODS['shared_goals']
    for day in assignments.day:
        if not isinstance(day, str) or dt.date.fromisoformat(day).isoformat() != day or not lo <= day <= hi:
            raise ValueError('Goal day lies outside the authorized period')
    if set(required_days) - set(assignments.day):
        raise ValueError('Scored or active days are missing locked goal assignments')
    intervals = classification.get('intervals', [])
    by_id = {interval['goal_id']: interval for interval in intervals}
    if len(by_id) != len(intervals) or set(assignments.primary_goal_id) - set(by_id):
        raise ValueError('Goal assignments refer to missing or duplicate goal IDs')
    for row in assignments.itertuples():
        if row.primary_category != by_id[row.primary_goal_id]['category']:
            raise ValueError('Goal category differs between day assignment and interval')
    assignments['transition_day'] = assignments.transition_day.astype(bool)
    return assignments, by_id


def goal_tables(predictions, saved_report, classification, *, activity=None, model_names=None):
    """Descriptive goal/category comparisons using the locked day assignment.

    Goal rows without primary-assigned days remain explicit zero-observation
    rows. Aggregated rates use summed numerators and denominators, never an
    unweighted mean of daily rates. No outcome-dependent regrouping is allowed.
    """
    names = list(saved_report['models']) if model_names is None else list(model_names)
    required_days = set(predictions.day)
    if activity is not None:
        required_days |= set(activity.day)
    assignments, goals = _goal_assignments(classification, required_days)
    # Verify every saved alert still uses its original threshold before grouping.
    if len(predictions):
        evaluate_predictions(predictions, saved_report, period='shared_goals', model_names=names)
    if activity is not None:
        _activity_counts(activity)
        _check_activity_support(activity, predictions)
    tables = {}
    for level in ('goal', 'category'):
        groups = list(goals) if level == 'goal' else list(dict.fromkeys(
            GOAL_CATEGORIES + [goal['category'] for goal in goals.values()]))
        rows = []
        for sensitivity in ('all_days', 'exclude_transitions'):
            available = assignments if sensitivity == 'all_days' else assignments[~assignments.transition_day]
            for group in groups:
                column = 'primary_goal_id' if level == 'goal' else 'primary_category'
                assigned = available[available[column] == group]
                days = set(assigned.day)
                subset = predictions[predictions.day.isin(days)]
                evaluation = (evaluate_predictions(subset, saved_report, period='shared_goals', model_names=names)
                              if len(subset) else None)
                activity_values = _activity_counts(activity, days)
                goal = goals[group] if level == 'goal' else None
                for model in names:
                    values = normalize_model_metrics(evaluation['models'][model] if evaluation else {
                        'rows': 0, 'alerts': 0, 'positives': 0, 'negatives': 0,
                        'true_positives': 0, 'false_positives': 0,
                        'false_negatives': 0, 'true_negatives': 0,
                        'threshold': saved_report['models'][model]['threshold']})
                    rows.append(dict(
                        grouping=level, group_id=group,
                        description=goal.get('description') if goal else None,
                        category=goal['category'] if goal else group,
                        classification_confidence=goal.get('confidence') if goal else None,
                        sensitivity=sensitivity, model=model,
                        status='available' if len(subset) else 'no_observed_pair_days',
                        assigned_calendar_days=len(days), observed_pair_days=len(subset),
                        days_with_pair_observations=int(subset.day.nunique()),
                        transition_days_included=int(assigned.transition_day.sum()),
                        goals_represented=int(assigned.primary_goal_id.nunique()),
                        **activity_values, **values,
                        alerts_per_1000_working_steps=_ratio(values['alerts'], activity_values['working_steps'], 1000)))
        tables[level] = pd.DataFrame(rows)
    return tables['goal'], tables['category']


def secondary_table(secondary_reports):
    """Keep unlike secondary estimands and unsupported cases visibly separate."""
    rows = []
    entries = [
        ('regimes', 'undirected_pair_day', ('regimes',)),
        ('marked_hawkes_all_pairs', 'undirected_pair_day', ('marked_hawkes_all_pairs',)),
        ('marked_hawkes_supported_actor_pairs', 'undirected_pair_day', ('marked_hawkes_supported_actor_pairs',)),
        ('legacy_alpha_pair_day', 'directed_pair_day', ('legacy_whole_development_network', 'alpha_pair_day')),
        ('legacy_excess_pair_day', 'directed_pair_day', ('legacy_whole_development_network', 'excess_pair_day')),
        ('legacy_alpha_any_request_pair', 'directed_pair', ('legacy_whole_development_network', 'alpha_any_request_pair')),
        ('legacy_excess_any_request_pair', 'directed_pair', ('legacy_whole_development_network', 'excess_any_request_pair')),
    ]
    for period in ('temporal', 'shared_goals'):
        report = secondary_reports.get(period)
        for method, unit, path in entries:
            value = report
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            present = isinstance(value, dict)
            rows.append(dict(period=period, method=method, unit=unit,
                             status='available' if present else 'missing',
                             fit_scope='whole_development' if method.startswith('legacy_') else 'July_training',
                             **normalize_model_metrics(value or {})))
    return pd.DataFrame(rows)


def plot_primary(table, destination):
    """Four rate panels with identical 0--1 scales and categorical periods."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), sharex=True)
    colors = ['#245b9e', '#228474', '#245b9e', '#228474']
    markers = ['o', 's', 'o', 's']
    offsets = np.linspace(-.22, .22, len(PLOT_MODELS))
    panels = [('auc', 'ROC AUC'), ('average_precision', 'Average precision'),
              ('recall', 'Recall at frozen threshold'), ('fpr', 'FPR at frozen threshold')]
    unsupported = set(table.loc[table.get('layer2_no_trace_matches', pd.Series(False, index=table.index)).eq(True), 'period'])
    period_labels = {period: label.replace('\n', '†\n', 1) if period in unsupported else label
                     for period, label in PERIOD_LABELS.items()}
    for axis, (metric, title) in zip(axes.flat, panels):
        for index, (model, label) in enumerate(PLOT_MODELS.items()):
            subset = table[table.model.eq(model)].set_index('period')
            for x, period in enumerate(PERIOD_ORDER):
                if period not in subset.index:
                    continue
                value = subset.loc[period, metric]
                if pd.notna(value):
                    axis.scatter(x + offsets[index], value, s=35, marker=markers[index],
                                 facecolors=colors[index] if index < 2 else 'white',
                                 edgecolors=colors[index], linewidths=1.2, zorder=3)
        if metric == 'auc':
            axis.axhline(.5, color='#aab0b7', linestyle=':', linewidth=.9)
        elif metric == 'fpr':
            axis.axhline(.1, color='#aab0b7', linestyle=':', linewidth=.9)
        elif metric == 'average_precision':
            for x, period in enumerate(PERIOD_ORDER):
                prevalence = table.loc[table.period.eq(period), 'positive_prevalence'].dropna()
                if len(prevalence):
                    axis.hlines(float(prevalence.iloc[0]), x - .35, x + .35,
                                color='#aab0b7', linestyle=':', linewidth=1)
        for x, period in enumerate(PERIOD_ORDER):
            if table.loc[table.period.eq(period) & table.model.isin(PLOT_MODELS), metric].isna().all():
                axis.text(x, .5, 'not available', ha='center', va='center', rotation=90,
                          color='#777777', fontsize=8)
        axis.set_title(title, loc='left', fontsize=10)
        # Equal padding preserves the common rate scale and shows full markers
        # when a measured rate is exactly zero or one.
        axis.set_ylim(-.025, 1.025)
        axis.set_xlim(-.55, len(PERIOD_ORDER) - .45)
        axis.set_yticks([0, .25, .5, .75, 1])
        axis.grid(axis='y', color='#e5e8eb', linewidth=.7)
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_xticks(range(len(PERIOD_ORDER)), [period_labels[p] for p in PERIOD_ORDER], fontsize=8)
    handles = [Line2D([], [], linestyle='none', marker=markers[i], markersize=5,
                      markerfacecolor=colors[i] if i < 2 else 'white', markeredgecolor=colors[i], label=label)
               for i, (model, label) in enumerate(PLOT_MODELS.items()) if table.model.eq(model).any()]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, -.005))
    if unsupported:
        fig.text(.5, -.045, '† No Layer 2 matches; model differences reflect separately fitted coefficients.',
                 ha='center', fontsize=8)
    fig.subplots_adjust(left=.07, right=.99, top=.94, bottom=.22, hspace=.36, wspace=.23)
    destination = Path(destination)
    fig.savefig(destination.with_suffix('.svg'), metadata={'Date': None}, bbox_inches='tight')
    fig.savefig(destination.with_suffix('.pdf'), metadata={'CreationDate': None, 'ModDate': None}, bbox_inches='tight')
    plt.close(fig)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_report(study, output=None, *, expected_goal_sha256=None):
    """Read only precomputed study artifacts and write tables/vector figures."""
    study = Path(study).resolve()
    output = Path(output or study / 'report').resolve()
    inputs, missing = {}, []

    def load(path, csv=False):
        path = Path(path)
        if not path.is_file():
            missing.append(str(path))
            return None
        inputs[str(path)] = _sha(path)
        return pd.read_csv(path) if csv else json.loads(path.read_text())

    frozen_root = study / 'frozen/outputs/research-layers'
    saved = load(frozen_root / 'scores.json')
    if saved is None:
        raise FileNotFoundError('The frozen source score report is required')
    historical = load(study / 'historical-freeze.json')
    known_agents = historical.get('training_actors') if historical else None
    reports, predictions, activities = {}, {}, {}
    for period in PERIOD_ORDER:
        if period == 'development_validation':
            source_report = saved
            prediction = load(frozen_root / 'predictions_dev.csv.gz', csv=True)
            activity = load(study / period / 'activity.csv.gz', csv=True)
        else:
            source_report = load(study / period / 'scores.json')
            prediction = load(study / period / 'predictions.csv.gz', csv=True)
            activity = load(study / period / 'activity.csv.gz', csv=True)
        if activity is not None:
            lo, hi = PERIODS[period]
            if not activity.day.between(lo, hi).all():
                raise ValueError('Activity file contains dates outside its declared period')
            _activity_counts(activity)
            activities[period] = activity
        if prediction is not None:
            evaluated = evaluate_predictions(prediction, saved, period=period, known_agents=known_agents)
            _check_activity_support(activity, prediction)
            if source_report:
                for name in saved['models']:
                    expected_hash = source_report['models'][name].get('frozen_model_sha256')
                    if expected_hash is not None and expected_hash != evaluated['models'][name]['frozen_model_sha256']:
                        raise ValueError('Saved target result references a different frozen model: ' + name)
                    existing = normalize_model_metrics(source_report['models'][name])
                    actual = normalize_model_metrics(evaluated['models'][name])
                    for key in COUNTS + METRICS + ['threshold']:
                        if existing[key] is not None and actual[key] is not None and not math.isclose(
                                existing[key], actual[key], rel_tol=1e-10, abs_tol=1e-10):
                            raise ValueError(f'Saved {period}/{name}/{key} disagrees with saved predictions')
                # Whole-development extraction coverage cannot be substituted
                # for narrower development-validation coverage.
                if period != 'development_validation':
                    for key in ('measurement', 'resources', 'labels', 'paired_comparisons', 'interpretation'):
                        if key in source_report:
                            evaluated[key] = source_report[key]
            reports[period] = evaluated
            predictions[period] = prediction
        else:
            reports[period] = source_report
    model_names = list(saved['models'])
    tables = {'primary-comparison': comparison_table(reports, model_names),
              'coverage': coverage_table(reports, activities),
              'novel-agent-strata': novel_agent_table(reports, model_names)}
    goal_path = study / 'goal-regime-classification.json'
    if expected_goal_sha256 is not None and goal_path.is_file() and _sha(goal_path) != expected_goal_sha256:
        raise ValueError('Locked goal classification hash does not match')
    classification = load(goal_path)
    goal_status = 'missing_predictions_or_classification'
    if classification is not None and 'shared_goals' in predictions:
        goals, categories = goal_tables(predictions['shared_goals'], saved, classification,
                                       activity=activities.get('shared_goals'))
        tables['goal-comparison'], tables['goal-category-comparison'] = goals, categories
        goal_status = 'available'
    secondary_reports, secondary_support = {}, {}
    for period in ('temporal', 'shared_goals'):
        secondary_reports[period] = load(study / period / 'secondary-label-evaluation.json')
        secondary_support[period] = load(study / period / 'secondary-summary.json')
    tables['secondary-comparison'] = secondary_table(secondary_reports)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False)
    for name in ('goal-comparison', 'goal-category-comparison'):
        if name not in tables:
            (output / (name + '.csv')).unlink(missing_ok=True)
    plot_primary(tables['primary-comparison'], output / 'primary-comparison')
    result = dict(schema_version=1, generator_sha256=_sha(Path(__file__)),
                  source_score_sha256=inputs[str(frozen_root / 'scores.json')],
                  inputs_sha256=inputs, missing_inputs=missing, goal_tables_status=goal_status,
                  periods={name: list(bounds) for name, bounds in PERIODS.items()},
                  notes=NOTES, tables={name: _json_safe(table.to_dict('records')) for name, table in tables.items()},
                  secondary_support=secondary_support,
                  paired_comparisons={period: (reports.get(period) or {}).get('paired_comparisons')
                                      for period in PERIOD_ORDER},
                  figure_notes='All four panels use a 0–1 scale. Dotted references mark AUC=0.5, '
                               'period-specific request prevalence for average precision, and the '
                               'training target FPR=0.10. Open markers include actor-name attribution. '
                               'A dagger denotes no Layer 2 trace matches: model differences there '
                               'reflect separately fitted coefficients, not evidence of workflow-rule benefit. '
                               'No confidence intervals are plotted.')
    (output / 'report.json').write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + '\n')
    (output / 'primary-comparison.json').write_text(json.dumps(result['tables']['primary-comparison'],
                                                              indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=config.OUT / 'generalization-20261106')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--goal-sha256', help='Require this previously locked classification digest')
    args = parser.parse_args()
    result = build_report(args.study, args.output, expected_goal_sha256=args.goal_sha256)
    print(json.dumps({'output': str(args.output or args.study / 'report'),
                      'goal_tables_status': result['goal_tables_status'],
                      'missing_input_count': len(result['missing_inputs'])}))


if __name__ == '__main__':
    main()
