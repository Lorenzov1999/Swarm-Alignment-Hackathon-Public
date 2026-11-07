"""Report aggregation preserves frozen operating points and real denominators."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from swarmcollab.transfer_report import (
    build_report, comparison_table, coverage_table, goal_tables,
    normalize_model_metrics, novel_agent_table, secondary_table,
)
from swarmcollab.transfer_scoring import evaluate_predictions


MODELS = ['layer1_behavior', 'layer1_plus_layer2_behavior']


def saved_models():
    model = dict(requested_features=['count'], features=['count'],
                 transformations={'count': 'log1p'}, threshold=.5,
                 fitted=dict(feature_names=['count'], mu=[0.], sd=[1.], w=[0., 1.], intercept_index=0))
    return {'models': {name: copy.deepcopy(model) for name in MODELS}}


def predictions(first='2026-06-01', second='2026-06-02'):
    rows = []
    for day, count in [(first, 2), (second, 6)]:
        for index in range(count):
            score = 1. if index < (1 if day == first else 2) else .1
            rows.append(dict(a='A', b='B' + str(index), day=day, y=day == first and index == 0,
                             **{name: score for name in MODELS},
                             **{name + '__alert': score > .5 for name in MODELS}))
    return pd.DataFrame(rows)


def goals():
    return dict(period=['2026-06-01', '2026-07-05'],
                intervals=[{'goal_id': 'g1', 'category': 'collaborative', 'description': 'Help the team', 'confidence': 'high'},
                           {'goal_id': 'g2', 'category': 'competitive', 'description': 'Compete', 'confidence': 'high'}],
                day_assignments=[
                    {'day': '2026-06-01', 'primary_goal_id': 'g1', 'primary_category': 'collaborative', 'transition_day': False},
                    {'day': '2026-06-02', 'primary_goal_id': 'g1', 'primary_category': 'collaborative', 'transition_day': True},
                    {'day': '2026-06-03', 'primary_goal_id': 'g2', 'primary_category': 'competitive', 'transition_day': False}])


def activity():
    return pd.DataFrame([
        {'agent': 'A', 'day': '2026-06-01', 'working': 5.},
        {'agent': 'B0', 'day': '2026-06-01', 'working': 2.5},
        {'agent': 'B1', 'day': '2026-06-01', 'working': 2.5},
        {'agent': 'A', 'day': '2026-06-02', 'working': 45.},
        *[{'agent': 'B' + str(i), 'day': '2026-06-02', 'working': 7.5} for i in range(6)],
        {'agent': 'A', 'day': '2026-06-03', 'working': 0.},
    ])


class AliasAndDenominatorTests(unittest.TestCase):
    def test_historical_nested_alias_uses_frozen_not_target_selected_recall(self):
        value = normalize_model_metrics({
            'auc': .8, 'descriptive_recall_at_10pct_fpr': .99, 'threshold': -1.2,
            'validation_at_train_threshold': {'rows': 20, 'positives': 5, 'negatives': 15,
                'alerts': 6, 'true_positives': 3, 'false_positives': 3,
                'false_negatives': 2, 'true_negatives': 12, 'recall': .6, 'fpr': .2,
                'precision': .5, 'alert_rate': .3, 'alerts_per_day': 2.}})
        self.assertEqual(value['recall'], .6)
        self.assertEqual(value['alerts_per_observed_day'], 2.)
        self.assertEqual(value['positive_prevalence'], .25)
        self.assertIsNone(value['average_precision'])

    def test_zero_denominators_and_missing_results_are_not_fabricated_zeros(self):
        value = normalize_model_metrics(dict(rows=0, positives=0, negatives=0, alerts=0,
                                            true_positives=0, false_positives=0))
        for name in ('recall', 'fpr', 'precision', 'alert_rate', 'positive_prevalence'):
            self.assertIsNone(value[name])
        table = comparison_table({}, MODELS)
        self.assertTrue(table.status.eq('missing').all())
        self.assertTrue(table.auc.isna().all())
        observed = normalize_model_metrics(dict(rows=10, positives=2, negatives=8, alerts=0,
                                               true_positives=0, false_positives=0, auc=0.))
        self.assertEqual(observed['auc'], 0.)
        self.assertEqual(observed['recall'], 0.)
        self.assertEqual(observed['fpr'], 0.)
        self.assertEqual(observed['alert_rate'], 0.)
        self.assertIsNone(observed['precision'])

    def test_invalid_rates_and_counts_reject(self):
        for value in ({'auc': 1.1}, {'rows': -1}, {'rows': .5}, {'rows': 10, 'positives': 4, 'negatives': 5}):
            with self.assertRaises(ValueError):
                normalize_model_metrics(value)

    def test_zero_layer2_coverage_is_explicit_and_distinct_from_missing(self):
        table = comparison_table({'shared_goals': {'models': {MODELS[0]: {'rows': 10}},
                                  'measurement': {'covered_steps_by_layer': {'2': 0}}}}, MODELS)
        self.assertTrue(table[table.period.eq('shared_goals')].layer2_no_trace_matches.eq(True).all())
        self.assertTrue(table[table.period.eq('development_validation')].layer2_no_trace_matches.isna().all())

    def test_coverage_uses_matching_cached_and_history_excluded_denominators(self):
        reports = {'temporal': {'measurement': {
            'cached_steps': 100, 'excluded_history_steps': 20, 'working_steps': 50,
            'covered_steps_by_layer': {'1': 40, '2': 8},
            'provenance': {'target_chains': 2, 'chain_levels': {'3': 2}}}}}
        row = coverage_table(reports).set_index('period').loc['temporal']
        self.assertEqual(row.layer1_coverage_per_cached_step, .4)
        self.assertEqual(row.layer1_coverage_after_history_exclusion, .5)
        self.assertEqual(row.level3_chains_per_1000_working_steps, 40.)
        self.assertEqual(row.level4_chains, 0.)
        missing = coverage_table({}).set_index('period').loc['temporal']
        self.assertTrue(pd.isna(missing.level4_chains))
        reports['temporal']['measurement']['working_steps'] = 0
        row = coverage_table(reports).set_index('period').loc['temporal']
        self.assertTrue(pd.isna(row.level3_chains_per_1000_working_steps))

    def test_empty_novel_stratum_is_distinct_from_missing_stratum(self):
        reports = {'temporal': {'agent_strata': {'both_known': {
            'rows': 0, 'models': {name: {'rows': 0, 'alerts': 0} for name in MODELS}}}}}
        table = novel_agent_table(reports, MODELS)
        selected = table[table.period.eq('temporal')]
        self.assertTrue(selected[selected.stratum.eq('both_known')].status.eq('empty_stratum').all())
        self.assertTrue(selected[selected.stratum.eq('both_unseen')].status.eq('missing').all())

    def test_secondary_methods_keep_units_and_missing_values_distinct(self):
        result = {'temporal': {'marked_hawkes_all_pairs': {'rows': 8, 'positives': 1, 'auc': .5},
                              'legacy_whole_development_network': {
                                  'alpha_any_request_pair': {'rows': 4, 'positives': 1, 'auc': .7}}}}
        table = secondary_table(result)
        row = table[(table.period == 'temporal') & (table.method == 'legacy_alpha_any_request_pair')].iloc[0]
        self.assertEqual(row.unit, 'directed_pair')
        self.assertEqual(row.fit_scope, 'whole_development')
        self.assertTrue(pd.isna(row.recall))
        self.assertTrue(table[table.period == 'shared_goals'].status.eq('missing').all())


class GoalAggregationTests(unittest.TestCase):
    def test_goal_rates_aggregate_counts_and_transition_sensitivity_exactly(self):
        by_goal, by_category = goal_tables(predictions(), saved_models(), goals(), activity=activity())
        group = by_goal[(by_goal.group_id == 'g1') & (by_goal.model == MODELS[0])]
        full = group[group.sensitivity == 'all_days'].iloc[0]
        strict = group[group.sensitivity == 'exclude_transitions'].iloc[0]
        self.assertEqual(full.rows, 8)
        self.assertEqual(full.alerts, 3)
        self.assertEqual(full.alert_rate, 3 / 8)
        self.assertEqual(full.working_steps, 100.)
        self.assertEqual(full.alerts_per_1000_working_steps, 30.)
        self.assertEqual(strict.rows, 2)
        self.assertEqual(strict.alerts, 1)
        self.assertEqual(strict.working_steps, 10.)
        self.assertEqual(strict.alerts_per_1000_working_steps, 100.)
        self.assertEqual(strict.transition_days_included, 0)
        cat = by_category[(by_category.group_id == 'collaborative') &
                          (by_category.model == MODELS[0]) & (by_category.sensitivity == 'all_days')].iloc[0]
        self.assertEqual(cat.alert_rate, full.alert_rate)

    def test_goal_without_observations_and_zero_working_denominator_stay_explicit(self):
        by_goal, _ = goal_tables(predictions(), saved_models(), goals(), activity=activity())
        row = by_goal[(by_goal.group_id == 'g2') & (by_goal.model == MODELS[0]) &
                      (by_goal.sensitivity == 'all_days')].iloc[0]
        self.assertEqual(row.rows, 0)
        self.assertEqual(row.assigned_calendar_days, 1)
        self.assertEqual(row.working_steps, 0.)
        self.assertEqual(row.status, 'no_observed_pair_days')
        self.assertTrue(pd.isna(row.alert_rate))
        self.assertTrue(pd.isna(row.alerts_per_1000_working_steps))

    def test_missing_activity_does_not_turn_alert_normalization_into_zero(self):
        by_goal, _ = goal_tables(predictions(), saved_models(), goals())
        self.assertTrue(by_goal.working_steps.isna().all())
        self.assertTrue(by_goal.alerts_per_1000_working_steps.isna().all())

    def test_incomplete_activity_denominator_is_rejected(self):
        incomplete = activity().query('agent != "B5"')
        with self.assertRaisesRegex(ValueError, 'every scored actor/day'):
            goal_tables(predictions(), saved_models(), goals(), activity=incomplete)

    def test_numeric_activity_strings_use_the_same_positive_support(self):
        counts = activity()
        counts['working'] = counts.working.astype(str)
        table, _ = goal_tables(predictions(), saved_models(), goals(), activity=counts)
        row = table[(table.group_id == 'g1') & (table.model == MODELS[0]) &
                    (table.sensitivity == 'all_days')].iloc[0]
        self.assertEqual(row.working_steps, 100.)

    def test_incomplete_or_conflicting_locked_assignments_rejected(self):
        modifications = [
            lambda value: value['day_assignments'].pop(1),
            lambda value: value['day_assignments'].append(copy.deepcopy(value['day_assignments'][0])),
            lambda value: value['day_assignments'][0].update(primary_category='competitive'),
            lambda value: value.update(period=['2026-09-16', '2026-10-31']),
        ]
        for modify in modifications:
            classification = goals()
            modify(classification)
            with self.assertRaises(ValueError):
                goal_tables(predictions(), saved_models(), classification, activity=activity())

    def test_goal_aggregation_cannot_replace_frozen_alerts(self):
        table = predictions()
        table.loc[0, MODELS[0] + '__alert'] = False
        with self.assertRaises(ValueError):
            goal_tables(table, saved_models(), goals())


class ReportArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='transfer-report-test-')
        self.study = Path(self.temporary.name) / 'study'
        frozen = self.study / 'frozen/outputs/research-layers'
        frozen.mkdir(parents=True)
        self.frozen = saved_models()
        development = predictions('2026-08-03', '2026-08-04')
        evaluated = evaluate_predictions(development, self.frozen, period='development_validation')
        for name in MODELS:
            self.frozen['models'][name].update(
                auc=evaluated['models'][name]['auc'],
                validation_at_train_threshold={key: value for key, value in evaluated['models'][name].items()
                                               if key not in ('frozen_model_sha256', 'threshold', 'auc')})
        (frozen / 'scores.json').write_text(json.dumps(self.frozen))
        development.to_csv(frozen / 'predictions_dev.csv.gz', index=False)
        (self.study / 'historical-freeze.json').write_text(json.dumps({'training_actors': ['A', 'B0']}))
        (self.study / 'goal-regime-classification.json').write_text(json.dumps(goals()))

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_target_results_are_reported_and_vector_files_build(self):
        result = build_report(self.study)
        output = self.study / 'report'
        self.assertEqual(result['goal_tables_status'], 'missing_predictions_or_classification')
        for name in ('primary-comparison.csv', 'primary-comparison.json', 'coverage.csv',
                     'novel-agent-strata.csv', 'secondary-comparison.csv', 'report.json',
                     'primary-comparison.svg', 'primary-comparison.pdf'):
            self.assertTrue((output / name).is_file(), name)
        table = pd.read_csv(output / 'primary-comparison.csv')
        self.assertTrue(table[table.period == 'temporal'].status.eq('missing').all())
        self.assertTrue(table[table.period == 'temporal'].auc.isna().all())
        self.assertTrue((output / 'primary-comparison.svg').read_text().startswith('<?xml'))
        json.dumps(result, allow_nan=False)

    def test_goal_tables_use_only_saved_predictions_and_activity(self):
        directory = self.study / 'shared_goals'
        directory.mkdir()
        table = predictions()
        table.to_csv(directory / 'predictions.csv.gz', index=False)
        activity().to_csv(directory / 'activity.csv.gz', index=False)
        report = evaluate_predictions(table, self.frozen, period='shared_goals')
        (directory / 'scores.json').write_text(json.dumps(report))
        result = build_report(self.study)
        self.assertEqual(result['goal_tables_status'], 'available')
        self.assertTrue((self.study / 'report/goal-category-comparison.csv').is_file())
        self.assertFalse(any('computer_use' in path or '/events/' in path for path in result['inputs_sha256']))

        (directory / 'predictions.csv.gz').unlink()
        result = build_report(self.study)
        self.assertEqual(result['goal_tables_status'], 'missing_predictions_or_classification')
        self.assertFalse((self.study / 'report/goal-category-comparison.csv').exists())
        self.assertFalse((self.study / 'report/goal-comparison.csv').exists())

    def test_mismatched_saved_metrics_or_goal_digest_fail(self):
        directory = self.study / 'temporal'
        directory.mkdir()
        table = predictions('2026-08-17', '2026-08-18')
        table.to_csv(directory / 'predictions.csv.gz', index=False)
        report = evaluate_predictions(table, self.frozen, period='temporal')
        report['models'][MODELS[0]]['auc'] = 0.
        (directory / 'scores.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            build_report(self.study)
        (directory / 'scores.json').unlink()
        with self.assertRaisesRegex(ValueError, 'classification hash'):
            build_report(self.study, expected_goal_sha256='not-the-locked-hash')


if __name__ == '__main__':
    unittest.main()
