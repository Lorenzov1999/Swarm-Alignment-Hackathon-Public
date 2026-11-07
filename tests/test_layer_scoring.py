"""Tests for frozen layer scoring and defensible operating-point comparisons."""
import json
import unittest

import numpy as np
import pandas as pd

from swarmcollab.layer_scoring import evaluate_layers, paired_day_comparison, training_threshold


def table():
    rows = []
    for day, offset in [('2026-07-06', 0), ('2026-07-07', 0),
                        ('2026-08-03', 1), ('2026-08-04', 2)]:
        for index in range(6):
            rows.append(dict(a='A', b='B' + str(index), day=day, y=index >= 3,
                             act_min=float(index), act_max=float(index + 2),
                             general=float(index + offset), village=float(index % 2),
                             constant=2.0))
    return pd.DataFrame(rows)


MODELS = {'general': ['act_min', 'general', 'constant'],
          'village': ['act_min', 'general', 'village', 'constant']}


class CalibrationTests(unittest.TestCase):
    def test_order_statistic_obeys_small_sample_budget(self):
        score = np.arange(5.)
        threshold = training_threshold(score, np.zeros(5, dtype=bool))
        self.assertEqual(threshold, 4)
        self.assertEqual(int((score > threshold).sum()), 0)
        score = np.arange(10.)
        threshold = training_threshold(score, np.zeros(10, dtype=bool))
        self.assertEqual(threshold, 8)
        self.assertEqual(int((score > threshold).sum()), 1)

    def test_tied_threshold_does_not_alert(self):
        threshold = training_threshold([1., 1., 1., 2.], [False, False, False, True])
        self.assertEqual(threshold, 1.)
        self.assertEqual((np.array([1., 2.]) > threshold).tolist(), [False, True])

    def test_threshold_requires_negatives_and_finite_aligned_scores(self):
        for score, labels in [([1], [True]), ([float('nan')], [False]), ([1, 2], [False])]:
            with self.assertRaises(ValueError):
                training_threshold(score, labels)

    def test_validation_cannot_change_fitted_transforms_or_threshold(self):
        before, _ = evaluate_layers(table(), MODELS)
        changed = table()
        validation = changed.day >= '2026-08-03'
        changed.loc[validation, 'general'] = 10_000
        changed.loc[validation, 'constant'] = 50_000
        changed.loc[validation, 'y'] = False
        after, predictions = evaluate_layers(changed, MODELS)
        for name in MODELS:
            self.assertEqual(before['models'][name]['fitted'], after['models'][name]['fitted'])
            self.assertEqual(before['models'][name]['threshold'], after['models'][name]['threshold'])
            self.assertEqual(after['models'][name]['dropped_constant_features'], ['constant'])
        self.assertTrue(predictions.day.ge('2026-08-03').all())

    def test_activity_is_not_logged_twice_and_parameters_replay_predictions(self):
        frame = table()
        report, predictions = evaluate_layers(frame, MODELS)
        model = report['models']['general']
        train = frame.day <= '2026-07-31'
        self.assertEqual(model['fitted']['mu'][0], frame.loc[train, 'act_min'].mean())
        self.assertAlmostEqual(model['fitted']['mu'][1], np.log1p(frame.loc[train, 'general']).mean())
        valid = frame[frame.day >= '2026-08-03']
        values = np.column_stack([valid.act_min, np.log1p(valid.general)])
        params = model['fitted']
        score = params['w'][0] + ((values - params['mu']) / params['sd']) @ np.array(params['w'][1:])
        np.testing.assert_allclose(score, predictions['general'])
        np.testing.assert_array_equal(score > model['threshold'], predictions['general__alert'])

    def test_all_constant_features_produce_intercept_and_no_tied_alerts(self):
        report, predictions = evaluate_layers(table(), {'constant_model': ['constant']})
        model = report['models']['constant_model']
        self.assertEqual(model['features'], [])
        self.assertEqual(model['fitted']['mu'], [])
        self.assertEqual(len(model['fitted']['w']), 1)
        self.assertFalse(predictions['constant_model__alert'].any())
        self.assertEqual(model['auc'], .5)

    def test_single_validation_class_is_json_safe(self):
        frame = table()
        frame.loc[frame.day >= '2026-08-03', 'y'] = False
        report, _ = evaluate_layers(frame, MODELS)
        for model in report['models'].values():
            self.assertIsNone(model['auc'])
            self.assertIsNone(model['descriptive_recall_at_10pct_fpr'])
            self.assertIsNone(model['validation_at_train_threshold']['recall'])
        json.dumps(report, allow_nan=False)

    def test_validation_metrics_use_fixed_training_threshold(self):
        report, predictions = evaluate_layers(table(), MODELS)
        for name in MODELS:
            model = report['models'][name]
            alerts = predictions[name] > model['threshold']
            expected_fpr = float(alerts[~predictions.y].mean())
            expected_recall = float(alerts[predictions.y].mean())
            self.assertEqual(model['validation_at_train_threshold']['fpr'], expected_fpr)
            self.assertEqual(model['validation_at_train_threshold']['recall'], expected_recall)
            self.assertLessEqual(model['training_at_train_threshold']['fpr'], .1)


class InputTests(unittest.TestCase):
    def test_training_requires_both_classes_and_both_periods(self):
        frame = table()
        frame.loc[frame.day <= '2026-07-31', 'y'] = False
        with self.assertRaises(ValueError):
            evaluate_layers(frame, MODELS)
        with self.assertRaises(ValueError):
            evaluate_layers(table().query('day < "2026-08-03"'), MODELS)

    def test_out_of_scope_or_noncanonical_dates_rejected(self):
        for day in ('2026-08-17', '2026-07-05', '2026-07-32', '20260706'):
            frame = table()
            frame.loc[0, 'day'] = day
            with self.assertRaises(ValueError):
                evaluate_layers(frame, MODELS)

    def test_missing_duplicate_nonfinite_or_negative_features_rejected(self):
        for features in (['missing'], ['general', 'general'], ['y']):
            with self.assertRaises(ValueError):
                evaluate_layers(table(), {'model': features})
        for value in (float('nan'), float('inf'), -1):
            frame = table()
            frame.loc[0, 'general'] = value
            with self.assertRaises(ValueError):
                evaluate_layers(frame, MODELS)
        frame = pd.concat([table(), table()[['general']]], axis=1)
        with self.assertRaises(ValueError):
            evaluate_layers(frame, MODELS)

    def test_duplicate_undirected_pairs_and_invalid_labels_rejected(self):
        frame = table()
        duplicate = frame.iloc[[0]].copy()
        duplicate['a'], duplicate['b'] = duplicate.b, duplicate.a
        with self.assertRaises(ValueError):
            evaluate_layers(pd.concat([frame, duplicate], ignore_index=True), MODELS)
        frame['y'] = frame.y.astype(str)
        with self.assertRaises(ValueError):
            evaluate_layers(frame, MODELS)

    def test_prediction_name_collisions_rejected(self):
        for feature_sets in ({'y': ['general']}, {'x': [], 'x__alert': []}):
            with self.assertRaises(ValueError):
                evaluate_layers(table(), feature_sets)

    def test_gap_days_are_not_used_for_fitting_or_validation(self):
        frame = table()
        gap = frame.iloc[[0]].copy()
        gap['day'] = '2026-08-01'
        gap['general'] = 1_000_000
        original, _ = evaluate_layers(frame, MODELS)
        report, predictions = evaluate_layers(pd.concat([frame, gap], ignore_index=True), MODELS)
        self.assertEqual(report['rows_between_splits'], 1)
        self.assertNotIn('2026-08-01', predictions.day.tolist())
        for name in MODELS:
            self.assertEqual(report['models'][name]['fitted'], original['models'][name]['fitted'])


class PairedComparisonTests(unittest.TestCase):
    def test_same_models_give_zero_deltas_and_deterministic_intervals(self):
        _, predictions = evaluate_layers(table(), MODELS)
        first = paired_day_comparison(predictions, 'general', 'general', repetitions=30)
        second = paired_day_comparison(predictions, 'general', 'general', repetitions=30)
        self.assertEqual(first, second)
        for key in ('delta_auc', 'delta_recall_at_fixed_threshold'):
            self.assertEqual(first[key]['estimate'], 0)
            self.assertEqual(first[key]['interval_95'], [0, 0])
            self.assertEqual(first[key]['valid_draws'], 30)

    def test_single_class_bootstrap_draws_are_counted_and_json_safe(self):
        _, predictions = evaluate_layers(table(), MODELS)
        predictions['y'] = predictions.day.eq('2026-08-03')
        report = paired_day_comparison(predictions, 'general', 'village', repetitions=100, seed=2)
        self.assertGreater(report['delta_auc']['skipped_single_class'], 0)
        self.assertEqual(report['delta_auc']['skipped_single_class'] + report['delta_auc']['valid_draws'], 100)
        self.assertGreater(report['delta_recall_at_fixed_threshold']['skipped_single_class'], 0)
        json.dumps(report, allow_nan=False)
        report = paired_day_comparison(predictions, 'general', 'village', repetitions=0)
        self.assertIsNone(report['delta_auc']['interval_95'])

    def test_bootstrap_does_not_refit_alerts_or_use_training_rows(self):
        _, predictions = evaluate_layers(table(), MODELS)
        predictions['general__alert'] = False
        predictions['village__alert'] = True
        report = paired_day_comparison(predictions, 'general', 'village', repetitions=20)
        self.assertEqual(report['delta_recall_at_fixed_threshold']['estimate'], 1)
        self.assertEqual(report['delta_recall_at_fixed_threshold']['interval_95'], [1, 1])
        predictions.loc[0, 'day'] = '2026-07-06'
        with self.assertRaises(ValueError):
            paired_day_comparison(predictions, 'general', 'village')


if __name__ == '__main__':
    unittest.main()
