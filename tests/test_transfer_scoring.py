"""Frozen transfer must reproduce development scores without target fitting."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from swarmcollab.layer_scoring import evaluate_layers
from swarmcollab.transfer_scoring import (
    average_precision, evaluate_predictions, paired_block_comparison, score_frozen,
)


def development_table():
    rows = []
    for day, offset in [('2026-07-06', 0), ('2026-07-07', 0),
                        ('2026-08-03', 1), ('2026-08-04', 2)]:
        for index in range(6):
            rows.append(dict(a='A', b='B' + str(index), day=day, y=index >= 3,
                             act_min=float(index), act_max=float(index + 2),
                             general=float(index + offset), village=float(index % 2),
                             constant=2.0))
    return pd.DataFrame(rows)


FEATURES = {'general': ['act_min', 'general', 'constant'],
            'village': ['act_min', 'general', 'village', 'constant']}


class FrozenModelTests(unittest.TestCase):
    def setUp(self):
        self.frame = development_table()
        self.saved, self.expected = evaluate_layers(self.frame, FEATURES)
        self.target = self.frame[self.frame.day >= '2026-08-03'].copy()
        self.target['day'] = self.target.day.replace({'2026-08-03': '2026-08-17',
                                                    '2026-08-04': '2026-08-18'})

    def test_exact_replay_never_fits_or_modifies_saved_parameters(self):
        saved_before = copy.deepcopy(self.saved)
        frame_before = self.target.copy(deep=True)
        with patch('swarmcollab.metrics.Logistic.fit', side_effect=AssertionError('Must not fit')):
            result = score_frozen(self.target, self.saved, period='temporal')
        for model in FEATURES:
            np.testing.assert_array_equal(result[model], self.expected[model])
            np.testing.assert_array_equal(result[model + '__alert'], self.expected[model + '__alert'])
        self.assertEqual(saved_before, self.saved)
        pd.testing.assert_frame_equal(frame_before, self.target)

    def test_unlabelled_prediction_and_label_changes_do_not_change_scores(self):
        original = score_frozen(self.target, self.saved, period='temporal')
        unlabelled = score_frozen(self.target.drop(columns='y'), self.saved, period='temporal')
        self.assertNotIn('y', unlabelled)
        changed = self.target.copy()
        changed['y'] = ~changed.y
        altered = score_frozen(changed, self.saved, period='temporal')
        for model in FEATURES:
            np.testing.assert_array_equal(original[model], unlabelled[model])
            np.testing.assert_array_equal(original[model], altered[model])
        report = evaluate_predictions(unlabelled, self.saved, period='temporal')
        self.assertFalse(report['labelled'])
        self.assertNotIn('auc', report['models']['general'])
        self.assertIn('alert_rate', report['models']['general'])

    def test_no_target_standardization_and_training_constant_stays_dropped(self):
        baseline = score_frozen(self.target, self.saved, period='temporal')
        changed = self.target.copy()
        changed.loc[changed.index[0], 'general'] = 10_000
        changed['constant'] = 500
        result = score_frozen(changed, self.saved, period='temporal')
        for model in FEATURES:
            self.assertNotEqual(baseline[model].iloc[0], result[model].iloc[0])
            np.testing.assert_array_equal(baseline[model].iloc[1:], result[model].iloc[1:])

    def test_intercept_only_model_and_strict_threshold_ties(self):
        report, _ = evaluate_layers(self.frame, {'constant_model': ['constant']})
        prediction = score_frozen(self.target, report, period='temporal')
        self.assertFalse(prediction.constant_model__alert.any())
        evaluated = evaluate_predictions(prediction, report, period='temporal')
        model = evaluated['models']['constant_model']
        self.assertEqual(model['auc'], .5)
        self.assertEqual(model['average_precision'], .5)

    def test_frozen_threshold_metrics_and_unseen_agent_strata(self):
        target = self.target.copy()
        target.loc[target.b.eq('B5'), 'a'] = 'newA'
        known = ['A', 'B0', 'B1', 'B2']
        prediction = score_frozen(target, self.saved, period='temporal', known_agents=known)
        report = evaluate_predictions(prediction, self.saved, period='temporal', known_agents=known)
        self.assertEqual(report['agent_strata']['both_known']['rows'], 6)
        self.assertEqual(report['agent_strata']['one_unseen']['rows'], 4)
        self.assertEqual(report['agent_strata']['both_unseen']['rows'], 2)
        self.assertEqual(prediction.novel_agent_count.tolist(), [0, 0, 0, 1, 1, 2] * 2)
        for model in FEATURES:
            alerts, labels = prediction[model + '__alert'], prediction.y
            values = report['models'][model]
            self.assertEqual(values['recall'], float(alerts[labels].mean()))
            self.assertEqual(values['fpr'], float(alerts[~labels].mean()))
            self.assertEqual(values['threshold'], self.saved['models'][model]['threshold'])
        json.dumps(report, allow_nan=False)

    def test_single_class_and_empty_strata_remain_json_safe(self):
        target = self.target.copy()
        target.y = False
        prediction = score_frozen(target, self.saved, period='temporal')
        report = evaluate_predictions(prediction, self.saved, period='temporal', known_agents=[])
        self.assertIsNone(report['models']['general']['auc'])
        self.assertIsNone(report['models']['general']['average_precision'])
        self.assertIsNone(report['models']['general']['recall'])
        self.assertEqual(report['agent_strata']['both_known']['rows'], 0)
        self.assertIsNone(report['agent_strata']['both_known']['models']['general']['alert_rate'])
        json.dumps(report, allow_nan=False)

    def test_shared_goal_period_is_explicit_and_reserve_stays_rejected(self):
        target = self.target.copy()
        target.day = target.day.replace({'2026-08-17': '2026-06-01', '2026-08-18': '2026-07-05'})
        score_frozen(target, self.saved, period='shared_goals')
        for period, day in [('shared_goals', '2026-07-06'), ('temporal', '2026-08-16'),
                            ('temporal', '2026-09-16'), ('temporal', '2026-10-01'),
                            ('development_validation', '2026-08-17'),
                            ('temporal', '20260817'), ('unknown', '2026-08-17')]:
            bad = self.target.copy()
            bad.loc[bad.index[0], 'day'] = day
            with self.subTest(period=period, day=day), self.assertRaises(ValueError):
                score_frozen(bad, self.saved, period=period)

    def test_missing_constant_nonfinite_negative_or_nonnumeric_features_fail(self):
        with self.assertRaises(ValueError):
            score_frozen(self.target.drop(columns='constant'), self.saved, period='temporal')
        for column in ('general', 'constant'):
            for value in (float('nan'), float('inf'), -1, 'invalid'):
                bad = self.target.copy()
                bad[column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    score_frozen(bad, self.saved, period='temporal')

    def test_invalid_model_parameters_or_transforms_fail(self):
        modifications = [
            lambda model: model['fitted'].update(sd=[0, 1]),
            lambda model: model['fitted'].update(mu=[float('nan'), 0]),
            lambda model: model['fitted'].update(w=[0]),
            lambda model: model['fitted'].update(feature_names=['general', 'act_min']),
            lambda model: model['transformations'].update(general='target_rank'),
            lambda model: model.update(threshold=float('nan')),
            lambda model: model.update(features=['general', 'act_min']),
        ]
        for modify in modifications:
            saved = copy.deepcopy(self.saved)
            modify(saved['models']['general'])
            with self.assertRaises(ValueError):
                score_frozen(self.target, saved, period='temporal')

    def test_invalid_pair_keys_duplicate_columns_labels_and_alerts_fail(self):
        reversed_pair = self.target.iloc[[0]].copy()
        reversed_pair['a'], reversed_pair['b'] = reversed_pair.b, reversed_pair.a
        duplicate = pd.concat([self.target, reversed_pair], ignore_index=True)
        with self.assertRaises(ValueError):
            score_frozen(duplicate, self.saved, period='temporal')
        with self.assertRaises(ValueError):
            score_frozen(pd.concat([self.target, self.target[['general']]], axis=1),
                         self.saved, period='temporal')
        prediction = score_frozen(self.target, self.saved, period='temporal')
        prediction.loc[0, 'general__alert'] = not prediction.loc[0, 'general__alert']
        with self.assertRaises(ValueError):
            evaluate_predictions(prediction, self.saved, period='temporal')
        prediction = score_frozen(self.target, self.saved, period='temporal')
        prediction.y = prediction.y.astype(str)
        with self.assertRaises(ValueError):
            evaluate_predictions(prediction, self.saved, period='temporal')

    def test_model_selection_and_metadata_collisions_are_validated(self):
        prediction = score_frozen(self.target, self.saved, period='temporal', model_names=['general'])
        self.assertNotIn('village', prediction)
        for names in ([], ['unknown'], ['general', 'general']):
            with self.assertRaises(ValueError):
                score_frozen(self.target, self.saved, period='temporal', model_names=names)
        saved = copy.deepcopy(self.saved)
        saved['models']['y'] = saved['models'].pop('general')
        with self.assertRaises(ValueError):
            score_frozen(self.target, saved, period='temporal')


class PrecisionRecallTests(unittest.TestCase):
    def test_tied_scores_are_grouped_and_permutation_invariant(self):
        self.assertEqual(average_precision([1, 1, 1, 1], [1, 0, 0, 0]), .25)
        self.assertAlmostEqual(average_precision([3, 2, 2, 1], [1, 1, 0, 0]), 5 / 6)
        self.assertAlmostEqual(average_precision([2, 3, 1, 2], [0, 1, 0, 1]), 5 / 6)

    def test_absent_positive_and_all_positive_cases(self):
        self.assertIsNone(average_precision([1, 2], [0, 0]))
        self.assertEqual(average_precision([1, 2], [1, 1]), 1.)
        self.assertIsNone(average_precision([], []))

    def test_extreme_finite_scores_do_not_overflow_tie_detection(self):
        with np.errstate(over='raise'):
            self.assertEqual(average_precision([1e308, -1e308], [1, 0]), 1.)


class BlockBootstrapTests(unittest.TestCase):
    def setUp(self):
        saved, prediction = evaluate_layers(development_table(), FEATURES)
        prediction.day = prediction.day.replace({'2026-08-03': '2026-08-17', '2026-08-04': '2026-08-18'})
        self.prediction = prediction

    def test_identity_comparison_gives_deterministic_zero_differences(self):
        options = dict(period='temporal', repetitions=25, seed=31)
        first = paired_block_comparison(self.prediction, 'general', 'general', **options)
        second = paired_block_comparison(self.prediction, 'general', 'general', **options)
        self.assertEqual(first, second)
        for key in ('delta_auc', 'delta_average_precision', 'delta_recall_at_fixed_threshold',
                    'delta_fpr_at_fixed_threshold', 'delta_precision_at_fixed_threshold', 'delta_alert_rate'):
            self.assertEqual(first[key]['estimate'], 0.)
            self.assertEqual(first[key]['interval_95'], [0., 0.])

    def test_days_resampled_as_whole_blocks_and_single_class_draws_counted(self):
        prediction = self.prediction.copy()
        prediction.y = prediction.day.eq('2026-08-17')
        prediction.general__alert = False
        prediction.village__alert = True
        report = paired_block_comparison(prediction, 'general', 'village',
                                         period='temporal', repetitions=100, seed=2)
        self.assertEqual(report['blocks'], 2)
        self.assertGreater(report['delta_auc']['skipped_undefined'], 0)
        self.assertGreater(report['delta_recall_at_fixed_threshold']['skipped_undefined'], 0)
        self.assertEqual(report['delta_recall_at_fixed_threshold']['interval_95'], [1., 1.])
        self.assertEqual(report['delta_alert_rate']['interval_95'], [1., 1.])
        self.assertEqual(report['delta_auc']['valid_draws'] + report['delta_auc']['skipped_undefined'], 100)
        json.dumps(report, allow_nan=False)

    def test_one_episode_does_not_claim_uncertainty_from_independent_calls(self):
        prediction = self.prediction.copy()
        prediction['episode'] = 'one_episode'
        report = paired_block_comparison(prediction, 'general', 'village', period='temporal',
                                         block_column='episode', repetitions=20)
        self.assertTrue(report['insufficient_blocks_for_uncertainty'])
        self.assertIsNone(report['delta_auc']['interval_95'])
        self.assertEqual(report['delta_auc']['skipped_insufficient_blocks'], 20)
        self.assertIsNotNone(report['delta_auc']['estimate'])

    def test_invalid_bootstrap_inputs_fail(self):
        for options in ({'repetitions': -1}, {'repetitions': True}, {'block_column': 'missing'}):
            with self.assertRaises(ValueError):
                paired_block_comparison(self.prediction, 'general', 'village', period='temporal', **options)
        with self.assertRaises(ValueError):
            paired_block_comparison(self.prediction.drop(columns='y'), 'general', 'village', period='temporal')


class SavedDevelopmentReplayTests(unittest.TestCase):
    def test_all_existing_saved_models_reproduce_development_predictions(self):
        root = Path(__file__).resolve().parents[1] / 'outputs/research-layers'
        names = ['scores.json', 'features_dev.csv.gz', 'predictions_dev.csv.gz']
        if not all((root / name).is_file() for name in names):
            self.skipTest('Local saved development outputs are not available')
        saved = json.loads((root / 'scores.json').read_text())
        features = pd.read_csv(root / 'features_dev.csv.gz')
        features = features[features.day.between('2026-08-03', '2026-08-14')].copy()
        expected = pd.read_csv(root / 'predictions_dev.csv.gz')
        with patch('swarmcollab.metrics.Logistic.fit', side_effect=AssertionError('Must not fit')):
            actual = score_frozen(features, saved, period='development_validation')
        actual, expected = [frame.sort_values(['a', 'b', 'day']).reset_index(drop=True)
                            for frame in (actual, expected)]
        pd.testing.assert_frame_equal(actual[['a', 'b', 'day', 'y']], expected[['a', 'b', 'day', 'y']])
        for name in saved['models']:
            np.testing.assert_allclose(actual[name], expected[name], atol=2e-12, rtol=2e-12)
            np.testing.assert_array_equal(actual[name + '__alert'], expected[name + '__alert'])


if __name__ == '__main__':
    unittest.main()
