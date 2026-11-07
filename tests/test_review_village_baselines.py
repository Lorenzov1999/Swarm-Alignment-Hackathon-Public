"""Verify scope, history isolation and the conditional AUROC estimand."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location('review_baselines',
    Path(__file__).resolve().parents[1] / 'scripts/review_village_baselines.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def row(a, b, day, label):
    return dict(a=a, b=b, day=day, y=label)


class BaselineReviewTests(unittest.TestCase):
    def test_scope_rejects_reserve_and_noncanonical_or_duplicate_dates(self):
        for rows in [[row('a', 'b', '2026-09-16', True)],
                     [row('a', 'b', '2026-08-17', True), row('b', 'a', '2026-08-17', False)]]:
            with self.assertRaises(ValueError):
                review.validate_rows(rows, '2026-08-17', '2026-09-15')
        with self.assertRaises(ValueError):
            review.fit_pair_history([row('a', 'b', '2026-08-03', True)])

    def test_history_never_reads_target_labels_and_unseen_pair_uses_global(self):
        training = [row('a', 'b', '2026-07-06', True), row('a', 'b', '2026-07-07', False),
                    row('a', 'c', '2026-07-06', False)]
        fitted = review.fit_pair_history(training)
        target = [dict(a='b', b='a'), dict(a='c', b='d')]
        first = review.history_scores(target, fitted)
        for item in target:
            item['y'] = 'unreadable and deliberately invalid'
        np.testing.assert_array_equal(first, review.history_scores(target, fitted))
        np.testing.assert_allclose(first, [(1 + 1 / 3) / 3, 1 / 3])
        self.assertEqual(review.history_scores(target, fitted, 0)[1], 1 / 3)
        with self.assertRaises(ValueError):
            review.history_scores(target, fitted, -1)

    def test_conditional_auc_only_compares_same_pair_and_counts_ties(self):
        rows = [row('a', 'b', '2026-08-17', True), row('a', 'b', '2026-08-18', False),
                row('c', 'd', '2026-08-17', True), row('c', 'd', '2026-08-18', False),
                row('c', 'd', '2026-08-19', False)]
        result, _, numerator, denominator = review.within_pair(rows, [1, 0, 10, 10, 12])
        self.assertEqual(result['positive_negative_comparisons'], 3)
        self.assertEqual(result['comparison_weighted_auc'], .5)
        self.assertEqual(result['macro_pair_auc'], .625)
        self.assertEqual(numerator.sum(), 1.5)
        self.assertEqual(denominator.sum(), 3)

    def test_pairwise_constant_score_shifts_cannot_change_conditional_auc(self):
        rows = [row('a', 'b', '2026-08-17', True), row('a', 'b', '2026-08-18', False),
                row('c', 'd', '2026-08-17', True), row('c', 'd', '2026-08-18', False)]
        left = review.within_pair(rows, [3, 1, 1, 4])[0]
        right = review.within_pair(rows, [103, 101, -999, -996])[0]
        self.assertEqual(left, right)
        constant = review.within_pair(rows, [99, 99, -40, -40])[0]
        self.assertEqual(constant['comparison_weighted_auc'], .5)

    def test_centered_pooled_auc_differs_from_true_conditional_comparisons(self):
        rows = [row('a', 'b', '2026-08-17', True), row('a', 'b', '2026-08-18', False),
                row('c', 'd', '2026-08-17', True), row('c', 'd', '2026-08-18', False),
                row('c', 'd', '2026-08-19', False)]
        scores = [1, 0, 10, 10, 12]
        conditional = review.within_pair(rows, scores)[0]['comparison_weighted_auc']
        self.assertNotEqual(conditional, review.centered_pooled_auc(rows, scores))

    def test_bootstrap_matrix_matches_literal_repeated_days(self):
        rows = [row('a', 'b', '2026-08-17', True), row('a', 'b', '2026-08-18', False),
                row('c', 'd', '2026-08-17', False), row('c', 'd', '2026-08-19', True)]
        scores = [4, 0, 10, 2]
        _, days, numerator, denominator = review.within_pair(rows, scores)
        weights = np.array([3, 1, 2])
        repeated, values = [], []
        for item, score in zip(rows, scores):
            for _ in range(weights[days.index(item['day'])]):
                repeated.append(item); values.append(score)
        expected = review.within_pair(repeated, values)[0]['comparison_weighted_auc']
        actual = float(weights @ numerator @ weights / (weights @ denominator @ weights))
        self.assertEqual(expected, actual)

    def test_single_class_pairs_are_explicitly_unavailable(self):
        rows = [row('a', 'b', '2026-08-17', True), row('c', 'd', '2026-08-17', False)]
        result = review.within_pair(rows, [4, 2])[0]
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertIsNone(result['comparison_weighted_auc'])
        with self.assertRaises(ValueError):
            review.within_pair(rows, [float('nan'), 2])

    def test_diagnostic_bootstrap_does_not_mutate_saved_inputs(self):
        training = [row('a', 'b', '2026-07-06', True), row('a', 'c', '2026-07-06', False)]
        fitted = review.fit_pair_history(training)
        rows = [row('a', 'b', '2026-08-17', True), row('a', 'b', '2026-08-18', False)]
        for i, item in enumerate(rows):
            item.update({name: float(1 - i) for name in review.MODELS})
        before = copy.deepcopy(rows)
        result = review.evaluate(rows, fitted, repetitions=8)
        self.assertEqual(rows, before)
        self.assertEqual(result['metrics']['july_pair_history']['auc'], .5)
        self.assertEqual(result['within_pair']['comparison_weighted_auc'], 1)
        self.assertGreater(result['within_pair']['day_block_interval']['valid_draws'], 0)


if __name__ == '__main__':
    unittest.main()
