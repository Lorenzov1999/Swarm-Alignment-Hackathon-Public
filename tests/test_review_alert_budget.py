"""Verify tie handling in the alert-budget curves and matched-budget flags."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location('review_alert_budget',
    Path(__file__).resolve().parents[1] / 'scripts/review_alert_budget.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class AlertBudgetTests(unittest.TestCase):
    def test_curve_ends_at_tie_blocks_and_interpolates_expected_recall(self):
        score = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
        y = np.array([True, True, False, False, True])
        alerts, found = review.budget_curve(score, y)
        np.testing.assert_array_equal(alerts, [0, 1, 4, 5])
        np.testing.assert_array_equal(found, [0, 1, 2, 3])
        # Two of the three tied rows flagged: expected 2/3 of the tied positive.
        self.assertAlmostEqual(np.interp(3, alerts, found), 1 + 2 / 3)

    def test_flag_probability_spends_exactly_the_budget(self):
        score = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
        probability = review.flag_probability(score, 3)
        np.testing.assert_allclose(probability, [1, 2 / 3, 2 / 3, 2 / 3, 0])
        self.assertAlmostEqual(probability.sum(), 3)
        # Expected recall from the flags equals the interpolated curve.
        y = np.array([True, True, False, False, True])
        alerts, found = review.budget_curve(score, y)
        self.assertAlmostEqual(probability[y].sum(), np.interp(3, alerts, found))

    def test_pair_history_uses_only_july_and_shrinks_unseen_pairs(self):
        import pandas as pd
        dev = pd.DataFrame(dict(a=['x', 'x', 'x', 'x'], b=['y', 'y', 'z', 'y'],
                                day=['2026-07-06', '2026-07-07', '2026-07-06', '2026-08-03'],
                                y=[True, False, False, True]))
        score = review.pair_history(dev)
        target = pd.DataFrame(dict(a=['x', 'x', 'u'], b=['y', 'z', 'v']))
        prevalence = 1 / 3
        np.testing.assert_allclose(score(target), [(1 + prevalence) / 3, prevalence / 2,
                                                   prevalence])


if __name__ == '__main__':
    unittest.main()
