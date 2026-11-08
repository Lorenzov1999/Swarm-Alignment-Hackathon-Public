"""Synthetic checks of ties and whole-day bootstrap weighting."""
import unittest
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import check_shared_resources as review


class SharedResourceChecks(unittest.TestCase):
    def test_reproduction_dry_run_is_standalone_and_does_not_write(self):
        script = Path(__file__).with_name('reproduce.py')
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / 'not-created'
            command = [sys.executable, str(script), '--stages', 'current-paper,swarm-ranking',
                       '--paper-output', str(destination), '--swarm-ranking-output', str(destination / 'ranking'),
                       '--dry-run']
            result = subprocess.run(command, cwd=folder, capture_output=True, text=True, check=True)
            self.assertIn('check_shared_resources.py', result.stdout)
            self.assertIn('swarmcollab.swarm_resource_overlap', result.stdout)
            self.assertNotIn('review_numeric_endpoint_fix.py', result.stdout)
            self.assertFalse(destination.exists())

    def test_correction_rejects_fresh_run_mode_even_for_dry_run(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('reproduce.py')),
                                 '--stages', 'numeric-correction', '--dry-run'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --paper-mode frozen-paper', result.stderr)

    def test_weighted_ranking_matches_explicit_resampling_with_ties(self):
        score = np.array([.2, .2, .8, .9])
        y = np.array([True, False, True, False])
        weights = np.array([2, 1, 1, 3])
        indices = np.repeat(np.arange(len(y)), weights)
        observed = review.ranking_parts(score, y)(weights)
        expected = [review.metrics.auc(score[indices], y[indices]),
                    review.average_precision(score[indices], y[indices])]
        np.testing.assert_allclose(observed, expected, atol=1e-12)

    def test_budget_ties_allocate_exact_expected_workload(self):
        score = np.array([3., 2., 2., 2., 1.])
        np.testing.assert_allclose(review.top_k_probability(score, 2),
                                   [1., 1/3, 1/3, 1/3, 0.])
        self.assertAlmostEqual(review.top_k_probability(score, 2).sum(), 2)

    def test_grouped_day_bootstrap_matches_explicit_day_copies(self):
        frame = pd.DataFrame({
            'a': ['A', 'A', 'A', 'A', 'A', 'A'],
            'b': ['B', 'C', 'B', 'C', 'B', 'C'],
            'day': ['d0', 'd0', 'd1', 'd1', 'd2', 'd2'],
            'y': [True, False, False, True, True, False],
            'shared_resources': [.8, .1, .9, .4, .3, .3],
            'activity': [.8, .8, .1, .2, .4, .7],
        })
        days, day_index = np.unique(frame.day, return_inverse=True)
        weights = np.array([2, 1, 3])
        copied = []
        for d, count in zip(days, weights):
            for replica in range(count):
                part = frame[frame.day == d].copy()
                part['day'] = d + '-' + str(replica)
                copied.append(part)
        explicit = pd.concat(copied, ignore_index=True)
        for columns in (['a', 'b'], ['day']):
            evaluate, _ = review.grouped_comparisons(frame, columns, day_index)
            expected = []
            for model in review.MODELS:
                wins = comparisons = 0
                for _, group in explicit.groupby(columns):
                    positive = group.loc[group.y, model].to_numpy()
                    negative = group.loc[~group.y, model].to_numpy()
                    comparisons += len(positive) * len(negative)
                    wins += ((positive[:, None] > negative).sum() +
                             .5 * (positive[:, None] == negative).sum())
                expected.append(wins / comparisons)
            np.testing.assert_allclose(evaluate(weights), expected, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
