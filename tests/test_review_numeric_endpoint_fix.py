"""Verify the additive cosine audit without reading the Village dataset."""
import importlib.util
import itertools
from collections import Counter
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd


SPEC = importlib.util.spec_from_file_location('review_numeric_endpoint_fix',
    Path(__file__).resolve().parents[1] / 'scripts/review_numeric_endpoint_fix.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def counts(rows):
    return pd.DataFrame(rows, columns=['day', 'agent', 'tid', 'n'])


def universe(agents, days):
    return pd.DataFrame([(a, b, day) for day in days for a, b in itertools.combinations(agents, 2)],
                        columns=['a', 'b', 'day'])


class NumericEndpointAuditTests(unittest.TestCase):
    def test_untouched_profiles_preserve_published_bits(self):
        day = '2026-07-06'
        original = counts([(day, a, 'shared', 1) for a in ['a', 'b', 'c']])
        added = counts([(day, 'a', 'private', 1)])
        frame = universe(['a', 'b', 'c'], [day])
        frame['published'] = np.nextafter(np.ones(len(frame)), np.zeros(len(frame)))
        before = frame.published.to_numpy().copy()
        actual = review.augment_cosine(original, added, frame, {'shared': 2., 'private': 3.}, 4.)
        untouched = (frame.a == 'b') & (frame.b == 'c')
        np.testing.assert_array_equal(actual[untouched].view(np.uint64), before[untouched].view(np.uint64))
        np.testing.assert_array_equal(frame.published.to_numpy().view(np.uint64), before.view(np.uint64))
        empty = review.augment_cosine(original, added.iloc[:0], frame, {'shared': 2.}, 4.)
        np.testing.assert_array_equal(empty.view(np.uint64), before.view(np.uint64))

    def test_new_shared_coordinate_creates_edge_for_previous_zero_profile(self):
        day = '2026-07-06'
        original = counts([(day, 'a', 'old', 1)])
        added = counts([(day, 'a', 'new', 1), (day, 'b', 'new', 1)])
        frame = universe(['a', 'b', 'c'], [day])
        frame['published'] = 0.
        actual = review.augment_cosine(original, added, frame, {'old': 2., 'new': 3.}, 4.)
        np.testing.assert_allclose(actual, [3 / np.sqrt(13), 0., 0.], rtol=1e-14)
        self.assertGreater(actual[0], review.THRESHOLD)
        self.assertEqual(len(actual), 3)  # Agent c remains in the complete universe.

    def test_unshared_coordinate_dilutes_existing_cosine(self):
        day = '2026-07-06'
        original = counts([(day, 'a', 'shared', 1), (day, 'b', 'shared', 1)])
        added = counts([(day, 'a', 'private', 1)])
        frame = universe(['a', 'b'], [day])
        frame['published'] = 1.
        actual = review.augment_cosine(original, added, frame, {'shared': 2., 'private': 3.}, 4.)
        self.assertAlmostEqual(actual[0], 2 / np.sqrt(13), places=14)
        self.assertLess(actual[0], frame.published.iloc[0])

    def test_incremental_update_matches_full_recomputation_with_log_counts(self):
        d1, d2 = '2026-07-06', '2026-07-07'
        original = counts([(d1, 'a', 'x', 3), (d1, 'a', 'y', 1), (d1, 'b', 'x', 2),
                           (d1, 'c', 'zero_weight', 4), (d2, 'a', 'x', 5),
                           (d2, 'b', 'y', 2), (d2, 'c', 'x', 1)])
        added = counts([(d1, 'a', 'z', 2), (d1, 'b', 'z', 5), (d1, 'c', 'unseen', 1),
                        (d2, 'a', 'zero_new', 7), (d2, 'b', 'z', 3), (d2, 'c', 'z', 1)])
        weights = {'x': 2., 'y': 1., 'z': 3., 'zero_weight': 0., 'zero_new': 0.}
        frame = universe(['a', 'b', 'c', 'd'], [d1, d2])
        frame['published'] = review.cosine(original, frame, weights, 4.)
        actual = review.augment_cosine(original, added, frame, weights, 4.)
        expected = review.cosine(pd.concat([original, added], ignore_index=True), frame, weights, 4.)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)

    def test_out_of_scope_labels_fail_before_the_loader_is_called(self):
        with mock.patch.object(review.labels, 'load_requests', side_effect=AssertionError('must not read labels')):
            for start, end in [('2026-09-16', '2026-10-01'), ('2026-06-01', '2026-09-16'),
                               ('2026-07-06', '2026-08-14')]:
                with self.assertRaises(ValueError):
                    review.scoped_labels(start, end)

    def test_replay_activity_rejects_missing_boundary_steps_and_extra_days(self):
        day = '2026-09-15'
        counter = Counter()
        review.record_activity(counter, 'shell')
        review.record_activity(counter, 'type')
        row = dict(agent='a', day=day, **{field: counter[field] for field in review.ACTIVITY_FIELDS})
        expected = pd.DataFrame([row])
        observed = {('a', day): counter}
        self.assertEqual(review.reconcile_activity(observed, expected, day, day), [row])
        with self.assertRaisesRegex(ValueError, 'Compact-step activity differs'):
            review.reconcile_activity({}, expected, day, day)
        duplicate = counter.copy()
        review.record_activity(duplicate, 'shell')
        with self.assertRaisesRegex(ValueError, 'Compact-step activity differs'):
            review.reconcile_activity({('a', day): duplicate}, expected, day, day)
        with self.assertRaisesRegex(ValueError, 'Compact-step activity differs'):
            review.reconcile_activity({**observed, ('a', '2026-09-16'): counter}, expected, day, day)

    def test_activity_categories_match_saved_legacy_definitions(self):
        counter = Counter()
        for kind in ['shell', 'type', next(iter(review.traces.GUI_KINDS - {'type'})),
                     next(iter(review.traces.WAITING)), 'unknown-tool']:
            review.record_activity(counter, kind)
        self.assertEqual(dict(counter), dict(steps=5, working=4, shell=1, gui=2,
                                             typed=1, waiting=1, other=1))


if __name__ == '__main__':
    unittest.main()
