"""Scientific regression checks: frozen fitting, dependent nulls, and time support."""
import copy
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from swarmcollab import metrics as M
from swarmcollab import pairs as P
from swarmcollab import regimes as R
from swarmcollab import spread as S
from swarmcollab import timescale as T


def mentions(rows):
    records = []
    for agent, day, kind, identity in rows:
        records.append({'agent': agent, 'day': day, 'type': kind, 'id': identity,
                        't': 1.0, 'op': 'seen', 'src': 'out', 'step': 1, 'k': 1})
    return pd.DataFrame(records)


def events(day, offset=0, repeat=1):
    base = pd.Timestamp(day, tz='America/Los_Angeles').timestamp() + 9 * 3600
    rows = []
    for n in range(repeat):
        for i, agent in enumerate(('A', 'B')):
            rows.append({'agent': agent, 'tid': 'repo|shared', 't': base + offset + n * 200 + i * 50,
                         'day': day, 'hour': 9})
    return pd.DataFrame(rows)


class FrozenFeatureTests(unittest.TestCase):
    def test_feature_names_unique_in_every_model(self):
        models = dict(P.model_features())
        for columns in models.values():
            self.assertEqual(len(columns), len(set(columns)))
        self.assertEqual(models['+ all names'].count('name_comment'), 1)
        fitted = M.Logistic().fit([[0, 1], [1, 0]], [0, 1])
        with self.assertRaises(ValueError):
            fitted.coefficients(['duplicate', 'duplicate'])

    def test_validation_does_not_resolve_training_project_ids(self):
        train = mentions([('A', '2026-07-06', 'project', '12'),
                          ('B', '2026-07-06', 'repo', 'gitlab:group/project')])
        future = mentions([('A', '2026-08-03', 'pidmap', '12|group/project'),
                           ('B', '2026-08-03', 'repo', 'gitlab:group/project/child')])
        before, _ = P.resolve_repos(train, reference=train)
        after, _ = P.resolve_repos(pd.concat([train, future]), reference=train)
        self.assertEqual(before.iloc[0].id, '12')
        self.assertEqual(after.iloc[0].id, '12')
        self.assertEqual(after.iloc[1].id, 'gitlab:group/project')
        self.assertEqual(after.iloc[-1].id, 'gitlab:group/project/child')

    def test_validation_frequency_and_agents_do_not_change_idf(self):
        train = pd.DataFrame({'agent': ['A', 'B', 'A'], 'day': ['2026-07-06'] * 3,
                              'tid': ['common', 'common', 'rare']})
        future = pd.DataFrame({'agent': ['C'] * 100 + ['B'], 'day': ['2026-08-03'] * 101,
                               'tid': ['rare'] * 100 + ['new']})
        before = P.trace_weights(train, ['A', 'B'], {'2026-07-06'})
        after = P.trace_weights(pd.concat([train, future]), ['A', 'B'], {'2026-07-06'})
        for key in before:
            self.assertEqual(before[key], after[key])
        self.assertAlmostEqual(after['new'], np.log(2))

    def test_continuous_synchrony_crosses_hour_boundary_once(self):
        tm = pd.DataFrame({'agent': ['A', 'A', 'B', 'B'], 'day': ['2026-07-06'] * 4,
                           'tid': pd.Categorical(['repo|x'] * 4), 't': [3599., 3599., 3601., 3601.]})
        _, sync = P.p1_features(tm, {'repo|x': 0.7}, ['A', 'B'])
        self.assertEqual(len(sync), 1)
        self.assertAlmostEqual(sync.iloc[0].p1_sync, 0.7)
        tm.loc[tm.agent == 'B', 't'] = 8000
        _, sync = P.p1_features(tm, {'repo|x': 0.7}, ['A', 'B'])
        self.assertEqual(len(sync), 0)

    def test_time_windows_exclude_outside_hours(self):
        keys = T.window_key(['2026-07-06'] * 5, [8, 9, 16, 17, 23], 2)
        self.assertEqual(keys.tolist(), [None, '2026-07-06#0', '2026-07-06#3', None, None])
        with self.assertRaises(ValueError):
            T.window_key(['2026-07-06'], [9], 3)

    def test_single_class_recall_is_undefined(self):
        self.assertTrue(np.isnan(M.recall_at_fpr([1, 2], [1, 1])))
        self.assertTrue(np.isnan(M.recall_at_fpr([1, 2], [0, 0])))


class FrozenHawkesTests(unittest.TestCase):
    def test_validation_does_not_refit_parameters_marks_or_hours(self):
        train = pd.concat([events('2026-07-06', repeat=3), events('2026-07-07', repeat=2)])
        model = S.MarkedHawkes(['A', 'B']).fit(train, iters=10)
        frozen = (model.alpha.copy(), model.mu.copy(), copy.deepcopy(model.mark_counts), model.hour_profile.copy())
        model.score(events('2026-08-03', repeat=10))
        np.testing.assert_array_equal(model.alpha, frozen[0])
        np.testing.assert_array_equal(model.mu, frozen[1])
        self.assertEqual(model.mark_counts, frozen[2])
        np.testing.assert_array_equal(model.hour_profile, frozen[3])
        self.assertTrue(np.isfinite(model.loglik()))
        self.assertEqual(model.fit_days, ['2026-07-06', '2026-07-07'])

    def test_future_sharedness_does_not_change_training_sessions(self):
        train = events('2026-07-06')
        train.loc[train.agent == 'B', 'tid'] = 'repo|other'
        future = events('2026-08-03')
        combined = pd.concat([train, future])
        filtered = S.trace_sessions(combined, shared_reference=train)
        self.assertTrue(filtered.empty)

    def test_event_weighting_uses_source_columns(self):
        model = S.MarkedHawkes(['A', 'B'])
        model.alpha = np.array([[0.1, 0.9], [0.2, 0.1]])
        model.fit_event_counts = np.array([90., 10.])
        weights = model.cross_excitation()
        self.assertAlmostEqual(weights['unweighted_per_agent'], 0.55)
        self.assertAlmostEqual(weights['source_event_weighted'], 0.27)

    def test_day_swap_preserves_blocks_and_observed_availability(self):
        early = events('2026-07-06', repeat=2)
        late = events('2026-07-07', repeat=2)
        late.loc[late.agent == 'B', 'tid'] = 'new-artifact'
        original = pd.concat([early, late], ignore_index=True)
        swapped = S.day_swap(original, np.random.default_rng(1))
        self.assertEqual(original.groupby(['agent', 'tid']).size().to_dict(),
                         swapped.groupby(['agent', 'tid']).size().to_dict())
        first_seen = original.groupby('tid').day.min().to_dict()
        self.assertTrue(all(day >= first_seen[tid] for day, tid in zip(swapped.day, swapped.tid)))


class BlockNullTests(unittest.TestCase):
    def setUp(self):
        self.days = ['2026-07-06', '2026-07-07', '2026-07-08']
        self.agents = ['A', 'B']
        self.vecs = {day: (sparse.csr_matrix([[1, 0, 0], [0, 1, 0]]), np.array([True, True]))
                     for day in self.days}
        self.active = {day: {'A', 'B'} for day in self.days + ['2026-08-03']}

    def test_training_permutation_preserves_agent_block_multiset(self):
        null = R.AgentDayBlockNull(self.vecs, self.agents, self.active, self.days)
        draw = null.training_permutation(np.random.default_rng(1))
        for ai in range(2):
            self.assertEqual(set(draw[:, ai]), {di * 2 + ai for di in range(3)})

    def test_availability_prevents_moving_a_new_artifact_into_the_past(self):
        availability = {(agent, day): day for day in self.days for agent in self.agents}
        null = R.AgentDayBlockNull(self.vecs, self.agents, self.active, self.days, availability)
        draw = null.training_permutation(np.random.default_rng(5))
        for target_di, row in enumerate(draw):
            self.assertTrue(all(donor // 2 <= target_di for donor in row))

    def test_validation_vectors_do_not_enter_calibration(self):
        one = dict(self.vecs, **{'2026-08-03': (sparse.csr_matrix([[1, 0, 0], [1, 0, 0]]), np.array([True, True]))})
        two = dict(self.vecs, **{'2026-08-03': (sparse.csr_matrix([[0, 0, 1], [0, 1, 0]]), np.array([True, True]))})
        a = R.AgentDayBlockNull(one, self.agents, self.active, self.days)
        b = R.AgentDayBlockNull(two, self.agents, self.active, self.days)
        da, _ = a.distributions(0.5, np.random.default_rng(7), n=10)
        db, _ = b.distributions(0.5, np.random.default_rng(7), n=10)
        for day in da:
            np.testing.assert_array_equal(da[day], db[day])

    def test_regime_pvalue_uses_empirical_whole_window_counts(self):
        frame = pd.DataFrame({'a': ['A', 'A', 'B'], 'b': ['B', 'C', 'C'],
                              'day': ['2026-08-03'] * 3, 'score': [1., 1., 1.], 'y': [True, False, False]})
        table = R.window_table(frame, 'score', 0.5, {'2026-08-03': np.array([0., 1., 0., 1.])}, 1)
        self.assertAlmostEqual(table.iloc[0].p_value, 3 / 5)
        self.assertFalse(table.iloc[0].collaborative)


if __name__ == '__main__':
    unittest.main()
