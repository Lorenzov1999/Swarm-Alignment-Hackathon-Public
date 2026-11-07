"""Synthetic checks for frozen secondary-method transfer and scope boundaries."""
import copy
import json
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from swarmcollab import secondary_transfer as T
from swarmcollab import spread as S


def observations(days=('2026-07-06', '2026-07-07', '2026-07-08')):
    mentions, activity = [], []
    for offset, day in enumerate(days):
        start = pd.Timestamp(day + ' 09:00:00', tz='America/Los_Angeles').timestamp()
        # A and B share two resources; C supplies nonzero IDF and a private one.
        for actor, lag, traces in [('A', 0, ('repo|shared', 'repo|second')),
                                   ('B', 90, ('repo|shared', 'repo|second')),
                                   ('C', 40, ('repo|private',))]:
            for number, tid in enumerate(traces):
                mentions.append(dict(agent=actor, day=day, type='repo', tid=tid,
                                     t=start + lag + number * 2400 + offset * 30))
            activity.append(dict(agent=actor, day=day, working=len(traces)))
    return pd.DataFrame(mentions), pd.DataFrame(activity)


def fitted(null_fits=0):
    mentions, activity = observations()
    return T.fit_secondary(mentions, activity, cross_day_samples=100, null_fits=null_fits)


class SecondaryFitTests(unittest.TestCase):
    def test_full_state_round_trips_and_matches_existing_spread_fit(self):
        mentions, activity = observations()
        tm, _ = T._inputs(mentions, activity, T.FIT_START, T.FIT_END)
        events = S.trace_sessions(tm, shared_reference=tm)
        existing = S.MarkedHawkes(sorted(events.agent.unique())).fit(events)
        reference = pd.DataFrame(existing.alpha, index=existing.agents, columns=existing.agents)
        bundle = T.fit_secondary(mentions, activity, reference_alpha=reference, cross_day_samples=100)
        self.assertTrue(bundle['spread']['original_alpha_comparison']['matches'])
        restored = T._unpack_model(json.loads(json.dumps(bundle))['spread']['model'])
        existing.score(events)
        restored.score(events)
        np.testing.assert_allclose(restored.r, existing.r)
        np.testing.assert_allclose(restored.r0, existing.r0)
        self.assertEqual(restored.mark_counts, existing.mark_counts)
        self.assertEqual(bundle['fit_scope']['end'], '2026-07-31')
        self.assertIsInstance(bundle['regimes']['threshold'], float)

    def test_reference_mismatch_stops_transfer_calibration(self):
        mentions, activity = observations()
        reference = pd.DataFrame(np.zeros((2, 2)), index=['A', 'B'], columns=['A', 'B'])
        with self.assertRaisesRegex(ValueError, 'differs from original'):
            T.fit_secondary(mentions, activity, reference_alpha=reference, cross_day_samples=50)

    def test_training_must_not_contain_validation_or_transfer_rows(self):
        for day in ('2026-06-30', '2026-08-03', '2026-08-17'):
            mentions, activity = observations((day,))
            with self.assertRaisesRegex(ValueError, 'outside declared'):
                T.fit_secondary(mentions, activity, cross_day_samples=10)

    def test_planner_does_not_fit_and_caps_prevent_allocation(self):
        mentions, activity = observations()
        with mock.patch.object(S.MarkedHawkes, 'fit', side_effect=AssertionError('unexpected fit')):
            plan = T.secondary_plan(mentions, activity, null_fits=10)
        self.assertEqual(plan['em_fits'], 11)
        self.assertGreater(plan['candidate_pairs_upper_bound'], 0)
        with self.assertRaisesRegex(ValueError, 'compute cap'):
            T.fit_secondary(mentions, activity, cross_day_samples=10, max_events=1)


class SecondaryApplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = fitted(null_fits=2)

    def apply(self, mentions, activity, **kwargs):
        return T.apply_secondary(self.bundle, mentions, activity,
                                 start='2026-08-17', end='2026-09-15', block_samples=19, **kwargs)

    def test_application_does_not_fit_or_mutate_any_parameters(self):
        mentions, activity = observations(('2026-08-17', '2026-08-18'))
        before = copy.deepcopy(self.bundle)
        with mock.patch.object(S.MarkedHawkes, 'fit', side_effect=AssertionError('target refit')):
            report, pairs, days = self.apply(mentions, activity)
        self.assertEqual(self.bundle, before)
        self.assertFalse(report['target_fitting'])
        self.assertTrue(report['spread']['available'])
        self.assertTrue((pairs.hk_trig >= 0).all())
        self.assertTrue(days.hk_other_share.between(0, 1).all())
        self.assertIn('hk_excess', days)
        json.dumps(report, allow_nan=False)

    def test_target_labels_are_never_used(self):
        mentions, activity = observations(('2026-08-17',))
        _, first, _ = self.apply(mentions, activity)
        target = first[['a', 'b', 'day']].copy()
        target['y'] = True
        _, second, _ = self.apply(mentions, activity, target_pair_days=target)
        pd.testing.assert_frame_equal(first, second)

    def test_unseen_actors_are_unsupported_not_negative_evidence(self):
        mentions, activity = observations(('2026-08-17',))
        extra = mentions[mentions.agent == 'A'].copy(); extra['agent'] = 'D'
        extra['t'] += 5
        extra_activity = activity[activity.agent == 'A'].copy(); extra_activity['agent'] = 'D'
        report, pairs, days = self.apply(pd.concat([mentions, extra], ignore_index=True),
                                        pd.concat([activity, extra_activity], ignore_index=True))
        self.assertGreater(report['spread']['unseen_agent_sessions'], 0)
        self.assertFalse(days.null_calibrated.iloc[0])
        self.assertIsNone(days.p_value.iloc[0])
        self.assertIsNotNone(days.known_agent_p_value.iloc[0])
        self.assertFalse(pairs.loc[(pairs.a == 'D') | (pairs.b == 'D'), 'hk_supported_pair'].any())

    def test_unseen_traces_use_frozen_regime_prior_but_not_hawkes_support(self):
        mentions, activity = observations(('2026-08-17',))
        mentions['tid'] = 'repo|brand-new'
        report, pairs, days = self.apply(mentions, activity)
        self.assertFalse(report['spread']['available'])
        self.assertEqual(report['spread']['supported_sessions'], 0)
        np.testing.assert_allclose(pairs.p1_cos, 1.)
        self.assertNotIn('hk_other_share', days)

    def test_june_is_allowed_and_described_as_retrospective(self):
        mentions, activity = observations(('2026-06-01', '2026-06-02'))
        report, pairs, days = T.apply_secondary(self.bundle, mentions, activity,
                                        start='2026-06-01', end='2026-07-05', block_samples=19)
        self.assertIn('retrospective', report['regimes']['interpretation'])
        self.assertTrue(days.flagged_share.notna().all())
        self.assertFalse(days.null_calibrated.any())
        self.assertFalse(days.known_agent_null_calibrated.any())
        self.assertTrue(days.p_value.isna().all())
        self.assertTrue(days.known_agent_p_value.isna().all())
        self.assertTrue(days.collaborative.isna().all())
        self.assertTrue(days.known_agent_collaborative.isna().all())
        self.assertTrue((days.known_training_agents == 3).all())
        self.assertTrue((days.known_agent_pairs == 3).all())
        self.assertTrue((days.available_donor_agents == 0).all())
        self.assertEqual(report['regimes']['fully_calibrated_days'], 0)
        self.assertEqual(len(pairs), 6)

    def test_donor_availability_filters_blocks_without_changing_identity_support(self):
        reference = self.bundle['regimes']
        early = T._available_donor_rows(reference, '2026-06-30')
        later = T._available_donor_rows(reference, '2026-08-17')
        self.assertEqual(set(early), set(reference['donor_rows']))
        self.assertTrue(all(not rows for rows in early.values()))
        self.assertEqual(later, reference['donor_rows'])
        mentions, activity = observations(('2026-08-17',))
        _, _, days = self.apply(mentions, activity)
        self.assertTrue(days.null_calibrated.all())
        self.assertTrue(days.p_value.notna().all())

    def test_donor_availability_uses_each_blocks_latest_artifact_date(self):
        reference = dict(agents=['A', 'B'], days=['2026-07-06', '2026-07-07'],
                         donor_rows={'A': [0, 2], 'B': [1, 3]},
                         block_availability=[['A', '2026-07-06', '2026-07-06'],
                                             ['A', '2026-07-07', '2026-07-07'],
                                             ['B', '2026-07-06', '2026-07-06'],
                                             ['B', '2026-07-07', '2026-07-07']])
        self.assertEqual(T._available_donor_rows(reference, '2026-07-06'), {'A': [0], 'B': [1]})
        self.assertEqual(T._available_donor_rows(reference, '2026-07-07'), reference['donor_rows'])

    def test_reserved_window_and_forged_dates_are_rejected(self):
        mentions, activity = observations(('2026-09-16',))
        with self.assertRaisesRegex(ValueError, 'not authorized'):
            T.apply_secondary(self.bundle, mentions, activity,
                              start='2026-09-16', end='2026-10-31', block_samples=10)
        with self.assertRaisesRegex(ValueError, 'outside declared'):
            self.apply(mentions, activity)
        mentions, activity = observations(('2026-08-17',))
        mentions.loc[0, 't'] -= 86400
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.apply(mentions, activity)

    def test_tampered_bundle_and_wrong_pair_universe_fail(self):
        mentions, activity = observations(('2026-08-17',))
        changed = copy.deepcopy(self.bundle)
        changed['regimes']['threshold'] += .5
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            T.apply_secondary(changed, mentions, activity,
                              start='2026-08-17', end='2026-09-15', block_samples=10)
        wrong = pd.DataFrame([dict(a='A', b='missing', day='2026-08-17')])
        with self.assertRaisesRegex(ValueError, 'eligible'):
            self.apply(mentions, activity, target_pair_days=wrong)

    def test_source_changes_after_calibration_are_rejected(self):
        mentions, activity = observations(('2026-08-17',))
        environment = T._environment()
        environment['source_sha256']['spread'] = 'changed'
        with mock.patch.object(T, '_environment', return_value=environment):
            with self.assertRaisesRegex(ValueError, 'source code changed'):
                self.apply(mentions, activity)


class LegacyNetworkTests(unittest.TestCase):
    def test_matrix_orientation_and_original_fit_scope_are_explicit(self):
        alpha = pd.DataFrame([[.1, .2], [.3, .4]], index=['A', 'B'], columns=['A', 'B'])
        excess = alpha - .1
        pairs = pd.DataFrame([dict(a='A', b='B', day='2026-08-17', y=True),
                              dict(a='A', b='new', day='2026-08-17', y=False)])
        report, scores = T.score_legacy_network(alpha, excess, pairs,
                                                start='2026-08-17', end='2026-09-15')
        self.assertEqual(report['original_fit_scope']['end'], '2026-08-14')
        self.assertEqual(report['supported_directed_pair_days'], 2)
        self.assertEqual(scores.iloc[0]['source'], 'A')
        self.assertEqual(scores.iloc[0]['alpha'], .3)
        self.assertEqual(scores.iloc[1]['alpha'], .2)
        self.assertTrue(scores.loc[~scores.supported, 'alpha'].isna().all())
        self.assertNotIn('y', scores)

    def test_legacy_rejects_reserved_dates_and_malformed_network(self):
        alpha = pd.DataFrame([[.1]], index=['A'], columns=['A'])
        pairs = pd.DataFrame([dict(a='A', b='B', day='2026-09-16')])
        with self.assertRaises(ValueError):
            T.score_legacy_network(alpha, alpha, pairs, start='2026-08-17', end='2026-09-15')
        pairs['day'] = '2026-08-17'
        with self.assertRaises(ValueError):
            T.score_legacy_network(alpha.rename(columns={'A': 'B'}), alpha, pairs,
                                   start='2026-08-17', end='2026-09-15')


if __name__ == '__main__':
    unittest.main()
