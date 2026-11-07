"""Frozen feature transfer preserves development semantics without target fitting."""
import copy
import math
import unittest

import numpy as np
import pandas as pd

from swarmcollab import pairs
from swarmcollab.research_layers import build_trace_features, feature_prefix
from swarmcollab.transfer_features import (
    TRACE_COLUMNS, combine_features, freeze_historical, historical_features,
    provenance_features, trace_features,
)


def trace(actor, turn, seconds, entity='url:shared', day='2026-07-06',
          layer=1, family='url', operation='read', status='unknown',
          verbal=False, related_actor='', weight=1.0):
    return dict(actor=actor, turn=turn, time=float(seconds), entity=entity, day=day,
                layer=layer, family=family, operation=operation, status=status,
                verbal=verbal, related_actor=related_actor, weight=weight)


def mention(actor, seconds, kind, identity, day='2026-07-06', op='read', src='cmd'):
    return dict(agent=actor, t=float(seconds), day=day, step=int(seconds), type=kind,
                id=identity, op=op, src=src, k=1)


def reference_mentions():
    return pd.DataFrame([
        mention('A', 1, 'repo', 'gitlab:group/project'),
        mention('A', 2, 'pidmap', '10|group/project'),
        mention('A', 3, 'commit', 'aabbccd', op='create'),
        mention('B', 4, 'commit', 'aabbccd'),
        mention('C', 5, 'repo', 'gitlab:group/other'),
        mention('A', 6, 'name', 'B', src='out'),
    ])


class FrozenTraceTests(unittest.TestCase):
    def setUp(self):
        self.actors = ['A', 'B', 'C', 'D']
        self.universe = pd.DataFrame([
            ('A', 'B', '2026-07-06'), ('A', 'B', '2026-08-03'),
        ], columns=['a', 'b', 'day'])
        self.rows = pd.DataFrame([
            trace('A', 'a', 1, operation='write'), trace('B', 'b', 2),
            trace('A', 'c', 3, family='roster_name', verbal=True, related_actor='B'),
            trace('A', 'd', 4, layer=2, family='village_project', related_actor='B'),
            trace('A', 'e', 101, day='2026-08-03', operation='write'),
            trace('B', 'f', 102, day='2026-08-03'),
            trace('A', 'g', 103, entity='url:unseen', day='2026-08-03'),
            trace('B', 'h', 104, entity='url:unseen', day='2026-08-03'),
            trace('A', 'i', 105, day='2026-08-03', family='roster_name',
                  verbal=True, related_actor='B', weight=.5),
            trace('A', 'j', 106, day='2026-08-03', layer=2, family='village_project', related_actor='B'),
        ])
        self.original, self.manifest = build_trace_features(self.rows, self.universe, self.actors)

    def test_saved_manifest_matches_original_validation_exactly(self):
        rows = self.rows[self.rows.day == '2026-08-03']
        universe = self.universe[self.universe.day == '2026-08-03']
        result, _ = trace_features(rows, universe, self.manifest, self.actors,
                                   '2026-08-03', '2026-08-03')
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                      self.original.iloc[1:].reset_index(drop=True))

    def test_june_rows_cannot_become_training_observations(self):
        june = self.rows[self.rows.day == '2026-08-03'].assign(day='2026-06-01')
        universe = self.universe.iloc[1:].assign(day='2026-06-01')
        before = copy.deepcopy(self.manifest)
        result, _ = trace_features(june, universe, self.manifest, self.actors,
                                   '2026-06-01', '2026-07-05')
        expected = self.original.iloc[1:].assign(day='2026-06-01').reset_index(drop=True)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)
        self.assertEqual(self.manifest, before)
        overlap = feature_prefix(1, 'url') + '_idf_overlap'
        self.assertAlmostEqual(result.iloc[0][overlap], math.log(4 / 2) + math.log(4))

    def test_unseen_families_do_not_change_vocabulary(self):
        rows = pd.DataFrame([trace('A', 'x', 1, day='2026-08-17', family='new_family')])
        universe = self.universe.iloc[1:].assign(day='2026-08-17')
        result, report = trace_features(rows, universe, self.manifest, self.actors,
                                        '2026-08-17', '2026-09-15')
        self.assertEqual(report['unseen_family_events'], 1)
        self.assertEqual(set(result), set(self.original))
        self.assertEqual(result.drop(columns=['a', 'b', 'day']).to_numpy().sum(), 0)

    def test_scope_and_manifest_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):
            trace_features(self.rows, self.universe, self.manifest, self.actors,
                           '2026-08-17', '2026-09-15')
        manifest = copy.deepcopy(self.manifest)
        next(iter(manifest['families'].values()))['features'] = ['invented']
        with self.assertRaises(ValueError):
            trace_features(self.rows, self.universe, manifest, self.actors,
                           '2026-07-06', '2026-08-14')

    def test_empty_rows_keep_all_frozen_columns(self):
        result, _ = trace_features(pd.DataFrame(columns=TRACE_COLUMNS), self.universe,
                                   self.manifest, self.actors, '2026-07-06', '2026-08-14')
        self.assertEqual(set(result), set(self.original))
        self.assertEqual(result.drop(columns=['a', 'b', 'day']).to_numpy().sum(), 0)


class FrozenHistoricalTests(unittest.TestCase):
    def setUp(self):
        self.reference = reference_mentions()
        self.actors = ['A', 'B', 'C']
        self.frozen = freeze_historical(self.reference, self.actors)
        self.activity = pd.DataFrame([
            dict(agent='A', day='2026-08-17', working=10),
            dict(agent='B', day='2026-08-17', working=20),
        ])
        self.target = pd.DataFrame([
            mention('A', 101, 'project', '10', day='2026-08-17'),
            mention('B', 102, 'repo', 'gitlab:group/project', day='2026-08-17'),
            mention('A', 103, 'commit', '1122334', day='2026-08-17', op='create'),
            mention('B', 104, 'commit', '1122334', day='2026-08-17'),
            mention('A', 105, 'name', 'B', day='2026-08-17', src='out'),
        ])

    def test_existing_historical_algorithms_are_preserved(self):
        result, _ = historical_features(self.target, self.activity, self.frozen,
                                        '2026-08-17', '2026-09-15')
        resolved, _ = pairs.resolve_repos(self.target, reference=self.reference)
        training, _ = pairs.resolve_repos(self.reference, reference=self.reference)
        target_traces = resolved[resolved.type.isin(pairs.TRACE_TYPES)].copy()
        target_traces['tid'] = target_traces.tid.astype(str).astype('category')
        all_traces = pd.concat([training[training.type.isin(pairs.TRACE_TYPES)], target_traces])
        idf = pairs.trace_weights(all_traces, self.actors, ['2026-07-06'])
        pieces = list(pairs.p1_features(target_traces, idf, ['A', 'B']))
        pieces.extend(pairs.p2_features(target_traces, idf)[:2])
        pieces.append(pairs.name_features(resolved)[0])
        expected = pairs.universe(self.activity)
        for piece in pieces:
            expected = expected.merge(piece, on=['a', 'b', 'day'], how='left')
        for c in pairs.FEATURES:
            if c not in expected:
                expected[c] = 0.0
        expected = expected.fillna(0.0)
        expected['act_min'], expected['act_max'] = np.log1p(10), np.log1p(20)
        pd.testing.assert_frame_equal(result[expected.columns], expected, check_dtype=False)

    def test_history_supports_handoff_without_refitting_identity(self):
        target = pd.DataFrame([mention('B', 101, 'commit', 'aabbccd', day='2026-08-17')])
        without, _ = historical_features(target, self.activity, self.frozen, '2026-08-17', '2026-09-15')
        with_history, report = historical_features(target, self.activity, self.frozen,
            '2026-08-17', '2026-09-15', history_mentions=self.reference)
        self.assertEqual(without.iloc[0].pou_created_commit, 0)
        self.assertEqual(with_history.iloc[0].pou_created_commit, 1)
        self.assertEqual(report['frozen_sha256'], self.frozen['sha256'])
        with self.assertRaises(ValueError):
            historical_features(target.assign(day='2026-06-01'), self.activity.assign(day='2026-06-01'),
                                self.frozen, '2026-06-01', '2026-07-05', history_mentions=self.reference)

    def test_training_scope_and_fingerprint_protect_freeze(self):
        with self.assertRaises(ValueError):
            freeze_historical(pd.concat([self.reference, self.target]), self.actors)
        altered = copy.deepcopy(self.frozen)
        altered['unseen_idf'] = 9.0
        with self.assertRaises(ValueError):
            historical_features(self.target, self.activity, altered, '2026-08-17', '2026-09-15')

    def test_no_identifier_or_no_mention_events_keep_zero_columns(self):
        for target in [self.target[self.target.type.isin(['repo', 'project'])], self.target.iloc[:0]]:
            result, _ = historical_features(target, self.activity, self.frozen,
                                            '2026-08-17', '2026-09-15')
            self.assertEqual(result.iloc[0].pou_created_commit, 0)
            self.assertEqual(result.iloc[0].pou_first_other, 0)
            self.assertGreater(result.iloc[0].act_min, 0)


class AssemblyTests(unittest.TestCase):
    def test_empty_provenance_is_explicit_zero_and_unknown_features_fail(self):
        universe = pd.DataFrame([('A', 'B', '2026-08-17')], columns=['a', 'b', 'day'])
        provenance, report = provenance_features(universe, [], ['prov_level_1', 'prov_level_4'],
                                                 '2026-08-17', '2026-09-15')
        self.assertEqual(report['target_chains'], 0)
        self.assertEqual(provenance.prov_level_4.iloc[0], 0)
        base = universe.assign(act_min=1.0)
        result = combine_features([base, provenance], ['act_min', 'prov_level_4'])
        self.assertEqual(list(result), ['a', 'b', 'day', 'act_min', 'prov_level_4'])
        with self.assertRaises(ValueError):
            combine_features([base, provenance], ['missing'])
        with self.assertRaises(ValueError):
            combine_features([base, base], ['act_min'])


if __name__ == '__main__':
    unittest.main()
