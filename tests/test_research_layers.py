"""Checks for research-layer isolation and conservative candidate aggregation."""
import unittest

import pandas as pd

from swarmcollab.research_layers import RosterMentions, build_trace_features, feature_prefix, is_history_dump, model_sets


def event(actor, turn, seconds, entity='artifact:one', day='2026-07-06', layer=1,
          family='file', operation='read', status='unknown', verbal=False, related_actor=''):
    return dict(actor=actor, turn=turn, time=float(seconds), entity=entity, day=day, layer=layer,
                family=family, operation=operation, status=status, verbal=verbal,
                related_actor=related_actor, weight=1.0)


def universe():
    return pd.DataFrame([('A', 'B', '2026-07-06'), ('A', 'B', '2026-08-03')], columns=['a', 'b', 'day'])


class LayerFeatureTests(unittest.TestCase):
    def test_echoes_do_not_multiply_shared_counts_or_cosine(self):
        rows = pd.DataFrame([event('A', 'a', 1), event('B', 'b', 2)])
        one, _ = build_trace_features(rows, universe(), ['A', 'B', 'C'])
        repeated, _ = build_trace_features(pd.concat([rows, rows, rows]), universe(), ['A', 'B', 'C'])
        pd.testing.assert_frame_equal(one, repeated)
        self.assertEqual(one.iloc[0][feature_prefix(1, 'file') + '_shared'], 1)

    def test_validation_cannot_change_training_idf_or_family_vocabulary(self):
        train = pd.DataFrame([event('A', 'a', 1), event('B', 'b', 2)])
        future = pd.DataFrame([event('C', 'future', 3, day='2026-08-03'),
                               event('B', 'new', 4, day='2026-08-03', family='future_only')])
        old, old_report = build_trace_features(train, universe(), ['A', 'B', 'C'])
        new, new_report = build_trace_features(pd.concat([train, future]), universe(), ['A', 'B', 'C'])
        pd.testing.assert_frame_equal(old.iloc[:1], new.iloc[:1])
        self.assertEqual(set(old_report['families']), set(new_report['families']))
        key = feature_prefix(1, 'file')
        self.assertEqual(old_report['families'][key]['fitted_idf'], new_report['families'][key]['fitted_idf'])
        self.assertEqual(new_report['unseen_family_events'], 1)

    def test_failure_simultaneous_and_reversed_time_do_not_support_ordered_candidate(self):
        column = feature_prefix(1, 'file') + '_ordered_candidate'
        for write_at, read_at, status in [(1, 2, 'failed'), (1, 1, 'unknown'), (2, 1, 'unknown')]:
            rows = pd.DataFrame([event('A', 'a', write_at, operation='write', status=status),
                                 event('B', 'b', read_at)])
            frame, _ = build_trace_features(rows, universe(), ['A', 'B', 'C'])
            self.assertEqual(frame.iloc[0][column], 0)
        rows = pd.DataFrame([event('A', 'a', 1, operation='write'), event('B', 'b', 2)])
        frame, _ = build_trace_features(rows, universe(), ['A', 'B', 'C'])
        self.assertGreater(frame.iloc[0][column], 0)
        self.assertFalse(any('confirmed' in c for c in frame))

    def test_layer2_does_not_change_layer1(self):
        general = pd.DataFrame([event('A', 'a', 1), event('B', 'b', 2)])
        village = pd.DataFrame([event('A', 'x', 1, layer=2, family='village_project', related_actor='B')])
        one, _ = build_trace_features(general, universe(), ['A', 'B', 'C'])
        both, report = build_trace_features(pd.concat([general, village]), universe(), ['A', 'B', 'C'])
        pd.testing.assert_frame_equal(one, both[one.columns])
        self.assertEqual(both.iloc[0][feature_prefix(2, 'village_project') + '_related_actor_candidate'], 1)
        for c in ['act_min', 'act_max', 'p1_cos', 'p1_sync', 'h_write', 'h_read', 'pou_created_commit',
                  'pou_created_other', 'pou_first_commit', 'pou_first_other']:
            both[c] = 0.0
        sets = model_sets(both, report)
        self.assertFalse(any(c.startswith('l2_') for c in sets['layer1_behavior']))
        self.assertTrue(any(c.startswith('l2_') for c in sets['layer1_plus_layer2_behavior']))

    def test_scoped_files_do_not_match_across_actors(self):
        rows = pd.DataFrame([event('A', 'a', 1, entity='local:A:report.csv'),
                             event('B', 'b', 2, entity='local:B:report.csv')])
        frame, _ = build_trace_features(rows, universe(), ['A', 'B'])
        self.assertEqual(frame.iloc[0][feature_prefix(1, 'file') + '_shared'], 0)

    def test_verbal_channel_is_not_behavior(self):
        rows = pd.DataFrame([event('A', 'a', 1, family='name', verbal=True, related_actor='B')])
        frame, report = build_trace_features(rows, universe(), ['A', 'B'])
        column = feature_prefix(1, 'name') + '_verbal_mentions'
        self.assertEqual(frame.iloc[0][column], 1)
        for c in ['act_min', 'act_max', 'p1_cos', 'p1_sync', 'h_write', 'h_read', 'pou_created_commit',
                  'pou_created_other', 'pou_first_commit', 'pou_first_other']:
            frame[c] = 0.0
        sets = model_sets(frame, report)
        self.assertNotIn(column, sets['layer1_behavior'])
        self.assertIn(column, sets['layer1_with_attribution'])

    def test_scope_guard_rejects_sealed_rows(self):
        with self.assertRaises(ValueError):
            build_trace_features(pd.DataFrame([event('A', 'a', 1, day='2026-08-17')]), universe(), ['A'])


class RosterTests(unittest.TestCase):
    def test_history_dump_mask_uses_executed_operands(self):
        def record(command):
            return dict(i='x', t='2026-07-06T17:00:00Z', k='shell', x=command)
        self.assertTrue(is_history_dump(record('bash /tmp/pre_send_chat.sh')))
        self.assertTrue(is_history_dump(record('cat /tmp/public_comms.md')))
        self.assertFalse(is_history_dump(record('echo "bash /tmp/pre_send_chat.sh"')))
        self.assertFalse(is_history_dump(record('cat ordinary_report.md')))

    def test_shorter_version_name_does_not_match_longer_actor(self):
        parser = RosterMentions('A', ['GPT 5', 'GPT 5.5'])
        record = dict(i='x', t='2026-07-06T17:00:00Z', s='s', k='type', x='Ask GPT 5.5', o='')
        self.assertEqual([r['related_actor'] for r in parser.extract(record)], ['GPT 5.5'])

    def test_supplied_full_names_only_with_no_family_aliases(self):
        parser = RosterMentions('A', ['A', 'Claude Sonnet 5', 'Grok 4.5'])
        record = dict(i='x', t='2026-07-06T17:00:00Z', s='s', k='shell',
                      x='echo "Claude Sonnet 5, Grok, Grok 4.5x"', o='')
        result = parser.extract(record)
        self.assertEqual([r['related_actor'] for r in result], ['Claude Sonnet 5'])
        self.assertTrue(result[0]['verbal_channel'])
        self.assertFalse(result[0]['relation_authority'])

    def test_out_of_scope_payload_never_read(self):
        class Guard(dict):
            def get(self, key, *args):
                raise AssertionError('payload accessed')
        self.assertEqual(RosterMentions('A', ['B']).extract(Guard(t='2026-09-01T17:00:00Z')), [])


if __name__ == '__main__':
    unittest.main()
