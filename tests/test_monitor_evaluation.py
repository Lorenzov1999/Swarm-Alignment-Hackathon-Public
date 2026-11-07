"""Chronological replay, saved models, and training-only calibration."""
import datetime as dt
import json
import unittest

import pandas as pd

from swarmcollab.monitor_evaluation import normalize_signals, replay


def stamp(day, hour=10, minute=0):
    return f'{day}T{hour + 7:02d}:{minute:02d}:00+00:00'


def signal(a='A', b='B', day='2026-07-06', hour=10, kind='reuse', weight=1., suffix=''):
    when = stamp(day, hour)
    return dict(source_actor=a, target_actor=b, time=when, kind=kind, weight=weight,
                signal_id='|'.join((day, a, b, str(hour), kind, suffix)),
                evidence_turns=[dict(actor=a, turn_id='src' + suffix, time=when),
                                dict(actor=b, turn_id='dst' + suffix, time=when)])


def fixtures():
    pairs, signals, activity, requests = [], [], [], []
    for day in ('2026-07-06', '2026-07-07', '2026-08-03', '2026-08-04'):
        for index in range(10):
            actor = 'B' + str(index)
            pairs.append(dict(a='A', b=actor, day=day, y=index >= 6))
            activity.append(dict(actor=actor, turn_id=day + actor, time=stamp(day, 9)))
            if index:
                signals.append(signal('A', actor, day, 10, weight=float(index), suffix=day + actor))
            if index >= 6:
                requests.append(dict(a='A', b=actor, time=stamp(day, 11 if index == 9 else 9)))
        activity.append(dict(actor='A', turn_id=day + 'A', time=stamp(day, 9)))
    return signals, pd.DataFrame(pairs), activity, requests


class SignalTests(unittest.TestCase):
    def test_support_availability_delays_signal_and_changes_day(self):
        row = signal(day='2026-07-31')
        row['evidence_turns'][1]['time'] = stamp('2026-08-03')
        normalized, info = normalize_signals([row])
        self.assertEqual(normalized[0]['day'], '2026-08-03')
        self.assertEqual(info['availability_delayed_to_support'], 1)

    def test_duplicates_equal_time_sort_and_self_signals(self):
        first, second = signal(suffix='x'), signal(kind='workflow', suffix='y')
        before, info = normalize_signals([first, second, first, signal(a='A', b='A')])
        after, _ = normalize_signals([second, first])
        self.assertEqual(before, after)
        self.assertEqual(info['duplicates_removed'], 1)
        self.assertEqual(info['self_signals_ignored'], 1)

    def test_conflicting_ids_bad_weights_and_unscoped_support_rejected(self):
        for rows in ([signal(), dict(signal(), weight=2)], [dict(signal(), weight=-1)],
                     [dict(signal(), weight=float('nan'))]):
            with self.assertRaises(ValueError):
                normalize_signals(rows)
        row = signal(); row['evidence_turns'][0]['time'] = '2026-09-01T17:00:00Z'
        with self.assertRaises(ValueError):
            normalize_signals([row])


class ReplayTests(unittest.TestCase):
    def test_duplicate_activity_with_iso_and_unix_times_is_counted_once(self):
        signals, pairs, activity, _ = fixtures()
        original, before, _ = replay(signals, pairs, activity_events=activity)
        duplicate = dict(activity[0], time=dt.datetime.fromisoformat(activity[0]['time']).timestamp())
        repeated, after, _ = replay(signals, pairs, activity_events=activity + [duplicate])
        self.assertEqual(original['observation']['activity_turns'], repeated['observation']['activity_turns'])
        pd.testing.assert_frame_equal(before, after)

    def test_conflicting_or_out_of_scope_activity_times_are_rejected(self):
        signals, pairs, activity, _ = fixtures()
        conflicting = dict(activity[0], time=stamp('2026-07-06', 11))
        outside = dict(actor='A', turn_id='outside', time='2026-08-17T17:00:00Z')
        for extra in (conflicting, outside):
            with self.assertRaises(ValueError):
                replay(signals, pairs, activity_events=activity + [extra])

    def test_train_threshold_is_daily_maximum_not_checkpoint_budget(self):
        signals, pairs, activity, requests = fixtures()
        report, snapshots, alerts = replay(signals, pairs, requests, activity)
        result = report['models']['all']
        self.assertLessEqual(result['training']['fpr'], .1)
        self.assertEqual(result['pair_days'], 20)
        self.assertEqual(result['alerts'], len(alerts))
        self.assertGreater(len(snapshots), len(pairs))
        json.dumps(report, allow_nan=False)

    def test_future_validation_append_cannot_change_earlier_features_fit_or_threshold(self):
        signals, pairs, activity, _ = fixtures()
        before, old, _ = replay(signals, pairs, activity_events=activity)
        future = signal('A', 'B0', '2026-08-03', 15, kind='validation_only', weight=1e6, suffix='late')
        after, new, _ = replay(signals + [future], pairs, activity_events=activity)
        self.assertEqual(before['fitted'], after['fitted'])
        cut = dt.datetime.fromisoformat(stamp('2026-08-03', 12)).timestamp()
        pd.testing.assert_frame_equal(old[old.time <= cut], new[new.time <= cut])
        self.assertEqual(after['observation']['unseen_kind_signals_ignored'], 1)

    def test_future_known_signal_only_changes_later_prefixes(self):
        signals, pairs, activity, _ = fixtures()
        before, old, _ = replay(signals, pairs, activity_events=activity)
        future = signal('A', 'B0', '2026-08-03', 15, kind='reuse', weight=100, suffix='laterknown')
        after, new, _ = replay(signals + [future], pairs, activity_events=activity)
        self.assertEqual(before['fitted'], after['fitted'])
        cut = dt.datetime.fromisoformat(stamp('2026-08-03', 12)).timestamp()
        pd.testing.assert_frame_equal(old[old.time <= cut], new[new.time <= cut])
        last = new[(new.day == '2026-08-03') & (new.b == 'B0') & new.is_close]
        self.assertEqual(last.iloc[0]['signal__reuse'], 100)

    def test_no_future_activity_or_pair_universe_can_create_early_alert(self):
        signals, pairs, activity, _ = fixtures()
        activity = [dict(row, time=stamp('2026-08-03', 15))
                    if row['actor'] == 'B9' and row['turn_id'].startswith('2026-08-03') else row for row in activity]
        _, snapshots, _ = replay(signals, pairs, activity_events=activity)
        restricted = snapshots[(snapshots.day == '2026-08-03') & (snapshots.b == 'B9')]
        first = dt.datetime.fromisoformat(stamp('2026-08-03', 15)).timestamp()
        self.assertTrue((restricted.time >= first).all())

    def test_simultaneous_updates_are_batched_order_independently(self):
        signals, pairs, activity, _ = fixtures()
        first, a, _ = replay(signals, pairs, activity_events=activity)
        second, b, _ = replay(list(reversed(signals)) + signals, pairs, activity_events=activity)
        self.assertEqual(first['fitted'], second['fitted'])
        pd.testing.assert_frame_equal(a, b)

    def test_saved_model_replays_empty_validation_stream_without_changing_denominator(self):
        signals, pairs, activity, requests = fixtures()
        original, _, _ = replay(signals, pairs, requests, activity)
        saved = json.loads(json.dumps(original['fitted']))
        validation = pairs[pairs.day >= '2026-08-03']
        changed, snapshots, alerts = replay([], validation, requests, [], fitted=saved)
        self.assertEqual(original['fitted'], changed['fitted'])
        self.assertEqual(changed['mode'], 'frozen model replay')
        self.assertEqual(changed['models']['all']['pair_days'], 20)
        self.assertEqual(changed['models']['all']['alerts'], 0)
        self.assertTrue(alerts.empty)
        val = snapshots[snapshots.day >= '2026-08-03']
        self.assertTrue(val.is_close.all())
        self.assertFalse(val.eligible.any())
        self.assertEqual(len(val), 20)
        retrospective = changed['models']['all']['retrospective_comparison']
        self.assertEqual(retrospective['pair_days'], 20)
        self.assertEqual(retrospective['alerts'], 0)

    def test_retrospective_comparison_has_same_population_and_separate_train_threshold(self):
        signals, pairs, activity, _ = fixtures()
        report, snapshots, _ = replay(signals, pairs, activity_events=activity)
        stream = report['models']['all']
        retrospective = stream['retrospective_comparison']
        self.assertEqual(stream['pair_days'], retrospective['pair_days'])
        self.assertEqual(stream['positives'], retrospective['positives'])
        self.assertEqual(stream['negatives'], retrospective['negatives'])
        self.assertGreaterEqual(stream['threshold'], retrospective['threshold'])
        close = snapshots[(snapshots.day >= '2026-08-03') & snapshots.is_close]
        expected = int((close.eligible & (close['score__all'] > retrospective['threshold'])).sum())
        self.assertEqual(retrospective['alerts'], expected)

    def test_validation_signal_magnitude_does_not_change_training_fit(self):
        signals, pairs, activity, _ = fixtures()
        original, _, _ = replay(signals, pairs, activity_events=activity)
        changed = [dict(row, weight=row['weight'] * 100) if row['time'] >= '2026-08' else row for row in signals]
        after, _, _ = replay(changed, pairs, activity_events=activity)
        self.assertEqual(original['fitted'], after['fitted'])

    def test_pre_request_alerts_report_leads_not_negative_delays(self):
        signals, pairs, activity, requests = fixtures()
        report, _, alerts = replay(signals, pairs, requests, activity)
        timing = report['models']['all']['timing']
        self.assertGreater(timing['pre_request_alerts']['count'], 0)
        self.assertGreater(timing['at_or_after_request_alerts']['count'], 0)
        self.assertGreaterEqual(timing['pre_request_alerts']['median_seconds'], 0)
        self.assertTrue((alerts.loc[alerts.relation_to_first_request == 'pre_request', 'seconds_from_first_request'] < 0).all())

    def test_no_signal_training_fits_intercept_and_never_alerts(self):
        _, pairs, _, _ = fixtures()
        report, _, alerts = replay([], pairs, activity_events=[])
        self.assertTrue(alerts.empty)
        self.assertEqual(report['models']['all']['alerts'], 0)
        self.assertEqual(report['fitted']['models']['all']['features'], [])

    def test_model_name_with_punctuation_and_explicit_feature_sets(self):
        signals, pairs, activity, _ = fixtures()
        report, snapshots, alerts = replay(signals, pairs, activity_events=activity,
                                          feature_sets={'general + workflow': ['reuse'], 'activity': ['activity_min']})
        self.assertIn('general + workflow', report['models'])
        self.assertIn('score__general + workflow', snapshots)

    def test_bad_pair_scope_labels_duplicate_and_schedule_rejected(self):
        signals, pairs, activity, _ = fixtures()
        for frame in (pd.concat([pairs, pairs.iloc[:1]]), pairs.assign(day='2026-08-17'), pairs.assign(y='yes')):
            with self.assertRaises(ValueError):
                replay(signals, frame, activity_events=activity)
        with self.assertRaises(ValueError):
            replay(signals, pairs, activity_events=activity, snapshot_seconds=700)


if __name__ == '__main__':
    unittest.main()
