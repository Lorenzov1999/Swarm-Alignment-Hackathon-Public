import datetime as dt
import random
import unittest

from swarmcollab.provenance_common import PACIFIC, make_event, timestamp
from swarmcollab.provenance_controls import direction_diagnostic, shift_actor_day_blocks, time_shift_control
from swarmcollab.provenance_links import build_links


VERSION = 'git:gitlab:team/project@' + 'abcdef01' * 5


def event(actor, local, operation, identity, inputs=None, outputs=None):
    when = dt.datetime.fromisoformat(local).replace(tzinfo=PACIFIC).astimezone(dt.timezone.utc)
    record = {'i': identity, 's': actor + '-session', 't': when.isoformat()}
    return make_event(actor, record, 'git', operation, 'success', 'gitlab:team/project',
                      inputs=inputs, outputs=outputs, relation_authority=True)


class TimeShiftControlTests(unittest.TestCase):
    def test_producer_cannot_move_after_an_unmovable_consumer(self):
        producer = event('A', '2026-07-06T22:30:00', 'generate', 'produce', outputs=[VERSION])
        consumer = event('B', '2026-07-06T23:15:00', 'acquire', 'consume', inputs=[VERSION])
        shifted, report = shift_actor_day_blocks([producer, consumer], random.Random(1), [3600])
        self.assertEqual(report['eligible_blocks'], 1)
        self.assertEqual(report['boundary_ineligible_blocks'], 1)
        self.assertEqual(report['availability_rejected_blocks'], 1)
        self.assertEqual(report['moved_blocks'], 0)
        self.assertEqual(report['availability_violations'], 0)
        self.assertEqual([e['time'] for e in shifted], [producer['time'], consumer['time']])

    def test_blocks_preserve_internal_order_and_never_cross_phase_boundary(self):
        a = event('A', '2026-07-31T09:00:00', 'generate', 'a', outputs=[VERSION])
        b = event('A', '2026-07-31T09:20:00', 'test', 'b', inputs=[VERSION])
        late = event('B', '2026-07-31T23:50:00', 'observe', 'late')
        valid = event('C', '2026-08-03T09:00:00', 'observe', 'valid')
        original = [a, b, late, valid]
        shifted, report = shift_actor_day_blocks(original, random.Random(4), [1800])
        self.assertEqual((timestamp(shifted[1]['time']) - timestamp(shifted[0]['time'])).total_seconds(), 1200)
        self.assertEqual(shifted[2]['time'], late['time'])
        self.assertEqual(report['moved_blocks_by_phase'], {'train': 1, 'validation': 1})
        for before, after in zip(original, shifted):
            self.assertEqual(before['day'], timestamp(after['time']).astimezone(PACIFIC).date().isoformat())
            for key in ('actor', 'operation', 'status', 'inputs', 'outputs', 'event_id'):
                self.assertEqual(before[key], after[key])
        self.assertEqual(a['time'], original[0]['time'])

    def test_preexisting_content_is_not_assigned_a_false_birth_time(self):
        old = event('B', '2026-07-06T09:00:00', 'observe', 'seen', inputs=[VERSION])
        repeated = event('A', '2026-07-06T10:00:00', 'generate', 'again', outputs=[VERSION])
        _, report = shift_actor_day_blocks([old, repeated], random.Random(1), [1800])
        self.assertEqual(report['preexisting_keys_without_birth_bound'], 1)
        self.assertEqual(report['known_availability_keys'], 0)

    def test_degenerate_count_and_edge_null_reported_despite_movement(self):
        producer = event('A', '2026-07-06T09:00:00', 'generate', 'produce', outputs=[VERSION])
        consumer = event('B', '2026-07-06T12:00:00', 'acquire', 'consume', inputs=[VERSION])
        report = time_shift_control([producer, consumer], repetitions=3, offsets_seconds=[300])
        self.assertEqual(report['original_confirmed_chains'], 1)
        self.assertTrue(report['count_degenerate'])
        self.assertTrue(report['edge_degenerate'])
        self.assertFalse(report['no_blocks_moved'])
        self.assertTrue(all(draw['moved_blocks'] == 2 for draw in report['draws']))

    def test_out_of_scope_events_are_rejected_before_shifting(self):
        record = event('A', '2026-07-06T09:00:00', 'observe', 'ok')
        record['time'] = '2026-08-18T16:00:00+00:00'
        record['day'] = '2026-08-18'
        with self.assertRaises(ValueError):
            time_shift_control([record])

    def test_no_repetitions_has_no_degeneracy_claim(self):
        report = time_shift_control([], repetitions=0)
        self.assertEqual(report['draws'], [])
        self.assertFalse(report['count_degenerate'])
        self.assertFalse(report['edge_degenerate'])


class DirectionDiagnosticTests(unittest.TestCase):
    def test_valid_chain_passes(self):
        producer = event('A', '2026-07-06T09:00:00', 'generate', 'produce', outputs=[VERSION])
        consumer = event('B', '2026-07-06T10:00:00', 'acquire', 'consume', inputs=[VERSION])
        chains, _ = build_links([producer, consumer], include_exposure=False)
        self.assertTrue(direction_diagnostic(chains)['valid'])

    def test_backwards_same_actor_and_invalid_status_are_detected(self):
        chain = dict(source_actor='A', target_actor='A', source_time='2026-07-06T17:00:00+00:00',
                     target_time='2026-07-06T16:00:00+00:00', delay_seconds=3600,
                     level=3, source_status='success', target_status='failed')
        report = direction_diagnostic([chain])
        self.assertFalse(report['valid'])
        self.assertEqual(report['violations']['backwards'], 1)
        self.assertEqual(report['violations']['same_actor'], 1)
        self.assertEqual(report['violations']['inconsistent_delay'], 1)
        self.assertEqual(report['violations']['confirmed_without_success'], 1)


if __name__ == '__main__':
    unittest.main()
