"""Synthetic-only checks for exact-window, metadata-first transfer extraction."""
import copy
import datetime as dt
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from swarmcollab import extract
from swarmcollab import transfer_extract as transfer


def record(identifier='turn', stamp='2026-08-17 07:00:00', session='session', action=None, **extra):
    return dict(id=identifier, session_id=session,
                agent_action={'command': 'echo hello'} if action is None else action,
                output='hello', **extra, created_at=stamp)


def encoded(row):
    return (json.dumps(row, ensure_ascii=False) + '\n').encode()


class TimestampGateTests(unittest.TestCase):
    def test_final_timestamp_uses_fast_suffix(self):
        row = record(output_metadata={'created_at': '2026-10-01 12:00:00'})
        stamp, route = transfer.top_level_created_at(encoded(row))
        self.assertEqual(stamp, row['created_at'])
        self.assertEqual(route, 'suffix')

    def test_nested_timestamp_and_strings_cannot_override_root_timestamp(self):
        row = {'created_at': '2026-10-01 12:00:00',
               'agent_action': {'created_at': '2026-08-17 12:00:00'},
               'output': '{"created_at":"2026-08-17 12:00:00"}'}
        stamp, route = transfer.top_level_created_at(encoded(row))
        self.assertEqual(stamp, '2026-10-01 12:00:00')
        self.assertEqual(route, 'structural_scan')
        row['created_at'] = '2026-08-17 12:00:00'
        row['agent_action']['created_at'] = '2026-10-01 12:00:00'
        self.assertEqual(transfer.top_level_created_at(encoded(row))[0], row['created_at'])

    def test_unicode_key_whitespace_and_nonfinal_root_timestamp(self):
        line = b'{ "created_\\u0061t" : "2026-08-17T12:00:00Z", "tail" : [1,{"x":"{}"}] }\n'
        self.assertEqual(transfer.top_level_created_at(line), ('2026-08-17T12:00:00Z', 'structural_scan'))

    def test_invalid_missing_or_nonstring_metadata_rejected(self):
        for line in (b'[]', b'', b'{"nested":{"created_at":"2026-08-17 12:00:00"}}',
                     b'{"created_at":null}', b'{"created_at":{}}', b'{"x":1'):
            with self.subTest(line=line), self.assertRaises(transfer.MetadataError):
                transfer.top_level_created_at(line)

    def test_duplicate_timestamp_gate_uses_last_value(self):
        line = b'{"created_at":"2026-08-17 12:00:00","created_at":"2026-10-01 12:00:00"}'
        self.assertEqual(transfer.top_level_created_at(line)[0], '2026-10-01 12:00:00')
        line = line[:-1] + b',"tail":1}'
        self.assertEqual(transfer.top_level_created_at(line)[0], '2026-10-01 12:00:00')


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='transfer-extraction-test-')
        self.root = Path(self.temporary.name)
        self.turns = self.root / 'turns.jsonl'
        self.agents = self.root / 'agents.jsonl.gz'
        self.sessions = self.root / 'sessions.jsonl.gz'
        self.output = self.root / 'steps'
        self.freeze = self.root / 'freeze.json'
        self.freeze.write_text('{"synthetic_protocol":true}')
        self.write_metadata([{'id': 'agent', 'name': 'Test Agent'}],
                            [{'id': 'session', 'agent_id': 'agent'}])

    def tearDown(self):
        self.temporary.cleanup()

    def write_metadata(self, agents, sessions):
        for path, rows in [(self.agents, agents), (self.sessions, sessions)]:
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                stream.writelines(json.dumps(row) + '\n' for row in rows)

    def write_rows(self, rows):
        self.turns.write_bytes(b''.join(encoded(row) if not isinstance(row, bytes) else row + b'\n'
                                      for row in rows))

    def run_extract(self, period='temporal', **kwargs):
        return transfer.extract_period(period, self.output, turns_path=self.turns,
                                       agents_path=self.agents, sessions_path=self.sessions,
                                       freeze_manifest=self.freeze, progress_every=0, **kwargs)

    def rows_for(self, summary, actor='Test Agent'):
        with gzip.open(self.output / summary['agents'][actor]['file'], 'rt', encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def test_exact_pacific_boundaries_and_excluded_payloads_never_decoded(self):
        rows = [record('before', '2026-08-17 06:59:59.999999'),
                record('first', '2026-08-17 07:00:00'),
                record('last', '2026-09-16 06:59:59.999999'),
                record('reserve', '2026-09-16 07:00:00'),
                record('october', '2026-10-01 16:00:00')]
        self.write_rows(rows)
        decoded = []
        original = transfer._decode_record

        def only_target(line):
            row = original(line)
            self.assertIn(row['id'], {'first', 'last'})
            decoded.append(row['id'])
            return row

        with patch.object(transfer, '_decode_record', side_effect=only_target):
            summary = self.run_extract()
        self.assertEqual(decoded, ['first', 'last'])
        self.assertEqual(summary['steps_in_period'], 2)
        self.assertEqual(summary['timestamp_gate']['outside_period'], 3)
        self.assertEqual([row['i'] for row in self.rows_for(summary)], ['first', 'last'])
        self.assertEqual(set(summary['steps_in_period_by_day']), {'2026-08-17', '2026-09-15'})
        self.assertEqual(summary['utc_bounds']['end_exclusive'], '2026-09-16T07:00:00+00:00')

    def test_nested_in_scope_timestamp_does_not_expose_reserved_payload(self):
        reserved = {'id': 'reserved', 'created_at': '2026-10-01 00:00:00',
                    'agent_action': {'created_at': '2026-08-17 12:00:00'},
                    'output': {'created_at': '2026-08-18 12:00:00'}}
        self.write_rows([reserved, record('allowed')])
        original = transfer._decode_record

        def checked(line):
            self.assertNotIn(b'reserved', line)
            return original(line)

        with patch.object(transfer, '_decode_record', side_effect=checked):
            summary = self.run_extract()
        self.assertEqual(summary['agents']['Test Agent']['kept'], 1)

    def test_shared_goal_window_and_timezone_offsets(self):
        rows = [record('before', '2026-06-01 06:59:59'),
                record('first', '2026-06-01T00:00:00-07:00'),
                record('last', '2026-07-06 06:59:59.999999'),
                record('after', '2026-07-06 07:00:00')]
        self.write_rows(rows)
        summary = self.run_extract('shared_goals')
        self.assertEqual([row['i'] for row in self.rows_for(summary)], ['first', 'last'])
        self.assertEqual(summary['period'], ['2026-06-01', '2026-07-05'])
        self.assertEqual(set(summary['steps_in_period_by_day']), {'2026-06-01', '2026-07-05'})

    def test_scan_continues_after_long_out_of_scope_run_and_sorts_only_disorder(self):
        self.write_rows([record('later', '2026-08-18 12:00:00')]
                        + [record('outside', '2026-10-01 12:00:00')] * 20
                        + [record('earlier', '2026-08-17 12:00:00')])
        with patch.object(extract, 'STOP_AFTER', 2), patch.object(transfer, '_resort', wraps=transfer._resort) as resort:
            summary = self.run_extract()
        resort.assert_called_once()
        self.assertEqual(summary['lines_read'], 22)
        self.assertTrue(summary['complete_source_scan'])
        self.assertIsNone(summary['stopped_early_at_date'])
        self.assertEqual([row['i'] for row in self.rows_for(summary)], ['earlier', 'later'])
        agent = summary['agents']['Test Agent']
        self.assertEqual(agent['first'], '2026-08-17 12:00:00')
        self.assertEqual(agent['out_of_order'], 1)
        with gzip.open(self.output / agent['file'], 'rb') as stream:
            self.assertEqual(hashlib.sha256(stream.read()).hexdigest(), agent['uncompressed_sha256'])

    def test_sorted_cache_does_not_run_resort(self):
        self.write_rows([record('a'), record('b', '2026-08-18 07:00:00')])
        with patch.object(transfer, '_resort', side_effect=AssertionError('Already sorted')):
            summary = self.run_extract()
        self.assertEqual(summary['agents']['Test Agent']['out_of_order'], 0)

    def test_original_action_filter_and_caps_are_reused_exactly(self):
        command, output, error = 'c' * 50010, 'o' * 8010, 'e' * 2010
        row = record('large', action={'command': command}, system='must not be copied',
                     agent_messages=['narration must not be copied'])
        row['output'], row['error'] = output, error
        self.write_rows([row, record('chat', action={'action': 'send_message_back_to_chat', 'text': 'secret'}),
                         record('goal', action={'command': 'echo test', 'sessionGoal': 'secret'}),
                         record('restart', action={'command': '', 'restart': True})])
        summary = self.run_extract()
        records = self.rows_for(summary)
        self.assertEqual(len(records), 2)
        first = records[0]
        self.assertEqual(first['x'], extract.kind_and_payload(row['agent_action'])[1])
        self.assertEqual(first['o'], extract.cap(output, extract.OUT_HEAD, extract.OUT_TAIL))
        self.assertEqual(first['e'], extract.cap(error, extract.ERR_HEAD))
        self.assertEqual(set(first), {'i', 't', 's', 'k', 'x', 'o', 'e'})
        self.assertEqual(records[1]['k'], 'restart')
        values = summary['agents']['Test Agent']
        self.assertEqual(values['truncated_fields'], {'action': 1, 'output': 1, 'error': 1})
        self.assertEqual(values['removed_characters'], {'action': 10, 'output': 10, 'error': 10})
        self.assertEqual(values['dropped'], 2)
        self.assertEqual(values['system_field_present'], 1)

    def test_duplicate_top_level_metadata_rejected_before_actions(self):
        line = b'{"created_at":"2026-08-17 12:00:00","created_at":"2026-08-18 12:00:00", "agent_action":{"command":"ignored"}}'
        self.write_rows([line])
        with patch.object(extract, 'kind_and_payload', side_effect=AssertionError('No action may be interpreted')):
            summary = self.run_extract()
        self.assertEqual(summary['schema_anomalies']['duplicate_top_level_keys'], 1)
        self.assertEqual(summary['agents'], {})

    def test_unknown_session_and_schema_anomalies_are_accounted_for(self):
        missing_id = record()
        missing_id.pop('id')
        self.write_rows([record('unknown', session='unknown'), missing_id,
                         {'id': 'missing_timestamp'}, record('badtime', stamp='not a date'),
                         b'{"invalid_json":,"created_at":"2026-08-17 12:00:00"}', record('valid')])
        summary = self.run_extract()
        self.assertEqual(summary['unknown_session'], 1)
        self.assertEqual(summary['agents']['Test Agent']['kept'], 1)
        self.assertEqual(summary['schema_anomalies'], {
            'missing_or_invalid_turn_id': 1, 'missing_top_level_created_at': 1,
            'invalid_created_at': 1, 'invalid_json_inside_period': 1})

    def test_decoded_timestamp_verification_precedes_action_interpretation(self):
        self.write_rows([record('valid')])
        wrong = transfer._Object(record('wrong', stamp='2026-10-01 12:00:00').items())
        with patch.object(transfer, '_decode_record', return_value=wrong), \
             patch.object(extract, 'kind_and_payload', side_effect=AssertionError('No action')):
            summary = self.run_extract()
        self.assertEqual(summary['schema_anomalies']['timestamp_verification_mismatch'], 1)
        self.assertEqual(summary['agents'], {})

    def test_malformed_action_schema_is_counted_without_losing_valid_rows(self):
        self.write_rows([record('invalid', action={'action': []}), record('valid')])
        summary = self.run_extract()
        self.assertEqual(summary['schema_anomalies']['invalid_action_schema'], 1)
        self.assertEqual([row['i'] for row in self.rows_for(summary)], ['valid'])

    def test_input_changed_during_scan_prevents_publishing(self):
        self.write_rows([record()])
        original = transfer._decode_record

        def change_metadata(line):
            with self.agents.open('ab') as stream:
                stream.write(b'changed')
            return original(line)

        with patch.object(transfer, '_decode_record', side_effect=change_metadata):
            with self.assertRaisesRegex(RuntimeError, 'input changed'):
                self.run_extract()
        self.assertFalse(self.output.exists())

    def test_resume_requires_explicit_matching_completed_cache(self):
        self.write_rows([record()])
        summary = self.run_extract()
        with self.assertRaises(FileExistsError):
            self.run_extract()
        with patch.object(transfer, '_turn_stream', side_effect=AssertionError('Must skip source scan')):
            self.assertEqual(self.run_extract(resume=True), summary)
        self.freeze.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            self.run_extract(resume=True)

    def test_resume_detects_same_size_changed_cache_content(self):
        self.write_rows([record()])
        summary = self.run_extract()
        path = self.output / summary['agents']['Test Agent']['file']
        with gzip.open(path, 'rb') as stream:
            changed = stream.read().replace(b'hello', b'jello')
        before = path.stat().st_size
        with gzip.open(path, 'wb', compresslevel=4) as stream:
            stream.write(changed)
        # gzip's text writer adds a flush block; preserve the size field in the
        # summary to test the stronger content-hash check independently of size.
        summary['agents']['Test Agent']['compressed_bytes'] = path.stat().st_size
        (self.output / 'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'content has changed'):
            self.run_extract(resume=True)

    def test_partial_outputs_original_cache_and_unrecognized_period_are_refused(self):
        self.write_rows([record()])
        self.output.mkdir()
        (self.output / 'existing.txt').write_text('preserve me')
        with self.assertRaises(FileExistsError):
            self.run_extract()
        self.assertEqual((self.output / 'existing.txt').read_text(), 'preserve me')
        with patch.object(transfer.config, 'STEPS', self.output), self.assertRaises(ValueError):
            self.run_extract()
        with self.assertRaises(ValueError):
            self.run_extract('reserve')

    def test_missing_freeze_manifest_refuses_extraction(self):
        self.write_rows([record()])
        self.freeze.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_extract()
        self.assertFalse(self.output.exists())

    def test_failure_does_not_publish_or_leave_partial_cache(self):
        self.write_rows([record()])
        with patch.object(extract, 'kind_and_payload', side_effect=RuntimeError('synthetic failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                self.run_extract()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('steps-partial-*')), [])

    def test_slug_collisions_keep_distinct_actor_caches(self):
        self.write_metadata([{'id': 'a', 'name': 'Same Name'}, {'id': 'b', 'name': 'Same-Name'}],
                            [{'id': 's1', 'agent_id': 'a'}, {'id': 's2', 'agent_id': 'b'}])
        self.write_rows([record('one', session='s1'), record('two', session='s2')])
        summary = self.run_extract()
        self.assertNotEqual(summary['agents']['Same Name']['file'], summary['agents']['Same-Name']['file'])
        self.assertEqual(self.rows_for(summary, 'Same Name')[0]['i'], 'one')
        self.assertEqual(self.rows_for(summary, 'Same-Name')[0]['i'], 'two')

    @unittest.skipUnless(shutil.which('pigz'), 'pigz is not installed')
    def test_compressed_stream_and_corrupt_gzip_failure(self):
        self.turns = self.root / 'turns.jsonl.gz'
        with gzip.open(self.turns, 'wb') as stream:
            stream.write(encoded(record()))
        summary = self.run_extract()
        self.assertEqual(summary['agents']['Test Agent']['kept'], 1)
        self.output = self.root / 'corrupt-output'
        self.turns.write_bytes(self.turns.read_bytes()[:-7])
        with self.assertRaisesRegex(RuntimeError, 'pigz failed'):
            self.run_extract()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
