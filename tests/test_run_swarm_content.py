"""Production orchestration with small synthetic compressed records only."""
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_swarm_content.py'
SPEC = importlib.util.spec_from_file_location('content_runner', SCRIPT)
runner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ContentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.jsonl.gz'
        self.operations = self.root / 'payload-operation-coverage.csv.gz'
        self.manifest = self.root / 'operation-manifest.json'
        self.development = self.root / 'development.json'
        self.protocol = self.root / 'protocol.md'; self.protocol.write_text('Synthetic frozen protocol.\n')
        self.output = self.root / 'census'
        artifact = '{"summary":"I am the coordinator", "path":"report.json"}'
        self.rows = [dict(id='a', kind='payload', parent_id=None, text='fetch("https://example.invalid")'),
                     dict(id='b', kind='payload', parent_id=None, text='fetch("https://example.invalid")'),
                     dict(id='c', kind='payload', parent_id=None, text='prefix ' + artifact + ' suffix'),
                     dict(id='d', kind='payload', parent_id=None, text='ordinary source'),
                     dict(id='r', kind='response', parent_id='c', text=artifact),
                     dict(id='n', kind='recovered_text', parent_id='r', text=None),
                     dict(id='u', kind='note', parent_id=None, text={'not': 'an eligible text field'})]
        with gzip.open(self.source, 'wt') as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + '\n')
        self.source_hash = runner.sha(self.source)
        self.write_operations()
        self.development.write_text(json.dumps(dict(source_sha256=self.source_hash,
            candidates=[dict(record_id='a', text_sha256=digest(self.rows[0]['text']))])))

    def tearDown(self):
        self.directory.cleanup()

    def write_operations(self, omit=()):
        with gzip.open(self.operations, 'wt', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['record_id', 'payload_sha256', 'operation_bearing_payload'])
            writer.writeheader()
            for row in self.rows:
                if row['kind'] == 'payload' and row['id'] not in omit:
                    writer.writerow(dict(record_id=row['id'], payload_sha256=digest(row['text']),
                                         operation_bearing_payload=row['id'] in {'a', 'b'}))
        self.manifest.write_text(json.dumps(dict(source_sha256=self.source_hash,
            outputs_sha256={self.operations.name: runner.sha(self.operations)})))

    def run_phase(self, **kwargs):
        return runner.run(self.source, self.operations, self.manifest, self.development,
                          self.protocol, self.output, expected_source=self.source_hash, **kwargs)

    def test_lock_precedes_source_decode_and_complete_root_census_is_separate_from_views(self):
        original = runner.records
        def checked(path):
            self.assertTrue((self.output / 'pre-census-lock.json').is_file())
            self.assertTrue((self.output / 'code-snapshot/src/swarmcollab/swarm_views.py').is_file())
            return original(path)
        with patch.object(runner, 'records', side_effect=checked):
            result = self.run_phase()
        self.assertEqual(result['views']['payload_records'], 4)
        self.assertEqual(result['views']['exact_unique_payload_bodies'], 3)
        self.assertEqual(result['views']['direct_child_records'], 1)
        self.assertEqual(result['views']['payload_roots_with_contained_artifact_fields'], 1)
        self.assertEqual(result['views']['excluded_record_counts'],
                         {'no_direct_payload_parent': 1, 'unsupported_record_kind': 1})
        self.assertTrue(result['source_file_bytes_verified'])
        self.assertIsNone(result['actual_cooperation_rate'])
        manifest = json.loads((self.output / 'manifest.json').read_text())
        for relative, expected in manifest['outputs_sha256'].items():
            self.assertEqual(runner.sha(self.output / relative), expected)
        lock = json.loads((self.output / 'pre-census-lock.json').read_text())
        self.assertEqual(lock['inputs']['protocol']['sha256'], runner.sha(self.protocol))

    def test_audit_context_has_complete_body_and_provenance_but_no_detector_flags(self):
        self.run_phase()
        selection = json.loads((self.output / 'audit-primary-selection.json').read_text())
        self.assertEqual({row['record_id'] for row in selection['selection']}, {'c', 'd'})
        self.assertEqual(selection['excluded_development_bodies'], 1)
        with gzip.open(self.output / 'audit-primary-context.jsonl.gz', 'rt') as stream:
            contexts = [json.loads(line) for line in stream]
        forbidden = {'claim_candidate', 'artifact_view_available', 'operation_bearing_payload',
                     'category', 'role_or_task_allocation', 'publication_intent', 'result_request_intent',
                     'stratum', 'candidate_member_ids'}
        def inspect(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & set(value))
                for item in value.values(): inspect(item)
            elif isinstance(value, list):
                for item in value: inspect(item)
        inspect(contexts)
        c = next(row for row in contexts if row['record_id'] == 'c')
        self.assertEqual(c['full_raw_body'], self.rows[2]['text'])
        self.assertTrue(c['related_contained_views'])
        for view in c['related_contained_views']:
            self.assertEqual(c['full_raw_body'][slice(*view['root_span'])], view['source_text'])
        forms = json.loads((self.output / 'audit-primary-annotations.json').read_text())['annotations']
        self.assertTrue(all(row['role_or_task_allocation'] == 'unreviewed' for row in forms))
        self.assertTrue(all(row['body_fully_reviewed'] is None for row in forms))

    def test_verified_but_incomplete_operation_population_fails_and_is_preserved(self):
        self.write_operations(omit={'d'})
        with self.assertRaisesRegex(ValueError, 'complete source payload population'):
            self.run_phase()
        self.assertTrue((self.output / 'failed.json').is_file())
        self.assertTrue((self.output / 'pre-census-lock.json').is_file())
        self.assertFalse((self.output / 'summary.json').exists())
        with self.assertRaises(FileExistsError):
            self.run_phase()

    def test_operation_source_and_csv_hash_mismatches_prevent_census(self):
        value = json.loads(self.manifest.read_text()); value['source_sha256'] = 'f' * 64
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'source identity differs'):
            self.run_phase()
        self.write_operations()
        value = json.loads(self.manifest.read_text()); value['outputs_sha256'][self.operations.name] = 'f' * 64
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.run_phase()
        self.assertFalse(self.output.exists())

    def test_source_identity_and_development_source_are_bound(self):
        with self.assertRaisesRegex(ValueError, 'Source bytes differ'):
            runner.run(self.source, self.operations, self.manifest, self.development,
                       self.protocol, self.output, expected_source='f' * 64)
        value = json.loads(self.development.read_text()); value['source_sha256'] = 'f' * 64
        self.development.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Development candidate source'):
            self.run_phase()

    def test_development_exclusions_require_real_unique_matching_payload_ids(self):
        original = json.loads(self.development.read_text())
        for candidates in ([dict(record_id='missing', text_sha256=digest(self.rows[0]['text']))],
                           [dict(record_id='a', text_sha256='f' * 64)],
                           original['candidates'] * 2):
            self.development.write_text(json.dumps(dict(source_sha256=self.source_hash, candidates=candidates)))
            with self.assertRaisesRegex(ValueError, 'candidate ID/hash'):
                self.run_phase()
        self.assertFalse(self.output.exists())

    def test_metadata_mutation_during_loading_prevents_source_census(self):
        original = runner.development_hashes
        def changed(*args):
            result = original(*args)
            self.protocol.write_text('Mutated protocol.\n')
            return result
        with patch.object(runner, 'development_hashes', side_effect=changed), \
                patch.object(runner, 'records', side_effect=AssertionError('No source census')):
            with self.assertRaisesRegex(ValueError, 'input changed while loading'):
                self.run_phase()
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
