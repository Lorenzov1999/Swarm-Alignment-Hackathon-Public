"""Synthetic records only: no Swarm payloads or investigation outcomes read."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from swarmcollab.swarm_static import analyze_records, extract_references, run_file


def configuration():
    return dict(format='jsonl',compression='gzip',source_sha256='a'*64,
                fields=dict(id='id',kind='kind',parent_id='parent_id',text='text'),
                payload_kind='payload',response_kind='response',excluded_kinds=['recovered_text'])


def record(record_id,text='',kind='payload',parent=None):
    return dict(id=record_id,kind=kind,parent_id=parent,text=text,tags='ignored',cite='ignored')


class ReferenceTests(unittest.TestCase):
    def test_redacted_host_path_query_and_fragment_never_leave_url_prefixes(self):
        samples = ['https://[SERVICE 2 URL 1]/a',
                   'https://example.invalid/[REDACTED:destination:000002]',
                   'https://example.invalid/[REDACTED:destination:000003]',
                   'https://example.invalid/?q=[SERVICE 2 URL 1]',
                   'https://example.invalid/#x=[REDACTED:runtime_identifier]',
                   'https://example.invalid/[REDACTED:destination:000002]/https://nested.invalid/x',
                   'https://example.invalid/[REDACTED:unfinished']
        for sample in samples:
            with self.subTest(sample=sample):
                found,excluded = extract_references(sample)
                self.assertEqual(found['url'],set())
                self.assertEqual(excluded.get('redacted_or_bracketed_url_spans'),1)

    def test_opaque_aliases_keep_case_numbering_and_leading_zeros_exact(self):
        text='[SERVICE 01 URL 002] [SERVICE 1 URL 2] [REDACTED:destination:000002] [REDACTED:destination:2] [service 1 URL 2]'
        refs,_=extract_references(text)
        self.assertEqual(refs['opaque_alias'],{'[SERVICE 01 URL 002]','[SERVICE 1 URL 2]',
                                             '[REDACTED:destination:000002]','[REDACTED:destination:2]'})
        self.assertEqual(refs['url'],set())

    def test_braces_escapes_and_encoded_redactions_never_leave_url_prefixes(self):
        samples=['fetch("https://example.invalid/a/{token}")',
                 'https://example.invalid/{token}/https://nested.invalid/a',
                 'https://example.invalid/a/{unfinished',
                 r'https://example.invalid/path\u003fquery',
                 'https://example.invalid/a/%5BREDACTED:destination:1%5D',
                 'https://example.invalid/a/%5BCREDENTIAL%201%5D']
        for sample in samples:
            with self.subTest(sample=sample):
                refs,_=extract_references(sample)
                self.assertFalse(refs['url'])
        refs,_=extract_references('https://example.invalid/a/%5Bordinary%5D')
        self.assertEqual(refs['url'],{'https://example.invalid/a/%5Bordinary%5D'})

    def test_credential_runtime_and_source_identifiers_do_not_become_resources(self):
        text='[CREDENTIAL 1] [REDACTED:runtime_identifier] [REDACTED:runtime_identifier:000001] [REDACTED:source_identifier:000002]'
        text+=' https://u:secret@example.invalid/a https://example.invalid/?token=hidden'
        refs,_=extract_references(text)
        self.assertEqual(refs,dict(url=set(),opaque_alias=set(),local_path=set()))

    def test_unredacted_url_canonicalization_preserves_resource_distinctions(self):
        refs,_=extract_references('HTTPS is not a URL; https://EXAMPLE.invalid:443/A?q=1#frag https://example.invalid/a?q=1#frag https://[::1]/x')
        self.assertEqual(refs['url'],{'https://example.invalid/A?q=1#frag','https://example.invalid/a?q=1#frag','https://[::1]/x'})

    def test_identifiable_dynamic_literals_are_not_complete_url_or_path_resources(self):
        samples=['fetch("https://example.invalid/" + variable)',
                 'prefix + "https://example.invalid/a"',
                 '`https://example.invalid/${identifier}`',
                 '"data.json" + suffix',
                 'prefix + "data.json"']
        for sample in samples:
            with self.subTest(sample=sample):
                refs,_=extract_references(sample)
                self.assertFalse(refs['url'])
                self.assertFalse(refs['local_path'])

    def test_quoted_path_references_use_existing_validation_without_operations(self):
        refs,_=extract_references('fetch("./results/data.json"); read("/tmp/results.csv"); "report.md"; "/dev/a.txt"; "~/private.json"; "$ROOT/a.py"; "not a path.json"')
        self.assertEqual(refs['local_path'],{'./results/data.json','/tmp/results.csv','report.md'})
        self.assertNotIn('operation',refs)


class AggregateTests(unittest.TestCase):
    def test_duplicate_accounting_has_its_own_complete_denominator_and_recurrence(self):
        text='fetch("https://example.invalid/a"); [SERVICE 1 URL 2]'
        result=analyze_records([record('b',text),record('a',text),record('c','empty')],configuration())
        summary=result['summary'];primary=summary['primary'];dedup=summary['duplicate_sensitivity']['summary']
        self.assertEqual(primary['payload_records'],3)
        self.assertEqual(primary['families']['url']['records_with_reference'],2)
        self.assertAlmostEqual(primary['families']['url']['reference_coverage_per_1000_payload_records'],2000/3)
        self.assertEqual(primary['families']['url']['recurrent_reference_participating_records'],2)
        self.assertEqual(dedup['payload_records'],2)
        self.assertEqual(dedup['families']['url']['records_with_reference'],1)
        self.assertEqual(dedup['families']['url']['reference_coverage_per_1000_payload_records'],500.)
        self.assertEqual(dedup['families']['url']['recurrent_reference_participating_records'],0)
        rows=result['payload_features'].set_index('record_id')
        self.assertEqual(rows.loc['b','duplicate_representative_id'],'a')
        self.assertTrue(rows.loc['a','included_in_duplicate_sensitivity'])
        self.assertFalse(rows.loc['b','included_in_duplicate_sensitivity'])
        self.assertEqual(summary['duplicate_sensitivity']['duplicate_payload_records'],1)

    def test_whole_payload_dedup_is_exact_and_does_not_restore_near_copy(self):
        result=analyze_records([record('a','https://example.invalid/a'),record('b','https://example.invalid/a ')],configuration())
        self.assertEqual(result['summary']['duplicate_sensitivity']['exact_unique_payloads'],2)
        self.assertEqual(result['summary']['primary']['families']['url']['recurrent_reference_participating_records'],2)

    def test_source_hash_namespaces_alias_and_url_identity(self):
        rows=[record('a','https://example.invalid/a [SERVICE 01 URL 002]')]
        one=analyze_records(rows,configuration())
        different=configuration();different['source_sha256']='b'*64
        two=analyze_records(rows,different)
        self.assertFalse(set(one['resource_counts'].identity_id)&set(two['resource_counts'].identity_id))
        self.assertEqual(one['resource_counts'].opaque_alias.dropna().tolist(),['[SERVICE 01 URL 002]'])

    def test_paths_are_scoped_to_payload_root_and_only_direct_response_inherits(self):
        rows=[record('a','"./data.json"'),record('b','"./data.json"'),
              record('ra','"./data.json"',kind='response',parent='a'),
              record('rb','"./data.json"',kind='response',parent='b'),
              record('rr','"./data.json"',kind='response',parent='ra')]
        result=analyze_records(rows,configuration())
        paths=result['resource_counts'].query('family == "local_path"')
        self.assertEqual(len(paths),3)
        self.assertEqual(sorted(paths.payload_record_count),[0,1,1])
        self.assertEqual(result['parent_associations'].local_path_intersection_count.tolist(),[1,1])
        self.assertEqual(result['summary']['parent_associations']['excluded_response_parents'],{'nonpayload_parent':1})
        self.assertIsNone(result['summary']['primary']['families']['local_path']['recurrent_reference_participating_records'])

    def test_parent_intersections_separate_families_and_keep_raw_denominators(self):
        rows=[record('p','https://example.invalid/a [SERVICE 1 URL 1] "x.csv"'),
              record('r','https://example.invalid/a [SERVICE 2 URL 1] "x.csv"',kind='response',parent='p'),
              record('rootresponse','https://example.invalid/a',kind='response')]
        result=analyze_records(rows,configuration());summary=result['summary']['parent_associations']
        self.assertEqual(summary['response_records'],2)
        self.assertEqual(summary['direct_payload_parent_associations'],1)
        self.assertEqual(summary['response_records_with_payload_parent_per_1000_responses'],500.)
        self.assertEqual(summary['families']['url']['associations_with_reference_intersection'],1)
        self.assertEqual(summary['families']['opaque_alias']['associations_with_reference_intersection'],0)
        self.assertEqual(summary['families']['local_path']['associations_with_reference_intersection'],1)
        self.assertEqual(summary['families']['url']['intersection_per_1000_all_response_records'],500.)
        self.assertEqual(summary['families']['url']['intersection_per_1000_direct_payload_parent_associations'],1000.)

    def test_recovered_text_tags_and_cite_are_never_accessed_for_content(self):
        class Guarded(dict):
            def __getitem__(self,key):
                if key in {'text','tags','cite'}:
                    raise AssertionError('Excluded content was accessed')
                return super().__getitem__(key)
        rows=[Guarded(record('derived',kind='recovered_text')),record('p','plain')]
        result=analyze_records(rows,configuration())
        self.assertEqual(result['summary']['excluded_content_records'],1)
        self.assertEqual(result['summary']['primary']['families']['url']['records_with_reference'],0)

    def test_zero_payload_or_parent_denominators_are_null(self):
        result=analyze_records([record('r','https://example.invalid/a',kind='response')],configuration())
        self.assertEqual(result['summary']['primary']['payload_records'],0)
        self.assertIsNone(result['summary']['primary']['families']['url']['reference_coverage_per_1000_payload_records'])
        self.assertIsNone(result['summary']['parent_associations']['families']['url']['intersection_per_1000_direct_payload_parent_associations'])
        self.assertTrue(all(value is None for value in result['summary']['unavailable_metrics'].values()))
        self.assertFalse(any('call' in column for column in result['payload_features']))
        json.dumps(result['summary'],allow_nan=False)

    def test_duplicate_dangling_or_cyclic_ids_fail_before_publication(self):
        bad=[[record('a'),record('a')],[record('a',parent='missing')],
             [record('a',parent='b'),record('b',parent='a')]]
        for rows in bad:
            with self.assertRaises(ValueError):
                analyze_records(rows,configuration())

    def test_record_order_does_not_change_static_results_or_make_time_claims(self):
        rows=[record('p','https://example.invalid/a'),record('r','https://example.invalid/a',kind='response',parent='p')]
        first=analyze_records(rows,configuration());second=analyze_records(list(reversed(rows)),configuration())
        self.assertEqual(first['summary'],second['summary'])
        for table in ['payload_features','resource_counts','parent_associations']:
            pd.testing.assert_frame_equal(first[table],second[table])
        self.assertTrue(all(value is None for value in first['summary']['unavailable_metrics'].values()))


class FileOutputTests(unittest.TestCase):
    def test_source_hash_is_verified_and_outputs_contain_no_raw_urls_or_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.jsonl.gz';cfg=root/'config.json';output=root/'report'
            rows=[record('p','fetch("https://example.invalid/private?q=anonymous"); [SERVICE 1 URL 2]'),
                  record('r','https://example.invalid/private?q=anonymous',kind='response',parent='p')]
            with gzip.open(source,'wt') as stream:
                for row in rows: stream.write(json.dumps(row)+'\n')
            config=configuration();config['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();cfg.write_text(json.dumps(config))
            manifest=run_file(source,cfg,output)
            self.assertTrue(json.loads((output/'summary.json').read_text())['input_byte_hash_validated'])
            self.assertEqual(set(manifest['outputs_sha256']),{'summary.json','payload-features.csv.gz','resource-counts.csv','parent-associations.csv.gz'})
            self.assertIn('swarm_static.py',manifest['code_sha256'])
            for name,expected in manifest['outputs_sha256'].items():
                path=output/name;self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
                value=gzip.open(path,'rt').read() if name.endswith('.gz') else path.read_text()
                self.assertNotIn('https://example.invalid',value)
                self.assertNotIn('fetch(',value)
            with self.assertRaises(FileExistsError):run_file(source,cfg,output)

    def test_wrong_source_hash_fails_before_reading_even_valid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.jsonl.gz';cfg=root/'config.json'
            source.write_bytes(b'not gzip');cfg.write_text(json.dumps(configuration()))
            with self.assertRaisesRegex(ValueError,'SHA-256'):
                run_file(source,cfg,root/'report')
            self.assertFalse((root/'report').exists())

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.jsonl.gz';cfg=root/'config.json'
            with gzip.open(source,'wt') as stream:
                stream.write('{"id":"a","id":"b","kind":"payload","parent_id":null,"text":""}\n')
            config=configuration();config['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();cfg.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Duplicate JSON'):
                run_file(source,cfg,root/'report')
            self.assertFalse((root/'report').exists())

    def test_configuration_changed_during_analysis_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic.jsonl.gz';cfg=root/'config.json'
            with gzip.open(source,'wt') as stream:stream.write(json.dumps(record('a'))+'\n')
            config=configuration();config['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest();cfg.write_text(json.dumps(config))
            def modified(records,loaded):
                result=analyze_records(records,loaded)
                cfg.write_text(json.dumps({**loaded,'changed':True}))
                return result
            with patch('swarmcollab.swarm_static.analyze_records',side_effect=modified):
                with self.assertRaisesRegex(ValueError,'configuration or implementation changed'):
                    run_file(source,cfg,root/'report')
            self.assertFalse((root/'report').exists())


if __name__ == '__main__':
    unittest.main()
