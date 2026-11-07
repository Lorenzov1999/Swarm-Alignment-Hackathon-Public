"""Synthetic artifact views only; no source export or external reports read."""
from dataclasses import replace
import json
import unittest

from swarmcollab.swarm_views import Limits, build_views, payload_views, structured_fields


def record(key, text='', kind='payload', parent=None):
    return dict(id=key, kind=kind, parent_id=parent, text=text)


class StructuredFieldTests(unittest.TestCase):
    def test_complete_structured_fields_keep_pointers_roles_and_exact_spans(self):
        text = json.dumps({'path': ['a.json', 'b.json'], 'version': 3,
                           'weird/key~': {'summary': 'We coordinate tasks.'},
                           'description': 'No operation is asserted', 'other': 'not retained'})
        result = structured_fields(text)
        self.assertEqual(result['source_class'], 'complete_json')
        fields = result['fields']
        self.assertEqual([f['field_role'] for f in fields], ['path', 'path', 'version', 'summary', 'description'])
        self.assertEqual(fields[3]['json_pointer'], '/weird~1key~0/summary')
        for field in fields:
            start, end = field['source_span']
            self.assertEqual(field['source_text'], text[start:end])
            self.assertEqual(field['source_text'], field['text'])
            self.assertTrue(field['text_is_exact_source_substring'])

    def test_escaped_unicode_and_surrogate_pair_maps_exact_source(self):
        text = r'{"summary":"we\u0020coordinate\n\ud83d\ude00","path":"a\\b.json"}'
        fields = structured_fields(text)['fields']
        self.assertEqual(fields[0]['text'], 'we coordinate\n😀')
        self.assertEqual(fields[1]['text'], 'a\\b.json')
        for field in fields:
            start, end = field['source_span']
            self.assertEqual(field['source_text'], text[start:end])
            self.assertFalse(field['text_is_exact_source_substring'])

    def test_nested_literal_json_keeps_source_mapping_and_pointer_chain(self):
        inner = json.dumps({'summary': 'we coordinate\n😀', 'version': 'v1'})
        text = json.dumps({'artifact': inner})
        fields = structured_fields(text)['fields']
        self.assertEqual(fields[0]['json_pointer_chain'], ['/artifact', '/summary'])
        self.assertEqual(fields[0]['nested_decode_depth'], 1)
        self.assertEqual(fields[0]['text'], 'we coordinate\n😀')
        for field in fields:
            start, end = field['source_span']
            self.assertEqual(field['source_text'], text[start:end])
            self.assertGreater(start, 0)
            self.assertLess(end, len(text))

    def test_second_nested_decode_and_empty_value_keep_valid_spans(self):
        text = json.dumps({'artifact': json.dumps({'result': json.dumps({'summary': '', 'path': 'x.json'})})})
        fields = structured_fields(text)['fields']
        self.assertEqual([f['text'] for f in fields], ['', 'x.json'])
        self.assertEqual(fields[0]['nested_decode_depth'], 2)
        self.assertEqual(fields[0]['source_span'][0], fields[0]['source_span'][1])
        for field in fields:
            self.assertEqual(field['source_text'], text[slice(*field['source_span'])])

    def test_ndjson_requires_every_nonempty_line_complete(self):
        text = '{"summary":"first"}\n\n{"path":"x.json"}\n'
        result = structured_fields(text)
        self.assertEqual(result['source_class'], 'complete_ndjson')
        self.assertEqual([f['ndjson_line_index'] for f in result['fields']], [0, 2])
        for field in result['fields']:
            self.assertEqual(field['source_text'], text[slice(*field['source_span'])])
        self.assertEqual(structured_fields(text + 'not JSON\n')['fields'], [])

    def test_no_prefix_substring_comment_or_program_recovery(self):
        samples = ['const x = {"summary":"we coordinate"};',
                   'prefix {"summary":"we coordinate"}',
                   '{"summary":"we coordinate"} trailing',
                   '/* {"summary":"we coordinate"} */',
                   '{"summary":"first","summary":"second"}',
                   '{"summary":"x","n":NaN}',
                   '{"summary":"x","n":1e999}',
                   '{"summary":"\\ud800"}',
                   '{"summary":"valid"}\n{"path": broken}']
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(structured_fields(text)['fields'], [])

    def test_program_fields_and_base64_never_decoded(self):
        text = json.dumps({'command': json.dumps({'summary': 'blocked'}),
                           'code': {'description': 'blocked too'},
                           'base64': 'eyJzdW1tYXJ5IjoiYmxvY2tlZCJ9',
                           'text': '__import__("os").system("false")',
                           'other': 'eyJzdW1tYXJ5IjoiYmxvY2tlZCJ9'})
        fields = structured_fields(text)['fields']
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0]['field_role'], 'text')
        self.assertEqual(fields[0]['nested_decode_depth'], 0)

    def test_depth_size_node_and_field_limits_are_explicit(self):
        text = '{"summary":"one","description":"two"}'
        result = structured_fields(text, limits=replace(Limits(), max_fields_per_record=1))
        self.assertEqual(len(result['fields']), 1)
        self.assertEqual(result['diagnostics']['field_limit_reached'], 1)
        self.assertEqual(structured_fields(text, limits=replace(Limits(), max_json_chars=10))['source_class'], 'json_size_limit')
        for limits in [replace(Limits(), max_json_depth=1), replace(Limits(), max_json_nodes=2)]:
            self.assertEqual(structured_fields('{"a":{"summary":"deep"}}', limits=limits)['fields'], [])
        nested = json.dumps({'artifact': json.dumps({'summary': 'nested'})})
        result = structured_fields(nested, limits=replace(Limits(), max_nested_decodes=0))
        self.assertFalse(result['fields'])
        self.assertEqual(result['diagnostics']['nested_decode_limit_reached'], 1)


class ParentViewTests(unittest.TestCase):
    def test_unique_child_span_and_artifact_availability_do_not_rebuild_parent(self):
        artifact = '{"summary":"We coordinate tasks", "path":"x.json"}'
        text = 'fetch("https://example.invalid")\n' + artifact + '\nmalformed tail'
        result = payload_views(record('p', text), [record('c', artifact, 'recovered_text', 'p')])
        root_view = result['views'][0]
        self.assertEqual(root_view['text'], text)
        self.assertEqual(root_view['source_class'], 'unstructured_source')
        self.assertFalse(result['payload']['root_structured_artifact_available'])
        self.assertTrue(result['payload']['artifact_view_available'])
        child_views = [v for v in result['views'] if v['origin_record_id'] == 'c']
        self.assertEqual(len(child_views), 3)
        for view in child_views:
            self.assertEqual(text[slice(*view['root_span'])], view['source_text'])
            self.assertEqual(view['execution_status'], 'not_established')

    def test_ambiguous_and_uncontained_child_views_never_get_unique_root_span(self):
        artifact = '{"summary":"same"}'
        rows = [record('p', artifact + '\nseparator\n' + artifact),
                record('a', artifact, 'recovered_text', 'p'),
                record('b', '{"description":"different"}', 'response', 'p')]
        result = build_views(rows)
        self.assertFalse(result['payloads'][0]['artifact_view_available'])
        self.assertEqual([a['containment'] for a in result['associations']], ['ambiguous', 'not_contained'])
        for view in result['views'][1:]:
            self.assertIsNone(view['root_span'])
            self.assertIsNone(view['physical_evidence_id'])
            self.assertTrue(view['association_only'])

    def test_duplicate_child_views_share_physical_identity_not_denominator(self):
        artifact = '{"summary":"We coordinate"}'
        rows = [record('p', artifact), record('a', artifact, 'response', 'p'),
                record('b', artifact, 'recovered_text', 'p'), record('q', artifact)]
        result = build_views(rows)
        self.assertEqual(result['summary']['payload_records'], 2)
        self.assertEqual(result['summary']['exact_unique_payload_bodies'], 1)
        source_views = [v for v in result['views'] if v['root_id'] == 'p' and v['view_kind'] != 'structured_field']
        self.assertEqual(len(source_views), 3)
        self.assertEqual(len({v['physical_evidence_id'] for v in source_views}), 1)
        self.assertEqual(len({v['view_id'] for v in source_views}), 3)
        self.assertNotEqual(source_views[0]['physical_evidence_id'], result['views'][-2]['physical_evidence_id'])

    def test_empty_and_overlapping_children_have_explicit_ambiguity(self):
        group = payload_views(record('p', 'aaaa'), [record('a', 'aa', 'response', 'p'),
                                                   record('b', '', 'recovered_text', 'p')],
                              limits=replace(Limits(), max_child_occurrences=2))
        self.assertEqual(group['associations'][0]['root_span_candidates'], [[0, 2], [1, 3]])
        self.assertTrue(group['associations'][0]['occurrences_truncated'])
        self.assertEqual(group['associations'][1]['containment'], 'empty_child')

    def test_non_direct_or_unknown_record_text_is_not_inspected(self):
        class Guarded(dict):
            def __getitem__(self, key):
                if key in {'text', 'tags', 'cite'}:
                    raise AssertionError('Excluded content accessed')
                return super().__getitem__(key)
        rows = [record('p', 'source'), record('r', 'response', 'response', 'p'),
                Guarded(record('nested', kind='recovered_text', parent='r')),
                Guarded(record('unknown', kind='investigator_note', parent='p'))]
        result = build_views(rows)
        self.assertEqual(result['summary']['direct_child_records'], 1)
        self.assertEqual(result['summary']['excluded_children'], {'no_direct_payload_parent': 1})

    def test_source_size_cap_keeps_root_denominator_and_source_hash(self):
        result = build_views([record('p', 'a' * 11)], limits=replace(Limits(), max_source_chars=10))
        self.assertEqual(result['summary']['payload_records'], 1)
        self.assertFalse(result['payloads'][0]['text_available'])
        self.assertFalse(result['views'])
        self.assertFalse(result['summary']['source_file_bytes_verified'])
        one = build_views([record('p', 'source')], source_sha256='a' * 64)
        two = build_views([record('p', 'source')], source_sha256='b' * 64)
        self.assertNotEqual(one['views'][0]['view_id'], two['views'][0]['view_id'])

    def test_invalid_graph_duplicates_and_source_identity_fail_closed(self):
        for rows in [[record('p'), record('p')], [record('p', parent='missing')],
                     [record('p', parent='q'), record('q', parent='p')]]:
            with self.assertRaises(ValueError):
                build_views(rows)
        with self.assertRaises(ValueError):
            build_views([], source_sha256='unverified')
        with self.assertRaises(ValueError):
            payload_views(record('p'), [record('a', 'x', 'response', 'other')])


if __name__ == '__main__':
    unittest.main()
