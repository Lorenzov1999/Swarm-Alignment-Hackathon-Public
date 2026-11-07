import copy
import hashlib
import json
import unittest

from swarmcollab.swarm_claims import (
    audit_plan, blank_annotations, classify_views, detect_claims,
    detector_specification, review_rubric, select_candidate_audit,
    select_unmarked_audit, summarize_claims)
from swarmcollab.swarm_views import build_views, Limits


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(identifier, text, kind='payload', parent=None):
    return dict(id=identifier, kind=kind, parent_id=parent, text=text)


def sample_row(index, syntax=False, artifact=False, marked=False, body=None):
    return dict(record_id='p' + str(index).zfill(4), payload_sha256=digest(str(index) if body is None else body),
                operation_bearing_payload=syntax, artifact_view_available=artifact, claim_candidate=marked)


class ClaimRuleTests(unittest.TestCase):
    def test_role_request_and_publication_are_separate_candidates(self):
        cases = {'teammate owns this task': 'role_or_task_allocation',
                 'I am the coordinator': 'role_or_task_allocation',
                 'delegate this task to another agent': 'role_or_task_allocation',
                 'I will handle the files and you can work on the tests': 'role_or_task_allocation',
                 'publish the findings': 'publication_intent',
                 'please send your results': 'result_request_intent',
                 'give me the summary': 'result_request_intent'}
        for text, category in cases.items():
            with self.subTest(text=text):
                self.assertIn(category, {m['category'] for m in detect_claims(text)})
                for match in detect_claims(text):
                    self.assertEqual(text[match['start']:match['end']], match['literal'])
                    self.assertEqual(match['status'], 'candidate_requires_review')

    def test_past_tense_negation_and_quotes_remain_unresolved_candidates(self):
        for verb in ('published', 'uploaded', 'shared', 'posted', 'submitted'):
            for text in (verb + ' the results', 'We never ' + verb + ' the results',
                         'Example: "' + verb + ' the results"'):
                self.assertIn('publication_intent', {m['category'] for m in detect_claims(text)})
                self.assertFalse(any('confirmed' in m for m in detect_claims(text)))

    def test_operations_keywords_and_resource_equality_are_not_claims(self):
        for text in ('POST /documents', 'shared library', 'api/collaborators',
                     'geometry.coordinates', 'peer socket', 'fetch("https://a.test/report")',
                     'upload_file(path)', 'version: 1.2.3'):
            self.assertEqual(detect_claims(text), [], text)

    def test_bounded_gap_and_literal_offsets(self):
        self.assertFalse(detect_claims('delegate ' + 'x' * 130 + ' to another agent'))
        text = '\N{SNOWMAN} TEAMMATE owns the task'
        self.assertEqual(detect_claims(text)[0]['start'], 2)
        self.assertEqual(detect_claims('teammate; owns task'), [])


class ClaimViewTests(unittest.TestCase):
    def test_decoded_structured_signal_has_no_invented_raw_match_offsets(self):
        text = '{"summary":"teammate\\u0020owns the task","path":"src/a.py","version":"v2"}'
        result = classify_views(build_views([record('p', text)]))
        self.assertEqual(result['summary']['cohorts']['all_payload_records']['denominator'], 1)
        row = result['payloads'][0]
        self.assertTrue(row['role_or_task_allocation'])
        self.assertTrue(row['artifact_view_available'])
        decoded = [e for e in result['evidence'] if not e['text_is_exact_source_substring']]
        self.assertTrue(decoded)
        self.assertIsNone(decoded[0]['raw_payload_match_span'])
        self.assertEqual({a['artifact_signal'] for a in result['artifact_signals']}, {'summary', 'path', 'version'})
        self.assertIsNone(result['summary']['actual_cooperation_rate'])

    def test_noncontained_or_ambiguous_children_never_expand_root_flags(self):
        child = '{"summary":"teammate\\u0020owns the task"}'
        for root in ('ordinary payload', child + '\n' + child):
            result = classify_views(build_views([record('p', root), record('c', child, 'response', 'p')]))
            # Repeated JSON lines independently yield root fields; ensure only
            # child evidence is association-only when containment is ambiguous.
            child_evidence = [e for e in result['association_evidence'] if e['origin_record_id'] == 'c']
            self.assertTrue(child_evidence)
            self.assertFalse(any(e['origin_record_id'] == 'c' for e in result['evidence']))
            if root == 'ordinary payload':
                self.assertFalse(result['payloads'][0]['claim_candidate'])
                self.assertFalse(result['payloads'][0]['artifact_view_available'])

    def test_unique_contained_child_adds_decoded_candidate_with_provenance(self):
        child = '{"description":"teammate\\u0020owns the task"}'
        root = 'not valid JS or JSON prefix\n' + child + '\nsuffix'
        result = classify_views(build_views([record('p', root), record('c', child, 'recovered_text', 'p')]))
        self.assertTrue(result['payloads'][0]['claim_candidate'])
        self.assertTrue(result['payloads'][0]['artifact_view_available'])
        self.assertEqual({e['origin_record_id'] for e in result['evidence']}, {'c'})
        self.assertEqual(result['summary']['cohorts']['all_payload_records']['denominator'], 1)

    def test_duplicate_bodies_with_different_children_union_only_in_body_summary(self):
        child = '{"summary":"teammate\\u0020owns the task"}'
        body = 'prefix ' + child + ' suffix'
        bundle = build_views([record('a', body), record('b', body), record('c', child, 'response', 'b')])
        ops = [dict(record_id=r, payload_sha256=digest(body), operation_bearing_payload=False) for r in ('a','b')]
        result = classify_views(bundle, ops)
        cohorts = result['summary']['cohorts']
        self.assertEqual(cohorts['all_payload_records']['claim_candidate']['count'], 1)
        self.assertEqual(cohorts['all_payload_records']['denominator'], 2)
        self.assertEqual(cohorts['exact_unique_payload_bodies']['claim_candidate']['count'], 1)
        audit = select_unmarked_audit(result['payloads'])
        self.assertEqual(audit['selection'][0]['record_id'], 'a')
        self.assertEqual(audit['selection'][0]['artifact_view_member_ids'], ['b'])
        self.assertEqual(audit['selection'][0]['stratum'], 'syntax_no__artifact_yes')
        candidates = select_candidate_audit(result['payloads'])
        self.assertEqual(candidates['selection'][0]['candidate_member_ids'], ['b'])

    def test_tampered_hash_span_and_operation_binding_fail(self):
        bundle = build_views([record('p', '{"summary":"hello"}')])
        for key, value in [('text_sha256', '0'*64), ('root_span', [0, 1])]:
            changed = copy.deepcopy(bundle)
            changed['views'][-1][key] = value
            with self.assertRaises(ValueError):
                classify_views(changed)
        with self.assertRaises(ValueError):
            classify_views(bundle, [dict(record_id='p', payload_sha256='0'*64, operation_bearing_payload=True)])
        with self.assertRaises(ValueError):
            classify_views(bundle, [])

    def test_limited_root_is_unknown_and_remains_denominator(self):
        bundle = build_views([record('p', 'long payload')], limits=Limits(max_source_chars=3))
        result = classify_views(bundle)
        self.assertFalse(result['payloads'][0]['text_available'])
        self.assertIn('unavailable', result['payloads'][0]['claim_scan_status'])
        self.assertEqual(result['summary']['cohorts']['all_payload_records']['denominator'], 1)


class AuditSelectionTests(unittest.TestCase):
    def rows(self):
        return [sample_row(1000*s+100*a+i, bool(s), bool(a), marked=(i % 2 == 0))
                for s in (0,1) for a in (0,1) for i in range(30,60)]

    def test_primary_ignores_new_flags_order_invariant_and_predeclared_96(self):
        rows = self.rows()
        first = select_unmarked_audit(rows)
        changed = [dict(row, claim_candidate=not row['claim_candidate']) for row in reversed(rows)]
        second = select_unmarked_audit(changed)
        self.assertEqual(first, second)
        self.assertEqual(len(first['selection']), 96)
        self.assertTrue(all(s['population'] == 30 and s['selected'] == 24 for s in first['strata'].values()))
        self.assertTrue(all(row['inclusion_probability'] == .8 for row in first['selection']))
        for name in first['strata']:
            population = [row for row in rows if
                          ('syntax_' + ('yes' if row['operation_bearing_payload'] else 'no') +
                           '__artifact_' + ('yes' if row['artifact_view_available'] else 'no')) == name]
            expected = sorted(population, key=lambda row: digest('20261107:' + name + ':' + row['payload_sha256']))[:24]
            self.assertEqual({r['payload_sha256'] for r in expected},
                             {r['payload_sha256'] for r in first['selection'] if r['stratum'] == name})

    def test_development_exclusion_is_whole_body_and_no_quota_redistribution(self):
        rows = [sample_row(1, body='same'), sample_row(2, body='same'), sample_row(3), sample_row(4, True)]
        result = select_unmarked_audit(rows, [digest('same')])
        self.assertEqual(result['excluded_development_bodies'], 1)
        self.assertEqual(result['exact_unique_payloads'], 3)
        self.assertEqual(len(result['selection']), 2)
        self.assertTrue(all(r['inclusion_probability'] == 1 for r in result['selection']))
        self.assertEqual(result['strata']['syntax_yes__artifact_yes']['selected'], 0)
        self.assertIsNone(result['strata']['syntax_yes__artifact_yes']['inclusion_probability'])

    def test_duplicate_syntax_conflict_and_unknown_support_rejected(self):
        with self.assertRaises(ValueError):
            select_unmarked_audit([sample_row(1, body='x'), sample_row(2, True, body='x')])
        with self.assertRaises(ValueError):
            select_unmarked_audit([dict(sample_row(1), operation_bearing_payload=None)])

    def test_targeted_candidates_are_separate_capped_and_union_members(self):
        rows = [sample_row(i, marked=True) for i in range(110)]
        rows += [sample_row(111, marked=False, body='0')]
        result = select_candidate_audit(rows, [digest('1')])
        self.assertEqual(result['population'], 109)
        self.assertEqual(result['selected'], 96)
        self.assertEqual(result, select_candidate_audit(list(reversed(rows)), [digest('1')]))
        self.assertTrue(all(r['inclusion_probability'] == 96/109 for r in result['selection']))
        self.assertNotIn(digest('1'), {r['payload_sha256'] for r in result['selection']})

    def test_empty_population_no_invented_rate_and_blank_form_no_detector_flags(self):
        result = select_unmarked_audit([])
        self.assertEqual(result['selection'], [])
        self.assertEqual(select_candidate_audit([])['selected'], 0)
        form = blank_annotations([sample_row(1)])[0]
        self.assertNotIn('claim_candidate', form)
        self.assertIsNone(form['body_fully_reviewed'])
        self.assertEqual(form['reviewed_extent'], [])
        self.assertEqual(form['role_or_task_allocation'], 'unreviewed')
        self.assertIsNone(summarize_claims([])['cohorts']['all_payload_records']['claim_candidate']['per_1000'])
        json.dumps([audit_plan(), detector_specification(), review_rubric()], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
