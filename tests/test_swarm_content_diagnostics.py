import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'scripts/summarize_swarm_content_diagnostics.py'
SPEC = importlib.util.spec_from_file_location('content_diagnostics', PATH)
module = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(module)


class DiagnosticTests(unittest.TestCase):
    def fixtures(self):
        rows = [dict(record_id=key, payload_sha256=body, operation_bearing_payload=syntax,
                     claim_candidate=True, role=True, artifact_view_available=key == 'b')
                for key, body, syntax in [('a', 'old', True), ('b', 'new', False)]]
        def event(key, kind, span, transformed=False):
            return dict(record_id=key, category='role', rule='rule', view_kind=kind,
                        origin_kind='payload' if kind == 'root_source' else 'response',
                        text_is_exact_source_substring=not transformed, raw_payload_match_span=span,
                        physical_evidence_id='physical', text_sha256='text', start=0, end=3,
                        counts_toward_payload_candidate=True, containment='unique')
        evidence = [event('a', 'root_source', [1, 4]), event('a', 'direct_child', [1, 4]),
                    event('b', 'structured_field', None, True)]
        artifacts = [dict(record_id='b', artifact_signal='summary', counts_toward_payload_candidate=True,
                          origin_kind='response', text_is_exact_source_substring=False)]
        return rows, evidence, artifacts

    def test_views_never_multiply_roots_and_duplicate_raw_matches_collapse(self):
        rows, evidence, artifacts = self.fixtures()
        result = module.calculate(rows, evidence, [], artifacts, [], ['old'], ['role'])
        self.assertEqual(result['claim_routes']['eligible_any_view']['payload_records'], 2)
        self.assertEqual(result['claim_routes']['extra_view_only']['payload_records'], 1)
        self.assertEqual(result['claim_routes']['extra_view_only']['bodies_outside_prior_development'], 1)
        self.assertEqual(result['claim_routes']['distinct_rule_raw_span_or_transformed_container_matches'], 2)
        self.assertEqual(result['claim_routes']['categories']['role']['syntax_coverage_among_category_candidates'], .5)
        self.assertEqual(result['artifacts']['extra_child_only_roots']['payload_records'], 1)

    def test_saved_flags_must_agree_with_evidence(self):
        rows, evidence, artifacts = self.fixtures()
        rows[0]['claim_candidate'] = False
        with self.assertRaisesRegex(ValueError, 'candidate flags'):
            module.calculate(rows, evidence, [], artifacts, [], [], ['role'])

    def test_duplicate_body_counts_stay_separate_from_record_counts(self):
        rows, evidence, artifacts = self.fixtures()
        rows[1]['payload_sha256'] = 'old'
        result = module.calculate(rows, evidence, [], artifacts, [], ['old'], ['role'])
        self.assertEqual(result['claim_routes']['eligible_any_view']['payload_records'], 2)
        self.assertEqual(result['claim_routes']['eligible_any_view']['exact_unique_bodies'], 1)
        self.assertEqual(result['claim_routes']['eligible_any_view']['bodies_outside_prior_development'], 0)

    def test_extra_view_category_is_not_hidden_by_a_different_raw_category(self):
        rows, evidence, artifacts = self.fixtures()
        for row in rows:
            row['second'] = row['record_id'] == 'a'
        extra = dict(evidence[0], category='second', view_kind='structured_field',
                     text_is_exact_source_substring=False, raw_payload_match_span=None)
        result = module.calculate(rows, evidence + [extra], [], artifacts, [], [], ['role', 'second'])
        self.assertEqual(result['claim_routes']['categories']['second']['raw_root_match_records'], 0)
        self.assertEqual(result['claim_routes']['categories']['second']['extra_view_only_records'], 1)


if __name__ == '__main__':
    unittest.main()
