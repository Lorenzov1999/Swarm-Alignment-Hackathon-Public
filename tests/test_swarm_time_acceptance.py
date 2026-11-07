import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/swarm_time_acceptance.py'
spec=importlib.util.spec_from_file_location('time_acceptance',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def candidate(family='iso_datetime',context='"created":"[[CANDIDATE:2020-01-01T00:00:00Z]]"',**kw):
    return dict(family=family,context_excerpt=context,parse_status='valid',timezone_status='explicit_offset',utc_value='2020-01-01T00:00:00+00:00',header_context=None,**kw)

class AcceptanceTests(unittest.TestCase):
    def test_http_date_remains_unattributed_not_execution_clock(self):
        r=candidate(family='rfc_datetime');r['header_context']='date';out=module.disposition(r)
        self.assertEqual(out['semantic_category'],'http_generation_clock_unattributed')
        self.assertFalse(out['accepted_original_action_clock'])

    def test_artifact_date_and_service_claim_are_not_payload_time(self):
        self.assertEqual(module.disposition(candidate())['semantic_category'],'artifact_metadata_or_authored_datetime')
        r=candidate(context='"create_time":"[[CANDIDATE:2020-01-01T00:00:00Z]]"')
        self.assertEqual(module.disposition(r)['semantic_category'],'embedded_service_or_document_event_claim')
        self.assertFalse(module.disposition(r)['normalized_literal_is_execution_time'])

    def test_expiry_duration_fixed_archive_and_projection_stay_nonclocks(self):
        for field,expected in [('Expires','expiry_or_duration_claim'),('mtime','archive_metadata_or_literal'),('created_at','unresolved_numeric_field')]:
            r=candidate(family='typed_numeric_time',field_name=field)
            self.assertEqual(module.disposition(r)['semantic_category'],expected)
            self.assertFalse(module.disposition(r)['accepted_original_action_clock'])

    def test_missing_timezone_is_preserved(self):
        r=candidate();r.update(utc_value=None,timezone_status='missing')
        out=module.disposition(r);self.assertIsNone(out['normalized_literal_utc']);self.assertEqual(out['timezone_status'],'missing')

    def test_signing_patterns_require_typed_field_and_complete_literal(self):
        text='id=20200101T000000Z&X-Amz-Date=20200101T000000Z&Timestamp=2020-01-01T00%3A00%3A00Z'
        matches=list(module.SIGNING.finditer(text));self.assertEqual(len(matches),2)
        self.assertFalse(list(module.SIGNING.finditer('Timestamp=2020-01-01T00%3A00%3A')))

if __name__=='__main__':unittest.main()
