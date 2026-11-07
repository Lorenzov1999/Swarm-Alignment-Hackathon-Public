"""Synthetic tests only; the profiler never treats strings as clock receipts."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/swarm_timestamp_profile.py'
spec=importlib.util.spec_from_file_location('timestamp_profile',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TimestampCandidateTests(unittest.TestCase):
    def test_explicit_offset_and_missing_zone_remain_distinct(self):
        rows=list(module.candidates('2020-01-02T03:04:05Z 2020-01-02 03:04:05'))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['utc_value'],'2020-01-02T03:04:05+00:00')
        self.assertIsNone(rows[1]['utc_value'])
        self.assertEqual(rows[1]['timezone_status'],'missing')

    def test_date_is_not_duplicated_inside_datetime(self):
        rows=list(module.candidates('2020-01-02T03:04:05.123456789+0200'))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['utc_value'],'2020-01-02T01:04:05.123456+00:00')

    def test_rfc_date_header_is_context_not_receipt(self):
        row=list(module.candidates('Date: Thu, 02 Jan 2020 03:04:05 GMT\nServer: test'))[0]
        self.assertEqual(row['header_context'],'date')
        self.assertEqual(row['utc_value'],'2020-01-02T03:04:05+00:00')
        self.assertNotIn('observation_time',row)

    def test_numeric_ids_are_ignored_and_units_not_invented(self):
        rows=list(module.candidates('{"task_id":1577934245,"timestamp":1577934245,"created_at_ms":1577934245000}'))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['field_name'],'timestamp')
        self.assertIsNone(rows[0]['explicit_unit'])
        self.assertEqual(rows[1]['explicit_unit'],'milliseconds')
        self.assertTrue(all(r['utc_value'] is None for r in rows))

    def test_invalid_date_and_ambiguous_timezone_retained_unresolved(self):
        rows=list(module.candidates('2020-02-31 02:03:04 02 Jan 2020 03:04:05 CST'))
        self.assertEqual(rows[0]['parse_status'],'invalid_calendar_or_clock')
        self.assertEqual(rows[1]['timezone_status'],'ambiguous_named_zone')
        self.assertTrue(all(r['utc_value'] is None for r in rows))

    def test_code_and_expiry_strings_are_candidates_not_executed(self):
        rows=list(module.candidates('const fixture="2020-01-02T03:04:05Z"; expires=1577934245'))
        self.assertEqual(len(rows),2)
        self.assertIn('fixture',rows[0]['lexical_context_hints'])
        self.assertEqual(rows[1]['field_name'],'expires')


if __name__=='__main__':unittest.main()
