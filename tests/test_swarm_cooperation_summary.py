import importlib.util
from pathlib import Path
import unittest
import hashlib
import json
import tempfile

import pandas as pd

spec=importlib.util.spec_from_file_location('swarm_summary',Path(__file__).resolve().parents[1]/'scripts/summarize_swarm_cooperation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CoverageSummaryTests(unittest.TestCase):
    def fixtures(self):
        operations=pd.DataFrame([
            dict(record_id='a',payload_sha256='x',operation_bearing_payload=True,operation_site_count=7),
            dict(record_id='b',payload_sha256='x',operation_bearing_payload=True,operation_site_count=7),
            dict(record_id='c',payload_sha256='y',operation_bearing_payload=False,operation_site_count=0)])
        review=dict(candidates=[dict(record_id=i,text_sha256='x',confirmed_cooperation=False,
            primary_explicit_work_sharing_claim=True,category='work_allocation_claim') for i in ['a','b']])
        return operations,review

    def test_record_not_site_denominator_and_duplicate_sensitivity(self):
        o,r=self.fixtures();out=module.summarize(o,r)
        self.assertEqual(out['cohorts']['all_payload_records']['all']['denominator'],3)
        self.assertEqual(out['cohorts']['all_payload_records']['operation_bearing']['explicit_role_or_allocation_claim']['count'],2)
        self.assertEqual(out['cohorts']['exact_unique_payload_bodies']['all']['denominator'],2)
        self.assertIsNone(out['true_cooperation_rate'])
        self.assertIsNone(out['timeline'])

    def test_missing_or_changed_payload_rejected(self):
        for field,value in [('record_id','missing'),('text_sha256','different')]:
            o,r=self.fixtures();r['candidates'][0][field]=value
            with self.assertRaises(ValueError):module.summarize(o,r)

    def test_duplicate_records_and_inconsistent_duplicate_decisions_rejected(self):
        o,r=self.fixtures()
        with self.assertRaises(ValueError):module.summarize(pd.concat([o,o.iloc[:1]]),r)
        r['candidates'][0]['primary_explicit_work_sharing_claim']=False
        with self.assertRaises(ValueError):module.summarize(o,r)

    def test_unsupported_operation_is_not_excluded_from_full_denominator(self):
        o,r=self.fixtures();o.operation_bearing_payload=False;o.operation_site_count=0
        out=module.summarize(o,r)['cohorts']['all_payload_records']
        self.assertEqual(out['all']['explicit_role_or_allocation_claim']['count'],2)
        self.assertIsNone(out['operation_bearing']['explicit_role_or_allocation_claim']['fraction'])

    def test_denominator_rows_and_source_bound_to_census_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'coverage.csv';path.write_text('record_id\na\nb\n')
            manifest=Path(folder)/'manifest.json'
            manifest.write_text(json.dumps(dict(source_sha256='source',outputs_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()})))
            module.validate_operation_manifest(path,manifest,'source')
            with self.assertRaises(ValueError):module.validate_operation_manifest(path,manifest,'different')
            path.write_text('record_id\na\n')
            with self.assertRaises(ValueError):module.validate_operation_manifest(path,manifest,'source')


if __name__=='__main__':unittest.main()
