"""Contract tests for reporting locked synthetic contextual annotations."""
import copy
import csv
import datetime as dt
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/summarize_swarm_content_audits.py'
SPEC=importlib.util.spec_from_file_location('content_audits',SCRIPT)
report=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(report)
SOURCE='f'*64


def packet(identifier='p', text='ordinary source'):
    return dict(record_id=identifier,payload_sha256=report.text_sha(text),source_sha256=SOURCE,
                full_raw_body=text,body_characters=len(text),exact_body_member_ids=[identifier],
                related_contained_views=[])


def selection(p):
    return dict(selection=[dict(record_id=p['record_id'],payload_sha256=p['payload_sha256'],
                                body_member_ids=[p['record_id']],stratum='s',inclusion_probability=1)],
                strata={'s':dict(population=1,selected=1,inclusion_probability=1)})


def annotation(p):
    return dict(record_id=p['record_id'],payload_sha256=p['payload_sha256'],
                **{field:'no_visible_evidence' for field in report.FIELDS},
                context_status='no_visible_explicit_claim',body_fully_reviewed=True,
                reviewed_extent=[dict(record_id=p['record_id'],coordinate_system='raw_body_unicode_characters',
                                      start=0,end=len(p['full_raw_body']))],evidence=[],rationale='Entire supplied body inspected.')


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self.p=packet(); self.s=selection(self.p)
        self.selected,self.contexts=report.validate_selection_context(self.s,[self.p],SOURCE)
        self.row=annotation(self.p)

    def validate(self,row):
        return report.validate_annotations(dict(annotations=[row]),self.selected,self.contexts)

    def test_completed_labels_and_exact_population_are_required(self):
        self.assertEqual(set(self.validate(self.row)),{'p'})
        for bad in [dict(self.row,publication_intent='unreviewed'),
                    {k:v for k,v in self.row.items() if k!='publication_intent'}]:
            with self.assertRaisesRegex(ValueError,'completed'):
                self.validate(bad)
        with self.assertRaisesRegex(ValueError,'missing selected'):
            report.validate_annotations(dict(annotations=[]),self.selected,self.contexts)

    def test_payload_and_context_hashes_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError,'payload hash'):
            self.validate(dict(self.row,payload_sha256='0'*64))
        changed=copy.deepcopy(self.p); changed['full_raw_body']='different source'
        with self.assertRaisesRegex(ValueError,'body/source identity'):
            report.validate_selection_context(self.s,[changed],SOURCE)

    def test_partial_reviews_cannot_receive_any_negative_decision(self):
        partial=dict(self.row,body_fully_reviewed=False,
                     reviewed_extent=[dict(source='full_raw_body',span=[0,3])])
        with self.assertRaisesRegex(ValueError,'Partial reviews'):
            self.validate(partial)
        partial.update({field:'unknown' for field in report.FIELDS})
        partial['context_status']='unresolved_or_truncated'
        self.assertEqual(self.validate(partial)['p']['artifact_evidence'],'unknown')

    def test_full_review_requires_covered_body_without_gaps(self):
        changed=copy.deepcopy(self.row)
        changed['reviewed_extent']=[dict(source='full_raw_body',span=[0,2]),dict(source='full_raw_body',span=[3,15])]
        with self.assertRaisesRegex(ValueError,'complete raw-body extents'):
            self.validate(changed)
        changed['reviewed_extent']=[dict(source='full_raw_body',span=[0,5]),dict(source='full_raw_body',span=[5,15])]
        self.validate(changed)

    def test_positive_requires_exact_field_specific_literal(self):
        changed=dict(self.row,artifact_evidence='yes',evidence=[])
        with self.assertRaisesRegex(ValueError,'field-specific'):
            self.validate(changed)
        e=dict(field='artifact_evidence',record_id='p',coordinate_system='full_raw_body_unicode_characters',
               span=[0,8],literal='ordinary',rationale='Visible quoted artifact term for this synthetic check.')
        changed['evidence']=[e]
        self.validate(changed)
        for broken in [dict(e,literal='invented'),dict(e,span=[0,7]),dict(e,span=[-1,8]),dict(e,record_id='other')]:
            with self.assertRaises(ValueError):
                self.validate(dict(changed,evidence=[broken]))

    def test_decoded_view_offsets_are_not_raw_payload_offsets(self):
        raw='{"summary":"hello\\u0020world"}'
        p=packet(text=raw); start=raw.index('hello'); end=raw.index('"',start)
        view=dict(view_id='v',root_id='p',origin_record_id='child',root_span=[start,end],
                  text='hello world',source_text=raw[start:end],text_sha256=report.text_sha('hello world'),
                  source_text_sha256=report.text_sha(raw[start:end]),origin_members=[])
        p['related_contained_views']=[view]
        selected,contexts=report.validate_selection_context(selection(p),[p],SOURCE)
        row=annotation(p);row['artifact_evidence']='yes'
        row['reviewed_extent'].append(dict(view_id='v',coordinate_system='view_text_unicode_characters',start=0,end=11))
        row['evidence']=[dict(field='artifact_evidence',view_id='v',record_id='child',
                             coordinate_system='view_text_unicode_characters',start=0,end=11,
                             literal='hello world',rationale='Decoded structured text.')]
        report.validate_annotations(dict(annotations=[row]),selected,contexts)
        row['evidence'][0]['coordinate_system']='raw_body_unicode_characters'
        with self.assertRaises(ValueError):
            report.validate_annotations(dict(annotations=[row]),selected,contexts)

    def test_positive_evidence_must_be_within_declared_review(self):
        row=dict(self.row,body_fully_reviewed=False,context_status='unresolved_or_truncated',
                 reviewed_extent=[dict(source='full_raw_body',span=[0,3])])
        row.update({field:'unknown' for field in report.FIELDS});row['artifact_evidence']='yes'
        row['evidence']=[dict(field='artifact_evidence',record_id='p',coordinate_system='raw_body_unicode_characters',
                              start=9,end=15,literal='source',rationale='Outside declared extent.')]
        with self.assertRaisesRegex(ValueError,'outside declared'):
            self.validate(row)

    def test_summaries_keep_unknowns_overlap_and_disagreement(self):
        a=dict(self.row); b=dict(self.row,publication_intent='unknown')
        primary={'p':b}
        result=report.summarize_reviews(primary,self.s,{'p':a},{'p':b},prediction_rows=[
            dict(record_id='p',payload_sha256=self.p['payload_sha256'],claim_candidate=True)])
        self.assertEqual(result['primary_by_stratum']['s']['annotations']['fields']['publication_intent']['unknown'],1)
        self.assertEqual(result['targeted_agreement']['publication_intent']['disagree'],1)
        self.assertEqual(result['sample_overlap']['exact_bodies'],1)
        self.assertIsNone(result['recall']);self.assertIsNone(result['actual_cooperation_rate'])


class ReportBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.census=self.base/'census';self.census.mkdir()
        self.p=packet();self.ann={};self.locks={}
        for group in ('primary','targeted'):
            context=self.census/f'audit-{group}-context.jsonl.gz'
            with gzip.open(context,'wt') as stream: stream.write(json.dumps(self.p)+'\n')
            (self.census/f'audit-{group}-selection.json').write_text(json.dumps(selection(self.p)))
        for name,group in [('primary','primary'),('targeted_a','targeted'),('targeted_b','targeted')]:
            path=self.base/(name+'.json');lock=self.base/(name+'-lock.json')
            context=self.census/f'audit-{group}-context.jsonl.gz'
            document=dict(annotations=[annotation(self.p)],input_sha256={str(context):report.sha(context)})
            if name=='targeted_b':
                document=dict(annotations=[annotation(self.p)],context_sha256=report.sha(context))
            path.write_text(json.dumps(document));self.ann[name]=path;self.locks[name]=lock
            if name=='targeted_b':
                lock.write_text(json.dumps(dict(path=path.name,sha256=report.sha(path),locked_utc=dt.datetime.now(dt.timezone.utc).isoformat())))
            else:
                lock.write_text(json.dumps(dict(annotation_file=path.name,annotation_sha256=report.sha(path),
                    input_sha256=document['input_sha256'],locked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())))
        with gzip.open(self.census/'payloads.csv.gz','wt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['record_id','payload_sha256','claim_candidate']);writer.writeheader()
            writer.writerow(dict(record_id='p',payload_sha256=self.p['payload_sha256'],claim_candidate=True))
        self.rehash_manifest();self.output=self.base/'summary.json'

    def tearDown(self):self.tmp.cleanup()

    def rehash_manifest(self):
        (self.census/'manifest.json').write_text(json.dumps(dict(source_sha256=SOURCE,
            outputs_sha256={path.name:report.sha(path) for path in self.census.iterdir() if path.name!='manifest.json'})))

    def build(self):
        return report.build(self.census,self.ann['primary'],self.locks['primary'],
                            self.ann['targeted_a'],self.locks['targeted_a'],
                            self.ann['targeted_b'],self.locks['targeted_b'],self.output)

    def test_complete_locked_report_and_no_overwrite(self):
        result=self.build()
        self.assertTrue(result['prediction_join_performed_after_all_annotation_locks_validated'])
        self.assertEqual(result['primary']['full_body_reviews'],1)
        self.assertEqual(result['targeted_agreement']['artifact_evidence']['agree'],1)
        with self.assertRaises(FileExistsError):self.build()

    def test_bad_primary_lock_blocks_prediction_read(self):
        lock=json.loads(self.locks['primary'].read_text());lock['annotation_sha256']='0'*64
        self.locks['primary'].write_text(json.dumps(lock))
        with patch.object(report,'load_predictions',side_effect=AssertionError('Predictions must not load')):
            with self.assertRaisesRegex(ValueError,'Annotation lock'):
                self.build()
        self.assertFalse(self.output.exists())

    def test_missing_primary_labels_blocks_prediction_read_even_with_valid_lock(self):
        doc=json.loads(self.ann['primary'].read_text());doc['annotations']=[]
        self.ann['primary'].write_text(json.dumps(doc))
        lock=json.loads(self.locks['primary'].read_text());lock['annotation_sha256']=report.sha(self.ann['primary'])
        self.locks['primary'].write_text(json.dumps(lock))
        with patch.object(report,'load_predictions',side_effect=AssertionError('Predictions must not load')):
            with self.assertRaisesRegex(ValueError,'missing selected'):
                self.build()

    def test_manifest_and_annotation_context_binding_required(self):
        path=self.census/'audit-primary-selection.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'manifest hash mismatch'):
            self.build()
        self.rehash_manifest()
        doc=json.loads(self.ann['primary'].read_text());doc['input_sha256']={}
        self.ann['primary'].write_text(json.dumps(doc))
        lock=json.loads(self.locks['primary'].read_text());lock['annotation_sha256']=report.sha(self.ann['primary'])
        self.locks['primary'].write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError,'context bytes'):
            self.build()


if __name__=='__main__':unittest.main()
