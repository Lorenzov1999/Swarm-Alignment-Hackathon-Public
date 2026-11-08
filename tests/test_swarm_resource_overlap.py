"""Synthetic-only checks of record overlap and publication boundaries."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from swarmcollab.swarm_resource_overlap import aggregate, analyze_records, run_file


def config():
    return dict(format='jsonl', compression='gzip', source_sha256='a'*64,
                fields=dict(id='id', kind='kind', parent_id='parent_id', text='text'),
                payload_kind='payload', response_kind='response', excluded_kinds=['recovered_text'])


def record(identifier, text='', kind='payload', parent=None):
    return dict(id=identifier, kind=kind, parent_id=parent, text=text)


class AggregateTests(unittest.TestCase):
    def test_sparse_formula_equals_dense_and_uniform_and_excludes_diagonal(self):
        incidence = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=float)
        labels = {str(j): ('url', f'https://example.invalid/{j}') for j in range(3)}
        payloads = [dict(record_id=str(i), root_id=str(i//2), payload_sha256=str(i),
                         resources={str(j) for j in range(3) if incidence[i,j]}) for i in range(4)]
        result = aggregate(payloads, labels)
        weights = np.log(len(incidence)/incidence.sum(axis=0))
        x = incidence*weights
        norms = np.linalg.norm(x, axis=1)
        z = np.divide(x, norms[:,None], out=np.zeros_like(x), where=norms[:,None] > 0)
        uniform = np.divide(incidence, np.linalg.norm(incidence,axis=1)[:,None],
                            out=np.zeros_like(incidence), where=norms[:,None]>0)
        rows = {r['resource_id']:r for r in result['resources']}
        for j in range(3):
            expected = sum(z[i,j]*z[k,j] for i in range(4) for k in range(i+1,4))
            cross = sum(z[i,j]*z[k,j] for i in range(4) for k in range(i+1,4) if i//2 != k//2)
            plain = sum(uniform[i,j]*uniform[k,j] for i in range(4) for k in range(i+1,4))
            self.assertAlmostEqual(rows[str(j)]['overlap_mass'],expected)
            self.assertAlmostEqual(rows[str(j)]['cross_root_overlap_mass'],cross)
            self.assertAlmostEqual(rows[str(j)]['uniform_overlap_mass'],plain)
        self.assertAlmostEqual(result['summary']['total_cosine_overlap_mass'],np.triu(z@z.T,1).sum())
        self.assertEqual(result['summary']['unordered_distinct_record_pairs'],6)

    def test_ubiquitous_zero_idf_empty_profiles_and_singletons(self):
        labels = {'r':('url','https://example.invalid/')}
        rows = [dict(record_id=str(i),root_id=str(i),payload_sha256=str(i),resources={'r'}) for i in range(3)]
        result=aggregate(rows,labels)
        self.assertEqual(result['resources'][0]['idf'],0)
        self.assertEqual(result['resources'][0]['overlap_mass'],0)
        self.assertEqual(result['resources'][0]['uniform_overlap_mass'],3)
        self.assertEqual(result['summary']['records_with_nonzero_weighted_profile'],0)
        self.assertEqual(result['summary']['top20_overlap_resources'],[])
        single=aggregate(rows[:1],labels)
        self.assertEqual(single['resources'][0]['possible_record_pairs'],0)
        self.assertEqual(single['resources'][0]['overlap_mass'],0)
        self.assertEqual(aggregate([],{} )['summary']['payload_records'],0)


class BoundaryTests(unittest.TestCase):
    def test_exact_dedup_and_raw_sensitivity_have_separate_weights(self):
        rows=[record('b','https://example.invalid/a'),record('a','https://example.invalid/a'),record('c','empty')]
        result=analyze_records(rows,config())
        self.assertEqual(result['summary']['payload_records'],3)
        self.assertEqual(result['summary']['exact_unique_payloads'],2)
        self.assertEqual(result['primary']['resources'][0]['supporting_records'],1)
        self.assertEqual(result['raw']['resources'][0]['supporting_records'],2)
        self.assertEqual(result['primary']['resources'][0]['overlap_mass'],0)
        self.assertEqual(result['raw']['resources'][0]['overlap_mass'],1)
        self.assertEqual([r['record_id'] for r in result['primary']['incidence']],['a','c'])
        self.assertNotEqual(result['primary']['resources'][0]['idf'],result['raw']['resources'][0]['idf'])

    def test_nonpayload_content_and_tags_are_never_accessed_and_paths_stay_excluded(self):
        class Guard(dict):
            def __getitem__(self,key):
                if key in {'text','tags','cite'}:
                    raise AssertionError('Excluded content accessed')
                return super().__getitem__(key)
        rows=[record('a','[SERVICE 1 URL 2] "./report.json"'),
              Guard(id='b',kind='response',parent_id='a'),
              Guard(id='c',kind='recovered_text',parent_id='b')]
        result=analyze_records(rows,config())
        self.assertEqual(result['summary']['excluded_content_records'],2)
        self.assertEqual(result['summary']['excluded_local_path_coverage']['records_with_local_path'],1)
        self.assertEqual({r['family'] for r in result['primary']['resources']},{'opaque_alias'})
        self.assertEqual(result['summary']['graph']['distinct_payload_roots'],1)

    def test_roots_are_structural_and_invalid_graphs_fail(self):
        rows=[record('a','https://example.invalid/a'),record('b','https://example.invalid/a other',parent='a'),record('c','empty')]
        result=analyze_records(rows,config())
        self.assertEqual(result['summary']['graph']['payloads_with_parent'],1)
        self.assertEqual(result['primary']['resources'][0]['overlap_mass'],1)
        self.assertEqual(result['primary']['resources'][0]['cross_root_overlap_mass'],0)
        for invalid in ([record('a',parent='missing')], [record('a',parent='b'),record('b',parent='a')],
                        [record('a'),record('a')]):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):analyze_records(invalid,config())

    def test_review_selection_is_score_based_deterministic_and_has_no_text(self):
        rows=[record(str(i),'https://example.invalid/a '+str(i)) for i in range(5)]
        rows += [record('z','empty'), record('duplicate',rows[0]['text'])]
        first=analyze_records(rows,config())
        second=analyze_records(list(reversed(rows)),config())
        self.assertEqual(first['review_selection'],second['review_selection'])
        selected=first['review_selection']['selected_resources'][0]['records']
        self.assertEqual([r['record_id'] for r in selected],['0','1','2'])
        self.assertNotIn('text',json.dumps(first['primary']['incidence']))

    def test_hashed_atomic_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.gz';cfg=root/'config.json';protocol=root/'protocol.md';out=root/'result'
            with gzip.open(source,'wt') as handle:
                for row in [record('a','https://example.invalid/a'),record('b','https://example.invalid/a b'),record('c','empty')]:
                    handle.write(json.dumps(row)+'\n')
            settings=config();settings['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
            cfg.write_text(json.dumps(settings));protocol.write_text('Synthetic protocol only.\n')
            manifest=run_file(source,cfg,out,protocol)
            self.assertEqual(manifest['protocol_sha256'],hashlib.sha256(protocol.read_bytes()).hexdigest())
            for name,digest in manifest['outputs_sha256'].items():
                self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(),digest)
            with self.assertRaises(FileExistsError):run_file(source,cfg,out,protocol)
            settings['source_sha256']='0'*64;cfg.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,'Source SHA'):run_file(source,cfg,root/'bad',protocol)
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
