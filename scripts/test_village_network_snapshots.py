"""Small graph cases test the scientific meaning of the plotted edges."""
import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from build_village_network_snapshots import graph_from_pairs
import build_corrected_village_network_snapshots as correction


def frame(names, weights):
    return pd.DataFrame([(a, b, weights.get((a, b), 0.0))
                         for a, b in itertools.combinations(names, 2)],
                        columns=['a', 'b', 'p1_cos'])


class NetworkSnapshots(unittest.TestCase):
    def test_locked_score_hash_is_checked_before_building(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scores = root / 'scores-before-labels.csv.gz'
            scores.write_bytes(b'changed')
            lock = root / 'prediction-lock.json'
            lock.write_text(json.dumps({'scores_sha256': hashlib.sha256(b'original').hexdigest()}))
            (root / 'summary.json').write_text(json.dumps({
                'prediction_lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError, 'Corrected score input changed'):
                correction.main(['--audit', str(root), '--original', str(root / 'unread.json')])

    def test_additive_correction_rejects_new_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'original.json'
            original.write_text(json.dumps({'reproduction_mode': 'current-run'}))
            scores = root / 'evaluated-pairs.csv.gz'
            scores.write_bytes(b'not read before provenance validation')
            (root / 'manifest.json').write_text(json.dumps({'files_sha256': {
                scores.name: hashlib.sha256(scores.read_bytes()).hexdigest()}}))
            (root / 'summary.json').write_text('{}')
            output = root / 'corrected.json'
            with self.assertRaisesRegex(ValueError, 'original frozen-paper snapshots'):
                correction.main(['--original', str(original), '--audit', str(root), '--output', str(output)])
            self.assertFalse(output.exists())

    def test_strict_threshold_and_isolates(self):
        g = graph_from_pairs(frame(['A', 'B', 'C', 'D'],
                                   {('A', 'B'): .2, ('B', 'C'): .1}), .1)
        self.assertEqual(g['nodes'], ['A', 'B', 'C', 'D'])
        self.assertEqual(g['components'], [['A', 'B'], ['C'], ['D']])
        self.assertEqual(g['metrics']['isolates'], 2)
        self.assertEqual(g['edges'], [dict(a='A', b='B', score=.2)])

    def test_triangle_and_disconnected_pair(self):
        names = ['A', 'B', 'C', 'D', 'E']
        weights = {('A', 'B'): .3, ('A', 'C'): .4, ('B', 'C'): .5, ('D', 'E'): .2}
        g = graph_from_pairs(frame(names, weights), .1)
        self.assertEqual(g['components'], [['A', 'B', 'C'], ['D', 'E']])
        self.assertEqual(g['metrics']['triangles'], 1)
        self.assertEqual(g['metrics']['transitivity'], 1.)
        self.assertEqual(g['degrees'], dict(A=2, B=2, C=2, D=1, E=1))

    def test_empty_edge_graph_keeps_agents(self):
        g = graph_from_pairs(frame(['A', 'B', 'C'], {}), .1)
        self.assertEqual(g['metrics']['isolates'], 3)
        self.assertEqual(g['metrics']['component_sizes'], [1, 1, 1])
        self.assertEqual(g['metrics']['transitivity'], 0.)

    def test_incomplete_or_duplicate_pair_universe_rejected(self):
        pairs = frame(['A', 'B', 'C'], {})
        with self.assertRaises(ValueError):
            graph_from_pairs(pairs.iloc[:2], .1)
        with self.assertRaises(ValueError):
            graph_from_pairs(pd.concat([pairs, pairs.iloc[:1]]), .1)


if __name__ == '__main__':
    unittest.main()
