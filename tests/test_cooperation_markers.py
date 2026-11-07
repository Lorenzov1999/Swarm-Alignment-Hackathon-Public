import unittest
from swarmcollab.cooperation_markers import census, markers


class CooperationMarkerTests(unittest.TestCase):
    def test_common_technical_words_are_not_markers(self):
        for text in ['shared library', 'socket peer', 'tool --help', 'api/collaborators',
                     'computed_geometry.coordinates', 'board controller']:
            self.assertEqual(markers(text), [], text)

    def test_markers_are_not_semantic_labels(self):
        for text in ['teammate owns the other task', 'Do not run without coordination',
                     'hardcoded coordinate', 'handoff watcher']:
            self.assertTrue(markers(text))
        self.assertNotIn('cooperation', markers('teammate')[0])

    def test_offsets_and_case(self):
        text = 'xx TEAMMATE yy other agents zz'
        for m in markers(text):
            self.assertEqual(text[m['start']:m['end']], m['literal'])

    def test_payload_unit_and_exact_duplicate_sensitivity(self):
        records = [dict(id='p1',kind='payload',text='teammate'),
                   dict(id='p2',kind='payload',text='teammate'),
                   dict(id='r1',kind='recovered_text'),
                   dict(id='o1',kind='response')]
        result = census(records)
        self.assertEqual(result['payload_records'], 2)
        self.assertEqual(result['exact_unique_payloads'], 1)
        self.assertEqual(result['lexical_candidate_payloads'], 2)
        self.assertEqual(result['lexical_candidate_unique_texts'], 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            census([dict(id='p',kind='payload',text=''),dict(id='p',kind='payload',text='')])


if __name__ == '__main__':
    unittest.main()
