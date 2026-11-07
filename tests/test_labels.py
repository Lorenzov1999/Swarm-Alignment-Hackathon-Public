"""Date-scoped request loading guards."""
import datetime as dt
import unittest
from unittest import mock

from swarmcollab import config
from swarmcollab.labels import load_requests


class ScopedRequestLabelsTests(unittest.TestCase):
    def test_scoped_labels_ignore_excluded_payload_and_recipient_fields(self):
        class TimestampOnly(dict):
            def __getitem__(self, key):
                if key != 'timestamp':
                    raise AssertionError('Excluded request payload was inspected: ' + key)
                return super().__getitem__(key)

        class ExcludedEdge(dict):
            def __getitem__(self, key):
                raise AssertionError('Excluded recipient payload was inspected: ' + key)

        selected = dict(timestamp='2026-08-14 23:59:00 PT', **{
            'from': 'Reader', 'accepted': 'yes', 'followedThrough': 'yes', 'weight': 1.})
        props = [dict(instances=[TimestampOnly(timestamp='2026-07-05 23:59:00 PT'),
                                 selected, TimestampOnly(timestamp='2026-08-17 00:00:00 PT'),
                                 dict(timestamp='2026-08-03 09:00:00 PT')],
                      options=[dict(label='All requests', edges=[
                          ExcludedEdge(instanceIds=[0, 2]),
                          {'instanceIds': [1], 'from': 'Reader', 'to': 'Author'}])])]
        with mock.patch('swarmcollab.labels.open', mock.mock_open()), \
                mock.patch('swarmcollab.labels.json.load', return_value=props):
            rows, requests, orphans = load_requests('synthetic', start=config.DEV_START, end=config.DEV_END)
        self.assertEqual((requests, orphans), (2, 1))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['request'], 1)
        self.assertEqual(rows[0]['to'], 'Author')
        self.assertEqual(rows[0]['day'], config.DEV_END)
        self.assertEqual(rows[0]['utc'].date(), dt.date(2026, 8, 15))

    def test_label_bounds_are_optional_inclusive_and_can_be_one_sided(self):
        instances = [dict(timestamp=day + ' 10:00:00 PT', **{
            'from': 'A', 'accepted': 'yes', 'followedThrough': 'yes', 'weight': 1.})
            for day in ('2026-07-06', '2026-07-07')]
        props = [dict(instances=instances, options=[dict(label='All', edges=[
            {'instanceIds': [0, 1], 'from': 'A', 'to': 'B'}])])]
        with mock.patch('swarmcollab.labels.open', mock.mock_open()), \
                mock.patch('swarmcollab.labels.json.load', return_value=props):
            legacy = load_requests('synthetic')
            self.assertEqual(legacy, load_requests('synthetic', start='2026-07-06', end='2026-07-07'))
            lower, count, orphan = load_requests('synthetic', start='2026-07-07')
            self.assertEqual(([r['request'] for r in lower], count, orphan), ([1], 1, 0))
            upper, count, orphan = load_requests('synthetic', end='2026-07-06')
            self.assertEqual(([r['request'] for r in upper], count, orphan), ([0], 1, 0))
            self.assertEqual(load_requests('synthetic', start='2026-08-03'), ([], 0, 0))

    def test_reversed_label_bounds_fail_before_file_access(self):
        with mock.patch('swarmcollab.labels.open') as opened:
            with self.assertRaisesRegex(ValueError, 'start date'):
                load_requests('synthetic', start='2026-08-04', end='2026-08-03')
        opened.assert_not_called()


if __name__ == "__main__":
    unittest.main()
