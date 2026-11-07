"""Authorized parser scopes and timestamp-first compact-cache filtering."""
import datetime as dt
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from swarmcollab import config
from swarmcollab.generalization import cached_records, observation_scope


class GeneralizationScopeTests(unittest.TestCase):
    def test_scope_restores_configuration_even_after_failure(self):
        before = config.DEV_START, config.DEV_END, config.TRAIN_END
        with self.assertRaises(RuntimeError):
            with observation_scope('2026-06-01', '2026-07-05'):
                self.assertEqual(config.DEV_START, dt.date(2026, 6, 1))
                self.assertEqual(config.DEV_END, dt.date(2026, 7, 5))
                self.assertEqual(config.TRAIN_END, before[2])
                raise RuntimeError('Synthetic parser failure')
        self.assertEqual((config.DEV_START, config.DEV_END, config.TRAIN_END), before)

    def test_reserve_and_unregistered_windows_are_rejected(self):
        for first, last in [('2026-09-16', '2026-10-31'),
                            ('2026-06-01', '2026-09-15'),
                            ('2026-08-17', '2026-10-01')]:
            with self.assertRaises(ValueError):
                with observation_scope(first, last):
                    self.fail('Unregistered window was opened')

    def test_pacific_bounds_filter_before_decoding_payload(self):
        rows = [
            # Invalid payloads outside the window must never be JSON-decoded.
            b'{"i":"before","t":"2026-08-17T06:59:59Z","x":INVALID}\n',
            json.dumps(dict(i='first', t='2026-08-17T07:00:00Z', k='shell', x='echo first')).encode() + b'\n',
            json.dumps(dict(i='last', t='2026-09-16T06:59:59Z', k='shell', x='echo last')).encode() + b'\n',
            b'{"i":"reserve","t":"2026-09-16T07:00:00Z","x":INVALID}\n',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.jsonl.gz'
            with gzip.open(path, 'wb') as stream:
                stream.writelines(rows)
            actual = list(cached_records(path, '2026-08-17', '2026-09-15'))
        self.assertEqual([row['i'] for row in actual], ['first', 'last'])

    def test_malformed_in_scope_payload_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.jsonl.gz'
            with gzip.open(path, 'wb') as stream:
                stream.write(b'{"i":"bad","t":"2026-06-02T16:00:00Z","x":INVALID}\n')
            with self.assertRaises(json.JSONDecodeError):
                list(cached_records(path, '2026-06-01', '2026-07-05'))


if __name__ == '__main__':
    unittest.main()
