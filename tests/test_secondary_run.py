"""Secondary orchestration must lock predictions before reading request labels."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from swarmcollab import pairs
from swarmcollab import secondary_run as R
from test_secondary_transfer import fitted, observations


class SecondaryOrchestrationTests(unittest.TestCase):
    def test_missing_or_invalid_lock_blocks_label_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(pd, 'read_csv', side_effect=AssertionError('labels opened')):
                with self.assertRaisesRegex(ValueError, 'locked before'):
                    R.evaluate_locked_predictions(output, output / 'labels.csv')
            (output / 'prediction.csv').write_text('anything\n')
            R._save(output / 'secondary-prediction-lock.json',
                    dict(prediction_files={'prediction.csv': 'incorrect'}))
            with mock.patch.object(pd, 'read_csv', side_effect=AssertionError('labels opened')):
                with self.assertRaisesRegex(ValueError, 'lock mismatch'):
                    R.evaluate_locked_predictions(output, output / 'labels.csv')

    def test_synthetic_application_locks_before_evaluation_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            output = study / 'temporal'; output.mkdir()
            bundle = fitted()
            R._save(study / 'secondary-frozen.json', bundle)
            legacy_root = study / 'frozen/outputs/hawkes'; legacy_root.mkdir(parents=True)
            alpha = pd.DataFrame(np.arange(9).reshape(3, 3) / 10.,
                                 index=['A', 'B', 'C'], columns=['A', 'B', 'C'])
            alpha.to_csv(legacy_root / 'alpha_dev.csv')
            (alpha - .2).to_csv(legacy_root / 'alpha_excess_dev.csv')
            mentions, activity = observations(('2026-08-17', '2026-08-18'))
            universe = pairs.universe(activity)
            universe.to_csv(output / 'features.csv.gz', index=False)
            labelled = universe.copy(); labelled['y'] = [True, False, False, False, True, False]
            labelled.to_csv(output / 'predictions.csv.gz', index=False)
            original_read = pd.read_csv
            label_reads = []

            def guarded_read(path, *args, **kwargs):
                if 'y' in kwargs.get('usecols', []):
                    self.assertTrue((output / 'secondary-prediction-lock.json').exists())
                    label_reads.append(str(path))
                return original_read(path, *args, **kwargs)

            with mock.patch.object(R, '_historical', return_value={'sha256': 'synthetic'}), \
                    mock.patch.object(R, '_load_inputs', return_value=(mentions, activity, {})), \
                    mock.patch.object(pd, 'read_csv', side_effect=guarded_read), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = R.apply_stage(study, 'temporal')
                self.assertEqual(len(label_reads), 1)
                self.assertIn('request_evaluation', result)
                with self.assertRaisesRegex(ValueError, 'already locked'):
                    R.apply_stage(study, 'temporal')
            lock = json.loads((output / 'secondary-prediction-lock.json').read_text())
            self.assertFalse(lock['target_labels_accessed'])
            self.assertEqual(len(lock['prediction_files']), 3)
            self.assertEqual(result['request_evaluation']['regimes']['rows'], len(universe))


if __name__ == '__main__':
    unittest.main()
