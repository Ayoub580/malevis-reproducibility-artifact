"""Small, CPU-only checks of split invariants and metric definitions."""
import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from train_full_malevis import (allocate_split, classification_metrics, validate_split,
                                prepare_output, prepare_run)


def fixture():
    rows = []
    for c in ('a', 'b'):
        for i in range(10):
            # Two repeated training files; one other training image matches evaluation.
            key = c + str(0 if i == 1 else i)
            rows.append(dict(sample_id=f'train/{c}/{i}.png', original_split='train',
                             class_label=c, rgb_sha256=key, input_sha256=key))
        rows.append(dict(sample_id=f'val/{c}/0.png', original_split='val',
                         class_label=c, rgb_sha256=c+'9', input_sha256=c+'9'))
    return rows


class ProtocolChecks(unittest.TestCase):
    def test_full_retains_all_samples_and_evaluation_exposure(self):
        rows = allocate_split(fixture(), .2, 42)
        self.assertEqual(len(rows), len(fixture()))
        for c in ('a', 'b'):
            match = next(r for r in rows if r['sample_id'] == f'train/{c}/9.png')
            self.assertEqual(match['partition'], 'fitting')
        validate_split(rows)

    def test_duplicate_groups_stay_together(self):
        rows = allocate_split(fixture(), .2, 42)
        for c in ('a', 'b'):
            self.assertEqual(len({r['partition'] for r in rows if r['input_sha256'] == c+'0'}), 1)

    def test_repeatable_selection_independent_of_input_order(self):
        a = allocate_split(fixture(), .2, 42)
        b = allocate_split(list(reversed(fixture())), .2, 42)
        self.assertEqual({r['sample_id']: r['partition'] for r in a},
                         {r['sample_id']: r['partition'] for r in b})

    def test_infeasible_development_stops(self):
        rows = fixture()
        for r in rows:
            r['rgb_sha256'] = r['input_sha256'] = r['class_label']
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            allocate_split(rows, .2, 42)

    def test_known_confusion_matrix_and_macro_f1(self):
        scores, per_class, cm = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ['a', 'b'])
        np.testing.assert_array_equal(cm, [[1, 1], [0, 2]])
        self.assertEqual(scores['accuracy'], .75)
        self.assertAlmostEqual(scores['macro_f1'], (2/3 + .8)/2)
        self.assertEqual(per_class[0]['recall'], .5)

    def test_empty_run_can_restart_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)/'resnet50'/'seed_42'
            run.mkdir(parents=True)
            self.assertTrue(prepare_run(run))

    def test_incomplete_files_preserved_on_explicit_restart(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)/'resnet50'/'seed_42'
            run.mkdir(parents=True)
            (run/'best.pt').write_bytes(b'checkpoint')
            with self.assertRaisesRegex(ValueError, 'restart-incomplete'):
                prepare_run(run)
            self.assertTrue(prepare_run(run, True))
            self.assertFalse(list(run.iterdir()))
            saved = list((run.parent/'_interrupted').glob('*/best.pt'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].read_bytes(), b'checkpoint')

    def test_completed_run_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)/'seed_42'
            run.mkdir()
            (run/'metrics.json').write_text('{"accuracy": 0.5}')
            self.assertFalse(prepare_run(run, True))
            self.assertTrue((run/'metrics.json').exists())

    def test_script_repair_after_empty_attempt_preserves_config(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'results'
            old = {'script_sha256': 'old', 'batch_size': 16}
            new = {**old, 'script_sha256': 'new'}
            prepare_output(output, old)
            (output/'resnet50'/'seed_42').mkdir(parents=True)
            prepare_output(output, new)
            self.assertEqual(json.loads((output/'config.json').read_text()), new)
            saved = list((Path(d)/'_interrupted').glob('*/config.json'))
            self.assertEqual(json.loads(saved[0].read_text()), old)

    def test_restart_does_not_allow_changed_training_settings(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'results'
            config = {'script_sha256': 'old', 'batch_size': 16}
            prepare_output(output, config)
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                prepare_output(output, {**config, 'batch_size': 8}, True)

    def test_script_change_cannot_mix_completed_runs(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/'results'
            config = {'script_sha256': 'old', 'batch_size': 16}
            prepare_output(output, config)
            run = output/'resnet50'/'seed_42'
            run.mkdir(parents=True)
            (run/'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                prepare_output(output, {**config, 'script_sha256': 'new'}, True)


if __name__ == '__main__':
    unittest.main()
