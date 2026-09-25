"""Protocol checks and a tiny CPU end-to-end training test (no downloads)."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tune_and_compare_malevis as study


def fixture():
    rows = []
    for cls in ('a', 'b'):
        for part, n in [('fitting', 6), ('development', 2), ('evaluation', 2)]:
            for i in range(n):
                # Both fitting copies of the evaluation image must be removed.
                key = cls + ('shared' if (part == 'fitting' and i < 2) or
                             (part == 'evaluation' and i == 0) else part + str(i))
                rows.append(dict(sample_id=f'{part}/{cls}/{i}.png', class_label=cls,
                                 partition=part, rgb_sha256=key, input_sha256=key))
    return rows


class ConditionsTest(unittest.TestCase):
    def test_all_counterparts_removed_and_control_counts_match(self):
        fits, dev, ev, removed, table = study.conditions(fixture(), 123)
        self.assertEqual(len(removed), 4)  # Two evaluation matches, four fitting copies.
        self.assertEqual([len(fits[c]) for c in study.CONDITIONS], [12, 8, 8])
        self.assertEqual(len(dev), 4)
        self.assertEqual(len(ev), 4)
        for row in table:
            self.assertEqual(row['clean_exact_evaluation_native_matches'], 0)
            self.assertEqual(row['clean_exact_fitting'], row['random_control_fitting'])
            self.assertEqual(row['full_evaluation_native_matches'], 1)

    def test_repeatable_regardless_of_input_order(self):
        self.assertEqual(study.conditions(fixture(), 123),
                         study.conditions(list(reversed(fixture())), 123))

    def test_cross_label_matches_are_also_removed(self):
        rows = fixture()
        next(r for r in rows if r['sample_id'] == 'fitting/b/5.png')['rgb_sha256'] = 'ashared'
        fits, _, _, removed, _ = study.conditions(rows, 123)
        self.assertIn('fitting/b/5.png', [r['sample_id'] for r in removed])
        self.assertEqual(len(fits['clean_exact']), 7)

    def test_exhausted_class_stops(self):
        rows = fixture()
        for r in rows:
            if r['partition'] == 'fitting' and r['class_label'] == 'a':
                r['rgb_sha256'] = 'ashared'
        with self.assertRaisesRegex(ValueError, 'exhausts'):
            study.conditions(rows, 123)

    def test_selection_ignores_evaluation_accuracy(self):
        results = [dict(recipe_name=r, development_loss=v, development_accuracy=.9,
                        evaluation_accuracy=1-v) for r, v in zip(study.RECIPES, [.3, .2, .4])]
        self.assertEqual(study.select_recipe(results)[0], 'faster')
        for r in results:
            r['evaluation_accuracy'] = 0 if r['recipe_name'] == 'faster' else 1
        self.assertEqual(study.select_recipe(results)[0], 'faster')

    def test_frozen_settings_detect_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'settings.json'
            study.freeze_json(path, {'lr': .01})
            with self.assertRaisesRegex(ValueError, 'Frozen settings differ'):
                study.freeze_json(path, {'lr': .1})

    def test_cpu_workflow_tunes_then_compares_and_preserves_outputs(self):
        import torch
        from torch import nn
        from PIL import Image
        torch.set_num_threads(1)

        def tiny_model(name, n):
            head = nn.Linear(4, n)
            return nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), head), head

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data, protocol, output = root/'data', root/'protocol', root/'output'
            protocol.mkdir()
            rows = fixture()
            for r in rows:
                path = data/r['sample_id']
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (300, 300), (80, 40, 30) if r['class_label'] == 'a'
                          else (30, 40, 80)).save(path)
            meta = dict(classes=['a', 'b'], manifest_sha256='synthetic',
                        split_settings=dict(development_fraction=.2, split_seed=42))
            (protocol/'protocol.json').write_text(json.dumps(meta))
            common = ['test', '--dataset', str(data), '--protocol-dir', str(protocol),
                      '--output', str(output), '--epochs', '3', '--head-epochs', '1',
                      '--batch-size', '2', '--workers', '0', '--allow-cpu', '--no-amp']
            with patch.object(study.base, 'prepare', return_value=(rows, meta)), \
                 patch.object(study, 'build_model', side_effect=tiny_model), \
                 patch.object(torch.cuda, 'is_available', return_value=False):
                with patch.object(sys, 'argv', common + ['--stage', 'tune']):
                    study.main()
                self.assertFalse((output/'evaluation').exists())
                self.assertEqual(len(list((output/'tuning').glob('*/seed_42/trained.json'))), 3)
                with patch.object(sys, 'argv', common + ['--stage', 'compare']):
                    study.main()
                evaluation_files = sorted((output/'evaluation').glob('*/seed_42/metrics.json'))
                self.assertEqual(len(evaluation_files), 3)
                results = [json.loads(p.read_text()) for p in evaluation_files]
                self.assertEqual(len({r['initial_state_sha256'] for r in results}), 1)
                self.assertEqual({r['evaluation_images'] for r in results}, {4})
                before = {p: p.read_bytes() for p in evaluation_files}
                with patch.object(sys, 'argv', common + ['--stage', 'compare']):
                    study.main()
                self.assertEqual(before, {p: p.read_bytes() for p in evaluation_files})
                with (output/'paired_differences_42.csv').open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 1)
                selected = output/'selected_recipe.json'
                corrupt = json.loads(selected.read_text())
                corrupt['recipe_name'] = 'not-the-winner'
                selected.write_text(json.dumps(corrupt))
                with patch.object(sys, 'argv', common + ['--stage', 'compare']):
                    with self.assertRaisesRegex(ValueError, 'Selection no longer matches'):
                        study.main()


if __name__ == '__main__':
    unittest.main()
