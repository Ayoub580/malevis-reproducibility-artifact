#!/usr/bin/env python3
"""Recompute reported five-seed performance summaries and paired contrasts."""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path(os.environ.get("MALEVIS_PROJECT_ROOT", ".")).resolve()
SEEDS = [42, 101, 7, 2024, 99]
STUDIES = {
    "ResNet50": (ROOT / "results/consensus_23_resnet50_v1/resnet50", "random_subsample23"),
    "ConvNeXt-Tiny": (ROOT / "results/consensus_23_convnext_tiny_v2/convnext_tiny", "random_subsample23"),
    "Swin-Tiny": (ROOT / "results/consensus_23_swin_tiny_v1/swin_tiny", "random_subsample23"),
    "Malimg": (ROOT / "results/malimg_resnet50_seed42_v1/resnet50", "random_subsample20"),
    "Virus-MNIST": (ROOT / "results/virusmnist_spatial_cnn_5seeds_v1/spatial_cnn", "random_subsample9"),
}
EXPECTED_MEANS = {
    "ResNet50": {"full": (96.55, 96.98, 96.74), "random": (95.15, 95.84, 95.02), "clean": (94.95, 95.24, 94.78)},
    "ConvNeXt-Tiny": {"full": (95.42, 95.84, 95.61), "random": (93.68, 94.69, 93.68), "clean": (92.45, 93.34, 92.76)},
    "Swin-Tiny": {"full": (93.09, 93.76, 93.94), "random": (92.63, 93.09, 93.00), "clean": (90.86, 91.31, 90.57)},
    "Malimg": {"full": (99.02, 97.16, 97.21), "random": (99.60, 98.84, 98.85), "clean": (98.65, 96.19, 96.12)},
    "Virus-MNIST": {"full": (93.79, 89.98, 89.23), "random": (91.61, 88.91, 88.31), "clean": (90.20, 86.80, 86.10)},
}
EXPECTED_SDS = {
    "ResNet50": {"full": (.21, .20, .23), "random": (.46, .41, .38), "clean": (.42, .19, .36)},
    "ConvNeXt-Tiny": {"full": (.65, .51, .56), "random": (.47, .43, .30), "clean": (.68, .71, .44)},
    "Swin-Tiny": {"full": (1.96, 1.79, 1.43), "random": (.48, .67, .76), "clean": (.79, .68, 1.39)},
    "Malimg": {"full": (.15, .44, .41), "random": (.16, .47, .44), "clean": (.21, .67, .60)},
    "Virus-MNIST": {"full": (.16, .28, .38), "random": (.16, .48, .39), "clean": (.19, .43, .48)},
}
EXPECTED_RANDOM_CLEAN = {"ResNet50": .20, "ConvNeXt-Tiny": 1.23, "Swin-Tiny": 1.77, "Malimg": .94, "Virus-MNIST": 1.41}
EXPECTED_FULL_CLEAN = {"ResNet50": 1.60, "ConvNeXt-Tiny": 2.97, "Swin-Tiny": 2.23}
EXPECTED_CONTRASTS = {
    ("ResNet50", "full"): (1.60, .60, .85, 2.35, 1.73, 1.97),
    ("ResNet50", "random"): (.20, .73, -.71, 1.11, .60, .25),
    ("ConvNeXt-Tiny", "full"): (2.97, .97, 1.77, 4.17, 2.50, 2.85),
    ("ConvNeXt-Tiny", "random"): (1.23, 1.03, -.05, 2.51, 1.35, .92),
    ("Swin-Tiny", "full"): (2.23, 1.58, .27, 4.19, 2.44, 3.37),
    ("Swin-Tiny", "random"): (1.77, .53, 1.10, 2.43, 1.78, 2.43),
    ("Malimg", "random"): (.94, .36, .50, 1.38, 2.64, 2.73),
    ("Virus-MNIST", "random"): (1.41, .28, 1.06, 1.75, 2.11, 2.22),
}


def metrics(base: Path, condition: str):
    return [json.loads((base / condition / f"seed_{seed}" / "metrics.json").read_text()) for seed in SEEDS]


for name, (base, random_name) in STUDIES.items():
    by_condition = {
        "full": metrics(base, "full"),
        "random": metrics(base, random_name),
        "clean": metrics(base, "clean_consensus"),
    }
    for condition, values in by_condition.items():
        means = tuple(statistics.mean(m[k] for m in values) * 100 for k in ("accuracy", "macro_f1", "macro_recall"))
        sds = tuple(statistics.stdev(m[k] for m in values) * 100 for k in ("accuracy", "macro_f1", "macro_recall"))
        assert tuple(round(v, 2) for v in means) == EXPECTED_MEANS[name][condition], (name, condition, means)
        assert tuple(round(v, 2) for v in sds) == EXPECTED_SDS[name][condition], (name, condition, sds)
    random_clean = [(r["accuracy"] - c["accuracy"]) * 100 for r, c in zip(by_condition["random"], by_condition["clean"])]
    assert round(statistics.mean(random_clean), 2) == EXPECTED_RANDOM_CLEAN[name]
    if name in EXPECTED_FULL_CLEAN:
        full_clean = [(f["accuracy"] - c["accuracy"]) * 100 for f, c in zip(by_condition["full"], by_condition["clean"])]
        assert round(statistics.mean(full_clean), 2) == EXPECTED_FULL_CLEAN[name]
    if name in {"ConvNeXt-Tiny", "Swin-Tiny", "Malimg", "Virus-MNIST"}:
        assert all(v > 0 for v in random_clean), (name, random_clean)
    if name == "ResNet50":
        assert min(random_clean) < 0 < max(random_clean)
    for short_condition in ("full", "random"):
        key = (name, short_condition)
        if key not in EXPECTED_CONTRASTS:
            continue
        left = by_condition[short_condition]
        clean = by_condition["clean"]
        differences = {
            metric: [(a[metric] - b[metric]) * 100 for a, b in zip(left, clean)]
            for metric in ("accuracy", "macro_f1", "macro_recall")
        }
        acc = differences["accuracy"]
        mean = statistics.mean(acc)
        sd = statistics.stdev(acc)
        margin = 2.7764451051977987 * sd / math.sqrt(5)
        found = (
            round(mean, 2), round(sd, 2), round(mean - margin, 2), round(mean + margin, 2),
            round(statistics.mean(differences["macro_f1"]), 2),
            round(statistics.mean(differences["macro_recall"]), 2),
        )
        assert found == EXPECTED_CONTRASTS[key], (key, found)
    print(name, "Random-Clean accuracy values:", ", ".join(f"{v:+.3f}" for v in random_clean))

print("PASS: all reported five-seed means and the principal paired contrasts reproduce from the 75 run-level metrics; all stated sign claims hold.")
