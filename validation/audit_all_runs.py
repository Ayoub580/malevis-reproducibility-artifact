#!/usr/bin/env python3
"""Independent consistency audit for the 75 manuscript training runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ.get("MALEVIS_PROJECT_ROOT", ".")).resolve()
SEEDS = {7, 42, 99, 101, 2024}
STUDIES = [
    ("MaleVis/ResNet50", ROOT / "results/consensus_23_resnet50_v1/resnet50", {"full", "random_subsample23", "clean_consensus"}),
    ("MaleVis/ConvNeXt-Tiny", ROOT / "results/consensus_23_convnext_tiny_v2/convnext_tiny", {"full", "random_subsample23", "clean_consensus"}),
    ("MaleVis/Swin-Tiny", ROOT / "results/consensus_23_swin_tiny_v1/swin_tiny", {"full", "random_subsample23", "clean_consensus"}),
    ("Malimg/ResNet50", ROOT / "results/malimg_resnet50_seed42_v1/resnet50", {"full", "random_subsample20", "clean_consensus"}),
    ("Virus-MNIST/Spatial-CNN", ROOT / "results/virusmnist_spatial_cnn_5seeds_v1/spatial_cnn", {"full", "random_subsample9", "clean_consensus"}),
]


def close(a: float, b: float, tol: float = 2e-8) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


errors: list[str] = []
checked = 0

for study_name, base, conditions in STUDIES:
    runs = []
    for condition in sorted(conditions):
        for seed_dir in sorted((base / condition).glob("seed_*")):
            seed = int(seed_dir.name.removeprefix("seed_"))
            runs.append((condition, seed, seed_dir))
    actual = {(c, s) for c, s, _ in runs}
    expected = {(c, s) for c in conditions for s in SEEDS}
    if actual != expected:
        errors.append(f"{study_name}: run set differs: missing={expected-actual}, extra={actual-expected}")

    initial_by_seed: dict[int, set[str]] = {s: set() for s in SEEDS}
    for condition, seed, run_dir in runs:
        checked += 1
        metrics = json.loads((run_dir / "metrics.json").read_text())
        initial_by_seed[seed].add(metrics["initial_state_sha256"])
        predictions = list(csv.DictReader((run_dir / "predictions.csv").open(newline="")))
        labels = [k.removeprefix("prob_") for k in predictions[0] if k.startswith("prob_")]
        true = [row["true_label"] for row in predictions]
        pred = [row["predicted_label"] for row in predictions]
        if len(predictions) != int(metrics["evaluation_images"]):
            errors.append(f"{run_dir}: evaluation row count mismatch")
        if any(t not in labels or p not in labels for t, p in zip(true, pred)):
            errors.append(f"{run_dir}: prediction labels differ from probability columns")

        cm = {t: Counter() for t in labels}
        for t, p in zip(true, pred):
            cm[t][p] += 1
        accuracy = sum(t == p for t, p in zip(true, pred)) / len(true)
        class_rows = []
        for label in labels:
            tp = cm[label][label]
            support = sum(cm[label].values())
            predicted = sum(cm[t][label] for t in labels)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_rows.append((label, support, precision, recall, f1))
        macro_f1 = sum(r[4] for r in class_rows) / len(class_rows)
        macro_recall = sum(r[3] for r in class_rows) / len(class_rows)
        for key, found in (("accuracy", accuracy), ("macro_f1", macro_f1), ("macro_recall", macro_recall)):
            if not close(metrics[key], found):
                errors.append(f"{run_dir}: {key} metrics={metrics[key]} recomputed={found}")

        stored_class = list(csv.DictReader((run_dir / "per_class.csv").open(newline="")))
        if [r["class_label"] for r in stored_class] != labels:
            errors.append(f"{run_dir}: per_class label order mismatch")
        for stored, calc in zip(stored_class, class_rows):
            _, support, precision, recall, f1 = calc
            if int(stored["support"]) != support or not all(
                close(stored[k], v) for k, v in (("precision", precision), ("recall", recall), ("f1", f1))
            ):
                errors.append(f"{run_dir}: per_class mismatch for {stored['class_label']}")

        stored_cm = list(csv.DictReader((run_dir / "confusion_matrix.csv").open(newline="")))
        if [r["true_class"] for r in stored_cm] != labels:
            errors.append(f"{run_dir}: confusion-matrix label order mismatch")
        for row in stored_cm:
            if any(int(row[p]) != cm[row["true_class"]][p] for p in labels):
                errors.append(f"{run_dir}: confusion matrix values differ")

        for filename, digest in metrics["artifacts_sha256"].items():
            path = run_dir / filename
            if not path.exists() or sha256(path) != digest:
                errors.append(f"{run_dir}: artifact digest mismatch for {filename}")

        history_path = run_dir / ("training_history.csv" if (run_dir / "training_history.csv").exists() else "finetuning_history.csv")
        history = list(csv.DictReader(history_path.open(newline="")))
        loss_key = "development_loss" if "development_loss" in history[0] else "val_loss"
        min_row = min(history, key=lambda row: float(row[loss_key]))
        selected_epoch = metrics.get("selected_epoch", metrics.get("selected_phase_epoch"))
        # Keras CSVLogger writes zero-based epoch indices, while the run
        # metadata reports the human-readable one-based epoch number. The
        # PyTorch/Swin logger writes one-based indices directly.
        history_epoch = int(float(min_row["epoch"])) + (1 if loss_key == "val_loss" else 0)
        if history_epoch != int(selected_epoch):
            errors.append(f"{run_dir}: selected epoch {selected_epoch}, minimum-history epoch {min_row['epoch']}")
        if not close(metrics["development_loss"], min_row[loss_key], tol=2e-6):
            errors.append(f"{run_dir}: selected development loss differs from minimum history")

    for seed, digests in initial_by_seed.items():
        if len(digests) != 1:
            errors.append(f"{study_name}: seed {seed} has {len(digests)} initial-state digests across conditions")
    print(f"{study_name}: {len(runs)} runs checked; paired initial states verified")

print(f"TOTAL: {checked} runs checked")
if errors:
    print(f"FAILURES: {len(errors)}")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)
print("PASS: predictions, aggregate metrics, per-class metrics, confusion matrices, artifact digests, checkpoint-selection histories, and paired initial states are internally consistent.")
