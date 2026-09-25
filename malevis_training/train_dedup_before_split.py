#!/usr/bin/env python3
"""Train the conventional and global-dedup-first MaleVis benchmarks.

Each condition uses its own 70/20/10 split.  Consequently, their score
difference is a benchmark-protocol difference; it is not a paired estimate on
the same evaluation images.  Use ``train_random_split_inflation.py`` alongside
this script for the fixed-evaluation causal comparison.
"""
import argparse
import csv
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np

import original_training_reference as reference
import train_full_malevis as audit
import train_resnet_preprocess_corrected as corrected
from tune_and_compare_malevis import digest_rows, freeze_csv, freeze_json


PARTITIONS = ("fitting", "development", "evaluation")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_protocol(directory, expected_rule=None):
    directory = Path(directory)
    manifest_path = directory / "manifest.csv"
    protocol_path = directory / "protocol.json"
    if not manifest_path.exists() or not protocol_path.exists():
        raise ValueError(f"Incomplete protocol directory: {directory}")
    meta = json.loads(protocol_path.read_text(encoding="utf-8"))
    if audit.sha(manifest_path.read_bytes()) != meta["manifest_sha256"]:
        raise ValueError(f"Manifest checksum mismatch in {directory}.")
    if expected_rule is not None and meta.get("rule") != expected_rule:
        raise ValueError(
            f"Expected {expected_rule} protocol in {directory}; found {meta.get('rule')}."
        )
    rows = read_csv(manifest_path)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError(f"Repeated sample IDs in {directory}.")
    if {r["partition"] for r in rows} != set(PARTITIONS):
        raise ValueError(f"Missing or unknown partitions in {directory}.")
    parts = {
        p: sorted((r for r in rows if r["partition"] == p),
                  key=lambda r: r["sample_id"])
        for p in PARTITIONS
    }
    classes = sorted({r["class_label"] for r in rows})
    for partition, subset in parts.items():
        if sorted({r["class_label"] for r in subset}) != classes:
            raise ValueError(f"{directory}/{partition} does not contain every class.")
    return parts, classes, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--full-protocol-dir", type=Path, required=True)
    parser.add_argument("--clean-protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions", nargs="+",
        choices=("full", "clean_exact", "clean_consensus"),
        help="Defaults to Full and the clean condition implied by its protocol rule.",
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 101, 7, 2024, 99])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=35)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    args.model = "resnet50"
    args.custom_epochs = 50  # Shared runner requirement; unused for ResNet50.
    if min(args.batch_size, args.epochs_phase1, args.epochs_phase2) < 1:
        parser.error("Batch size and epoch limits must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(
            seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be unique integers between 0 and 2**32-1.")
    args.dataset = args.dataset.resolve()
    args.full_protocol_dir = args.full_protocol_dir.resolve()
    args.clean_protocol_dir = args.clean_protocol_dir.resolve()
    args.output = args.output.resolve()
    if args.output in (args.dataset, args.full_protocol_dir, args.clean_protocol_dir):
        parser.error("Output must differ from the dataset and protocol directories.")

    full, full_classes, full_meta = load_protocol(args.full_protocol_dir)
    clean, clean_classes, clean_meta = load_protocol(args.clean_protocol_dir)
    clean_rule = clean_meta.get("rule")
    if clean_rule not in {"exact_input", "hash_consensus"}:
        raise ValueError(f"Unsupported clean protocol rule: {clean_rule}")
    if not set(clean_classes) <= set(full_classes):
        raise ValueError("Clean protocol contains classes absent from the Full protocol.")
    # A reduced-class clean benchmark must be compared with a Full baseline using
    # exactly the same label space. The original frozen Full split is filtered,
    # never resampled.
    full = {
        partition: [r for r in subset if r["class_label"] in set(clean_classes)]
        for partition, subset in full.items()
    }
    for partition, subset in full.items():
        if sorted({r["class_label"] for r in subset}) != clean_classes:
            raise ValueError(f"Filtered Full {partition} lacks a clean-protocol class.")

    clean_name = "clean_exact" if clean_rule == "exact_input" else "clean_consensus"
    if clean_rule == "exact_input":
        clean_values = [r["input_sha256"] for p in PARTITIONS for r in clean[p]]
    else:
        clean_values = [r["duplicate_group_id"] for p in PARTITIONS for r in clean[p]]
    if len(clean_values) != len(set(clean_values)):
        raise ValueError(f"{clean_name} manifest repeats a deduplication group.")

    conditions = {"full": full, clean_name: clean}
    requested = args.conditions or ["full", clean_name]
    invalid = sorted(set(requested) - set(conditions))
    if invalid:
        parser.error(
            f"Conditions {invalid} do not match clean rule {clean_rule}; "
            f"choose from {sorted(conditions)}."
        )
    args.conditions = requested
    args.output.mkdir(parents=True, exist_ok=True)
    for name, parts in conditions.items():
        for partition in PARTITIONS:
            freeze_csv(args.output / f"{name}_{partition}.csv", parts[partition])
    freeze_json(args.output / "conditions.json", {
        "interpretation": (
            "Each condition has its own class-stratified 70/20/10 evaluation population. "
            "The score contrast measures the complete benchmark-protocol change and is not "
            "a paired fixed-test causal estimate."
        ),
        "classes": clean_classes,
        "clean_rule": clean_rule,
        "full_protocol_sha256": audit.sha(
            (args.full_protocol_dir / "protocol.json").read_bytes()),
        "clean_protocol_sha256": audit.sha(
            (args.clean_protocol_dir / "protocol.json").read_bytes()),
        "counts": {
            name: {p: len(parts[p]) for p in PARTITIONS}
            for name, parts in conditions.items()
        },
    })
    print("Frozen dedup-before-split comparison:", {
        name: {p: len(parts[p]) for p in PARTITIONS}
        for name, parts in conditions.items()
    }, flush=True)
    if args.prepare_only:
        return

    import keras
    import tensorflow as tf

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not args.allow_cpu:
        raise RuntimeError("TensorFlow cannot see a GPU. Check the CUDA environment.")
    reference.setup_gpu()
    experiment = {
        "model": args.model,
        "classes": clean_classes,
        "protocol": {
            "full": full_meta,
            clean_name: clean_meta,
        },
        "batch_size": args.batch_size,
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "custom_epochs": args.custom_epochs,
        "pretrained": "Keras ResNet50 ImageNet weights",
        "normalization": (
            "keras.applications.resnet50.preprocess_input: RGB 0..255 to BGR minus "
            "ImageNet means; no /255"
        ),
        "interpolation": "bicubic 224x224",
        "augmentation": None,
        "precision": "float32",
        "selection": "best phase-2 development loss; evaluation requested explicitly",
        "comparison_note": (
            f"Global {clean_rule} deduplication precedes the clean split; Full is restricted "
            "to the same classes, and evaluation images differ between conditions."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": audit.PIL.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "gpus": [tf.config.experimental.get_device_details(g) for g in gpus],
        "source_sha256": {
            Path(f).name: audit.sha(Path(f).read_bytes())
            for f in (__file__, corrected.__file__, reference.__file__, audit.__file__)
        },
    }
    freeze_json(args.output / "experiment.json", experiment)
    for seed in args.seeds:
        for condition in args.conditions:
            parts = conditions[condition]
            print(f"Dedup-before-split: ResNet50, {condition}, seed {seed}", flush=True)
            corrected.run_one(
                args, experiment, parts["fitting"], parts["development"],
                parts["evaluation"], condition, seed, reference, tf
            )

    completed = [json.loads(path.read_text()) for path in
                 sorted((args.output / args.model).glob("*/seed_*/metrics.json"))]
    if not completed:
        print("Development runs completed; add --evaluate to score evaluation sets.", flush=True)
        return
    for seed in {r["seed"] for r in completed}:
        states = {r["initial_state_sha256"] for r in completed if r["seed"] == seed}
        if len(states) != 1:
            raise ValueError(f"Initial model state differs between conditions for seed {seed}.")
    audit.write_csv(args.output / "results.csv", [
        {k: v for k, v in row.items() if k != "artifacts_sha256"}
        for row in completed
    ])
    contrasts = []
    for seed in sorted({r["seed"] for r in completed}):
        by_condition = {r["condition"]: r for r in completed if r["seed"] == seed}
        if set(by_condition) >= {"full", clean_name}:
            row = {"seed": seed}
            for metric in ("accuracy", "macro_f1", "macro_recall"):
                row[f"full_minus_clean_{metric}_pp"] = 100 * (
                    by_condition["full"][metric] - by_condition[clean_name][metric]
                )
            contrasts.append(row)
    if contrasts:
        audit.write_csv(args.output / "benchmark_protocol_differences.csv", contrasts)
        print("Saved protocol differences; evaluation populations differ by design.", flush=True)


if __name__ == "__main__":
    main()
