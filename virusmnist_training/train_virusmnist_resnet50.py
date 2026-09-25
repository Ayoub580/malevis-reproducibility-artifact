#!/usr/bin/env python3
"""Train ResNet50 for the Virus-MNIST redundancy experiment.

Conditions
----------
full
    Conventional pooled 70/20/10 image split restricted to the nine eligible
    classes after mixed-label consensus groups are excluded.
clean_consensus
    One representative per exact (aHash, dHash, pHash) tuple, followed by a
    70/20/10 split.
random_subsample9
    A random image-level subset of Full-9 with exactly the same class and
    partition counts as Clean-Consensus-9. Consensus duplicates are retained
    when selected.

The script never overwrites changed manifests or completed artifacts.
"""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

MALEVIS_TRAINING = Path(__file__).resolve().parents[1] / "malevis_training"
sys.path.insert(0, str(MALEVIS_TRAINING))

import original_training_reference as reference
import train_full_malevis as audit
import train_resnet_preprocess_corrected as corrected
from train_dedup_before_split import load_protocol
from tune_and_compare_malevis import freeze_csv, freeze_json


PARTITIONS = ("fitting", "development", "evaluation")
DISPLAY = {
    "full": "Full-9",
    "clean_consensus": "Clean-Consensus-9",
    "random_subsample9": "Random-Subsample-9",
}
ORDER = tuple(DISPLAY)


def prefer_cached_resnet_weights():
    """Use the verified local ImageNet file when Keras cache writes are blocked."""
    cached = Path.home() / ".keras" / "models" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    expected = "66c8b43daff3fcc15bc4f30e3d2a167e21a14d9c9598a5394e5516471f4af504"
    if not cached.is_file():
        return
    if audit.sha(cached.read_bytes()) != expected:
        raise ValueError(f"Cached ResNet50 weights have unexpected SHA-256: {cached}")

    def build(num_classes, img_size):
        base = reference.ResNet50(
            weights=str(cached), include_top=False, input_shape=img_size + (3,)
        )
        base.trainable = False
        x = reference.GlobalAveragePooling2D(name="avg_pool")(base.output)
        x = reference.Dense(
            512, kernel_regularizer=reference.regularizers.l2(1e-4)
        )(x)
        x = reference.BatchNormalization()(x)
        x = reference.Activation("relu")(x)
        x = reference.Dropout(0.5)(x)
        predictions = reference.Dense(num_classes, activation="softmax")(x)
        return reference.Model(inputs=base.input, outputs=predictions), base

    reference.MODELS["resnet50"] = build


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_rank(seed, purpose, sample_id):
    return hashlib.sha256(
        f"{seed}\0{purpose}\0{sample_id}".encode("utf-8")
    ).hexdigest()


def flatten(parts):
    return [r for partition in PARTITIONS for r in parts[partition]]


def hash_index(path):
    rows = read_csv(path)
    result = {r["sample_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Repeated sample IDs in hash manifest.")
    return result


def make_random_subsample(source_parts, clean_parts, selection_seed, split_seed):
    """Match every clean class and partition count without deduplicating."""
    source = flatten(source_parts)
    clean = flatten(clean_parts)
    classes = sorted({r["class_label"] for r in clean})
    selected_parts = {p: [] for p in PARTITIONS}
    for cls in classes:
        candidates = sorted(
            (r for r in source if r["class_label"] == cls),
            key=lambda r: (stable_rank(selection_seed, "select", r["sample_id"]),
                           r["sample_id"]),
        )
        targets = {
            p: sum(r["class_label"] == cls for r in clean_parts[p])
            for p in PARTITIONS
        }
        needed = sum(targets.values())
        if needed > len(candidates):
            raise ValueError(f"Cannot sample {needed} images from class {cls}.")
        chosen = candidates[:needed]
        chosen.sort(
            key=lambda r: (stable_rank(split_seed, "split", r["sample_id"]),
                           r["sample_id"])
        )
        start = 0
        for partition in PARTITIONS:
            stop = start + targets[partition]
            selected_parts[partition].extend(
                {**r, "partition": partition} for r in chosen[start:stop]
            )
            start = stop
    for partition in PARTITIONS:
        selected_parts[partition].sort(key=lambda r: r["sample_id"])
    return selected_parts


def consensus_value(sample_id, hashes):
    row = hashes[sample_id]
    return (row["ahash"], row["dhash"], row["phash"])


def condition_summary(name, parts, classes, hashes):
    fit_values = {consensus_value(r["sample_id"], hashes)
                  for r in parts["fitting"]}
    all_rows = flatten(parts)
    return {
        "condition": name,
        "display_name": DISPLAY[name],
        "classes": len(classes),
        "images": len(all_rows),
        "fitting": len(parts["fitting"]),
        "development": len(parts["development"]),
        "evaluation": len(parts["evaluation"]),
        "distinct_consensus_groups": len({
            consensus_value(r["sample_id"], hashes) for r in all_rows
        }),
        "development_images_matching_fitting_group": sum(
            consensus_value(r["sample_id"], hashes) in fit_values
            for r in parts["development"]
        ),
        "evaluation_images_matching_fitting_group": sum(
            consensus_value(r["sample_id"], hashes) in fit_values
            for r in parts["evaluation"]
        ),
    }


def class_count_table(conditions):
    classes = sorted({r["class_label"] for _, parts, _ in conditions.values()
                      for r in flatten(parts)})
    rows = []
    for cls in classes:
        row = {"class_label": cls}
        for name, parts, condition_classes in conditions.values():
            for partition in PARTITIONS:
                row[f"{name}_{partition}"] = sum(
                    r["class_label"] == cls for r in parts[partition]
                ) if cls in condition_classes else 0
        rows.append(row)
    return rows


def validate_files(dataset, conditions):
    for _, parts, _ in conditions.values():
        for row in flatten(parts):
            path = dataset / row["sample_id"]
            if not path.is_file():
                raise ValueError(f"Missing dataset image: {path}")
            if audit.sha(path.read_bytes()) != row["file_sha256"]:
                raise ValueError(f"Dataset image changed: {path}")


def common_experiment(args, classes, protocol, condition, tf, keras):
    return {
        "model": args.model,
        "classes": classes,
        "protocol": protocol,
        "condition": condition,
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
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": audit.PIL.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "gpus": [tf.config.experimental.get_device_details(g)
                 for g in tf.config.list_physical_devices("GPU")],
        "source_sha256": {
            Path(f).name: audit.sha(Path(f).read_bytes())
            for f in (__file__, corrected.__file__, reference.__file__, audit.__file__)
        },
    }


def summarize_results(output):
    completed = []
    for condition in ORDER:
        for path in sorted((output / "resnet50" / condition).glob("seed_*/metrics.json")):
            completed.append(json.loads(path.read_text(encoding="utf-8")))
    if completed:
        audit.write_csv(output / "results.csv", [
            {k: v for k, v in row.items() if k != "artifacts_sha256"}
            for row in completed
        ])

    # Paired conditions for a seed must start from identical weights.
    for seed in sorted({r["seed"] for r in completed}):
        rows = [r for r in completed
                if r["seed"] == seed and r["condition"] in
                set(ORDER)]
        if len(rows) > 1 and len({r["initial_state_sha256"] for r in rows}) != 1:
            raise ValueError(f"Nine-class initialization differs for seed {seed}.")

    contrasts = []
    required = set(ORDER)
    for seed in sorted({r["seed"] for r in completed}):
        by = {r["condition"]: r for r in completed if r["seed"] == seed}
        if not required <= set(by):
            continue
        row = {"seed": seed}
        for metric in ("accuracy", "macro_f1", "macro_recall"):
            row[f"full_minus_clean_{metric}_pp"] = 100 * (
                by["full"][metric] - by["clean_consensus"][metric])
            row[f"full_minus_subsample_{metric}_pp"] = 100 * (
                by["full"][metric] - by["random_subsample9"][metric])
            row[f"subsample_minus_clean_{metric}_pp"] = 100 * (
                by["random_subsample9"][metric] - by["clean_consensus"][metric])
        contrasts.append(row)
    if contrasts:
        audit.write_csv(output / "seed_contrasts.csv", contrasts)

    metric_rows = []
    for condition in ORDER:
        rows = [r for r in completed if r["condition"] == condition]
        if not rows:
            continue
        for metric in ("accuracy", "macro_f1", "macro_recall"):
            values = [100 * float(r[metric]) for r in rows]
            metric_rows.append({
                "condition": condition,
                "display_name": DISPLAY[condition],
                "metric": metric,
                "completed_seeds": len(values),
                "mean_percent": statistics.mean(values),
                "sample_sd_pp": statistics.stdev(values) if len(values) > 1 else "",
                "minimum_percent": min(values),
                "maximum_percent": max(values),
            })
    if metric_rows:
        audit.write_csv(output / "metric_summary.csv", metric_rows)

    contrast_rows = []
    if contrasts:
        for key in contrasts[0]:
            if key == "seed":
                continue
            values = [float(r[key]) for r in contrasts]
            contrast_rows.append({
                "contrast": key,
                "completed_seeds": len(values),
                "mean_pp": statistics.mean(values),
                "sample_sd_pp": statistics.stdev(values) if len(values) > 1 else "",
                "minimum_pp": min(values),
                "maximum_pp": max(values),
            })
        audit.write_csv(output / "contrast_summary.csv", contrast_rows)
    return completed, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--full-protocol-dir", type=Path, required=True)
    parser.add_argument("--clean-protocol-dir", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=ORDER, default=list(ORDER))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 101, 7, 2024, 99])
    parser.add_argument("--subsample-seed", type=int, default=20260923)
    parser.add_argument("--subsample-split-seed", type=int, default=20260920)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=35)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    args.model = "resnet50"
    args.custom_epochs = 50
    if min(args.batch_size, args.epochs_phase1, args.epochs_phase2) < 1:
        parser.error("Batch size and epoch limits must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(
            seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be unique integers between 0 and 2**32-1.")
    args.dataset = args.dataset.resolve()
    args.full_protocol_dir = args.full_protocol_dir.resolve()
    args.clean_protocol_dir = args.clean_protocol_dir.resolve()
    args.hash_manifest = args.hash_manifest.resolve()
    args.output = args.output.resolve()

    full10, classes10, full_meta = load_protocol(args.full_protocol_dir)
    clean9, classes9, clean_meta = load_protocol(
        args.clean_protocol_dir, expected_rule="hash_consensus"
    )
    if not set(classes9) < set(classes10):
        raise ValueError("Expected Clean-Consensus to contain a strict subset of Full classes.")
    full9 = {
        p: [r for r in full10[p] if r["class_label"] in set(classes9)]
        for p in PARTITIONS
    }
    random9 = make_random_subsample(
        full10, clean9, args.subsample_seed, args.subsample_split_seed
    )
    for p in PARTITIONS:
        if Counter(r["class_label"] for r in random9[p]) != Counter(
                r["class_label"] for r in clean9[p]):
            raise AssertionError(f"Random and clean class counts differ in {p}.")

    conditions = {
        "full": ("full", full9, classes9),
        "clean_consensus": ("clean_consensus", clean9, classes9),
        "random_subsample9": ("random_subsample9", random9, classes9),
    }
    hashes = hash_index(args.hash_manifest)
    ids = {r["sample_id"] for _, parts, _ in conditions.values() for r in flatten(parts)}
    if not ids <= set(hashes):
        raise ValueError("Hash manifest lacks at least one experiment image.")
    validate_files(args.dataset, conditions)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, parts, _ in conditions.values():
        for partition in PARTITIONS:
            freeze_csv(args.output / f"{name}_{partition}.csv", parts[partition])
    summaries = [condition_summary(name, parts, classes, hashes)
                 for name, parts, classes in conditions.values()]
    freeze_csv(args.output / "dataset_summary.csv", summaries)
    freeze_csv(args.output / "class_counts.csv",
               class_count_table(conditions))
    study = {
        "conditions": {
            name: {
                "display_name": DISPLAY[name],
                "classes": classes,
                "partition_counts": {p: len(parts[p]) for p in PARTITIONS},
            }
            for name, parts, classes in conditions.values()
        },
        "subsample_definition": (
            "Class-size- and partition-size-matched individual-image sample from Full-9; "
            "consensus duplicates are intentionally retained when selected."
        ),
        "subsample_seed": args.subsample_seed,
        "subsample_split_seed": args.subsample_split_seed,
        "full_protocol_sha256": audit.sha(
            (args.full_protocol_dir / "protocol.json").read_bytes()),
        "clean_protocol_sha256": audit.sha(
            (args.clean_protocol_dir / "protocol.json").read_bytes()),
        "hash_manifest_sha256": audit.sha(args.hash_manifest.read_bytes()),
        "interpretation": {
            "full_minus_clean": "total benchmark-protocol difference",
            "full_minus_subsample": "sample-quantity and evaluation-population difference",
            "subsample_minus_clean": (
                "size- and class-count-adjusted consensus-deduplication difference"
            ),
        },
    }
    freeze_json(args.output / "study.json", study)
    print("Frozen Virus-MNIST study:", json.dumps(summaries, indent=2), flush=True)
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
    prefer_cached_resnet_weights()

    paired_experiment_path = args.output / "experiment.json"
    if paired_experiment_path.exists():
        paired_experiment = json.loads(paired_experiment_path.read_text(encoding="utf-8"))
        if paired_experiment.get("classes") != classes9:
            raise ValueError("Existing paired experiment uses a different class set.")
        for key, actual in (("batch_size", args.batch_size),
                            ("epochs_phase1", args.epochs_phase1),
                            ("epochs_phase2", args.epochs_phase2)):
            if paired_experiment.get(key) != actual:
                raise ValueError(f"Existing paired experiment has different {key}.")
    else:
        paired_experiment = common_experiment(
            args, classes9, {"full": full_meta, "clean_consensus": clean_meta},
            "paired_9_class_conditions", tf, keras
        )
        freeze_json(paired_experiment_path, paired_experiment)

    experiments = {
        "full": paired_experiment,
        "clean_consensus": paired_experiment,
        "random_subsample9": common_experiment(
            args, classes9, {
                "definition": study["subsample_definition"],
                "selection_seed": args.subsample_seed,
                "split_seed": args.subsample_split_seed,
            }, "random_subsample9", tf, keras),
    }
    freeze_json(args.output / "experiment_random_subsample9.json",
                experiments["random_subsample9"])

    for seed in args.seeds:
        for condition in ORDER:
            if condition not in args.conditions:
                continue
            _, parts, _ = conditions[condition]
            print(f"Virus-MNIST study: {DISPLAY[condition]}, seed {seed}", flush=True)
            corrected.run_one(
                args, experiments[condition], parts["fitting"], parts["development"],
                parts["evaluation"], condition, seed, reference, tf
            )
    completed, contrasts = summarize_results(args.output)
    print(f"Completed metric files: {len(completed)}; complete seed contrasts: "
          f"{len(contrasts)}.", flush=True)


if __name__ == "__main__":
    main()
