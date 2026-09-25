#!/usr/bin/env python3
"""Train a native-resolution spatial CNN on frozen Virus-MNIST manifests.

Virus-MNIST images are 32x32 grayscale thumbnails.  This LeNet-style model
preserves absolute spatial positions through a flattened dense head instead of
global average pooling.  Checkpoints are selected only by development loss.
Evaluation is accessed only when
``--evaluate`` is supplied, and a development-only checkpoint can later be
evaluated without retraining.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np

MALEVIS_TRAINING = Path(__file__).resolve().parents[1] / "malevis_training"
sys.path.insert(0, str(MALEVIS_TRAINING))

import train_full_malevis as audit
from tune_and_compare_malevis import digest_rows, freeze_csv, freeze_json


PARTITIONS = ("fitting", "development", "evaluation")
CONDITIONS = ("full", "clean_consensus", "random_subsample9")
DISPLAY = {
    "full": "Full-9",
    "clean_consensus": "Clean-Consensus-9",
    "random_subsample9": "Random-Subsample-9",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_condition(directory: Path, condition: str) -> tuple[dict, list[str]]:
    parts = {
        partition: read_csv(directory / f"{condition}_{partition}.csv")
        for partition in PARTITIONS
    }
    classes = sorted({row["class_label"] for row in parts["fitting"]})
    if len(classes) != 9:
        raise ValueError(f"{condition} must contain nine classes; found {classes}")
    identifiers = []
    for partition, rows in parts.items():
        labels = sorted({row["class_label"] for row in rows})
        if labels != classes:
            raise ValueError(f"{condition}/{partition} lacks a class")
        current = {row["sample_id"] for row in rows}
        if len(current) != len(rows):
            raise ValueError(f"Repeated sample ID in {condition}/{partition}")
        identifiers.append(current)
    if (identifiers[0] & identifiers[1] or identifiers[0] & identifiers[2]
            or identifiers[1] & identifiers[2]):
        raise ValueError(f"A sample occurs in multiple {condition} partitions")
    return parts, classes


def verify_dataset(root: Path, conditions: dict) -> None:
    checked = set()
    for parts, _ in conditions.values():
        for partition in PARTITIONS:
            for row in parts[partition]:
                sample = row["sample_id"]
                if sample in checked:
                    continue
                checked.add(sample)
                path = root / sample
                if not path.is_file():
                    raise ValueError(f"Missing image: {path}")
                if audit.sha(path.read_bytes()) != row["file_sha256"]:
                    raise ValueError(f"Image checksum changed: {path}")


def make_generators(reference, root, parts, classes, batch_size, seed):
    import pandas as pd

    frames = [
        pd.DataFrame({
            "filename": [str(root / row["sample_id"]) for row in parts[p]],
            "class": [row["class_label"] for row in parts[p]],
        })
        for p in PARTITIONS
    ]
    # The files are already 32x32 grayscale.  Rescaling changes only numeric
    # range; no geometric or stochastic augmentation is applied.
    datagen = reference.ImageDataGenerator(rescale=1.0 / 255.0)
    generators = tuple(
        datagen.flow_from_dataframe(
            frame,
            x_col="filename",
            y_col="class",
            target_size=(32, 32),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=index == 0,
            seed=seed,
            classes=classes,
            interpolation="nearest",
        )
        for index, frame in enumerate(frames)
    )
    expected = {label: index for index, label in enumerate(classes)}
    for generator, partition in zip(generators, PARTITIONS):
        if generator.class_indices != expected:
            raise ValueError("Generator changed the frozen class mapping")
        if generator.samples != len(parts[partition]):
            raise ValueError("Generator dropped a manifest sample")
    return generators


def residual_block(keras, x, filters, stride, weight_decay, name):
    regularizer = keras.regularizers.l2(weight_decay)
    shortcut = x
    y = keras.layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        kernel_initializer="he_normal", kernel_regularizer=regularizer,
        name=f"{name}_conv1",
    )(x)
    y = keras.layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = keras.layers.Activation("relu", name=f"{name}_relu1")(y)
    y = keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_initializer="he_normal", kernel_regularizer=regularizer,
        name=f"{name}_conv2",
    )(y)
    y = keras.layers.BatchNormalization(name=f"{name}_bn2")(y)
    if stride != 1 or int(shortcut.shape[-1]) != filters:
        shortcut = keras.layers.Conv2D(
            filters, 1, strides=stride, use_bias=False,
            kernel_initializer="he_normal", kernel_regularizer=regularizer,
            name=f"{name}_projection",
        )(shortcut)
        shortcut = keras.layers.BatchNormalization(
            name=f"{name}_projection_bn"
        )(shortcut)
    y = keras.layers.Add(name=f"{name}_add")([shortcut, y])
    return keras.layers.Activation("relu", name=f"{name}_out")(y)


def build_model(keras, class_count, weight_decay, dropout):
    regularizer = keras.regularizers.l2(weight_decay)
    inputs = keras.Input((32, 32, 1), name="image")
    x = inputs
    for stage, filters in enumerate((32, 64, 128), 1):
        for convolution in (1, 2):
            x = keras.layers.Conv2D(
                filters, 3, padding="same", use_bias=False,
                kernel_initializer="he_normal", kernel_regularizer=regularizer,
                name=f"stage{stage}_conv{convolution}",
            )(x)
            x = keras.layers.BatchNormalization(
                name=f"stage{stage}_bn{convolution}"
            )(x)
            x = keras.layers.Activation(
                "relu", name=f"stage{stage}_relu{convolution}"
            )(x)
        x = keras.layers.MaxPooling2D(2, name=f"stage{stage}_pool")(x)
        x = keras.layers.Dropout(
            0.10 + 0.05 * stage, name=f"stage{stage}_dropout"
        )(x)
    # Flattening retains the location of discriminative byte regions.  The
    # Virus-MNIST baseline paper reports position-dependent pixel structure.
    x = keras.layers.Flatten(name="spatial_flatten")(x)
    x = keras.layers.Dense(
        512, use_bias=False, kernel_initializer="he_normal",
        kernel_regularizer=regularizer, name="spatial_dense",
    )(x)
    x = keras.layers.BatchNormalization(name="spatial_dense_bn")(x)
    x = keras.layers.Activation("relu", name="spatial_dense_relu")(x)
    x = keras.layers.Dropout(dropout, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        class_count, activation="softmax", kernel_regularizer=regularizer,
        name="predictions",
    )(x)
    return keras.Model(inputs, outputs, name="virusmnist_spatial_cnn")


def initial_state_digest(model) -> str:
    digest = hashlib.sha256()
    for weight in model.get_weights():
        digest.update(audit.payload(weight))
    return digest.hexdigest()


def class_weights(rows, classes, mode):
    if mode == "none":
        return {index: 1.0 for index in range(len(classes))}
    counts = Counter(row["class_label"] for row in rows)
    total = len(rows)
    raw = {
        index: np.sqrt(total / (len(classes) * counts[label]))
        for index, label in enumerate(classes)
    }
    # Preserve average example weight one and avoid extreme optimization shifts.
    average = sum(raw[index] * counts[label]
                  for index, label in enumerate(classes)) / total
    return {
        index: float(np.clip(value / average, 0.5, 2.0))
        for index, value in raw.items()
    }


def train_model(tf, keras, model, train, development, weights, args, run):
    optimizer = keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.optimizer_weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing
        ),
        metrics=["accuracy"],
    )
    history = model.fit(
        train,
        validation_data=development,
        epochs=args.epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.early_stopping_patience,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.25,
                patience=args.lr_patience, min_lr=args.minimum_learning_rate,
            ),
            keras.callbacks.CSVLogger(str(run / "training_history.csv")),
        ],
        verbose=2,
    )
    selected = int(np.argmin(history.history["val_loss"]))
    return {
        "selected_epoch": selected + 1,
        "development_loss": float(history.history["val_loss"][selected]),
        "development_accuracy": float(
            history.history["val_accuracy"][selected]
        ),
        "epochs_completed": len(history.epoch),
    }


def prediction_outputs(model, generator, rows, classes, audit_module, verbose):
    generator.reset()
    probabilities = np.asarray(model.predict(generator, verbose=verbose))
    if probabilities.shape != (len(rows), len(classes)):
        raise ValueError("Prediction array has the wrong shape")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain nonfinite values")
    actual = generator.classes
    predicted = probabilities.argmax(1)
    metrics, per_class, confusion = audit_module.classification_metrics(
        actual, predicted, classes
    )
    records = [
        {
            "sample_id": row["sample_id"],
            "true_label": classes[int(target)],
            "predicted_label": classes[int(guess)],
            **{
                f"prob_{label}": float(score)
                for label, score in zip(classes, scores)
            },
        }
        for row, target, guess, scores in zip(
            rows, actual, predicted, probabilities
        )
    ]
    matrix = [
        {"true_class": label, **dict(zip(classes, map(int, values)))}
        for label, values in zip(classes, confusion)
    ]
    return metrics, per_class, matrix, records


def run_one(args, experiment, parts, classes, condition, seed, reference, tf, keras):
    run = args.output / "spatial_cnn" / condition / f"seed_{seed}"
    config = {
        "experiment_sha256": digest_rows(experiment),
        "condition": condition,
        "seed": seed,
        **{
            f"{partition}_sha256": digest_rows(parts[partition])
            for partition in PARTITIONS
        },
    }
    if (run / "metrics.json").exists():
        freeze_json(run / "run.json", config)
        metrics = json.loads((run / "metrics.json").read_text())
        for name, digest in metrics["artifacts_sha256"].items():
            if audit.sha((run / name).read_bytes()) != digest:
                raise ValueError(f"Changed completed artifact: {run / name}")
        print(f"Reusing completed {condition}, seed {seed}.", flush=True)
        return metrics

    resume_evaluation = (run / "trained.json").exists()
    if not resume_evaluation and run.exists() and any(run.iterdir()):
        if not args.restart_incomplete:
            raise ValueError(
                f"Incomplete run at {run}; use --restart-incomplete to archive it"
            )
        audit.archive_attempt(run)
    run.mkdir(parents=True, exist_ok=True)
    freeze_json(run / "run.json", config)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    train, development, evaluation = make_generators(
        reference, args.dataset, parts, classes, args.batch_size, seed
    )
    weights = class_weights(parts["fitting"], classes, args.class_balance)
    start = time.time()
    if resume_evaluation:
        trained = json.loads((run / "trained.json").read_text())
        if audit.sha((run / "selected.keras").read_bytes()) != trained["checkpoint_sha256"]:
            raise ValueError("Saved checkpoint changed")
        model = keras.models.load_model(run / "selected.keras", compile=False)
    else:
        model = build_model(
            keras, len(classes), args.model_weight_decay, args.dropout
        )
        state_hash = initial_state_digest(model)
        with (run / "model_summary.txt").open("w", encoding="utf-8") as handle:
            model.summary(print_fn=lambda line: handle.write(line + "\n"))
        trained = train_model(
            tf, keras, model, train, development, weights, args, run
        )
        model.save(run / "selected.keras")
        trained.update({
            "initial_state_sha256": state_hash,
            "checkpoint_sha256": audit.sha((run / "selected.keras").read_bytes()),
            "training_seconds": time.time() - start,
            "class_weights": {classes[index]: value
                              for index, value in weights.items()},
        })
        audit.write_json(run / "trained.json", trained)

    dev_metrics, dev_per_class, _, dev_records = prediction_outputs(
        model, development, parts["development"], classes, audit, verbose=0
    )
    audit.write_json(run / "development_metrics.json", dev_metrics)
    audit.write_csv(run / "development_per_class.csv", dev_per_class)
    audit.write_csv(run / "development_predictions.csv", dev_records)
    print("Selected checkpoint development metrics:",
          json.dumps(dev_metrics), flush=True)
    if not args.evaluate:
        print("Development-only run complete; evaluation remains unread.", flush=True)
        return {"status": "development_complete", **dev_metrics}

    metrics, per_class, confusion, records = prediction_outputs(
        model, evaluation, parts["evaluation"], classes, audit, verbose=1
    )
    audit.write_csv(run / "predictions.csv", records)
    audit.write_csv(run / "per_class.csv", per_class)
    audit.write_csv(run / "confusion_matrix.csv", confusion)
    metrics.update({
        "model": "spatial_cnn",
        "condition": condition,
        "seed": seed,
        "fitting_images": len(parts["fitting"]),
        "development_images": len(parts["development"]),
        "evaluation_images": len(parts["evaluation"]),
        **trained,
        "artifacts_sha256": {
            name: audit.sha((run / name).read_bytes())
            for name in (
                "run.json", "trained.json", "selected.keras",
                "predictions.csv", "per_class.csv", "confusion_matrix.csv",
            )
        },
    })
    audit.write_json(run / "metrics.json", metrics)
    print(json.dumps(
        {key: value for key, value in metrics.items()
         if key != "artifacts_sha256"}, indent=2
    ), flush=True)
    return metrics


def summarize(output: Path) -> None:
    rows = []
    for condition in CONDITIONS:
        for path in sorted(
            (output / "spatial_cnn" / condition).glob("seed_*/metrics.json")
        ):
            rows.append(json.loads(path.read_text()))
    if rows:
        audit.write_csv(output / "results.csv", [
            {key: value for key, value in row.items()
             if key not in {"artifacts_sha256", "class_weights"}}
            for row in rows
        ])
        metric_summary = []
        for condition in CONDITIONS:
            condition_rows = [
                row for row in rows if row["condition"] == condition
            ]
            for metric in ("accuracy", "macro_f1", "macro_recall"):
                values = [float(row[metric]) for row in condition_rows]
                if not values:
                    continue
                metric_summary.append({
                    "condition": condition,
                    "metric": metric,
                    "completed_seeds": len(values),
                    "mean": statistics.mean(values),
                    "sample_sd": (
                        statistics.stdev(values) if len(values) > 1 else ""
                    ),
                    "minimum": min(values),
                    "maximum": max(values),
                })
        audit.write_csv(output / "condition_summary.csv", metric_summary)
    for seed in sorted({row["seed"] for row in rows}):
        paired = [row for row in rows if row["seed"] == seed]
        if len(paired) > 1 and len({row["initial_state_sha256"] for row in paired}) != 1:
            raise ValueError(f"Condition initialization differs for seed {seed}")
    contrasts = []
    for seed in sorted({row["seed"] for row in rows}):
        by = {row["condition"]: row for row in rows if row["seed"] == seed}
        if set(CONDITIONS) <= set(by):
            result = {"seed": seed}
            for metric in ("accuracy", "macro_f1", "macro_recall"):
                result[f"full_minus_clean_{metric}_pp"] = 100 * (
                    by["full"][metric] - by["clean_consensus"][metric]
                )
                result[f"subsample_minus_clean_{metric}_pp"] = 100 * (
                    by["random_subsample9"][metric]
                    - by["clean_consensus"][metric]
                )
            contrasts.append(result)
    if contrasts:
        audit.write_csv(output / "seed_contrasts.csv", contrasts)
        summary = []
        # Two-sided 95% Student-t critical values for the planned five paired
        # seeds. Partial summaries remain valid while a long study is running.
        t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
        for key in contrasts[0]:
            if key == "seed":
                continue
            values = [float(row[key]) for row in contrasts]
            sample_sd = statistics.stdev(values) if len(values) > 1 else None
            half_width = (
                t_critical[len(values)] * sample_sd / len(values) ** 0.5
                if len(values) in t_critical else None
            )
            mean = statistics.mean(values)
            summary.append({
                "contrast": key,
                "completed_seeds": len(values),
                "mean_pp": mean,
                "sample_sd_pp": sample_sd if sample_sd is not None else "",
                "ci95_low_pp": (
                    mean - half_width if half_width is not None else ""
                ),
                "ci95_high_pp": (
                    mean + half_width if half_width is not None else ""
                ),
                "minimum_pp": min(values),
                "maximum_pp": max(values),
            })
        audit.write_csv(output / "contrast_summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--optimizer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-weight-decay", type=float, default=5e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.50)
    parser.add_argument("--class-balance", choices=("none", "sqrt"),
                        default="none")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.learning_rate,
           args.minimum_learning_rate) <= 0:
        parser.error("Batch size, epochs, and learning rates must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    args.dataset = args.dataset.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    args.output = args.output.resolve()

    # Freeze all three condition manifests in one experiment record even when
    # the current command requests only Full-9.  This permits later Clean and
    # Random runs in the same output directory without changing provenance.
    conditions = {
        condition: load_condition(args.manifest_dir, condition)
        for condition in CONDITIONS
    }
    verify_dataset(args.dataset, conditions)
    args.output.mkdir(parents=True, exist_ok=True)

    import keras
    import tensorflow as tf
    import original_training_reference as reference

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and not args.allow_cpu:
        raise RuntimeError("TensorFlow cannot see a GPU")
    reference.setup_gpu()
    experiment = {
        "model": "spatial_cnn",
        "input": "native 32x32 grayscale scaled to [0,1]",
        "augmentation": None,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "optimizer_weight_decay": args.optimizer_weight_decay,
        "model_weight_decay": args.model_weight_decay,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "class_balance": args.class_balance,
        "selection": "minimum development cross-entropy; evaluation after selection",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "gpus": [tf.config.experimental.get_device_details(gpu) for gpu in gpus],
        # Freeze all condition manifests regardless of the subset requested in
        # this invocation. This keeps provenance identical when a completed
        # study is resumed with another condition.
        "manifest_sha256": {
            condition: {
                partition: audit.sha(
                    (args.manifest_dir / f"{condition}_{partition}.csv").read_bytes()
                )
                for partition in PARTITIONS
            }
            for condition in CONDITIONS
        },
        "source_sha256": audit.sha(Path(__file__).read_bytes()),
    }
    freeze_json(args.output / "experiment.json", experiment)

    for seed in args.seeds:
        for condition in CONDITIONS:
            if condition not in args.conditions:
                continue
            parts, classes = conditions[condition]
            print(
                f"Virus-MNIST spatial CNN: {DISPLAY[condition]}, seed {seed}",
                flush=True,
            )
            run_one(
                args, experiment, parts, classes, condition, seed,
                reference, tf, keras,
            )
    summarize(args.output)


if __name__ == "__main__":
    main()
