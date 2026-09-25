#!/usr/bin/env python3
"""Train ImageNet-pretrained ConvNeXt-Tiny on frozen MaleVis 23-class manifests.

The first recommended run is Full-23 with seed 42.  The same script can later
run Clean-Consensus-23 and Random-Subsample-23 without changing the model,
preprocessing, class mapping, or checkpoint-selection rule.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import original_training_reference as reference
import train_full_malevis as audit
from tune_and_compare_malevis import digest_rows, freeze_csv, freeze_json


PARTITIONS = ("fitting", "development", "evaluation")
CONDITIONS = ("full", "clean_consensus", "random_subsample23")
DISPLAY = {
    "full": "Full-23",
    "clean_consensus": "Clean-Consensus-23",
    "random_subsample23": "Random-Subsample-23",
}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_condition(directory, condition):
    parts = {}
    for partition in PARTITIONS:
        path = Path(directory) / f"{condition}_{partition}.csv"
        if not path.exists():
            raise ValueError(
                f"Missing {path}. Run train_four_condition_study.py --prepare-only "
                "to create all condition manifests."
            )
        parts[partition] = read_csv(path)
    classes = sorted({r["class_label"] for r in parts["fitting"]})
    if len(classes) != 23:
        raise ValueError(f"{condition} must contain 23 classes; found {len(classes)}.")
    for partition, rows in parts.items():
        if sorted({r["class_label"] for r in rows}) != classes:
            raise ValueError(f"{condition}/{partition} lacks at least one class.")
        if len({r["sample_id"] for r in rows}) != len(rows):
            raise ValueError(f"Repeated sample ID in {condition}/{partition}.")
    ids = [{r["sample_id"] for r in parts[p]} for p in PARTITIONS]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError(f"A sample occurs in multiple {condition} partitions.")
    return parts, classes


def verify_dataset(root, conditions):
    checked = set()
    for parts in conditions.values():
        for partition in PARTITIONS:
            for row in parts[partition]:
                if row["sample_id"] in checked:
                    continue
                checked.add(row["sample_id"])
                path = root / row["sample_id"]
                if not path.is_file():
                    raise ValueError(f"Missing image: {path}")
                if audit.sha(path.read_bytes()) != row["file_sha256"]:
                    raise ValueError(f"Image checksum changed: {path}")


def build_model(keras, class_count):
    backbone = keras.applications.ConvNeXtTiny(
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None,
    )
    backbone.trainable = False
    inputs = keras.Input(shape=(224, 224, 3), name="image")
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = keras.layers.Dense(
        class_count, activation="softmax", name="predictions"
    )(x)
    return keras.Model(inputs, outputs, name="convnext_tiny_malevis"), backbone


def make_generators(reference_module, root, parts, classes, batch_size):
    import pandas as pd

    frames = [
        pd.DataFrame({
            "filename": [str(root / r["sample_id"]) for r in parts[p]],
            "class": [r["class_label"] for r in parts[p]],
        })
        for p in PARTITIONS
    ]
    # ConvNeXt(include_preprocessing=True) expects float RGB values in [0,255].
    datagen = reference_module.ImageDataGenerator()
    generators = tuple(
        datagen.flow_from_dataframe(
            frame,
            x_col="filename",
            y_col="class",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=(index == 0),
            classes=classes,
            interpolation="bicubic",
        )
        for index, frame in enumerate(frames)
    )
    expected = {label: index for index, label in enumerate(classes)}
    for generator, partition in zip(generators, PARTITIONS):
        if generator.class_indices != expected:
            raise ValueError("Generator changed the frozen class mapping.")
        if generator.samples != len(parts[partition]):
            raise ValueError("Generator dropped at least one manifest sample.")
        if generator.interpolation != "bicubic":
            raise ValueError("Generator changed interpolation.")
    return generators


def train_phases(tf, model, backbone, train, development, args, run):
    callbacks = tf.keras.callbacks
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.head_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(
        f"Phase 1: frozen ConvNeXt; Adam learning rate {args.head_learning_rate:g}.",
        flush=True,
    )
    warm = model.fit(
        train,
        validation_data=development,
        epochs=args.epochs_phase1,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            callbacks.CSVLogger(str(run / "warmup_history.csv")),
        ],
        verbose=2,
    )

    backbone.trainable = True
    split = len(backbone.layers) // 2
    for layer in backbone.layers[:split]:
        layer.trainable = False
    audit.write_json(
        run / "finetuning_layers.json",
        [{"name": layer.name, "trainable": layer.trainable}
         for layer in backbone.layers],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.backbone_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(
        "Phase 2: upper half of ConvNeXt unfrozen; Adam learning rate "
        f"{args.backbone_learning_rate:g}.",
        flush=True,
    )
    fine = model.fit(
        train,
        validation_data=development,
        epochs=args.epochs_phase2,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7
            ),
            callbacks.CSVLogger(str(run / "finetuning_history.csv")),
        ],
        verbose=2,
    )
    selected = int(np.argmin(fine.history["val_loss"]))
    return {
        "selected_phase": "finetuning",
        "selected_phase_epoch": selected + 1,
        "development_loss": float(fine.history["val_loss"][selected]),
        "development_accuracy": float(fine.history["val_accuracy"][selected]),
        "warmup_epochs_completed": len(warm.epoch),
        "finetuning_epochs_completed": len(fine.epoch),
    }


def run_one(args, experiment, parts, condition, seed, keras, tf):
    run = args.output / "convnext_tiny" / condition / f"seed_{seed}"
    config = {
        "experiment_sha256": digest_rows(experiment),
        "condition": condition,
        "seed": seed,
        **{f"{p}_sha256": digest_rows(parts[p]) for p in PARTITIONS},
    }
    if (run / "metrics.json").exists():
        freeze_json(run / "run.json", config)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for name, digest in metrics["artifacts_sha256"].items():
            if audit.sha((run / name).read_bytes()) != digest:
                raise ValueError(f"Changed completed artifact: {run / name}")
        print(f"Reusing completed {condition}, seed {seed}.", flush=True)
        return metrics

    resume_evaluation = (run / "trained.json").exists()
    if not resume_evaluation and run.exists() and any(run.iterdir()):
        if not args.restart_incomplete:
            raise ValueError(
                f"Incomplete run at {run}; add --restart-incomplete to archive and restart it."
            )
        audit.archive_attempt(run)
    run.mkdir(parents=True, exist_ok=True)
    freeze_json(run / "run.json", config)
    tf.keras.backend.clear_session()
    reference.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    train, development, evaluation = make_generators(
        reference, args.dataset, parts, experiment["classes"], args.batch_size
    )
    start = time.time()
    if resume_evaluation:
        trained = json.loads((run / "trained.json").read_text(encoding="utf-8"))
        if audit.sha((run / "selected.keras").read_bytes()) != trained["checkpoint_sha256"]:
            raise ValueError("Saved ConvNeXt checkpoint changed.")
        model = tf.keras.models.load_model(run / "selected.keras", compile=False)
    else:
        model, backbone = build_model(keras, len(experiment["classes"]))
        initial = hashlib.sha256()
        for weight in model.get_weights():
            initial.update(audit.payload(weight))
        with (run / "model_summary.txt").open("w", encoding="utf-8") as handle:
            model.summary(print_fn=lambda line: handle.write(line + "\n"))
        trained = train_phases(
            tf, model, backbone, train, development, args, run
        )
        model.save(run / "selected.keras")
        trained.update({
            "initial_state_sha256": initial.hexdigest(),
            "checkpoint_sha256": audit.sha((run / "selected.keras").read_bytes()),
            "training_seconds": time.time() - start,
        })
        audit.write_json(run / "trained.json", trained)

    development.reset()
    dev_probabilities = np.asarray(model.predict(development, verbose=0))
    dev_scores, dev_per_class, _ = audit.classification_metrics(
        development.classes,
        dev_probabilities.argmax(1),
        experiment["classes"],
    )
    audit.write_csv(run / "development_per_class.csv", dev_per_class)
    audit.write_json(run / "development_metrics.json", dev_scores)
    print("Selected checkpoint development metrics:", json.dumps(dev_scores), flush=True)
    if not args.evaluate:
        print("Development-only run complete; evaluation was not scored.", flush=True)
        del model, train, development, evaluation
        tf.keras.backend.clear_session()
        return None

    evaluation.reset()
    probabilities = np.asarray(model.predict(evaluation, verbose=1))
    if probabilities.shape != (len(parts["evaluation"]), len(experiment["classes"])):
        raise ValueError("Invalid ConvNeXt prediction shape.")
    actual, predicted = evaluation.classes, probabilities.argmax(1)
    metrics, per_class, confusion = audit.classification_metrics(
        actual, predicted, experiment["classes"]
    )
    records = [
        {
            "sample_id": row["sample_id"],
            "true_label": experiment["classes"][int(y)],
            "predicted_label": experiment["classes"][int(guess)],
            **{f"prob_{label}": float(probability)
               for label, probability in zip(experiment["classes"], scores)},
        }
        for row, y, guess, scores in zip(
            parts["evaluation"], actual, predicted, probabilities
        )
    ]
    audit.write_csv(run / "predictions.csv", records)
    audit.write_csv(run / "per_class.csv", per_class)
    audit.write_csv(
        run / "confusion_matrix.csv",
        [{"true_class": label,
          **dict(zip(experiment["classes"], map(int, row)))}
         for label, row in zip(experiment["classes"], confusion)],
    )
    metrics.update({
        "model": "convnext_tiny",
        "condition": condition,
        "seed": seed,
        "fitting_images": len(parts["fitting"]),
        "development_images": len(parts["development"]),
        "evaluation_images": len(parts["evaluation"]),
        **trained,
    })
    metrics["artifacts_sha256"] = {
        name: audit.sha((run / name).read_bytes())
        for name in (
            "run.json", "trained.json", "selected.keras", "predictions.csv",
            "per_class.csv", "confusion_matrix.csv"
        )
    }
    audit.write_json(run / "metrics.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "artifacts_sha256"}, indent=2), flush=True)
    del model, train, development, evaluation
    tf.keras.backend.clear_session()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=["full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=35)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    args.output = args.output.resolve()
    if min(args.batch_size, args.epochs_phase1, args.epochs_phase2) < 1:
        parser.error("Batch size and epoch limits must be positive.")
    if min(args.head_learning_rate, args.backbone_learning_rate) <= 0:
        parser.error("Learning rates must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(
            seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be unique integers between 0 and 2**32-1.")

    conditions, classes = {}, None
    for condition in args.conditions:
        parts, current_classes = load_condition(args.manifest_dir, condition)
        if classes is None:
            classes = current_classes
        elif current_classes != classes:
            raise ValueError("Requested conditions use different class mappings.")
        conditions[condition] = parts
    verify_dataset(args.dataset, conditions)
    args.output.mkdir(parents=True, exist_ok=True)
    for condition, parts in conditions.items():
        for partition in PARTITIONS:
            freeze_csv(args.output / f"{condition}_{partition}.csv", parts[partition])
    settings = {
        "model": "ConvNeXt-Tiny",
        "classes": classes,
        "class_count": len(classes),
        "input": "224x224 RGB bicubic, float32 0..255",
        "preprocessing": "ConvNeXt built-in ImageNet normalization",
        "pretrained": "ImageNet",
        "classification_head": "global average pooling, dropout 0.3, softmax",
        "batch_size": args.batch_size,
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "head_learning_rate": args.head_learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "augmentation": None,
        "selection": "minimum phase-2 development loss",
        "source_sha256": audit.sha(Path(__file__).read_bytes()),
    }
    freeze_json(args.output / "experiment_settings.json", settings)
    print("Verified ConvNeXt 23-class manifests:", {
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
        raise RuntimeError("TensorFlow cannot see a GPU.")
    reference.setup_gpu()
    experiment = {
        **settings,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "gpus": [tf.config.experimental.get_device_details(g) for g in gpus],
    }
    freeze_json(args.output / "experiment.json", experiment)

    for seed in args.seeds:
        for condition in args.conditions:
            print(f"ConvNeXt-Tiny: {DISPLAY[condition]}, seed {seed}", flush=True)
            run_one(args, experiment, conditions[condition], condition, seed, keras, tf)

    completed = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output / "convnext_tiny").glob("*/seed_*/metrics.json"))
    ]
    if completed:
        for seed in {r["seed"] for r in completed}:
            rows = [r for r in completed if r["seed"] == seed]
            if len(rows) > 1 and len({r["initial_state_sha256"] for r in rows}) != 1:
                raise ValueError(f"ConvNeXt initialization differs for seed {seed}.")
        audit.write_csv(args.output / "results.csv", [
            {k: v for k, v in row.items() if k != "artifacts_sha256"}
            for row in completed
        ])


if __name__ == "__main__":
    main()
