#!/usr/bin/env python3
"""Train paired MaleVis conditions on a frozen pooled 70/20/10 split.

Conditions share the same development and evaluation images:
  full            conventional class-stratified individual-image split;
  clean_exact     fitting counterparts of held-out exact inputs removed;
  random_control  same per-class number of images removed uniformly at random.

The default exact identity is the verified bicubic 224x224 float32 model input.
Run multiple seeds and interpret random_control - clean_exact as the adjusted
effect of held-out exact exposure.  The script does not promise a positive drop.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np

import original_training_reference as reference
import random_split_protocol as split
import train_full_malevis as audit
import train_resnet_preprocess_corrected as corrected
from tune_and_compare_malevis import digest_rows, freeze_csv, freeze_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--protocol-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--conditions", nargs="+",
                   choices=("full", "clean_exact", "random_control"),
                   default=["full", "clean_exact", "random_control"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--split-seed", type=int, default=20260920)
    p.add_argument("--removal-seed", type=int, default=20260921)
    p.add_argument("--identity-key", choices=split.IDENTITY_KEYS,
                   default="input_sha256")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs-phase1", type=int, default=15)
    p.add_argument("--epochs-phase2", type=int, default=35)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--restart-incomplete", action="store_true")
    args = p.parse_args()
    args.model = "resnet50"
    args.custom_epochs = 50  # Required by the shared runner; unused for ResNet50.
    if min(args.batch_size, args.epochs_phase1, args.epochs_phase2) < 1:
        p.error("Batch size and epoch limits must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error("Seeds must be unique integers between 0 and 2**32-1.")
    args.dataset, args.protocol_dir, args.output = (
        x.resolve() for x in (args.dataset, args.protocol_dir, args.output)
    )
    if any(args.output == x or x in args.output.parents
           for x in (args.dataset, args.protocol_dir)):
        p.error("Output must be outside the dataset and frozen protocol directory.")
    if not (args.dataset / "train").is_dir() or not (args.dataset / "val").is_dir():
        p.error("Dataset must contain the original train/ and val/ directories.")

    rows, meta = split.prepare(args.dataset, args.protocol_dir,
                               train_fraction=0.70, development_fraction=0.20,
                               seed=args.split_seed)
    fitting, development, evaluation, removed, table = split.paired_conditions(
        rows, args.removal_seed, args.identity_key
    )
    args.output.mkdir(parents=True, exist_ok=True)
    freeze_json(args.output / "conditions.json", {
        "protocol_manifest_sha256": meta["manifest_sha256"],
        "identity_key": args.identity_key,
        "removal_seed": args.removal_seed,
        "condition_fitting_sha256": {k: digest_rows(v) for k, v in fitting.items()},
        "development_sha256": digest_rows(development),
        "evaluation_sha256": digest_rows(evaluation),
        "removed_exact_sha256": digest_rows(removed),
        "definition": (
            "Clean-Exact removes all fitting images whose selected exact identity occurs in "
            "development or evaluation. Random-Control removes the same number per class "
            "uniformly from the full fitting partition. Held-out images are fixed."
        ),
    })
    for name, subset in fitting.items():
        freeze_csv(args.output / f"{name}_fitting.csv", subset)
    freeze_csv(args.output / "development.csv", development)
    freeze_csv(args.output / "evaluation.csv", evaluation)
    freeze_csv(args.output / "removed_exact.csv", removed)
    freeze_csv(args.output / "condition_class_counts.csv", table)
    print("Frozen pooled 70/20/10 paired protocol; fitting sizes:",
          {k: len(v) for k, v in fitting.items()}, flush=True)
    print(f"Development={len(development)}, evaluation={len(evaluation)}, "
          f"removed exact fitting images={len(removed)}.", flush=True)
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
        "classes": meta["classes"],
        "protocol": meta,
        "conditions_sha256": audit.sha((args.output / "conditions.json").read_bytes()),
        "identity_key": args.identity_key,
        "removal_seed": args.removal_seed,
        "fitting_sha256": {k: digest_rows(v) for k, v in fitting.items()},
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
        "split_note": "Pooled class-stratified random 70/20/10 individual-image split",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": audit.PIL.__version__,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "tensorflow_build": tf.sysconfig.get_build_info(),
        "gpus": [tf.config.experimental.get_device_details(g) for g in gpus],
        "source_sha256": {
            Path(f).name: audit.sha(Path(f).read_bytes())
            for f in (__file__, split.__file__, corrected.__file__, reference.__file__, audit.__file__)
        },
    }
    freeze_json(args.output / "experiment.json", experiment)
    for seed in args.seeds:
        for condition in args.conditions:
            print(f"Pooled 70/20/10: ResNet50, {condition}, seed {seed}", flush=True)
            corrected.run_one(args, experiment, fitting[condition], development,
                              evaluation, condition, seed, reference, tf)

    completed = [json.loads(f.read_text()) for f in
                 sorted((args.output / args.model).glob("*/seed_*/metrics.json"))]
    if not completed:
        print("Development runs completed; add --evaluate to score the fixed test set.", flush=True)
        return
    for seed in {r["seed"] for r in completed}:
        states = {r["initial_state_sha256"] for r in completed if r["seed"] == seed}
        if len(states) != 1:
            raise ValueError(f"Paired initialization differs for seed {seed}.")
    audit.write_csv(args.output / "results.csv",
                    [{k: v for k, v in r.items() if k != "artifacts_sha256"}
                     for r in completed])
    contrasts = []
    for seed in sorted({r["seed"] for r in completed}):
        by = {r["condition"]: r for r in completed if r["seed"] == seed}
        if set(by) == {"full", "clean_exact", "random_control"}:
            row = {"seed": seed}
            for metric in ("accuracy", "macro_f1", "macro_recall"):
                row[f"full_minus_clean_{metric}_pp"] = 100 * (
                    by["full"][metric] - by["clean_exact"][metric])
                row[f"full_minus_random_{metric}_pp"] = 100 * (
                    by["full"][metric] - by["random_control"][metric])
                row[f"adjusted_random_minus_clean_{metric}_pp"] = 100 * (
                    by["random_control"][metric] - by["clean_exact"][metric])
            contrasts.append(row)
    if contrasts:
        audit.write_csv(args.output / "paired_differences.csv", contrasts)
        print("Paired differences saved; adjusted effect is Random-Control minus Clean-Exact.",
              flush=True)


if __name__ == "__main__":
    main()
