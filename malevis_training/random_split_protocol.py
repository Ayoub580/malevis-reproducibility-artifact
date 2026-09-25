#!/usr/bin/env python3
"""Freeze and verify a stratified image-level 70/20/10 MaleVis protocol.

The original train/ and val/ folders are treated only as file provenance.  All
14,226 images are pooled and partitioned within class.  The resulting test set
is fixed for the paired Full, Clean-Exact and Random-Control experiments.
Dataset files are never copied, moved, relabeled or modified.
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import PIL

import train_full_malevis as audit

PARTITIONS = ("fitting", "development", "evaluation")
IDENTITY_KEYS = ("file_sha256", "rgb_sha256", "input_sha256")


def _counts(n, train_fraction, development_fraction):
    """Allocate a class deterministically; the residual goes to evaluation."""
    n_train = round(n * train_fraction)
    n_development = round(n * development_fraction)
    n_evaluation = n - n_train - n_development
    if min(n_train, n_development, n_evaluation) < 1:
        raise ValueError("Every class must contribute to all three partitions.")
    return n_train, n_development, n_evaluation


def stratified_image_split(rows, train_fraction, development_fraction, seed):
    """Randomly partition individual images within each supplied class."""
    if not (0 < train_fraction < 1 and 0 < development_fraction < 1 and
            train_fraction + development_fraction < 1):
        raise ValueError("Fractions must be positive and sum to less than one.")
    rng = random.Random(seed)
    result = []
    for cls in sorted({r["class_label"] for r in rows}):
        subset = sorted((r for r in rows if r["class_label"] == cls),
                        key=lambda r: r["sample_id"])
        rng.shuffle(subset)
        n_train, n_dev, _ = _counts(len(subset), train_fraction, development_fraction)
        for i, row in enumerate(subset):
            partition = ("fitting" if i < n_train else
                         "development" if i < n_train + n_dev else "evaluation")
            result.append({**row, "partition": partition})
    return sorted(result, key=lambda r: r["sample_id"])


def validate_manifest(rows):
    if len(rows) != 14226:
        raise ValueError(f"Expected 14,226 images, found {len(rows)}.")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated sample identifiers.")
    if {r["partition"] for r in rows} != set(PARTITIONS):
        raise ValueError("Unknown or missing partition.")
    classes = {r["class_label"] for r in rows}
    if len(classes) != 26:
        raise ValueError(f"Expected 26 classes, found {len(classes)}.")
    for partition in PARTITIONS:
        if {r["class_label"] for r in rows if r["partition"] == partition} != classes:
            raise ValueError(f"{partition} lacks at least one class.")
    if any(r["original_split"] not in ("train", "val") for r in rows):
        raise ValueError("Unexpected source partition.")


def _overlap_summary(rows, key):
    parts = {p: [r for r in rows if r["partition"] == p] for p in PARTITIONS}
    keys = {p: {r[key] for r in subset} for p, subset in parts.items()}
    return {
        "identity_key": key,
        "development_matching_fitting_images": sum(r[key] in keys["fitting"] for r in parts["development"]),
        "evaluation_matching_fitting_images": sum(r[key] in keys["fitting"] for r in parts["evaluation"]),
        "evaluation_matching_development_images": sum(r[key] in keys["development"] for r in parts["evaluation"]),
    }


def _class_counts(rows):
    table = []
    for cls in sorted({r["class_label"] for r in rows}):
        line = {"class_label": cls}
        for partition in PARTITIONS:
            subset = [r for r in rows
                      if r["class_label"] == cls and r["partition"] == partition]
            line[f"{partition}_images"] = len(subset)
            for key in IDENTITY_KEYS:
                line[f"{partition}_{key}_distinct"] = len({r[key] for r in subset})
        table.append(line)
    return table


def prepare(root, directory, train_fraction=0.70, development_fraction=0.20,
            seed=20260920):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    settings = {
        "scheme": "class-stratified random split of individual pooled images",
        "train_fraction": train_fraction,
        "development_fraction": development_fraction,
        "evaluation_fraction": 1.0 - train_fraction - development_fraction,
        "split_seed": seed,
        "Pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    if directory.exists():
        protocol_path, manifest_path = directory / "protocol.json", directory / "manifest.csv"
        if not protocol_path.exists() or not manifest_path.exists():
            raise ValueError(f"Incomplete frozen protocol directory: {directory}")
        meta = json.loads(protocol_path.read_text())
        if meta["split_settings"] != settings:
            raise ValueError("Existing random-split settings differ; use a new protocol directory.")
        if audit.sha(manifest_path.read_bytes()) != meta["manifest_sha256"]:
            raise ValueError("Frozen manifest checksum mismatch.")
        with manifest_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        validate_manifest(rows)
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*.png")}
        if actual != {r["sample_id"] for r in rows}:
            raise ValueError("Dataset membership changed after the protocol was frozen.")
        for row in rows:
            rel = Path(row["sample_id"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe path in frozen manifest.")
            if audit.sha((root / rel).read_bytes()) != row["file_sha256"]:
                raise ValueError(f"Dataset file changed: {rel}")
        print("Verified and reused frozen pooled 70/20/10 protocol.", flush=True)
        return rows, meta

    rows = stratified_image_split(audit.scan(root), train_fraction,
                                  development_fraction, seed)
    validate_manifest(rows)
    directory.mkdir(parents=True, exist_ok=False)
    audit.write_csv(directory / "manifest.csv", rows)
    audit.write_csv(directory / "class_counts.csv", _class_counts(rows))
    overlaps = [_overlap_summary(rows, key) for key in IDENTITY_KEYS]
    audit.write_csv(directory / "split_overlap.csv", overlaps)
    snapshot = b"".join(
        json.dumps([r["sample_id"], r["file_sha256"]], separators=(",", ":")).encode() + b"\n"
        for r in sorted(rows, key=lambda r: r["sample_id"])
    )
    manifest_path = directory / "manifest.csv"
    meta = {
        "split_settings": settings,
        "classes": sorted({r["class_label"] for r in rows}),
        "snapshot_sha256": audit.sha(snapshot),
        "manifest_sha256": audit.sha(manifest_path.read_bytes()),
        "partition_counts": dict(Counter(r["partition"] for r in rows)),
        "source_partition_counts": dict(Counter(r["original_split"] for r in rows)),
        "overlap_before_cleaning": overlaps,
        "interpretation": (
            "Conventional individual-image split. Cross-partition identity overlap is measured, "
            "not prevented. Use paired conditions with a fixed evaluation set to estimate its effect."
        ),
    }
    audit.write_json(directory / "protocol.json", meta)
    print(json.dumps(meta, indent=2), flush=True)
    return rows, meta


def paired_conditions(rows, removal_seed=20260921, identity_key="input_sha256"):
    """Return Full, exact-cleaned and class-size-matched control fitting sets.

    Clean-Exact removes every fitting image whose identity occurs in development
    or evaluation. Random-Control removes the same count per class uniformly
    from the complete fitting partition. It can therefore remove some exposed
    counterparts by chance; the resulting overlap is recorded explicitly.
    Development and evaluation are identical in all conditions.
    """
    if identity_key not in IDENTITY_KEYS:
        raise ValueError(f"identity_key must be one of {IDENTITY_KEYS}")
    parts = {p: sorted((r for r in rows if r["partition"] == p),
                       key=lambda r: r["sample_id"]) for p in PARTITIONS}
    heldout_keys = {r[identity_key] for r in parts["development"] + parts["evaluation"]}
    removed = [r for r in parts["fitting"] if r[identity_key] in heldout_keys]
    removed_ids = {r["sample_id"] for r in removed}
    clean = [r for r in parts["fitting"] if r["sample_id"] not in removed_ids]
    remove_counts = Counter(r["class_label"] for r in removed)

    rng = random.Random(removal_seed)
    control_ids = set()
    classes = sorted({r["class_label"] for r in rows})
    for cls in classes:
        eligible = [r["sample_id"] for r in parts["fitting"]
                    if r["class_label"] == cls]
        need = remove_counts[cls]
        if need >= sum(r["class_label"] == cls for r in parts["fitting"]):
            raise ValueError(f"Exact cleaning exhausts fitting class {cls}.")
        control_ids.update(rng.sample(eligible, need))
    control = [r for r in parts["fitting"] if r["sample_id"] not in control_ids]

    fitting = {"full": parts["fitting"], "clean_exact": clean,
               "random_control": control}
    if {r[identity_key] for r in clean} & heldout_keys:
        raise AssertionError("Clean-Exact retains a held-out identity.")
    if Counter(r["class_label"] for r in clean) != Counter(r["class_label"] for r in control):
        raise AssertionError("Random-Control does not match Clean-Exact class counts.")
    if not removed:
        raise ValueError("No cross-partition exact exposure was found.")

    table = []
    dev_keys = {r[identity_key] for r in parts["development"]}
    eval_keys = {r[identity_key] for r in parts["evaluation"]}
    for cls in classes:
        dev = [r for r in parts["development"] if r["class_label"] == cls]
        evaluation = [r for r in parts["evaluation"] if r["class_label"] == cls]
        removed_cls = [r for r in removed if r["class_label"] == cls]
        line = {
            "class_label": cls,
            "development": len(dev),
            "evaluation": len(evaluation),
            "removed_exact": len(removed_cls),
            "removed_matching_development": sum(r[identity_key] in dev_keys for r in removed_cls),
            "removed_matching_evaluation": sum(r[identity_key] in eval_keys for r in removed_cls),
        }
        for name, subset in fitting.items():
            class_subset = [r for r in subset if r["class_label"] == cls]
            keys = {r[identity_key] for r in class_subset}
            line[f"{name}_fitting"] = len(class_subset)
            line[f"{name}_development_matches"] = sum(r[identity_key] in keys for r in dev)
            line[f"{name}_evaluation_matches"] = sum(r[identity_key] in keys for r in evaluation)
        table.append(line)
    return fitting, parts["development"], parts["evaluation"], removed, table
