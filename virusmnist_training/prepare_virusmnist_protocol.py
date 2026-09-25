#!/usr/bin/env python3
"""Freeze Full and consensus-clean Virus-MNIST protocols before training.

The input is the completed external audit. Every mixed-label consensus group is
excluded. Full uses an individual-image 70/20/10 split over the remaining ten
classes. Clean excludes classes with fewer than 50 pure consensus groups,
retains one deterministic representative per complete aHash/dHash/pHash tuple,
and only then makes a class-stratified 70/20/10 split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PARTITIONS = ("fitting", "development", "evaluation")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(n: int, fit_fraction: float, development_fraction: float) -> tuple[int, int, int]:
    fit = round(n * fit_fraction)
    development = round(n * development_fraction)
    evaluation = n - fit - development
    if min(fit, development, evaluation) < 1:
        raise ValueError(f"Cannot place {n} samples in all three partitions")
    return fit, development, evaluation


def split_rows(rows: list[dict], fit_fraction: float, development_fraction: float, seed: int) -> list[dict]:
    result = []
    for label in sorted({row["class_label"] for row in rows}):
        subset = sorted((row for row in rows if row["class_label"] == label), key=lambda row: row["sample_id"])
        rng = random.Random(f"{seed}:{label}")
        rng.shuffle(subset)
        n_fit, n_dev, _ = counts(len(subset), fit_fraction, development_fraction)
        for index, row in enumerate(subset):
            partition = "fitting" if index < n_fit else "development" if index < n_fit + n_dev else "evaluation"
            result.append({**row, "partition": partition})
    return sorted(result, key=lambda row: row["sample_id"])


def stable_rank(seed: int, sample_id: str) -> str:
    return sha(f"{seed}\0{sample_id}".encode())


def standard_row(row: dict) -> dict:
    value = row["hash_consensus"]
    return {
        "sample_id": row["sample_id"],
        "original_split": row["original_split"],
        "class_label": row["class_label"],
        "file_sha256": row["sha256"],
        "rgb_sha256": row["rgb_sha256"],
        "ahash": row["ahash"],
        "dhash": row["dhash"],
        "phash": row["phash"],
        "duplicate_group_id": sha(f"hash_consensus\0{value}".encode()),
    }


def validate_images(dataset: Path, rows: list[dict]) -> None:
    for number, row in enumerate(rows, 1):
        path = dataset / row["sample_id"]
        if not path.is_file() or sha(path.read_bytes()) != row["file_sha256"]:
            raise ValueError(f"Missing or changed image: {path}")
        if number % 10000 == 0:
            print(f"Verified {number}/{len(rows)} source images", flush=True)


def write_protocol(directory: Path, rows: list[dict], metadata: dict) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory / "manifest.csv"
    write_csv(manifest, rows)
    metadata = {
        **metadata,
        "manifest_sha256": sha(manifest.read_bytes()),
        "partition_counts": dict(Counter(row["partition"] for row in rows)),
        "classes": sorted({row["class_label"] for row in rows}),
        "images": len(rows),
    }
    (directory / "protocol.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    audit = args.audit_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {output}")
    metadata = json.loads((audit / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("layout") != "split_class_folders":
        raise ValueError("Audit metadata is incomplete or has the wrong layout")
    hashes_path = audit / "hashes.csv"
    rows = read_csv(hashes_path)
    if len(rows) != metadata["images"] or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Audit hash manifest membership is invalid")

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["hash_consensus"]].append(row)
    mixed_values = {
        value for value, members in groups.items()
        if len({row["class_label"] for row in members}) > 1
    }
    safe = [row for row in rows if row["hash_consensus"] not in mixed_values]
    group_counts = Counter(row["class_label"] for value, members in groups.items()
                           if value not in mixed_values for row in members[:1])
    eligible = sorted(label for label, number in group_counts.items() if number >= args.minimum_groups_per_class)
    excluded = sorted(set(group_counts) - set(eligible))
    eligible_set = set(eligible)
    clean_source = [row for row in safe if row["class_label"] in eligible_set]

    representatives = []
    for value, all_members in sorted(groups.items()):
        if value in mixed_values:
            continue
        members = [row for row in all_members if row["class_label"] in eligible_set]
        if not members:
            continue
        labels = {row["class_label"] for row in members}
        if len(labels) != 1:
            raise AssertionError("Mixed-label group survived exclusion")
        representatives.append(min(members, key=lambda row: (stable_rank(args.representative_seed, row["sample_id"]), row["sample_id"])))

    full_manifest = split_rows([standard_row(row) for row in safe], args.fit_fraction, args.development_fraction, args.split_seed)
    clean_manifest = split_rows([standard_row(row) for row in representatives], args.fit_fraction, args.development_fraction, args.split_seed)
    if len({row["duplicate_group_id"] for row in clean_manifest}) != len(clean_manifest):
        raise AssertionError("Clean manifest repeats a consensus group")
    for partition in PARTITIONS:
        if {row["class_label"] for row in clean_manifest if row["partition"] == partition} != set(eligible):
            raise AssertionError(f"Clean {partition} lacks an eligible class")

    validate_images(dataset, [standard_row(row) for row in rows])
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "dataset": "Virus-MNIST local image snapshot",
        "dataset_snapshot_sha256": metadata["dataset_snapshot_sha256"],
        "hash_manifest_sha256": sha(hashes_path.read_bytes()),
        "mixed_label_groups_excluded": len(mixed_values),
        "mixed_label_images_excluded": len(rows) - len(safe),
        "minimum_groups_per_class": args.minimum_groups_per_class,
        "excluded_classes": excluded,
        "fit_fraction": args.fit_fraction,
        "development_fraction": args.development_fraction,
        "evaluation_fraction": 1 - args.fit_fraction - args.development_fraction,
        "split_seed": args.split_seed,
        "representative_seed": args.representative_seed,
    }
    write_protocol(output / "full_safe_10", full_manifest, {
        **common,
        "rule": "individual_image_split_after_excluding_mixed_label_consensus_groups",
        "interpretation": "Full source condition; the trainer restricts it to the clean nine-class label space.",
    })
    write_protocol(output / "clean_consensus_9", clean_manifest, {
        **common,
        "rule": "hash_consensus",
        "definition": "One representative per complete 64-bit aHash/dHash/pHash tuple, then split",
        "interpretation": "Perceptual-signature sensitivity condition; tuple equality does not prove binary identity.",
    })
    summary = {
        "source_images": len(rows),
        "safe_full_images": len(safe),
        "safe_full_classes": sorted({row["class_label"] for row in safe}),
        "eligible_classes": eligible,
        "excluded_classes": excluded,
        "clean_representatives": len(representatives),
        "consensus_excess_removed_from_eligible_safe_pool": len(clean_source) - len(representatives),
        "mixed_label_groups_excluded": len(mixed_values),
        "mixed_label_images_excluded": len(rows) - len(safe),
        "protocol_directories": {"full": "full_safe_10", "clean": "clean_consensus_9"},
    }
    (output / "study_protocol.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-groups-per-class", type=int, default=50)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--development-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260920)
    parser.add_argument("--representative-seed", type=int, default=20260922)
    args = parser.parse_args()
    if not (0 < args.fit_fraction < 1 and 0 < args.development_fraction < 1 and args.fit_fraction + args.development_fraction < 1):
        parser.error("Split fractions must be positive and sum to less than one")
    if args.minimum_groups_per_class < 3:
        parser.error("minimum groups must be at least three")
    prepare(args)


if __name__ == "__main__":
    main()
