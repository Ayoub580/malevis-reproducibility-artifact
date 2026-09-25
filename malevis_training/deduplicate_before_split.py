#!/usr/bin/env python3
"""Deduplicate pooled MaleVis before making a class-stratified split.

The script consumes the frozen pooled manifest produced by
``random_split_protocol.py`` and the image hashes produced by the audit.  The
old partition column is ignored.  It supports two operational definitions:

* exact_input: equality of the verified 224x224 model-input representation;
* hash_consensus: equality of the complete (aHash, dHash, pHash) tuple.

The latter is a perceptual-signature sensitivity analysis, not proof of pixel
identity.  Dataset files are never copied, moved, or changed.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PARTITIONS = ("fitting", "development", "evaluation")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty output {path}.")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def split_counts(n, train_fraction, development_fraction):
    if n < 3:
        raise ValueError(
            f"A class with {n} retained groups cannot populate all three partitions."
        )
    # Keep all partitions represented for small sensitivity-analysis classes.
    n_fit = max(1, min(round(n * train_fraction), n - 2))
    n_dev = max(1, min(round(n * development_fraction), n - n_fit - 1))
    n_eval = n - n_fit - n_dev
    return n_fit, n_dev, n_eval


def stable_rank(seed, sample_id):
    return hashlib.sha256(f"{seed}\0{sample_id}".encode()).hexdigest()


def load_rows(source_manifest, hash_manifest):
    rows = read_csv(source_manifest)
    hashes = {r["sample_id"]: r for r in read_csv(hash_manifest)}
    if len(hashes) != len(rows) or set(hashes) != {r["sample_id"] for r in rows}:
        raise ValueError("Source and hash manifests describe different dataset memberships.")
    merged = []
    for row in rows:
        h = hashes[row["sample_id"]]
        if row["class_label"] != h["class_label"]:
            raise ValueError(f"Class mismatch for {row['sample_id']}.")
        if row["file_sha256"] != h["sha256"] or row["rgb_sha256"] != h["rgb_sha256"]:
            raise ValueError(f"Hash-audit mismatch for {row['sample_id']}.")
        merged.append({**row, "ahash": h["ahash"], "dhash": h["dhash"],
                       "phash": h["phash"]})
    return merged


def group_key(row, rule):
    if rule == "exact_input":
        return row["input_sha256"]
    if rule == "hash_consensus":
        return ":".join((row["ahash"], row["dhash"], row["phash"]))
    raise ValueError(f"Unknown rule {rule}.")


def deduplicate(rows, rule, representative_seed):
    by_value = defaultdict(list)
    for row in rows:
        by_value[group_key(row, rule)].append(row)

    conflicts = [members for members in by_value.values()
                 if len({r["class_label"] for r in members}) > 1]
    if conflicts:
        examples = [[r["sample_id"] for r in g[:3]] for g in conflicts[:3]]
        raise ValueError(
            f"{len(conflicts)} duplicate groups span supplied labels; examples={examples}"
        )

    kept, removed, groups = [], [], []
    for key, members in sorted(by_value.items()):
        ordered = sorted(members,
                         key=lambda r: (stable_rank(representative_seed, r["sample_id"]),
                                        r["sample_id"]))
        representative = ordered[0]
        group_id = sha256_bytes(f"{rule}\0{key}".encode())
        common = {
            "dedup_rule": rule,
            "duplicate_group_id": group_id,
            "duplicate_group_size": str(len(ordered)),
            "representative_sample_id": representative["sample_id"],
        }
        kept.append({**representative, **common})
        for row in ordered[1:]:
            removed.append({**row, **common})
        groups.append({
            "duplicate_group_id": group_id,
            "class_label": representative["class_label"],
            "group_size": len(ordered),
            "representative_sample_id": representative["sample_id"],
            "dedup_value": key,
        })
    return kept, removed, groups


def stratified_split(rows, train_fraction, development_fraction, split_seed):
    result = []
    for cls in sorted({r["class_label"] for r in rows}):
        subset = sorted((r for r in rows if r["class_label"] == cls),
                        key=lambda r: r["sample_id"])
        rng = random.Random(f"{split_seed}:{cls}")
        rng.shuffle(subset)
        n_fit, n_dev, _ = split_counts(
            len(subset), train_fraction, development_fraction
        )
        for index, row in enumerate(subset):
            partition = ("fitting" if index < n_fit else
                         "development" if index < n_fit + n_dev else "evaluation")
            result.append({**row, "partition": partition})
    return sorted(result, key=lambda r: r["sample_id"])


def per_class_table(original, kept, removed, split_rows):
    table = []
    for cls in sorted({r["class_label"] for r in original}):
        before = [r for r in original if r["class_label"] == cls]
        after = [r for r in kept if r["class_label"] == cls]
        gone = [r for r in removed if r["class_label"] == cls]
        row = {
            "class_label": cls,
            "original_images": len(before),
            "retained_groups": len(after),
            "removed_excess_copies": len(gone),
            "removed_percent": f"{100 * len(gone) / len(before):.6f}",
        }
        for partition in PARTITIONS:
            row[partition] = sum(
                r["class_label"] == cls and r["partition"] == partition
                for r in split_rows
            )
        table.append(row)
    return table


def prepare(args):
    rows = load_rows(args.source_manifest, args.hash_manifest)
    kept, removed, groups = deduplicate(rows, args.rule, args.representative_seed)
    retained_counts = Counter(r["class_label"] for r in kept)
    insufficient = sorted(
        cls for cls, n in retained_counts.items()
        if n < args.minimum_groups_per_class
    )
    if insufficient and not args.exclude_insufficient_classes:
        details = ", ".join(f"{c}={retained_counts[c]}" for c in insufficient)
        raise ValueError(
            f"Deduplication leaves fewer than {args.minimum_groups_per_class} groups for: "
            + details + ". Re-run with --exclude-insufficient-classes only for an "
            "explicitly labeled reduced-class sensitivity analysis."
        )
    excluded = insufficient if args.exclude_insufficient_classes else []
    kept_for_split = [r for r in kept if r["class_label"] not in excluded]
    split_rows = stratified_split(kept_for_split, args.train_fraction,
                                  args.development_fraction, args.split_seed)
    table = per_class_table(rows, kept, removed, split_rows)

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "manifest.csv", split_rows)
    write_csv(args.output / "removed_duplicates.csv", removed,
              fieldnames=list(kept[0]))
    write_csv(args.output / "duplicate_groups.csv", groups)
    write_csv(args.output / "per_class.csv", table)
    manifest_sha = sha256_bytes((args.output / "manifest.csv").read_bytes())
    metadata = {
        "rule": args.rule,
        "definition": (
            "Verified equality of the serialized 224x224 float32 model input"
            if args.rule == "exact_input" else
            "Equality of the complete 64-bit aHash, dHash, and pHash tuple; a "
            "perceptual-signature sensitivity analysis, not verified pixel identity"
        ),
        "order": "pool original folders, deduplicate globally, then split retained representatives",
        "source_images": len(rows),
        "retained_representatives": len(kept),
        "removed_excess_copies": len(removed),
        "excluded_classes": excluded,
        "minimum_groups_per_class": args.minimum_groups_per_class,
        "split_images": len(split_rows),
        "partition_counts": dict(Counter(r["partition"] for r in split_rows)),
        "train_fraction": args.train_fraction,
        "development_fraction": args.development_fraction,
        "evaluation_fraction": 1 - args.train_fraction - args.development_fraction,
        "representative_seed": args.representative_seed,
        "split_seed": args.split_seed,
        "source_manifest_sha256": sha256_bytes(args.source_manifest.read_bytes()),
        "hash_manifest_sha256": sha256_bytes(args.hash_manifest.read_bytes()),
        "manifest_sha256": manifest_sha,
    }
    (args.output / "protocol.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule", choices=("exact_input", "hash_consensus"),
                        default="exact_input")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--development-fraction", type=float, default=0.20)
    parser.add_argument("--representative-seed", type=int, default=20260922)
    parser.add_argument("--split-seed", type=int, default=20260920)
    parser.add_argument("--exclude-insufficient-classes", action="store_true")
    parser.add_argument(
        "--minimum-groups-per-class", type=int, default=3,
        help=("Minimum retained duplicate groups required for a class. Use 50 for "
              "the recommended near-duplicate sensitivity benchmark, which gives "
              "approximately five evaluation groups per retained class."),
    )
    args = parser.parse_args()
    args.source_manifest = args.source_manifest.resolve()
    args.hash_manifest = args.hash_manifest.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory.")
    if not (0 < args.train_fraction < 1 and 0 < args.development_fraction < 1 and
            args.train_fraction + args.development_fraction < 1):
        parser.error("Split fractions must be positive and sum to less than one.")
    if args.minimum_groups_per_class < 3:
        parser.error("--minimum-groups-per-class must be at least 3.")
    prepare(args)


if __name__ == "__main__":
    main()
