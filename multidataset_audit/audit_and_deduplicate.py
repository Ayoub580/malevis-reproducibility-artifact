#!/usr/bin/env python3
"""Audit and globally deduplicate class-folder image datasets.

The operational perceptual rule matches the MaleVis sensitivity experiment:
equality of the complete 64-bit (aHash, dHash, pHash) tuple.  This rule defines
perceptual-signature groups; it does not prove equal pixels or source binaries.
Dataset files are read only.  Outputs contain hashes, groups, deterministic
representatives, removed excess copies, per-class statistics, and, when an
original train/evaluation split exists, cross-split overlap statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
import scipy
from PIL import Image
from scipy.fftpack import dct


SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SIMPLE_METHODS = ("sha256", "rgb_sha256", "ahash", "dhash", "phash")
MATCH_METHODS = SIMPLE_METHODS + ("hash_consensus",)
REFERENCE = "https://github.com/JohannesBuchner/imagehash/blob/v4.3.2/imagehash/__init__.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    if bits.size != 64:
        raise ValueError("Perceptual hashes must contain 64 bits")
    return np.packbits(bits.reshape(-1), bitorder="big").tobytes().hex()


def perceptual_hashes(image: Image.Image) -> dict[str, str]:
    gray = image.convert("L")
    average = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS))
    difference = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    frequency = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS))
    frequency = dct(
        dct(frequency, type=2, axis=0, norm=None),
        type=2,
        axis=1,
        norm=None,
    )[:8, :8]
    return {
        "ahash": bits_to_hex(average > average.mean()),
        "dhash": bits_to_hex(difference[:, 1:] > difference[:, :-1]),
        "phash": bits_to_hex(frequency > np.median(frequency)),
    }


def rgb_payload(image: Image.Image) -> bytes:
    pixels = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    header = json.dumps(
        {"shape": list(pixels.shape), "dtype": pixels.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return header + b"\n" + pixels.tobytes(order="C")


def inventory(root: Path, layout: str, train_split: str, eval_split: str) -> list[dict]:
    rows = []
    if layout == "class_folders":
        candidates = sorted(p for p in root.glob("*/*") if p.is_file())
        for path in candidates:
            if path.suffix.lower() in SUPPORTED:
                rows.append({
                    "path": path,
                    "sample_id": path.relative_to(root).as_posix(),
                    "original_split": "pooled",
                    "class_label": path.parent.name,
                })
    else:
        for split in (train_split, eval_split):
            split_root = root / split
            if not split_root.is_dir():
                raise ValueError(f"Missing split directory: {split_root}")
            for path in sorted(p for p in split_root.glob("*/*") if p.is_file()):
                if path.suffix.lower() in SUPPORTED:
                    rows.append({
                        "path": path,
                        "sample_id": path.relative_to(root).as_posix(),
                        "original_split": split,
                        "class_label": path.parent.name,
                    })
    if not rows:
        raise ValueError(f"No supported images found under {root}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Sample identifiers are not unique")
    return rows


def hash_one(item: dict) -> dict:
    path = item["path"]
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Multiple frames require an explicit policy: {path}")
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"EXIF orientation requires an explicit policy: {path}")
        rgb = image.convert("RGB")
        hashes = perceptual_hashes(image)
        return {
            "sample_id": item["sample_id"],
            "original_split": item["original_split"],
            "class_label": item["class_label"],
            "file_size_bytes": len(raw),
            "source_mode": image.mode,
            "width": image.width,
            "height": image.height,
            "sha256": sha256_bytes(raw),
            "rgb_sha256": sha256_bytes(rgb_payload(rgb)),
            **hashes,
            "hash_consensus": ":".join((hashes["ahash"], hashes["dhash"], hashes["phash"])),
        }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_rank(seed: int, sample_id: str) -> str:
    return sha256_bytes(f"{seed}\0{sample_id}".encode())


def make_groups(rows: list[dict], method: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row[method]].append(row)
    return result


def verify_exact_groups(root: Path, groups: dict[str, dict[str, list[dict]]]) -> dict[str, int]:
    checks = {}
    for method in ("sha256", "rgb_sha256"):
        count = 0
        for members in groups[method].values():
            if len(members) < 2:
                continue
            representative = None
            for row in members:
                raw = (root / row["sample_id"]).read_bytes()
                if method == "rgb_sha256":
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        payload = rgb_payload(image)
                else:
                    payload = raw
                if sha256_bytes(payload) != row[method]:
                    raise RuntimeError(f"Image changed during audit: {row['sample_id']}")
                if representative is None:
                    representative = payload
                elif payload != representative:
                    raise RuntimeError(f"Direct comparison failed for {method}")
                else:
                    count += 1
        checks[method] = count
        print(f"Verified {method}: {count} repeated payload comparisons", flush=True)
    return checks


def global_summary(rows: list[dict], groups: dict[str, dict[str, list[dict]]]) -> list[dict]:
    result = []
    for method in MATCH_METHODS:
        buckets = groups[method]
        mixed = [members for members in buckets.values() if len({r["class_label"] for r in members}) > 1]
        result.append({
            "method": method,
            "images": len(rows),
            "distinct_groups": len(buckets),
            "excess_copies": len(rows) - len(buckets),
            "redundancy_fraction": (len(rows) - len(buckets)) / len(rows),
            "nonsingleton_groups": sum(len(members) > 1 for members in buckets.values()),
            "images_in_nonsingleton_groups": sum(len(members) for members in buckets.values() if len(members) > 1),
            "mixed_label_groups": len(mixed),
            "images_in_mixed_label_groups": sum(len(members) for members in mixed),
        })
    return result


def cross_split_summary(
    rows: list[dict],
    groups: dict[str, dict[str, list[dict]]],
    train_split: str,
    eval_split: str,
) -> list[dict]:
    eval_rows = [row for row in rows if row["original_split"] == eval_split]
    if not eval_rows:
        return []
    result = []
    for method in MATCH_METHODS:
        training_values = {
            row[method] for row in rows if row["original_split"] == train_split
        }
        matched = [row for row in eval_rows if row[method] in training_values]
        same_label = 0
        other_label = 0
        for row in matched:
            counterparts = [
                r for r in groups[method][row[method]]
                if r["original_split"] == train_split
            ]
            same_label += any(r["class_label"] == row["class_label"] for r in counterparts)
            other_label += any(r["class_label"] != row["class_label"] for r in counterparts)
        result.append({
            "method": method,
            "train_images": sum(r["original_split"] == train_split for r in rows),
            "evaluation_images": len(eval_rows),
            "evaluation_matched": len(matched),
            "evaluation_overlap_fraction": len(matched) / len(eval_rows),
            "same_label_matched": same_label,
            "other_label_matched": other_label,
        })
    return result


def per_class_summary(rows: list[dict], groups: dict[str, dict[str, list[dict]]], minimum_groups: int) -> list[dict]:
    result = []
    for label in sorted({row["class_label"] for row in rows}):
        subset = [row for row in rows if row["class_label"] == label]
        counts = {method: len({row[method] for row in subset}) for method in MATCH_METHODS}
        tuple_groups = counts["hash_consensus"]
        mixed_memberships = sum(
            any(len({r["class_label"] for r in groups["hash_consensus"][row["hash_consensus"]]}) > 1 for _ in [0])
            for row in subset
        )
        result.append({
            "class_label": label,
            "original_images": len(subset),
            "sha256_groups": counts["sha256"],
            "rgb_groups": counts["rgb_sha256"],
            "ahash_groups": counts["ahash"],
            "dhash_groups": counts["dhash"],
            "phash_groups": counts["phash"],
            "consensus_groups": tuple_groups,
            "consensus_removed_excess": len(subset) - tuple_groups,
            "consensus_removed_percent": f"{100 * (len(subset) - tuple_groups) / len(subset):.6f}",
            "images_in_mixed_label_consensus_groups": mixed_memberships,
            "eligible_at_minimum_groups": int(tuple_groups >= minimum_groups),
        })
    return result


def deduplicate(
    rows: list[dict], groups: dict[str, dict[str, list[dict]]], seed: int
) -> tuple[list[dict], list[dict], list[dict]]:
    kept, removed, group_rows = [], [], []
    for value, members in sorted(groups["hash_consensus"].items()):
        ordered = sorted(members, key=lambda r: (stable_rank(seed, r["sample_id"]), r["sample_id"]))
        representative = ordered[0]
        group_id = sha256_bytes(f"hash_consensus\0{value}".encode())
        labels = sorted({row["class_label"] for row in ordered})
        common = {
            "duplicate_group_id": group_id,
            "duplicate_group_size": len(ordered),
            "representative_sample_id": representative["sample_id"],
            "group_class_count": len(labels),
        }
        kept.append({**representative, **common})
        for row in ordered[1:]:
            removed.append({**row, **common})
        group_rows.append({
            "duplicate_group_id": group_id,
            "group_size": len(ordered),
            "class_count": len(labels),
            "classes_json": json.dumps(labels),
            "distinct_rgb_arrays": len({row["rgb_sha256"] for row in ordered}),
            "representative_sample_id": representative["sample_id"],
            "consensus_value": value,
        })
    return kept, removed, group_rows


def run(args: argparse.Namespace) -> None:
    root = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = datetime.now(timezone.utc).isoformat()
    inventory_rows = inventory(root, args.layout, args.train_split, args.eval_split)
    print(f"Inventory: {len(inventory_rows)} images", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number, row in enumerate(pool.map(hash_one, inventory_rows), 1):
            rows.append(row)
            if number % 1000 == 0 or number == len(inventory_rows):
                print(f"Hashed {number}/{len(inventory_rows)}", flush=True)
    rows.sort(key=lambda row: row["sample_id"])
    groups = {method: make_groups(rows, method) for method in MATCH_METHODS}
    checks = verify_exact_groups(root, groups)
    summary = global_summary(rows, groups)
    cross_split = cross_split_summary(rows, groups, args.train_split, args.eval_split)
    by_class = per_class_summary(rows, groups, args.minimum_groups_per_class)
    kept, removed, group_rows = deduplicate(rows, groups, args.representative_seed)

    snapshot = hashlib.sha256()
    for row in rows:
        snapshot.update(json.dumps([row["sample_id"], row["sha256"]], separators=(",", ":")).encode() + b"\n")
    mixed_tuple_groups = sum(row["class_count"] > 1 for row in group_rows)
    metadata = {
        "status": "completed",
        "dataset_root": str(root),
        "layout": args.layout,
        "images": len(rows),
        "classes": len({row["class_label"] for row in rows}),
        "original_split_counts": dict(Counter(row["original_split"] for row in rows)),
        "dataset_snapshot_sha256": snapshot.hexdigest(),
        "deduplication_rule": "equality of complete 64-bit aHash, dHash, and pHash tuple",
        "interpretation": "perceptual-signature sensitivity analysis, not verified pixel or binary identity",
        "retained_representatives": len(kept),
        "removed_excess_copies": len(removed),
        "mixed_label_consensus_groups": mixed_tuple_groups,
        "safe_to_construct_label_preserving_clean_split": mixed_tuple_groups == 0,
        "minimum_groups_per_class": args.minimum_groups_per_class,
        "classes_below_minimum": [row["class_label"] for row in by_class if not row["eligible_at_minimum_groups"]],
        "representative_seed": args.representative_seed,
        "hash_bits": 64,
        "hamming_threshold": 0,
        "perceptual_hash_reference": REFERENCE,
        "direct_equality_checks": checks,
        "versions": {"python": sys.version, "Pillow": PIL.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "platform": platform.platform(),
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
    }
    output.mkdir(parents=True)
    write_csv(output / "hashes.csv", rows)
    write_csv(output / "global_summary.csv", summary)
    write_csv(output / "per_class.csv", by_class)
    write_csv(output / "consensus_groups.csv", group_rows)
    write_csv(output / "retained_representatives.csv", kept)
    write_csv(output / "removed_excess_copies.csv", removed, list(kept[0]))
    if cross_split:
        write_csv(output / "original_cross_split_overlap.csv", cross_split)
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            checksums.append(f"{sha256_bytes(path.read_bytes())}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("class_folders", "split_class_folders"), required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--minimum-groups-per-class", type=int, default=50)
    parser.add_argument("--representative-seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.minimum_groups_per_class < 1:
        parser.error("workers and minimum groups must be positive")
    run(args)


if __name__ == "__main__":
    main()
