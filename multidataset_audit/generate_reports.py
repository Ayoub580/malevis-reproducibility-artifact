#!/usr/bin/env python3
"""Generate readable reports from completed multi-dataset audits."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(
    os.environ.get(
        "MALEVIS_AUDIT_RESULTS",
        Path(__file__).resolve().parent / "results",
    )
).resolve()
DATASETS = {
    "Malimg": ROOT / "malimg_consensus_v1",
    "Virus-MNIST": ROOT / "virusmnist_consensus_v1",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(value: str | float) -> str:
    return f"{100 * float(value):.2f}%"


def global_table(rows: list[dict]) -> list[str]:
    result = [
        "| Method | Distinct groups | Excess copies | Redundancy | Mixed-label groups |",
        "|---|---:|---:|---:|---:|",
    ]
    names = {
        "sha256": "File SHA-256",
        "rgb_sha256": "Native RGB SHA-256",
        "ahash": "aHash",
        "dhash": "dHash",
        "phash": "pHash",
        "hash_consensus": "Complete tuple",
    }
    for row in rows:
        result.append(
            f"| {names[row['method']]} | {int(row['distinct_groups']):,} | "
            f"{int(row['excess_copies']):,} | {pct(row['redundancy_fraction'])} | "
            f"{int(row['mixed_label_groups']):,} |"
        )
    return result


def class_table(rows: list[dict]) -> list[str]:
    result = [
        "| Class | Images | File groups | RGB groups | aHash | dHash | pHash | Tuple groups | Within-class excess | Removed | Eligible >=50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        result.append(
            f"| {row['class_label']} | {int(row['original_images']):,} | "
            f"{int(row['sha256_groups']):,} | {int(row['rgb_groups']):,} | "
            f"{int(row['ahash_groups']):,} | {int(row['dhash_groups']):,} | "
            f"{int(row['phash_groups']):,} | {int(row['consensus_groups']):,} | "
            f"{int(row['consensus_removed_excess']):,} | "
            f"{float(row['consensus_removed_percent']):.2f}% | "
            f"{'yes' if row['eligible_at_minimum_groups'] == '1' else 'no'} |"
        )
    return result


def collision_stats(path: Path) -> dict:
    groups = [row for row in read_csv(path / "consensus_groups.csv") if int(row["class_count"]) > 1]
    return {
        "groups": len(groups),
        "images": sum(int(row["group_size"]) for row in groups),
        "exact_rgb_groups": sum(int(row["distinct_rgb_arrays"]) == 1 for row in groups),
        "pure_groups": sum(1 for row in read_csv(path / "consensus_groups.csv") if int(row["class_count"]) == 1),
        "details": groups,
    }


def conservative_cleaning_stats(path: Path, classes: list[dict]) -> dict:
    """Exclude mixed-label tuples and classes below the declared threshold."""
    rows = read_csv(path / "hashes.csv")
    mixed_values = {
        row["consensus_value"]
        for row in read_csv(path / "consensus_groups.csv")
        if int(row["class_count"]) > 1
    }
    eligible_labels = {
        row["class_label"] for row in classes
        if row["eligible_at_minimum_groups"] == "1"
    }
    eligible_original = [row for row in rows if row["class_label"] in eligible_labels]
    safe_pool = [
        row for row in eligible_original
        if row["hash_consensus"] not in mixed_values
    ]
    retained = len({row["hash_consensus"] for row in safe_pool})
    return {
        "eligible_classes": len(eligible_labels),
        "excluded_classes": sorted({row["class_label"] for row in classes} - eligible_labels),
        "eligible_original_images": len(eligible_original),
        "mixed_label_images_removed": len(eligible_original) - len(safe_pool),
        "safe_pool_images": len(safe_pool),
        "retained_pure_groups": retained,
        "within_safe_pool_excess": len(safe_pool) - retained,
        "within_safe_pool_redundancy_fraction": (len(safe_pool) - retained) / len(safe_pool),
    }


def refresh_checksums(path: Path) -> None:
    target = path / "checksums.sha256"
    lines = []
    for file in sorted(path.iterdir()):
        if file.is_file() and file != target:
            lines.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.name}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    overview_rows = []
    combined = [
        "# Malimg and Virus-MNIST consensus audit",
        "",
        "The audit uses the MaleVis perceptual sensitivity rule: equality of the complete 64-bit aHash, dHash, and pHash tuple at Hamming distance zero. File and native-RGB equality are verified separately. Tuple equality is a lossy signature criterion and is not proof of equal images, binaries, behavior, or provenance.",
        "",
        "## Dataset comparison",
        "",
        "| Dataset | Images | Classes | Exact RGB excess | Tuple groups | Tuple excess | Tuple redundancy | Mixed-label tuple groups | Classes below 50 groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    details = []
    for name, path in DATASETS.items():
        metadata = json.loads((path / "run_metadata.json").read_text())
        global_rows = read_csv(path / "global_summary.csv")
        by_method = {row["method"]: row for row in global_rows}
        classes = read_csv(path / "per_class.csv")
        collisions = collision_stats(path)
        conservative = conservative_cleaning_stats(path, classes)
        tuple_row = by_method["hash_consensus"]
        exact_row = by_method["rgb_sha256"]
        combined.append(
            f"| {name} | {metadata['images']:,} | {metadata['classes']} | "
            f"{int(exact_row['excess_copies']):,} ({pct(exact_row['redundancy_fraction'])}) | "
            f"{int(tuple_row['distinct_groups']):,} | {int(tuple_row['excess_copies']):,} | "
            f"{pct(tuple_row['redundancy_fraction'])} | {collisions['groups']} | "
            f"{len(metadata['classes_below_minimum'])} |"
        )
        overview_rows.append({
            "dataset": name,
            "images": metadata["images"],
            "classes": metadata["classes"],
            "exact_rgb_groups": int(exact_row["distinct_groups"]),
            "exact_rgb_excess": int(exact_row["excess_copies"]),
            "exact_rgb_redundancy_fraction": exact_row["redundancy_fraction"],
            "consensus_groups": int(tuple_row["distinct_groups"]),
            "consensus_excess": int(tuple_row["excess_copies"]),
            "consensus_redundancy_fraction": tuple_row["redundancy_fraction"],
            "mixed_label_consensus_groups": collisions["groups"],
            "images_in_mixed_label_consensus_groups": collisions["images"],
            "exact_rgb_mixed_label_consensus_groups": collisions["exact_rgb_groups"],
            "pure_consensus_groups_after_excluding_mixed": collisions["pure_groups"],
            "classes_below_50_groups": ";".join(metadata["classes_below_minimum"]),
            **conservative,
        })

        section = [
            f"## {name}",
            "",
            *global_table(global_rows),
            "",
        ]
        cross_file = path / "original_cross_split_overlap.csv"
        if cross_file.exists():
            section += [
                "### Original train/test overlap",
                "",
                "| Method | Test matched | Test images | Overlap | Same-label | Other-label |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            names = {
                "sha256": "File SHA-256", "rgb_sha256": "Native RGB SHA-256",
                "ahash": "aHash", "dhash": "dHash", "phash": "pHash",
                "hash_consensus": "Complete tuple",
            }
            for row in read_csv(cross_file):
                section.append(
                    f"| {names[row['method']]} | {int(row['evaluation_matched']):,} | "
                    f"{int(row['evaluation_images']):,} | {pct(row['evaluation_overlap_fraction'])} | "
                    f"{int(row['same_label_matched']):,} | {int(row['other_label_matched']):,} |"
                )
            section.append("")
        section += [
            "### Cross-label tuple groups",
            "",
            f"There are **{collisions['groups']}** mixed-label tuple groups containing **{collisions['images']:,}** images. "
            f"Of these, **{collisions['exact_rgb_groups']}** groups contain a single native RGB array. "
            "A strict one-representative-per-tuple operation would discard labels from these groups, so a label-preserving clean split has not been constructed.",
            "",
            "| Size | Classes | Distinct RGB arrays | Representative |",
            "|---:|---|---:|---|",
        ]
        for row in collisions["details"]:
            section.append(
                f"| {int(row['group_size']):,} | {', '.join(json.loads(row['classes_json']))} | "
                f"{int(row['distinct_rgb_arrays'])} | `{row['representative_sample_id']}` |"
            )
        section += [
            "",
            "### Conservative label-preserving scenario",
            "",
            f"After excluding mixed-label tuple groups and classes below 50 tuple groups, "
            f"{conservative['eligible_classes']} classes remain. The eligible classes contain "
            f"{conservative['eligible_original_images']:,} original images; removing "
            f"{conservative['mixed_label_images_removed']:,} images in mixed-label groups leaves "
            f"{conservative['safe_pool_images']:,} images and "
            f"{conservative['retained_pure_groups']:,} pure tuple groups. Retaining one image per "
            f"group would remove {conservative['within_safe_pool_excess']:,} additional excess "
            f"copies ({pct(conservative['within_safe_pool_redundancy_fraction'])} of the safe pool). "
            f"Excluded classes: {', '.join(conservative['excluded_classes'])}.",
            "",
            "### Per-class statistics",
            "",
            *class_table(classes),
            "",
        ]
        (path / "REPORT.md").write_text("\n".join(section) + "\n", encoding="utf-8")
        refresh_checksums(path)
        details.extend(section)

    combined += [
        "",
        "## Interpretation for a follow-up classifier study",
        "",
        "The raw tuple counts are suitable as an audit result. They are not yet sufficient to create label-preserving cleaned benchmarks because both datasets contain cross-label tuple groups. A conservative next protocol would exclude every mixed-label tuple group, exclude classes with fewer than 50 remaining groups, retain one deterministic representative from each remaining pure group, and only then create stratified fitting/development/evaluation partitions. The tuple rule should also be manually validated on samples from each dataset before its removals are called near-duplicates.",
        "",
        *details,
    ]
    (ROOT / "COMPARISON.md").write_text("\n".join(combined) + "\n", encoding="utf-8")
    with (ROOT / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overview_rows[0]))
        writer.writeheader()
        writer.writerows(overview_rows)


if __name__ == "__main__":
    main()
