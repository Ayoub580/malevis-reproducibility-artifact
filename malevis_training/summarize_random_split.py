#!/usr/bin/env python3
"""Summarize pooled MaleVis and paired 70/20/10 cleaning manifests."""
import argparse
import csv
from collections import Counter
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


def summarize(protocol_dir, condition_dir, output_dir, identity_key):
    protocol_dir, condition_dir, output_dir = map(Path, (protocol_dir, condition_dir, output_dir))
    manifest = read_csv(protocol_dir / "manifest.csv")
    full = read_csv(condition_dir / "full_fitting.csv")
    clean = read_csv(condition_dir / "clean_exact_fitting.csv")
    control = read_csv(condition_dir / "random_control_fitting.csv")
    development = read_csv(condition_dir / "development.csv")
    evaluation = read_csv(condition_dir / "evaluation.csv")
    removed = read_csv(condition_dir / "removed_exact.csv")
    classes = sorted({r["class_label"] for r in manifest})
    if identity_key not in manifest[0]:
        raise ValueError(f"Unknown identity column: {identity_key}")
    if {r["sample_id"] for r in full} - {r["sample_id"] for r in manifest}:
        raise ValueError("Condition manifest contains an unknown sample.")
    if {r["sample_id"] for r in clean} | {r["sample_id"] for r in removed} != {
            r["sample_id"] for r in full}:
        raise ValueError("Clean and removed files do not reconstruct Full fitting.")

    rows = []
    for cls in classes:
        all_c = [r for r in manifest if r["class_label"] == cls]
        full_c = [r for r in full if r["class_label"] == cls]
        clean_c = [r for r in clean if r["class_label"] == cls]
        control_c = [r for r in control if r["class_label"] == cls]
        dev_c = [r for r in development if r["class_label"] == cls]
        eval_c = [r for r in evaluation if r["class_label"] == cls]
        removed_c = [r for r in removed if r["class_label"] == cls]
        distinct = len({r[identity_key] for r in all_c})
        full_keys = {r[identity_key] for r in full_c}
        clean_keys = {r[identity_key] for r in clean_c}
        control_keys = {r[identity_key] for r in control_c}
        rows.append({
            "class_label": cls,
            "pooled_images": len(all_c),
            "pooled_distinct_exact_inputs": distinct,
            "pooled_excess_exact_copies": len(all_c) - distinct,
            "pooled_exact_redundancy_percent": f"{pct(len(all_c)-distinct, len(all_c)):.4f}",
            "full_fitting": len(full_c),
            "clean_exact_fitting": len(clean_c),
            "random_control_fitting": len(control_c),
            "removed_from_fitting": len(removed_c),
            "fitting_removal_percent": f"{pct(len(removed_c), len(full_c)):.4f}",
            "development": len(dev_c),
            "evaluation": len(eval_c),
            "full_evaluation_images_with_fitting_match": sum(r[identity_key] in full_keys for r in eval_c),
            "clean_evaluation_images_with_fitting_match": sum(r[identity_key] in clean_keys for r in eval_c),
            "control_evaluation_images_with_fitting_match": sum(r[identity_key] in control_keys for r in eval_c),
            "full_development_images_with_fitting_match": sum(r[identity_key] in full_keys for r in dev_c),
            "clean_development_images_with_fitting_match": sum(r[identity_key] in clean_keys for r in dev_c),
            "control_development_images_with_fitting_match": sum(r[identity_key] in control_keys for r in dev_c),
        })

    distinct_global = len({r[identity_key] for r in manifest})
    clean_total = len(clean) + len(development) + len(evaluation)
    summary = [{
        "scope": "pooled_full_dataset",
        "images": len(manifest),
        "distinct_exact_inputs": distinct_global,
        "excess_exact_copies": len(manifest) - distinct_global,
        "redundancy_percent": f"{pct(len(manifest)-distinct_global, len(manifest)):.4f}",
        "fitting": len(full), "development": len(development), "evaluation": len(evaluation),
        "removed_from_fitting": 0,
    }, {
        "scope": "experimental_clean_exact_dataset",
        "images": clean_total,
        "distinct_exact_inputs": "",
        "excess_exact_copies": "",
        "redundancy_percent": "",
        "fitting": len(clean), "development": len(development), "evaluation": len(evaluation),
        "removed_from_fitting": len(removed),
    }, {
        "scope": "random_control_dataset",
        "images": len(control) + len(development) + len(evaluation),
        "distinct_exact_inputs": "",
        "excess_exact_copies": "",
        "redundancy_percent": "",
        "fitting": len(control), "development": len(development), "evaluation": len(evaluation),
        "removed_from_fitting": len(full) - len(control),
    }]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dataset_statistics_per_class.csv", rows)
    write_csv(output_dir / "dataset_statistics_summary.csv", summary)

    total_eval_full = sum(int(r["full_evaluation_images_with_fitting_match"]) for r in rows)
    total_eval_clean = sum(int(r["clean_evaluation_images_with_fitting_match"]) for r in rows)
    total_eval_control = sum(int(r["control_evaluation_images_with_fitting_match"]) for r in rows)
    total_dev_full = sum(int(r["full_development_images_with_fitting_match"]) for r in rows)
    total_dev_clean = sum(int(r["clean_development_images_with_fitting_match"]) for r in rows)
    total_dev_control = sum(int(r["control_development_images_with_fitting_match"]) for r in rows)
    lines = [
        "# Pooled 70/20/10 MaleVis statistics", "",
        f"Exact identity: `{identity_key}`.", "",
        "## Whole-dataset and experiment totals", "",
        "| Scope | Images | Fitting | Development | Evaluation | Removed from fitting |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Pooled full dataset | {len(manifest)} | {len(full)} | {len(development)} | {len(evaluation)} | 0 |",
        f"| Experimental Clean-Exact | {clean_total} | {len(clean)} | {len(development)} | {len(evaluation)} | {len(removed)} |",
        f"| Random-Control | {len(control)+len(development)+len(evaluation)} | {len(control)} | {len(development)} | {len(evaluation)} | {len(full)-len(control)} |",
        "",
        f"The pooled dataset contains {distinct_global} distinct exact model inputs and "
        f"{len(manifest)-distinct_global} excess copies ({pct(len(manifest)-distinct_global, len(manifest)):.2f}%).",
        "Clean-Exact removes training counterparts of fixed development or evaluation inputs; "
        "it is not a global one-copy-per-group dataset.", "",
        "## Cross-partition exposure", "",
        "| Condition | Development images matched in fitting | Evaluation images matched in fitting |",
        "|---|---:|---:|",
        f"| Full | {total_dev_full} | {total_eval_full} |",
        f"| Clean-Exact | {total_dev_clean} | {total_eval_clean} |",
        f"| Random-Control | {total_dev_control} | {total_eval_control} |",
        "",
        "## Per-class statistics", "",
        "| Class | Pooled | Distinct | Excess | Full fit | Clean fit | Removed | Dev | Eval | Full eval matched | Clean eval matched | Control eval matched |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {class_label} | {pooled_images} | {pooled_distinct_exact_inputs} | "
                     "{pooled_excess_exact_copies} | {full_fitting} | {clean_exact_fitting} | "
                     "{removed_from_fitting} | {development} | {evaluation} | "
                     "{full_evaluation_images_with_fitting_match} | "
                     "{clean_evaluation_images_with_fitting_match} | "
                     "{control_evaluation_images_with_fitting_match} |".format(**r))
    (output_dir / "DATASET_STATISTICS.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} class rows and three summary rows to {output_dir}.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol-dir", type=Path, required=True)
    p.add_argument("--condition-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--identity-key", default="input_sha256")
    args = p.parse_args()
    summarize(args.protocol_dir, args.condition_dir,
              args.output_dir or args.condition_dir, args.identity_key)


if __name__ == "__main__":
    main()
