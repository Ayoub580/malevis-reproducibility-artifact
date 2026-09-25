#!/usr/bin/env python3
"""Verify and summarize the completed 45-run Virus-MNIST study."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t


CONDITIONS = ("full", "clean_consensus", "random_subsample9")
METRICS = ("accuracy", "macro_f1", "macro_recall")
MODELS = {
    "resnet50": ("resnet50",),
    "convnext_tiny": ("convnext_tiny",),
    "swin_tiny": ("swin_tiny",),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_model(root: Path, folder: str, seeds: list[int]) -> list[dict]:
    rows = []
    missing = []
    for condition in CONDITIONS:
        for seed in seeds:
            path = root / folder / condition / f"seed_{seed}" / "metrics.json"
            if not path.exists():
                missing.append(str(path))
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["condition"] != condition or int(row["seed"]) != seed:
                raise ValueError(f"Metric identity mismatch: {path}")
            rows.append(row)
    if missing:
        raise ValueError("Missing completed metric files:\n" + "\n".join(missing))
    for seed in seeds:
        selected = [row for row in rows if int(row["seed"]) == seed]
        if len({row["initial_state_sha256"] for row in selected}) != 1:
            raise ValueError(f"Initial state differs across conditions for seed {seed}")
    return rows


def ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean
    half = t.ppf(0.975, len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet-output", type=Path, required=True)
    parser.add_argument("--convnext-output", type=Path, required=True)
    parser.add_argument("--swin-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 7, 2024, 99])
    args = parser.parse_args()
    roots = {
        "resnet50": args.resnet_output.resolve(),
        "convnext_tiny": args.convnext_output.resolve(),
        "swin_tiny": args.swin_output.resolve(),
    }
    all_rows = {
        model: load_model(roots[model], folders[0], args.seeds)
        for model, folders in MODELS.items()
    }

    performance = []
    contrasts = []
    for model, rows in all_rows.items():
        for condition in CONDITIONS:
            selected = [row for row in rows if row["condition"] == condition]
            for metric in METRICS:
                values = [100 * float(row[metric]) for row in selected]
                performance.append({
                    "model": model,
                    "condition": condition,
                    "metric": metric,
                    "seeds": len(values),
                    "mean_percent": f"{statistics.mean(values):.6f}",
                    "sample_sd_pp": f"{statistics.stdev(values):.6f}",
                    "minimum_percent": f"{min(values):.6f}",
                    "maximum_percent": f"{max(values):.6f}",
                })
        for left, right, label in (
            ("full", "clean_consensus", "full_minus_clean"),
            ("random_subsample9", "clean_consensus", "random_minus_clean"),
            ("full", "random_subsample9", "full_minus_random"),
        ):
            by = {(row["condition"], int(row["seed"])): row for row in rows}
            for metric in METRICS:
                values = [100 * (float(by[(left, seed)][metric]) - float(by[(right, seed)][metric])) for seed in args.seeds]
                lower, upper = ci95(values)
                contrasts.append({
                    "model": model,
                    "contrast": label,
                    "metric": metric,
                    "seeds": len(values),
                    "mean_pp": f"{statistics.mean(values):.6f}",
                    "sample_sd_pp": f"{statistics.stdev(values):.6f}",
                    "ci95_lower_pp": f"{lower:.6f}",
                    "ci95_upper_pp": f"{upper:.6f}",
                    "seed_values_pp": ";".join(f"{seed}:{value:.6f}" for seed, value in zip(args.seeds, values)),
                })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "model_performance.csv", performance)
    write_csv(args.output / "paired_contrasts.csv", contrasts)
    report = [
        "# Virus-MNIST training summary", "",
        f"Verified {sum(len(rows) for rows in all_rows.values())} completed runs: three architectures, three conditions, and {len(args.seeds)} paired seeds.", "",
        "## Accuracy", "",
        "| Model | Full | Random | Clean | Full-Clean | Random-Clean |", "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        means = {(row["condition"], row["metric"]): row for row in performance if row["model"] == model}
        diffs = {(row["contrast"], row["metric"]): row for row in contrasts if row["model"] == model}
        report.append(
            f"| {model} | {float(means[('full','accuracy')]['mean_percent']):.2f}% | "
            f"{float(means[('random_subsample9','accuracy')]['mean_percent']):.2f}% | "
            f"{float(means[('clean_consensus','accuracy')]['mean_percent']):.2f}% | "
            f"{float(diffs[('full_minus_clean','accuracy')]['mean_pp']):+.2f} pp | "
            f"{float(diffs[('random_minus_clean','accuracy')]['mean_pp']):+.2f} pp |"
        )
    report += [
        "",
        "Full-Clean combines redundancy removal, training-size change, and different evaluation images. Random-Clean is the size- and class-count-matched sensitivity contrast. Intervals in paired_contrasts.csv describe variation over the five fixed optimization seeds and do not include split or representative uncertainty.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Wrote verified Virus-MNIST summary to", args.output, flush=True)


if __name__ == "__main__":
    main()
