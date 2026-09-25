#!/usr/bin/env python3
"""Recompute the manuscript's principal audit and construction claims."""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("MALEVIS_PROJECT_ROOT", ".")).resolve()


def rows(rel: str):
    with (ROOT / rel).open(newline="") as handle:
        return list(csv.DictReader(handle))


exact = rows("results/exact_audit_v1/manifest.csv")
assert len(exact) == 14226
assert Counter(r["original_split"] for r in exact) == {"train": 9100, "val": 5126}
assert {(r["width"], r["height"], r["source_mode"]) for r in exact} == {("300", "300", "RGB")}

for field, expected_unique, expected_train_excess, expected_val_excess, expected_match in [
    ("file_sha256", 13746, 286, 87, 173),
    ("rgb_pixels_sha256", 13746, 286, 87, 173),
    ("model_input_sha256", 13671, 339, 99, 193),
]:
    train = [r for r in exact if r["original_split"] == "train"]
    val = [r for r in exact if r["original_split"] == "val"]
    train_values = {r[field] for r in train}
    matched = [r for r in val if r[field] in train_values]
    assert len({r[field] for r in exact}) == expected_unique
    assert len(train) - len(train_values) == expected_train_excess
    assert len(val) - len({r[field] for r in val}) == expected_val_excess
    assert len(matched) == expected_match

train_rgb = {r["rgb_pixels_sha256"] for r in exact if r["original_split"] == "train"}
native_matches = [r for r in exact if r["original_split"] == "val" and r["rgb_pixels_sha256"] in train_rgb]
assert Counter(r["class_label"] for r in native_matches) == {"Fasong": 117, "Amonetize": 50, "Other": 6}

hashes = rows("results/hash_audit_v1/hashes.csv")
train_hash = [r for r in hashes if r["original_split"] == "train"]
val_hash = [r for r in hashes if r["original_split"] == "val"]
assert len(hashes) == 14226 and len(train_hash) == 9100 and len(val_hash) == 5126

def hash_stats(fields):
    by_key = defaultdict(list)
    for r in train_hash:
        by_key[tuple(r[f] for f in fields)].append(r)
    matched = []
    pairs = 0
    other = 0
    exact_rgb = 0
    for r in val_hash:
        candidates = by_key.get(tuple(r[f] for f in fields), [])
        if candidates:
            matched.append(r)
            pairs += len(candidates)
            other += any(c["class_label"] != r["class_label"] for c in candidates)
            exact_rgb += any(c["rgb_sha256"] == r["rgb_sha256"] for c in candidates)
    return len(matched), pairs, other, exact_rgb

assert hash_stats(["ahash"])[:3] == (3808, 744583, 2298)
assert hash_stats(["dhash"])[:3] == (2064, 161106, 2)
assert hash_stats(["phash"])[:3] == (1974, 153620, 0)
assert hash_stats(["ahash", "dhash", "phash"]) == (1691, 144613, 0, 173)

# Same-pair union and majority counts.
indices = {}
for f in ("ahash", "dhash", "phash"):
    d = defaultdict(list)
    for r in train_hash:
        d[r[f]].append(r)
    indices[f] = d
union_val = majority_val = union_pairs = majority_pairs = union_other = majority_other = 0
for v in val_hash:
    counts = Counter()
    train_labels = {}
    for f in indices:
        for t in indices[f].get(v[f], []):
            counts[t["sample_id"]] += 1
            train_labels[t["sample_id"]] = t["class_label"]
    union_val += bool(counts)
    majority_val += any(n >= 2 for n in counts.values())
    union_pairs += sum(n >= 1 for n in counts.values())
    majority_pairs += sum(n >= 2 for n in counts.values())
    union_other += any(train_labels[sample_id] != v["class_label"] for sample_id, n in counts.items() if n >= 1)
    majority_other += any(train_labels[sample_id] != v["class_label"] for sample_id, n in counts.items() if n >= 2)
assert (union_val, majority_val, union_pairs, majority_pairs) == (3841, 2141, 753266, 161430)
assert (union_other, majority_other) == (2299, 0)

pair_manifest = rows("results/pair_review_v1/sample_manifest.csv")
assert Counter(r["sample_kind"] for r in pair_manifest) == {"candidate": 500, "nonmatching_control": 50}
candidate_strata = {r["stratum"] for r in pair_manifest if r["sample_kind"] == "candidate"}
assert len(candidate_strata) == 368
pair_metrics = rows("results/pair_review_v1/pair_metrics.csv")
all_three = [r for r in pair_metrics if r["sample_kind"] == "candidate" and r["hash_mask"] == "7"]
nonexact_three = [r for r in all_three if r["native_exact_rgb"] == "0"]
assert len(all_three) == 78 and len(nonexact_three) == 70
assert round(statistics.median(float(r["native_ssim_rgb"]) for r in nonexact_three), 6) == 0.999995
assert round(statistics.median(float(r["native_mean_abs_channel_diff_255"]) for r in nonexact_three), 4) == 0.0073

analysis = json.loads((ROOT / "results/pair_review_v1/annotations_completed/analysis.json").read_text())
ann = analysis["annotation"]
assert round(ann["observed_agreement"] * 100, 2) == 98.36
assert round(ann["cohen_kappa"], 2) == 0.96
assert ann["sample_groups"]["All three hashes"]["n"] == 78
assert ann["sample_groups"]["No-hash controls"]["annotator_1"]["visually_indistinguishable"] + ann["sample_groups"]["No-hash controls"]["annotator_1"]["near_duplicate"] == 11
annotator_1 = {r["pair_id"]: r["judgement"].strip().lower() for r in rows("results/pair_review_v1/annotations_completed/annotator_1.csv")}
annotator_2 = {r["pair_id"]: r["judgement"].strip().lower() for r in rows("results/pair_review_v1/annotations_completed/annotator_2.csv")}
disagreements = [(annotator_1[k], annotator_2[k]) for k in annotator_1 if annotator_1[k] != annotator_2[k]]
assert len(disagreements) == 9
assert all(set(pair) == {"visually_indistinguishable", "near_duplicate"} for pair in disagreements)
assert all(value != "uncertain" for value in [*annotator_1.values(), *annotator_2.values()])
all_three_ids = {r["pair_id"] for r in all_three}
assert all(annotator_1[k] in {"visually_indistinguishable", "near_duplicate"} and annotator_2[k] in {"visually_indistinguishable", "near_duplicate"} for k in all_three_ids)
controls = [r for r in pair_metrics if r["sample_kind"] == "nonmatching_control"]
assert len(controls) == 50 and all(r["same_class"] == "1" and r["hash_mask"] == "0" for r in controls)

clean = rows("results/dedup_before_split_consensus_23class_v1/per_class.csv")
assert sum(int(r["original_images"]) for r in clean) == 14226
assert sum(int(r["retained_groups"]) for r in clean) == 8981
assert sum(int(r["removed_excess_copies"]) for r in clean) == 5245
excluded = {r["class_label"]: int(r["retained_groups"]) for r in clean if int(r["retained_groups"]) < 50}
assert excluded == {"Snarasite": 5, "VBA": 1, "Vilsel": 13}
eligible = [r for r in clean if int(r["retained_groups"]) >= 50]
assert len(eligible) == 23
assert sum(int(r["original_images"]) for r in eligible) == 12730
assert sum(int(r["retained_groups"]) for r in eligible) == 8962

for dataset, expected in [("malimg_consensus_v1", (9339, 1320, 1858, 1)), ("virusmnist_consensus_v1", (51880, 17395, 18351, 10))]:
    summary = {r["method"]: r for r in rows(f"results/external_consensus_audit_v1/{dataset}/global_summary.csv")}
    images, rgb_excess, tuple_excess, mixed = expected
    assert int(summary["rgb_sha256"]["images"]) == images
    assert int(summary["rgb_sha256"]["excess_copies"]) == rgb_excess
    assert int(summary["hash_consensus"]["excess_copies"]) == tuple_excess
    assert int(summary["hash_consensus"]["mixed_label_groups"]) == mixed
if True:
    virus_summary = {r["method"]: r for r in rows("results/external_consensus_audit_v1/virusmnist_consensus_v1/global_summary.csv")}
    assert int(virus_summary["hash_consensus"]["images_in_mixed_label_groups"]) == 23

print("PASS: all principal dataset, overlap, sampling, pixel, annotation, cleaning, and external-audit claims reproduce from the frozen row-level files.")
