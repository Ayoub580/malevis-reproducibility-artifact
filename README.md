# MaleVis representational-redundancy artifact

This repository contains the essential code used to audit image redundancy,
construct group-disjoint data partitions, run the controlled classifier
experiments, and verify the numerical claims reported in:

> **Representational Redundancy in MaleVis and Its Evaluation Effects: A
> Reproducible Audit with Controlled Retraining**

The artifact also contains the Malimg and Virus-MNIST replication code. It does
not contain malware images, APKs, pretrained weights, trained checkpoints, or
the full per-image result archive.

## Repository layout

- `multidataset_audit/`: file, RGB-pixel, and aHash/dHash/pHash tuple audits.
- `malevis_training/`: MaleVis protocol construction and the ResNet50,
  ConvNeXt-Tiny, and Swin-Tiny experiments.
- `malimg_training/`: Malimg protocol construction and ResNet50 replication.
- `virusmnist_training/`: Virus-MNIST protocol construction and Spatial CNN
  replication.
- `validation/`: independent checks for archived run outputs and manuscript
  claims.
- `manuscript_tools/`: scripts that regenerate the reported tables and plots
  from frozen outputs.

## Data

Obtain each dataset from its original distributor and keep its original class
labels. Expected layouts are:

```text
MaleVis/
  train/<class>/*.png
  val/<class>/*.png

Malimg/
  <class>/*.png

Virus-MNIST/
  train/<class>/*.{jpg,jpeg,png}
  test/<class>/*.{jpg,jpeg,png}
```

Dataset licenses and access conditions remain those of the original sources.
The scripts read source images without modifying or redistributing them.

## Environment

The completed audit used Python 3.10.19, NumPy 1.26.4, Pillow 12.0.0, and
SciPy 1.15.3. Install the audit dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-audit.txt
```

Training additionally used TensorFlow 2.20.0/Keras 3.12.0 and PyTorch
2.10.0/torchvision 0.25.0. Install a PyTorch build compatible with the local
CUDA driver, then install the remaining packages in
`requirements-training.txt`. Exact bitwise equality across different GPU,
CUDA, and library versions is not assumed.

## 1. Audit the datasets

The common audit computes SHA-256 values for files and decoded RGB arrays,
64-bit aHash/dHash/pHash values, tuple groups, class summaries, and original
cross-split overlap where a supplied split exists.

```bash
python multidataset_audit/audit_and_deduplicate.py \
  --dataset /data/MaleVis \
  --layout split_class_folders \
  --train-split train --eval-split val \
  --minimum-groups-per-class 50 \
  --output outputs/audits/malevis

python multidataset_audit/audit_and_deduplicate.py \
  --dataset /data/Malimg \
  --layout class_folders \
  --minimum-groups-per-class 50 \
  --output outputs/audits/malimg

python multidataset_audit/audit_and_deduplicate.py \
  --dataset /data/Virus-MNIST \
  --layout split_class_folders \
  --train-split train --eval-split test \
  --minimum-groups-per-class 50 \
  --output outputs/audits/virusmnist
```

Tuple equality is an operational perceptual-grouping rule. It does not prove
that source executables, behavior, provenance, or even native pixels are equal.

## 2. Construct the MaleVis conditions

First create the pooled individual-image protocol and its exact identities:

```bash
python malevis_training/train_random_split_inflation.py \
  --dataset /data/MaleVis \
  --protocol-dir outputs/malevis/full_protocol \
  --output outputs/malevis/exact_conditions \
  --prepare-only
```

Then retain one representative per complete aHash/dHash/pHash tuple and split
the retained groups:

```bash
python malevis_training/deduplicate_before_split.py \
  --source-manifest outputs/malevis/full_protocol/manifest.csv \
  --hash-manifest outputs/audits/malevis/hashes.csv \
  --output outputs/malevis/clean_consensus_protocol \
  --rule hash_consensus \
  --minimum-groups-per-class 50 \
  --exclude-insufficient-classes
```

Freeze Full-23, Clean-Consensus-23, and count-matched
Random-Subsample-23 manifests before starting expensive training:

```bash
python malevis_training/train_four_condition_study.py \
  --dataset /data/MaleVis \
  --full-protocol-dir outputs/malevis/full_protocol \
  --clean-protocol-dir outputs/malevis/clean_consensus_protocol \
  --hash-manifest outputs/audits/malevis/hashes.csv \
  --output outputs/malevis/resnet50 \
  --conditions full clean_consensus random_subsample23 \
  --seeds 42 101 7 2024 99 \
  --prepare-only
```

Remove `--prepare-only` and add `--evaluate` to train ResNet50. The
ConvNeXt-Tiny and Swin-Tiny scripts consume the frozen condition manifests in
that output directory:

```bash
python malevis_training/train_convnext_tiny_23.py \
  --dataset /data/MaleVis \
  --manifest-dir outputs/malevis/resnet50 \
  --output outputs/malevis/convnext_tiny \
  --conditions full clean_consensus random_subsample23 \
  --seeds 42 101 7 2024 99 --batch-size 8 --evaluate

python malevis_training/train_swin_tiny_23.py \
  --dataset /data/MaleVis \
  --manifest-dir outputs/malevis/resnet50 \
  --output outputs/malevis/swin_tiny \
  --conditions full clean_consensus random_subsample23 \
  --seeds 42 101 7 2024 99 --batch-size 8 --evaluate
```

Every script records its configuration, source digests, split manifests,
checkpoint-selection history, predictions, per-class metrics, and confusion
matrix. Existing outputs are reused only when their recorded configuration
matches.

## 3. Run the replications

Create Malimg and Virus-MNIST protocols from the corresponding audit folders:

```bash
python malimg_training/prepare_malimg_protocol.py \
  --dataset /data/Malimg \
  --audit-dir outputs/audits/malimg \
  --output outputs/malimg/protocol

python virusmnist_training/prepare_virusmnist_protocol.py \
  --dataset /data/Virus-MNIST \
  --audit-dir outputs/audits/virusmnist \
  --output outputs/virusmnist/protocol
```

Use `python <script> --help` for the final training commands. The paper reports
Malimg with `train_malimg_resnet50.py` and Virus-MNIST with
`train_virusmnist_spatial_cnn.py`, each under five paired seeds.

## 4. Validate archived results

If the complete archived `results/` directory is placed under a project root,
run:

```bash
export MALEVIS_PROJECT_ROOT=/path/to/project
python validation/audit_all_runs.py
python validation/audit_manuscript_claims.py
python validation/audit_performance_claims.py
```

The validation scripts fail immediately when an expected run, digest, metric,
prediction, or stated numerical result differs. Human-review exports and full
run artifacts are required for these final checks and are separate from this
code-only repository.

## Tests

The protocol tests do not train a network:

```bash
cd malevis_training
python -m unittest test_protocol test_tune_and_compare -v
```

`test_resnet_preprocess_corrected.py` additionally requires TensorFlow.

## Reproducibility scope

The repository reproduces the implemented audit and training protocols. It
does not redistribute third-party datasets or model weights, and it does not
turn an image-level match into evidence of executable identity. The five seeds
measure optimization variation for fixed constructions; they do not represent
five independent dataset samples.
