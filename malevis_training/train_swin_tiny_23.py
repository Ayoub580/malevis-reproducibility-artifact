#!/usr/bin/env python3
"""Train ImageNet-pretrained Swin-Tiny on frozen MaleVis 23-class manifests.

The script reuses the Full-23, Clean-Consensus-23, and Random-Subsample-23
manifests produced by ``train_four_condition_study.py``.  It trains every
requested condition from the same seeded initialization, selects checkpoints
only by development loss, and accesses evaluation labels only after selection.

Interrupted attempts are never overwritten.  Pass ``--restart-incomplete`` to
archive an incomplete run and restart it; completed runs are verified and
reused.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import PIL
from PIL import Image

import train_full_malevis as audit
from tune_and_compare_malevis import digest_rows, freeze_csv, freeze_json


PARTITIONS = ("fitting", "development", "evaluation")
CONDITIONS = ("full", "clean_consensus", "random_subsample23")
DISPLAY = {
    "full": "Full-23",
    "clean_consensus": "Clean-Consensus-23",
    "random_subsample23": "Random-Subsample-23",
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_condition(directory, condition):
    parts = {}
    for partition in PARTITIONS:
        path = Path(directory) / f"{condition}_{partition}.csv"
        if not path.exists():
            raise ValueError(
                f"Missing {path}. Run train_four_condition_study.py "
                "--prepare-only first."
            )
        parts[partition] = read_csv(path)

    classes = sorted({row["class_label"] for row in parts["fitting"]})
    if len(classes) != 23:
        raise ValueError(
            f"{condition} must contain exactly 23 classes; found {len(classes)}."
        )
    ids = []
    for partition, rows in parts.items():
        if sorted({row["class_label"] for row in rows}) != classes:
            raise ValueError(f"{condition}/{partition} lacks at least one class.")
        current = {row["sample_id"] for row in rows}
        if len(current) != len(rows):
            raise ValueError(f"Repeated sample ID in {condition}/{partition}.")
        ids.append(current)
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError(f"A sample occurs in multiple {condition} partitions.")
    return parts, classes


def verify_dataset(root, conditions):
    checked = set()
    for parts in conditions.values():
        for partition in PARTITIONS:
            for row in parts[partition]:
                sample = row["sample_id"]
                if sample in checked:
                    continue
                checked.add(sample)
                path = root / sample
                if not path.is_file():
                    raise ValueError(f"Missing image: {path}")
                if audit.sha(path.read_bytes()) != row["file_sha256"]:
                    raise ValueError(f"Image checksum changed: {path}")


class Images:
    def __init__(self, root, rows, classes, transform):
        self.root = root
        self.rows = rows
        self.transform = transform
        self.indices = {label: index for index, label in enumerate(classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["sample_id"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.indices[row["class_label"]], row["sample_id"]


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(audit.payload(tensor.detach().cpu().numpy()))
    return digest.hexdigest()


def build_model(models, nn, class_count, pretrained=True):
    weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.swin_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, class_count)
    return model, str(weights)


def set_phase(model, phase):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if phase == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        return
    if phase != "finetuning":
        raise ValueError(f"Unknown training phase: {phase}")

    # torchvision Swin-T has eight feature groups.  Groups 0--3 contain the
    # patch projection and the first two stages; groups 4--7 contain the upper
    # two stages and their patch-merging transitions.
    split = len(model.features) // 2
    for module in model.features[split:]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    for module in (model.norm, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def set_training_mode(model, phase):
    if phase == "head":
        model.eval()
        model.head.train()
        return
    model.train()
    split = len(model.features) // 2
    for module in model.features[:split]:
        module.eval()


def make_loaders(args, parts, classes, transform, torch, DataLoader):
    loaders = {}
    for partition in PARTITIONS:
        generator = torch.Generator().manual_seed(args.current_seed)
        loaders[partition] = DataLoader(
            Images(args.dataset, parts[partition], classes, transform),
            batch_size=args.batch_size,
            shuffle=partition == "fitting",
            num_workers=args.workers,
            pin_memory=args.device.type == "cuda",
            persistent_workers=args.workers > 0,
            drop_last=False,
            generator=generator,
        )
    return loaders


def evaluate(model, loader, classes, device, torch, nn, use_amp, collect=False):
    model.eval()
    loss_total, count = 0.0, 0
    records, truth, predictions = [], [], []
    with torch.inference_mode():
        for images, labels, sample_ids in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite evaluation loss.")
            loss_total += float(loss) * len(labels)
            count += len(labels)
            actual = labels.cpu().tolist()
            guessed = logits.argmax(1).cpu().tolist()
            truth.extend(actual)
            predictions.extend(guessed)
            if collect:
                probabilities = logits.float().softmax(1).cpu().numpy()
                for sample, target, guess, scores in zip(
                    sample_ids, actual, guessed, probabilities
                ):
                    records.append({
                        "sample_id": sample,
                        "true_label": classes[target],
                        "predicted_label": classes[guess],
                        **{
                            f"prob_{label}": float(score)
                            for label, score in zip(classes, scores)
                        },
                    })
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader.")
    return loss_total / count, records, truth, predictions


def train_phase(
    model, loaders, classes, args, run, phase, epochs, learning_rate,
    patience, checkpoint_name, history_name, torch, nn,
):
    set_phase(model, phase)
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters in phase {phase}.")
    optimizer = torch.optim.Adam(
        trainable, lr=learning_rate, weight_decay=args.weight_decay
    )
    scheduler = None
    if phase == "finetuning":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=3,
            min_lr=args.minimum_learning_rate,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)
    best_loss, best_accuracy, best_epoch, stale = float("inf"), 0.0, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        set_training_mode(model, phase)
        running, seen, correct = 0.0, 0, 0
        for images, labels, _ in loaders["fitting"]:
            images = images.to(args.device, non_blocking=args.device.type == "cuda")
            labels = labels.to(args.device, non_blocking=args.device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=args.device.type,
                enabled=args.use_amp,
            ):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * len(labels)
            seen += len(labels)
            correct += int((logits.argmax(1) == labels).sum())

        dev_loss, _, actual, guessed = evaluate(
            model, loaders["development"], classes, args.device,
            torch, nn, args.use_amp, collect=False,
        )
        dev_accuracy = float(
            np.mean(np.asarray(actual) == np.asarray(guessed))
        )
        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": running / seen,
            "train_accuracy": correct / seen,
            "development_loss": dev_loss,
            "development_accuracy": dev_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        audit.write_csv(run / history_name, history)
        print(
            f"swin_tiny {phase} epoch={epoch}: "
            f"train_loss={row['train_loss']:.5f} "
            f"train_accuracy={row['train_accuracy']:.5f} "
            f"dev_loss={dev_loss:.5f} dev_accuracy={dev_accuracy:.5f}",
            flush=True,
        )

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_accuracy = dev_accuracy
            best_epoch = epoch
            stale = 0
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "phase": phase,
                "epoch": epoch,
            }, run / checkpoint_name)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(dev_loss)
        if stale >= patience:
            break

    if best_epoch == 0:
        raise RuntimeError(f"No checkpoint was selected in phase {phase}.")
    checkpoint = torch.load(
        run / checkpoint_name, map_location=args.device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "best_epoch": best_epoch,
        "best_development_loss": best_loss,
        "best_development_accuracy": best_accuracy,
        "epochs_completed": len(history),
    }


def run_one(args, experiment, parts, condition, seed, torch, torchvision):
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    run = args.output / "swin_tiny" / condition / f"seed_{seed}"
    config = {
        "experiment_sha256": digest_rows(experiment),
        "condition": condition,
        "seed": seed,
        **{f"{partition}_sha256": digest_rows(parts[partition])
           for partition in PARTITIONS},
    }
    if (run / "metrics.json").exists():
        freeze_json(run / "run.json", config)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for name, digest in metrics["artifacts_sha256"].items():
            if audit.sha((run / name).read_bytes()) != digest:
                raise ValueError(f"Changed completed artifact: {run / name}")
        print(f"Reusing completed {condition}, seed {seed}.", flush=True)
        return metrics

    resume_evaluation = (run / "trained.json").exists()
    if not resume_evaluation and run.exists() and any(run.iterdir()):
        if not args.restart_incomplete:
            raise ValueError(
                f"Incomplete run at {run}; add --restart-incomplete to "
                "archive and restart it."
            )
        audit.archive_attempt(run)
    run.mkdir(parents=True, exist_ok=True)
    freeze_json(run / "run.json", config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    args.current_seed = seed
    transform = transforms.Compose([
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    loaders = make_loaders(
        args, parts, experiment["classes"], transform, torch, DataLoader
    )
    started = time.time()

    if resume_evaluation:
        trained = json.loads((run / "trained.json").read_text(encoding="utf-8"))
        checkpoint_path = run / "selected.pt"
        if audit.sha(checkpoint_path.read_bytes()) != trained["checkpoint_sha256"]:
            raise ValueError("Saved Swin-Tiny checkpoint changed.")
        model, _ = build_model(
            models, nn, len(experiment["classes"]), pretrained=False
        )
        model.to(args.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=args.device, weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model, weights_name = build_model(
            models, nn, len(experiment["classes"]), pretrained=True
        )
        initial_state_sha256 = state_digest(model)
        model.to(args.device)
        with (run / "model_summary.txt").open("w", encoding="utf-8") as handle:
            handle.write(str(model) + "\n")
            handle.write(
                f"parameters={sum(p.numel() for p in model.parameters())}\n"
            )

        print(
            f"Phase 1: frozen Swin-Tiny backbone; Adam learning rate "
            f"{args.head_learning_rate:g}.", flush=True,
        )
        warmup = train_phase(
            model, loaders, experiment["classes"], args, run,
            "head", args.epochs_phase1, args.head_learning_rate,
            4, "warmup_best.pt", "warmup_history.csv", torch, nn,
        )
        print(
            "Phase 2: upper half of Swin-Tiny unfrozen; Adam learning rate "
            f"{args.backbone_learning_rate:g}.", flush=True,
        )
        finetuning = train_phase(
            model, loaders, experiment["classes"], args, run,
            "finetuning", args.epochs_phase2, args.backbone_learning_rate,
            6, "selected.pt", "finetuning_history.csv", torch, nn,
        )
        audit.write_json(run / "finetuning_layers.json", [
            {"name": name, "trainable": parameter.requires_grad}
            for name, parameter in model.named_parameters()
        ])
        trained = {
            "selected_phase": "finetuning",
            "selected_phase_epoch": finetuning["best_epoch"],
            "development_loss": finetuning["best_development_loss"],
            "development_accuracy": finetuning["best_development_accuracy"],
            "warmup_epochs_completed": warmup["epochs_completed"],
            "finetuning_epochs_completed": finetuning["epochs_completed"],
            "initial_state_sha256": initial_state_sha256,
            "pretrained_weights": weights_name,
            "training_seconds": time.time() - started,
        }
        selected = torch.load(
            run / "selected.pt", map_location="cpu", weights_only=True
        )
        # Re-save the selected checkpoint with provenance needed for inspection.
        torch.save({
            **selected,
            "model": "swin_tiny",
            "condition": condition,
            "seed": seed,
            "experiment_sha256": config["experiment_sha256"],
        }, run / "selected.pt")
        trained["checkpoint_sha256"] = audit.sha(
            (run / "selected.pt").read_bytes()
        )
        audit.write_json(run / "trained.json", trained)

    dev_loss, dev_records, dev_actual, dev_predicted = evaluate(
        model, loaders["development"], experiment["classes"], args.device,
        torch, nn, args.use_amp, collect=True,
    )
    dev_scores, dev_per_class, _ = audit.classification_metrics(
        dev_actual, dev_predicted, experiment["classes"]
    )
    dev_scores["loss"] = dev_loss
    audit.write_csv(run / "development_predictions.csv", dev_records)
    audit.write_csv(run / "development_per_class.csv", dev_per_class)
    audit.write_json(run / "development_metrics.json", dev_scores)
    print(
        "Selected checkpoint development metrics: " + json.dumps(dev_scores),
        flush=True,
    )
    if not args.evaluate:
        print(
            "Development-only run complete; evaluation was not scored.",
            flush=True,
        )
        del model, loaders
        if args.device.type == "cuda":
            torch.cuda.empty_cache()
        return None

    evaluation_loss, records, actual, predicted = evaluate(
        model, loaders["evaluation"], experiment["classes"], args.device,
        torch, nn, args.use_amp, collect=True,
    )
    metrics, per_class, confusion = audit.classification_metrics(
        actual, predicted, experiment["classes"]
    )
    audit.write_csv(run / "predictions.csv", records)
    audit.write_csv(run / "per_class.csv", per_class)
    audit.write_csv(run / "confusion_matrix.csv", [
        {
            "true_class": label,
            **dict(zip(experiment["classes"], map(int, row))),
        }
        for label, row in zip(experiment["classes"], confusion)
    ])
    metrics.update({
        "model": "swin_tiny",
        "condition": condition,
        "seed": seed,
        "fitting_images": len(parts["fitting"]),
        "development_images": len(parts["development"]),
        "evaluation_images": len(parts["evaluation"]),
        "evaluation_loss": evaluation_loss,
        **trained,
    })
    metrics["artifacts_sha256"] = {
        name: audit.sha((run / name).read_bytes())
        for name in (
            "run.json", "trained.json", "selected.pt", "predictions.csv",
            "per_class.csv", "confusion_matrix.csv",
        )
    }
    audit.write_json(run / "metrics.json", metrics)
    print(json.dumps({
        key: value for key, value in metrics.items()
        if key != "artifacts_sha256"
    }, indent=2), flush=True)
    del model, loaders
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def write_summaries(output):
    completed = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (output / "swin_tiny").glob("*/seed_*/metrics.json")
        )
    ]
    if not completed:
        return
    for seed in sorted({row["seed"] for row in completed}):
        rows = [row for row in completed if row["seed"] == seed]
        if len(rows) > 1 and len({
            row["initial_state_sha256"] for row in rows
        }) != 1:
            raise ValueError(f"Swin-Tiny initialization differs for seed {seed}.")
    audit.write_csv(output / "results.csv", [
        {key: value for key, value in row.items()
         if key != "artifacts_sha256"}
        for row in completed
    ])

    contrasts = []
    for seed in sorted({row["seed"] for row in completed}):
        by_condition = {
            row["condition"]: row for row in completed if row["seed"] == seed
        }
        if not set(CONDITIONS) <= set(by_condition):
            continue
        row = {"seed": seed}
        for metric in ("accuracy", "macro_f1", "macro_recall"):
            row[f"full_minus_clean_{metric}_pp"] = 100 * (
                by_condition["full"][metric]
                - by_condition["clean_consensus"][metric]
            )
            row[f"full_minus_subsample_{metric}_pp"] = 100 * (
                by_condition["full"][metric]
                - by_condition["random_subsample23"][metric]
            )
            row[f"subsample_minus_clean_{metric}_pp"] = 100 * (
                by_condition["random_subsample23"][metric]
                - by_condition["clean_consensus"][metric]
            )
        contrasts.append(row)
    if contrasts:
        audit.write_csv(output / "seed_contrasts.csv", contrasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 101, 7, 2024, 99]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=35)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    args.output = args.output.resolve()

    if min(
        args.batch_size, args.epochs_phase1, args.epochs_phase2,
        args.head_learning_rate, args.backbone_learning_rate,
        args.minimum_learning_rate,
    ) <= 0:
        parser.error("Batch size, epoch limits, and learning rates must be positive.")
    if args.workers < 0 or args.weight_decay < 0:
        parser.error("Workers and weight decay cannot be negative.")
    if len(set(args.seeds)) != len(args.seeds) or any(
        seed < 0 or seed >= 2**32 for seed in args.seeds
    ):
        parser.error("Seeds must be unique integers between 0 and 2**32-1.")

    conditions, classes = {}, None
    for condition in args.conditions:
        parts, current_classes = load_condition(args.manifest_dir, condition)
        if classes is None:
            classes = current_classes
        elif classes != current_classes:
            raise ValueError("Requested conditions use different class mappings.")
        conditions[condition] = parts
    verify_dataset(args.dataset, conditions)
    args.output.mkdir(parents=True, exist_ok=True)
    for condition, parts in conditions.items():
        for partition in PARTITIONS:
            freeze_csv(
                args.output / f"{condition}_{partition}.csv", parts[partition]
            )

    settings = {
        "model": "Swin-Tiny",
        "classes": classes,
        "class_count": len(classes),
        "input": "224x224 RGB direct bicubic resize",
        "normalization": {"mean": MEAN, "std": STD},
        "pretrained": "torchvision Swin_T_Weights.IMAGENET1K_V1",
        "classification_head": "Swin-T linear head with 23 outputs",
        "phase1": "frozen backbone; train classification head",
        "phase2": "unfreeze upper half of Swin feature groups plus norm/head",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "head_learning_rate": args.head_learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "optimizer": "Adam",
        "weight_decay": args.weight_decay,
        "augmentation": None,
        "checkpoint_selection": "minimum phase-2 development cross-entropy",
        "source_sha256": audit.sha(Path(__file__).read_bytes()),
    }
    freeze_json(args.output / "experiment_settings.json", settings)
    print("Verified Swin-Tiny 23-class manifests:", {
        name: {partition: len(parts[partition]) for partition in PARTITIONS}
        for name, parts in conditions.items()
    }, flush=True)
    if args.prepare_only:
        return

    import torch
    import torchvision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "PyTorch cannot see a CUDA GPU. Check the CUDA environment or "
            "explicitly pass --allow-cpu."
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.device = device
    args.use_amp = device.type == "cuda" and not args.no_amp
    experiment = {
        **settings,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name() if device.type == "cuda"
            else platform.processor()
        ),
        "automatic_mixed_precision": args.use_amp,
    }
    freeze_json(args.output / "experiment.json", experiment)

    for seed in args.seeds:
        for condition in args.conditions:
            print(
                f"Swin-Tiny: {DISPLAY[condition]}, seed {seed}", flush=True
            )
            run_one(
                args, experiment, conditions[condition], condition, seed,
                torch, torchvision,
            )
            write_summaries(args.output)
    print("All requested Swin-Tiny runs completed.", flush=True)


if __name__ == "__main__":
    main()
