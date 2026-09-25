#!/usr/bin/env python3
"""Run the 23-class ResNet50 study, then ConvNeXt-Tiny, sequentially.

The launcher waits for every requested ResNet50 condition/seed to complete
successfully before starting ConvNeXt-Tiny. Both stages reuse verified completed
runs and stop immediately if a stage fails.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


CONDITIONS = ("full", "clean_consensus", "random_subsample23")


def run(command, label):
    print(f"\n===== {label} =====", flush=True)
    print(" ".join(shlex.quote(str(part)) for part in command), flush=True)
    try:
        subprocess.run([str(part) for part in command], check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"{label} failed with exit status {error.returncode}. "
            "The useful error is printed immediately above this message. "
            "If it reports an incomplete run, rerun this launcher with "
            "--restart-incomplete; completed runs will still be reused."
        ) from None
    print(f"===== {label} completed successfully =====\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--full-protocol-dir", type=Path, required=True)
    parser.add_argument("--clean-protocol-dir", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path, required=True)
    parser.add_argument("--resnet-output", type=Path, required=True)
    parser.add_argument("--convnext-output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[42, 101, 7, 2024, 99])
    parser.add_argument("--resnet-batch-size", type=int, default=16)
    parser.add_argument("--convnext-batch-size", type=int, default=8)
    parser.add_argument("--epochs-phase1", type=int, default=15)
    parser.add_argument("--epochs-phase2", type=int, default=35)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print both commands without starting training.")
    args = parser.parse_args()

    if min(args.resnet_batch_size, args.convnext_batch_size,
           args.epochs_phase1, args.epochs_phase2) < 1:
        parser.error("Batch sizes and epoch limits must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(
            seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be unique integers between 0 and 2**32-1.")

    script_dir = Path(__file__).resolve().parent
    python = Path(sys.executable).resolve()
    common_seed_args = ["--seeds", *map(str, args.seeds)]
    common_condition_args = ["--conditions", *args.conditions]
    common_epoch_args = [
        "--epochs-phase1", str(args.epochs_phase1),
        "--epochs-phase2", str(args.epochs_phase2),
    ]

    resnet_command = [
        python,
        script_dir / "train_four_condition_study.py",
        "--dataset", args.dataset.resolve(),
        "--full-protocol-dir", args.full_protocol_dir.resolve(),
        "--clean-protocol-dir", args.clean_protocol_dir.resolve(),
        "--hash-manifest", args.hash_manifest.resolve(),
        "--output", args.resnet_output.resolve(),
        *common_condition_args,
        *common_seed_args,
        "--batch-size", str(args.resnet_batch_size),
        *common_epoch_args,
        "--evaluate",
    ]
    if args.restart_incomplete:
        resnet_command.append("--restart-incomplete")

    convnext_command = [
        python,
        script_dir / "train_convnext_tiny_23.py",
        "--dataset", args.dataset.resolve(),
        "--manifest-dir", args.resnet_output.resolve(),
        "--output", args.convnext_output.resolve(),
        *common_condition_args,
        *common_seed_args,
        "--batch-size", str(args.convnext_batch_size),
        *common_epoch_args,
        "--evaluate",
    ]
    if args.restart_incomplete:
        convnext_command.append("--restart-incomplete")

    print(
        "Execution order: ResNet50 completes all requested runs, then "
        "ConvNeXt-Tiny starts. Models are never trained concurrently.",
        flush=True,
    )
    if args.dry_run:
        print("\nResNet50 command:\n" + " ".join(
            shlex.quote(str(part)) for part in resnet_command), flush=True)
        print("\nConvNeXt-Tiny command:\n" + " ".join(
            shlex.quote(str(part)) for part in convnext_command), flush=True)
        return
    run(resnet_command, "STAGE 1/2: ResNet50")
    run(convnext_command, "STAGE 2/2: ConvNeXt-Tiny")
    print("All ResNet50 and ConvNeXt-Tiny runs completed.", flush=True)


if __name__ == "__main__":
    main()
