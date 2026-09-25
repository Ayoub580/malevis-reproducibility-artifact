#!/usr/bin/env python3
"""Tune on development only, freeze a recipe, compare Full/Clean-Exact/Random.

Uses the existing train_full_malevis.py beside this file for verified inventory,
splits and metrics. Does not promise a target accuracy or modify dataset files.
"""
import argparse
import csv
import json
import math
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import PIL
import train_full_malevis as base

# Small, fixed search budget. Every candidate runs; no evaluation-based stopping.
RECIPES = {
    'conservative': dict(backbone_lr=1e-5, head_lr=1e-4, freeze_bn=False),
    'faster': dict(backbone_lr=1e-4, head_lr=1e-3, freeze_bn=False),
    'frozen_bn': dict(backbone_lr=3e-5, head_lr=3e-4, freeze_bn=True),
}
CONDITIONS = ('full', 'clean_exact', 'random_control')


def digest_rows(rows):
    return base.sha(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode())


def freeze_json(path, value):
    """Never silently mix experiments or overwrite altered provenance."""
    value = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen settings differ: {path}. Use a new output directory.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        base.write_json(path, value)


def freeze_csv(path, rows):
    if path.exists():
        with path.open() as f:
            actual = list(csv.DictReader(f))
        expected = [{k: str(v) for k, v in r.items()} for r in rows]
        if actual != expected:
            raise ValueError(f'Frozen manifest differs: {path}')
    else:
        base.write_csv(path, rows)


def conditions(rows, removal_seed):
    """Remove ALL fitting counterparts of fixed evaluation native RGB values."""
    parts = {p: sorted((r for r in rows if r['partition'] == p), key=lambda r: r['sample_id'])
             for p in ('fitting', 'development', 'evaluation')}
    evaluation_keys = {r['rgb_sha256'] for r in parts['evaluation']}
    removed = [r for r in parts['fitting'] if r['rgb_sha256'] in evaluation_keys]
    removed_ids = {r['sample_id'] for r in removed}
    clean = [r for r in parts['fitting'] if r['sample_id'] not in removed_ids]
    remove_counts = Counter(r['class_label'] for r in removed)
    rng = random.Random(removal_seed)
    random_ids = set()
    for cls in sorted({r['class_label'] for r in parts['fitting']}):
        eligible = [r['sample_id'] for r in parts['fitting'] if r['class_label'] == cls]
        if remove_counts[cls] >= len(eligible):
            raise ValueError(f'Cleaning exhausts fitting class {cls}; stop and revise scope.')
        random_ids.update(rng.sample(eligible, remove_counts[cls]))
    control = [r for r in parts['fitting'] if r['sample_id'] not in random_ids]
    fitting = dict(full=parts['fitting'], clean_exact=clean, random_control=control)
    if {r['rgb_sha256'] for r in clean} & evaluation_keys:
        raise AssertionError('Clean-Exact still overlaps evaluation.')
    if Counter(r['class_label'] for r in clean) != Counter(r['class_label'] for r in control):
        raise AssertionError('Control class counts differ.')
    table = []
    for cls in sorted({r['class_label'] for r in rows}):
        ev = [r for r in parts['evaluation'] if r['class_label'] == cls]
        line = dict(class_label=cls, removed_exact=remove_counts[cls],
                    development=len([r for r in parts['development'] if r['class_label'] == cls]),
                    evaluation=len(ev))
        for name, subset in fitting.items():
            line[name + '_fitting'] = sum(r['class_label'] == cls for r in subset)
            for key, suffix in [('rgb_sha256', 'native'), ('input_sha256', 'input')]:
                keys = {r[key] for r in subset}
                line[name + '_evaluation_' + suffix + '_matches'] = sum(r[key] in keys for r in ev)
        table.append(line)
    return fitting, parts['development'], parts['evaluation'], removed, table


def select_recipe(results):
    """Equal weight to each tuning seed; never consult evaluation metrics."""
    scores = []
    for recipe in RECIPES:
        subset = [r for r in results if r['recipe_name'] == recipe]
        if not subset or any(not math.isfinite(r['development_loss']) for r in subset):
            raise ValueError('Incomplete or nonfinite tuning results.')
        scores.append(dict(recipe_name=recipe,
                           mean_development_loss=float(np.mean([r['development_loss'] for r in subset])),
                           mean_development_accuracy=float(np.mean([r['development_accuracy'] for r in subset]))))
    winner = min(scores, key=lambda r: (r['mean_development_loss'], r['recipe_name']))
    return winner['recipe_name'], scores


def build_model(name, class_count):
    from torch import nn
    from torchvision import models
    if name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model, model.fc
    if name == 'convnext_tiny':
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        return model, model.classifier
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, class_count)
    return model, model.heads


def transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(), transforms.Normalize(base.MEAN, base.STD)])


def loader(args, rows, classes, seed, shuffle=False):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(base.Images(args.dataset, rows, classes, transform()),
                      batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=args.device == 'cuda', drop_last=False,
                      generator=torch.Generator().manual_seed(seed))


def score(model, batches, classes, device, amp, collect=False):
    import torch
    from torch.nn.functional import cross_entropy
    model.eval()
    total, n, truth, predicted, records = 0., 0, [], [], []
    with torch.inference_mode():
        for x, y, ids in batches:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, enabled=amp):
                logits = model(x)
                loss = cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss.')
            total += loss.item() * len(y)
            n += len(y)
            probs = logits.float().softmax(1).cpu().numpy()
            labels, guesses = y.cpu().tolist(), probs.argmax(1).tolist()
            truth.extend(labels)
            predicted.extend(guesses)
            if collect:
                for sid, actual, guess, ps in zip(ids, labels, guesses, probs):
                    records.append(dict(sample_id=sid, true_label=classes[actual], predicted_label=classes[guess],
                                        **{f'prob_{c}': float(p) for c, p in zip(classes, ps)}))
    metrics, per_class, confusion = base.classification_metrics(truth, predicted, classes)
    return dict(loss=total/n, **metrics), per_class, confusion, records


def verify_trained(run):
    result = json.loads((run / 'trained.json').read_text())
    if base.sha((run / 'best.pt').read_bytes()) != result['checkpoint_sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {run}')
    if base.sha((run / 'run.json').read_bytes()) != result['run_config_sha256']:
        raise ValueError(f'Run configuration checksum mismatch: {run}')
    return result


def fit(args, experiment, fitting, development, recipe_name, seed, run):
    """No evaluation rows accepted here: tuning cannot score evaluation images."""
    import torch
    from torch import nn
    if any(r['partition'] != 'fitting' for r in fitting):
        raise ValueError('Fitting loader contains non-fitting images.')
    if any(r['partition'] != 'development' for r in development):
        raise ValueError('Development loader contains non-development images.')
    recipe = RECIPES[recipe_name]
    config = dict(experiment_sha256=digest_rows(experiment), fitting_sha256=digest_rows(fitting),
                  development_sha256=digest_rows(development), recipe_name=recipe_name,
                  recipe=recipe, model=args.model, seed=seed)
    if (run / 'trained.json').exists():
        freeze_json(run / 'run.json', config)
        return verify_trained(run)
    if run.exists() and any(run.iterdir()):
        if not args.restart_incomplete:
            raise ValueError(f'Incomplete run: {run}. Add --restart-incomplete to archive and restart.')
        base.archive_attempt(run)
    run.mkdir(parents=True, exist_ok=True)
    freeze_json(run / 'run.json', config)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model, head = build_model(args.model, len(experiment['classes']))
    model.to(args.device)
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    # Saved for checking paired initialization across training conditions.
    initial_sha = base.sha(b''.join(t.detach().cpu().numpy().tobytes()
                                   for t in model.state_dict().values()))
    train_loader = loader(args, fitting, experiment['classes'], seed, True)
    dev_loader = loader(args, development, experiment['classes'], seed)
    optimizer = torch.optim.AdamW([
        dict(params=backbone, lr=recipe['backbone_lr']),
        dict(params=list(head.parameters()), lr=recipe['head_lr'])], weight_decay=args.weight_decay)
    # Fine-tuning gets its own schedule after the constant-LR head phase.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.head_epochs)
    amp = args.device == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best, best_epoch, stale, history = float('inf'), 0, 0, []
    started = time.time()
    print(f'Training {run}: {len(fitting)} fitting images; recipe={recipe_name}.', flush=True)
    for epoch in range(1, args.epochs + 1):
        warmup = epoch <= args.head_epochs
        for p in backbone:
            p.requires_grad_(not warmup)
        if warmup:
            model.eval()
            head.train()
        else:
            model.train()
            if recipe['freeze_bn']:
                # Freeze running moments, but keep affine parameters trainable.
                for layer in model.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
        if epoch == args.head_epochs + 1:
            stale = 0
        total, seen, correct = 0., 0, 0
        for x, y, _ in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            seen += len(y)
            correct += int((logits.argmax(1) == y).sum().item())
        dev, _, _, _ = score(model, dev_loader, experiment['classes'], args.device, amp)
        history.append(dict(epoch=epoch, phase='head' if warmup else 'finetune',
                            train_loss=total/seen, train_accuracy=correct/seen,
                            **{'development_' + k: v for k, v in dev.items()},
                            backbone_lr=optimizer.param_groups[0]['lr'],
                            head_lr=optimizer.param_groups[1]['lr'], optimizer_steps=len(train_loader)))
        base.write_csv(run / 'history.csv', history)
        print(f'{run.name} {recipe_name} epoch={epoch}/{args.epochs} '
              f'train_loss={total/seen:.5f} train_acc={correct/seen:.2%} dev_acc={dev["accuracy"]:.2%} '
              f'dev_loss={dev["loss"]:.5f}', flush=True)
        if dev['loss'] < best:
            best, best_epoch, stale = dev['loss'], epoch, 0
            torch.save(dict(state_dict=model.state_dict(), classes=experiment['classes'],
                            model=args.model, seed=seed, epoch=epoch,
                            fitting_sha256=config['fitting_sha256']), run / 'best.pt')
        elif not warmup:
            stale += 1
        if not warmup:
            scheduler.step()
        if not warmup and stale >= args.patience:
            break
    selected = history[best_epoch-1]
    result = dict(recipe_name=recipe_name, seed=seed, selected_epoch=best_epoch,
                  development_loss=best, development_accuracy=selected['development_accuracy'],
                  development_macro_f1=selected['development_macro_f1'], epochs_completed=len(history),
                  fitting_images=len(fitting), seconds=time.time()-started,
                  initial_state_sha256=initial_sha, checkpoint_sha256=base.sha((run/'best.pt').read_bytes()),
                  run_config_sha256=base.sha((run/'run.json').read_bytes()))
    base.write_json(run / 'trained.json', result)
    del model, head, backbone, optimizer, scheduler, scaler, train_loader, dev_loader
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return result


def evaluate_run(args, experiment, evaluation, run, destination, condition, seed, selection_sha):
    import torch
    trained = verify_trained(run)
    destination.mkdir(parents=True, exist_ok=True)
    freeze_json(destination / 'evaluation_config.json', dict(
        checkpoint_sha256=trained['checkpoint_sha256'], evaluation_sha256=digest_rows(evaluation),
        selection_sha256=selection_sha, condition=condition, seed=seed))
    if (destination / 'metrics.json').exists():
        result = json.loads((destination / 'metrics.json').read_text())
        for name, digest in result['artifacts_sha256'].items():
            if base.sha((destination / name).read_bytes()) != digest:
                raise ValueError(f'Evaluation artifact changed: {destination / name}')
        return result
    classes = experiment['classes']
    model, _ = build_model(args.model, len(classes))
    checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    scores, per_class, cm, predictions = score(model, loader(args, evaluation, classes, seed),
        classes, args.device, args.device == 'cuda' and not args.no_amp, collect=True)
    base.write_csv(destination / 'predictions.csv', predictions)
    base.write_csv(destination / 'per_class.csv', per_class)
    base.write_csv(destination / 'confusion_matrix.csv',
        [dict(true_class=c, **dict(zip(classes, map(int, row)))) for c, row in zip(classes, cm)])
    scores.update(condition=condition, seed=seed, model=args.model, evaluation_images=len(evaluation),
                  selected_epoch=trained['selected_epoch'], recipe_name=trained['recipe_name'],
                  fitting_images=trained['fitting_images'], initial_state_sha256=trained['initial_state_sha256'],
                  artifacts_sha256={n: base.sha((destination/n).read_bytes()) for n in
                      ('predictions.csv', 'per_class.csv', 'confusion_matrix.csv', 'evaluation_config.json')})
    base.write_json(destination / 'metrics.json', scores)
    print(json.dumps({k: v for k, v in scores.items() if k != 'artifacts_sha256'}, indent=2), flush=True)
    del model, checkpoint
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', 'tune', 'compare'), required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--protocol-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=base.ARCHITECTURES, default='resnet50')
    p.add_argument('--tune-seeds', nargs='+', type=int, default=[42])
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--removal-seed', type=int, default=20260918)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--head-epochs', type=int, default=5)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--restart-incomplete', action='store_true')
    args = p.parse_args()
    if not 0 <= args.head_epochs < args.epochs or min(args.batch_size, args.patience) < 1:
        p.error('Require epochs > head-epochs >= 0, positive batch-size and patience.')
    if args.workers < 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error('Invalid workers or weight decay.')
    if any(s < 0 or s >= 2**32 for s in args.seeds + args.tune_seeds):
        p.error('Seeds must be between 0 and 2**32-1.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.tune_seeds)) != len(args.tune_seeds):
        p.error('Repeated seeds.')
    args.dataset, args.protocol_dir, args.output = (
        v.resolve() for v in (args.dataset, args.protocol_dir, args.output))
    if args.output == args.dataset or args.dataset in args.output.parents:
        p.error('Output must be outside dataset.')
    if args.output == args.protocol_dir or args.protocol_dir in args.output.parents:
        p.error('Use a separate experiment output outside the frozen protocol directory.')
    if not (args.protocol_dir/'protocol.json').exists():
        p.error('Reuse the existing Full protocol directory; do not silently create a new split.')
    meta = json.loads((args.protocol_dir/'protocol.json').read_text())
    settings = meta['split_settings']
    rows, meta = base.prepare(args.dataset, args.protocol_dir,
                              settings['development_fraction'], settings['split_seed'])
    fitting, development, evaluation, removed, table = conditions(rows, args.removal_seed)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(parent_manifest_sha256=meta['manifest_sha256'], removal_seed=args.removal_seed,
                    condition_fitting_sha256={k: digest_rows(v) for k, v in fitting.items()},
                    development_sha256=digest_rows(development), evaluation_sha256=digest_rows(evaluation),
                    definition='Remove fitting members matching any evaluation native RGB identity; direct-verified upstream.')
    freeze_json(args.output / 'conditions.json', protocol)
    for name, subset in fitting.items():
        freeze_csv(args.output / (name + '_fitting.csv'), subset)
    freeze_csv(args.output / 'development.csv', development)
    freeze_csv(args.output / 'evaluation.csv', evaluation)
    freeze_csv(args.output / 'class_counts.csv', table)
    if removed:
        freeze_csv(args.output / 'removed_exact.csv', removed)
    print('Fitting sizes:', {k: len(v) for k, v in fitting.items()}, flush=True)
    print(f'Fixed development={len(development)}, evaluation={len(evaluation)}; removed={len(removed)}.', flush=True)
    if args.stage == 'prepare':
        return
    import torch
    import torchvision
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cpu' and not args.allow_cpu:
        raise RuntimeError('CUDA unavailable; check the conda/GPU environment.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    experiment = dict(classes=meta['classes'], protocol=protocol, model=args.model,
        epochs=args.epochs, head_epochs=args.head_epochs, patience=args.patience,
        batch_size=args.batch_size, workers=args.workers, weight_decay=args.weight_decay,
        tune_seeds=args.tune_seeds, recipes=RECIPES, no_amp=args.no_amp, device=args.device,
        python=platform.python_version(), numpy=np.__version__, Pillow=PIL.__version__,
        torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device_name=torch.cuda.get_device_name() if args.device == 'cuda' else platform.processor(),
        script_sha256=base.sha(Path(__file__).read_bytes()), helper_sha256=base.sha(Path(base.__file__).read_bytes()),
        preprocessing='RGB, direct Pillow bicubic224, ToTensor, ImageNet normalization; no augmentation',
        optimizer='AdamW, unweighted cross-entropy, gradient norm clipped to 1',
        schedule='constant learning rates during head phase; cosine during fine-tuning',
        selection='minimum mean selected development cross-entropy across tuning seeds; no evaluation scoring')
    freeze_json(args.output / 'experiment.json', experiment)
    tuning_dir = args.output / 'tuning'
    if args.stage == 'tune':
        results = []
        for recipe in RECIPES:
            for seed in args.tune_seeds:
                results.append(fit(args, experiment, fitting['full'], development, recipe, seed,
                                   tuning_dir / recipe / f'seed_{seed}'))
        winner, scores = select_recipe(results)
        base.write_csv(args.output / 'tuning_summary.csv', scores)
        freeze_json(args.output / 'selected_recipe.json', dict(recipe_name=winner, recipe=RECIPES[winner],
            experiment_sha256=digest_rows(experiment), scores=scores,
            selected_by='minimum mean development loss; evaluation was not scored',
            tuning_results_sha256=digest_rows(results)))
        print(f'Frozen recipe: {winner}. Next run --stage compare with the same settings.', flush=True)
        return
    selection_path = args.output / 'selected_recipe.json'
    if not selection_path.exists():
        raise ValueError('First complete --stage tune.')
    selection = json.loads(selection_path.read_text())
    if selection['experiment_sha256'] != digest_rows(experiment):
        raise ValueError('Selection was made using different settings.')
    tuning_results = [verify_trained(tuning_dir / recipe / f'seed_{seed}')
                      for recipe in RECIPES for seed in args.tune_seeds]
    winner, scores = select_recipe(tuning_results)
    if (selection['recipe_name'] != winner or selection['scores'] != scores or
            selection['tuning_results_sha256'] != digest_rows(tuning_results) or
            selection['recipe'] != RECIPES[winner]):
        raise ValueError('Selection no longer matches the completed development search.')
    deltas = []
    for seed in args.seeds:
        paired = {}
        for condition in CONDITIONS:
            run = args.output / 'training' / condition / f'seed_{seed}'
            if condition == 'full' and seed in args.tune_seeds:
                run = tuning_dir / winner / f'seed_{seed}'
            fit(args, experiment, fitting[condition], development, winner, seed, run)
            paired[condition] = evaluate_run(args, experiment, evaluation, run,
                args.output / 'evaluation' / condition / f'seed_{seed}', condition, seed,
                base.sha(selection_path.read_bytes()))
        if len({r['initial_state_sha256'] for r in paired.values()}) != 1:
            raise ValueError('Paired runs did not start from identical model states.')
        line = dict(seed=seed)
        for metric in ('accuracy', 'macro_f1', 'macro_recall'):
            line['full_minus_clean_' + metric + '_pp'] = 100*(paired['full'][metric]-paired['clean_exact'][metric])
            line['full_minus_random_' + metric + '_pp'] = 100*(paired['full'][metric]-paired['random_control'][metric])
            line['random_minus_clean_' + metric + '_pp'] = 100*(paired['random_control'][metric]-paired['clean_exact'][metric])
        deltas.append(line)
    # Rebuild reports from all completed paired seeds, including earlier invocations.
    complete = []
    for path in sorted((args.output/'evaluation'/'full').glob('seed_*/metrics.json')):
        seed = int(path.parent.name.removeprefix('seed_'))
        paths = [args.output/'evaluation'/c/f'seed_{seed}'/'metrics.json' for c in CONDITIONS]
        if all(p.exists() for p in paths):
            complete.extend(json.loads(p.read_text()) for p in paths)
    base.write_csv(args.output / 'comparison.csv',
        [{k: v for k, v in r.items() if k != 'artifacts_sha256'} for r in complete])
    # The per-invocation contrast file names its seeds; no unseen or missing runs are pooled.
    base.write_csv(args.output / ('paired_differences_' + '_'.join(map(str, args.seeds)) + '.csv'), deltas)
    print('Comparison saved. Differences are percentage points; positive drops are not assumed.', flush=True)


if __name__ == '__main__':
    main()
