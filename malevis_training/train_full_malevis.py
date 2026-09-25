#!/usr/bin/env python3
"""Full MaleVis baseline: ResNet50, ConvNeXt-Tiny, ViT-B/16.

Python 3.10+, numpy, Pillow; training additionally needs matching torch>=2.3
and torchvision installations. No dataset files are changed.
--prepare-only freezes manifests without importing PyTorch or training.
All original val images are evaluation data. Full means no deduplication of
the fitting pool; a common development subset is reserved from original train.
Development is disjoint from fitting/evaluation under native RGB and the
specified 224x224 bicubic input identity. Original fitting/evaluation overlap
is deliberately preserved in this baseline.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import PIL
from PIL import Image

ARCHITECTURES = ('resnet50', 'convnext_tiny', 'vit_b_16')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_attempt(path):
    """Move an interrupted attempt aside without deleting its contents."""
    archive = path.parent / '_interrupted'
    archive.mkdir(exist_ok=True)
    target = archive / f'{path.name}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:8]}'
    path.rename(target)
    print(f'Preserved previous attempt at {target}', flush=True)
    return target


def prepare_output(output, config, restart_incomplete=False):
    """Keep configuration checks while allowing safe recovery from setup failure."""
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        changed = {k for k in previous.keys() | config.keys() if previous.get(k) != config.get(k)}
        if changed:
            # A script repair can change its digest before any training starts.
            # Never silently mix different settings or completed script revisions.
            completed = list(output.glob('*/seed_*/metrics.json'))
            has_artifacts = any(any(run.iterdir()) for run in output.glob('*/seed_*') if run.is_dir())
            if changed == {'script_sha256'} and not completed and (restart_incomplete or not has_artifacts):
                archive_attempt(output)
                output.mkdir(parents=True)
            else:
                raise ValueError('Output configuration differs (' + ', '.join(sorted(changed)) +
                                 '). Use a fresh output directory; for a script-only change with no '
                                 'completed runs, --restart-incomplete preserves and restarts the old attempt.')
    write_json(config_path, config)
    return config_path


def prepare_run(run, restart_incomplete=False):
    if (run / 'metrics.json').exists():
        return False
    if run.exists() and any(run.iterdir()):
        if not restart_incomplete:
            raise ValueError(f'Incomplete run at {run}. Use --restart-incomplete to archive it '
                             'and restart from epoch 1, or use a fresh output directory.')
        archive_attempt(run)
    # Empty folders left by setup/download failures are safe to reuse.
    run.mkdir(parents=True, exist_ok=True)
    return True


def payload(array):
    header = json.dumps({'shape': list(array.shape), 'dtype': array.dtype.str},
                        sort_keys=True, separators=(',', ':')).encode('ascii')
    return header + b'\n' + array.tobytes(order='C')


def representations(path):
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        if im.mode != 'RGB' or im.size != (300, 300):
            raise ValueError(f'Expected native MaleVis RGB 300x300: {path}')
        if getattr(im, 'n_frames', 1) != 1 or im.getexif().get(274, 1) != 1:
            raise ValueError(f'Unsupported orientation or frame count: {path}')
        rgb = im.convert('RGB')
        native = payload(np.asarray(rgb, dtype=np.uint8))
        resized = rgb.resize((224, 224), Image.Resampling.BICUBIC)
        # Matches the audit's serialized float32 input; later normalization
        # is the same invertible per-channel affine transform for all samples.
        inp = payload(np.asarray(resized, dtype='<f4') / np.float32(255.0))
    return {'file_sha256': raw, 'rgb_sha256': native, 'input_sha256': inp}


def scan(root):
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    rows, representatives = [], {k: {} for k in ('file_sha256', 'rgb_sha256', 'input_sha256')}
    for p in paths:
        if p.suffix.lower() != '.png':
            continue
        rel = p.relative_to(root)
        if len(rel.parts) != 3 or rel.parts[0] not in ('train', 'val'):
            raise ValueError(f'Expected train|val/class/image.png: {rel}')
        row = {'sample_id': rel.as_posix(), 'original_split': rel.parts[0],
               'class_label': rel.parts[1]}
        for key, value in representations(p).items():
            digest = sha(value)
            previous = representatives[key].get(digest)
            if previous is not None:
                if representations(previous)[key] != value:
                    raise ValueError(f'Digest group failed direct verification: {p}')
            else:
                representatives[key][digest] = p
            row[key] = digest
        rows.append(row)
        if len(rows) % 1000 == 0:
            print(f'Inventoried {len(rows)} images', flush=True)
    counts = Counter(r['original_split'] for r in rows)
    if counts != {'train': 9100, 'val': 5126} or len({r['class_label'] for r in rows}) != 26:
        raise ValueError(f'This script expects the original 26-class MaleVis snapshot: {counts}')
    return rows


def allocate_split(rows, fraction, seed):
    """Reserve approximately fraction/class by indivisible input-identity groups.

    Sample shuffled eligible groups until the requested image count is reached.
    Achieved counts may slightly exceed targets. Fail on infeasible classes.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r['input_sha256']].append(r)
    for group in groups.values():
        if len({r['class_label'] for r in group}) != 1:
            raise ValueError('Mixed-label input identity group: resolve explicitly before splitting.')
    chosen = set()
    rng = random.Random(seed)
    for cls in sorted({r['class_label'] for r in rows}):
        train = [r for r in rows if r['original_split'] == 'train' and r['class_label'] == cls]
        target = max(1, round(fraction * len(train)))
        eligible = sorted(k for k, g in groups.items()
                          if g[0]['class_label'] == cls and all(r['original_split'] == 'train' for r in g))
        if sum(len(groups[k]) for k in eligible) < target:
            raise ValueError(f'Insufficient evaluation-disjoint development images in {cls}; no fallback.')
        rng.shuffle(eligible)
        count = 0
        for k in eligible:
            chosen.add(k)
            count += len(groups[k])
            if count >= target:
                break
        if count >= len(train):
            raise ValueError(f'Development selection exhausts training class {cls}.')
    result = []
    for r in rows:
        part = ('evaluation' if r['original_split'] == 'val' else
                'development' if r['input_sha256'] in chosen else 'fitting')
        result.append({**r, 'partition': part})
    validate_split(result)
    return result


def validate_split(rows):
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated sample IDs in manifest.')
    parts = {s: [r for r in rows if r['partition'] == s]
             for s in ('fitting', 'development', 'evaluation')}
    if sum(map(len, parts.values())) != len(rows):
        raise ValueError('Unknown partition.')
    classes = {r['class_label'] for r in rows}
    for part in parts.values():
        if {r['class_label'] for r in part} != classes:
            raise ValueError('A partition lacks a class.')
    for r in rows:
        if (r['original_split'] == 'val') != (r['partition'] == 'evaluation'):
            raise ValueError('The original evaluation population changed.')
    for key in ('rgb_sha256', 'input_sha256'):
        dev = {r[key] for r in parts['development']}
        other = {r[key] for r in parts['fitting'] + parts['evaluation']}
        if dev & other:
            raise ValueError('Development identity overlap.')


def prepare(root, directory, fraction, seed):
    identity = {'development_fraction': fraction, 'split_seed': seed,
                'grouping': 'verified Pillow RGB bicubic224 float32/255 identity',
                'Pillow': PIL.__version__, 'numpy': np.__version__}
    if directory.exists():
        meta = json.loads((directory / 'protocol.json').read_text())
        if meta['split_settings'] != identity:
            raise ValueError('Existing protocol settings differ; use the original settings/environment.')
        manifest = directory / 'manifest.csv'
        if sha(manifest.read_bytes()) != meta['manifest_sha256']:
            raise ValueError('Manifest checksum mismatch.')
        with manifest.open() as f:
            rows = list(csv.DictReader(f))
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*.png')}
        if actual != {r['sample_id'] for r in rows}:
            raise ValueError('Dataset file membership changed.')
        for r in rows:
            rel = Path(r['sample_id'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe manifest path.')
            if sha((root / rel).read_bytes()) != r['file_sha256']:
                raise ValueError(f'Dataset changed: {rel}')
        validate_split(rows)
        print('Verified and reused frozen protocol.', flush=True)
        return rows, meta
    rows = allocate_split(scan(root), fraction, seed)
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / 'manifest.csv', rows)
    classes = sorted({r['class_label'] for r in rows})
    counts = []
    fitting = [r for r in rows if r['partition'] == 'fitting']
    fit_native = {r['rgb_sha256'] for r in fitting}
    fit_input = {r['input_sha256'] for r in fitting}
    for cls in classes:
        line = {'class_label': cls}
        for part in ('fitting', 'development', 'evaluation'):
            subset = [r for r in rows if r['class_label'] == cls and r['partition'] == part]
            line[part + '_images'] = len(subset)
            line[part + '_native_distinct'] = len({r['rgb_sha256'] for r in subset})
            line[part + '_input_distinct'] = len({r['input_sha256'] for r in subset})
        ev = [r for r in rows if r['class_label'] == cls and r['partition'] == 'evaluation']
        line['evaluation_native_matches_fitting'] = sum(r['rgb_sha256'] in fit_native for r in ev)
        line['evaluation_input_matches_fitting'] = sum(r['input_sha256'] in fit_input for r in ev)
        counts.append(line)
    write_csv(directory / 'class_counts.csv', counts)
    snapshot = b''.join(json.dumps([r['sample_id'], r['file_sha256']], separators=(',', ':')).encode() + b'\n'
                        for r in sorted(rows, key=lambda r: r['sample_id']))
    meta = {'split_settings': identity, 'classes': classes, 'snapshot_sha256': sha(snapshot),
            'manifest_sha256': sha((directory / 'manifest.csv').read_bytes()),
            'partition_counts': dict(Counter(r['partition'] for r in rows)),
            'condition': 'Full: no removal from fitting pool beyond development reservation'}
    write_json(directory / 'protocol.json', meta)
    print(json.dumps(meta, indent=2), flush=True)
    return rows, meta


def classification_metrics(true, predicted, classes):
    n = len(classes)
    cm = np.bincount(np.asarray(true) * n + np.asarray(predicted), minlength=n*n).reshape(n, n)
    support, positives, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(n, float), where=support != 0)
    precision = np.divide(tp, positives, out=np.zeros(n, float), where=positives != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n, float), where=(precision + recall) != 0)
    metrics = {'accuracy': float(tp.sum() / cm.sum()), 'macro_f1': float(f1.mean()),
               'macro_recall': float(recall.mean())}
    per_class = [{'class_label': cls, 'support': int(support[i]), 'precision': float(precision[i]),
                  'recall': float(recall[i]), 'f1': float(f1[i])} for i, cls in enumerate(classes)]
    return metrics, per_class, cm


class Images:
    def __init__(self, root, rows, classes, transform):
        self.root, self.rows, self.transform = root, rows, transform
        self.indices = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row['sample_id']) as im:
            x = self.transform(im.convert('RGB'))
        return x, self.indices[row['class_label']], row['sample_id']


def train(args, rows, protocol):
    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        raise RuntimeError('CUDA unavailable. Use a GPU environment, or explicitly pass --allow-cpu.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    classes = protocol['classes']
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    config = {k: v for k, v in vars(args).items() if k not in
              ('models', 'seeds', 'prepare_only', 'output', 'dataset', 'protocol_dir', 'restart_incomplete')}
    config.update({'condition': 'full', 'protocol': protocol, 'script_sha256': sha(Path(__file__).read_bytes()),
                   'torch': torch.__version__, 'torchvision': torchvision.__version__,
                   'python': platform.python_version(), 'numpy': np.__version__, 'Pillow': PIL.__version__,
                   'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'device': str(device), 'device_name': torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor(),
                   'preprocessing': {'RGB': True, 'resize': [224, 224], 'interpolation': 'Pillow BICUBIC',
                                     'crop': None, 'mean': MEAN, 'std': STD, 'augmentation': None},
                   'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR over total epochs',
                   'checkpoint_selection': 'minimum development cross-entropy; evaluation only after selection'})
    builders = {
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        'vit_b_16': (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1)}
    output = Path(args.output).resolve()
    # JSON round trip canonicalizes tuples for comparison when continuing completed runs.
    config = json.loads(json.dumps(config))
    config_path = prepare_output(output, config, args.restart_incomplete)

    def evaluate(model, loader, collect=False):
        model.eval()
        loss_total, count, records, truth, predictions = 0., 0, [], [], []
        with torch.inference_mode():
            for x, y, ids in loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda' and not args.no_amp):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite evaluation loss.')
                loss_total += loss.item() * len(y)
                count += len(y)
                if collect:
                    probabilities = logits.float().softmax(1).cpu().numpy()
                    labels = y.cpu().tolist()
                    guessed = probabilities.argmax(1).tolist()
                    truth.extend(labels)
                    predictions.extend(guessed)
                    for sample, actual, guess, probs in zip(ids, labels, guessed, probabilities):
                        records.append({'sample_id': sample, 'true_label': classes[actual],
                                        'predicted_label': classes[guess],
                                        **{f'prob_{c}': float(p) for c, p in zip(classes, probs)}})
        return loss_total/count, records, truth, predictions

    for name in args.models:
        for seed in args.seeds:
            run = output / name / f'seed_{seed}'
            if not prepare_run(run, args.restart_incomplete):
                print(f'Skipping completed run {name}/{seed}', flush=True)
                continue
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)
            loaders = {}
            for part in ('fitting', 'development', 'evaluation'):
                subset = [r for r in rows if r['partition'] == part]
                gen = torch.Generator().manual_seed(seed)
                loaders[part] = DataLoader(Images(Path(args.dataset), subset, classes, transform),
                    batch_size=args.batch_size, shuffle=part == 'fitting', num_workers=args.workers,
                    pin_memory=device.type == 'cuda', drop_last=False, generator=gen)
            if len(loaders['fitting'].dataset) % args.batch_size == 1:
                raise ValueError('Final training batch has one image. Choose another batch size (BatchNorm).')
            builder, weights = builders[name]
            model = builder(weights=weights)
            if name == 'resnet50':
                model.fc = nn.Linear(model.fc.in_features, len(classes))
                head = model.fc
            elif name == 'convnext_tiny':
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
                head = model.classifier
            else:
                model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
                head = model.heads
            model.to(device)
            head_ids = {id(p) for p in head.parameters()}
            backbone = [p for p in model.parameters() if id(p) not in head_ids]
            optimizer = torch.optim.AdamW([
                {'params': backbone, 'lr': args.backbone_lr},
                {'params': head.parameters(), 'lr': args.head_lr}], weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not args.no_amp)
            best, best_epoch, stale = float('inf'), 0, 0
            history = []
            started = time.time()
            write_json(run / 'run.json', {'model': name, 'seed': seed, 'weights': str(weights),
                                          'config_sha256': sha(config_path.read_bytes()),
                                          'parameters': sum(p.numel() for p in model.parameters())})
            for epoch in range(1, args.epochs + 1):
                warmup = epoch <= args.head_epochs
                for p in backbone:
                    p.requires_grad_(not warmup)
                if warmup:
                    model.eval()  # Freeze backbone BatchNorm statistics and stochastic layers too.
                    head.train()
                else:
                    model.train()
                if epoch == args.head_epochs + 1:
                    stale = 0
                running, seen = 0., 0
                for x, y, _ in loaders['fitting']:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda' and not args.no_amp):
                        loss = nn.functional.cross_entropy(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss.')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    running += loss.item() * len(y)
                    seen += len(y)
                dev_loss, _, _, _ = evaluate(model, loaders['development'])
                history.append({'epoch': epoch, 'phase': 'head' if warmup else 'full_finetuning',
                                'train_loss': running/seen, 'development_loss': dev_loss,
                                'backbone_lr': optimizer.param_groups[0]['lr'],
                                'head_lr': optimizer.param_groups[1]['lr'],
                                'optimizer_steps_this_epoch': len(loaders['fitting'])})
                write_csv(run / 'history.csv', history)
                print(f'{name} seed={seed} epoch={epoch}: train={running/seen:.5f} dev={dev_loss:.5f}', flush=True)
                if dev_loss < best:
                    best, best_epoch, stale = dev_loss, epoch, 0
                    torch.save({'state_dict': model.state_dict(), 'classes': classes, 'epoch': epoch,
                                'model': name, 'seed': seed, 'manifest_sha256': protocol['manifest_sha256']},
                               run / 'best.pt')
                elif not warmup:
                    stale += 1
                scheduler.step()
                if not warmup and stale >= args.patience:
                    break
            checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            loss, records, actual, guessed = evaluate(model, loaders['evaluation'], collect=True)
            scores, per_class, confusion = classification_metrics(actual, guessed, classes)
            write_csv(run / 'predictions.csv', records)
            write_csv(run / 'per_class.csv', per_class)
            write_csv(run / 'confusion_matrix.csv', [{'true_class': c, **dict(zip(classes, map(int, row)))}
                                                   for c, row in zip(classes, confusion)])
            scores.update({'model': name, 'seed': seed, 'condition': 'full', 'evaluation_loss': loss,
                           'evaluation_images': len(actual), 'selected_epoch': best_epoch,
                           'best_development_loss': best, 'epochs_completed': len(history),
                           'seconds': time.time()-started, 'manifest_sha256': protocol['manifest_sha256']})
            write_json(run / 'metrics.json', scores)  # Completion marker, written last.
            completed = [json.loads(p.read_text()) for p in sorted(output.glob('*/seed_*/metrics.json'))]
            write_csv(output / 'results.csv', completed)
            print(json.dumps(scores, indent=2), flush=True)
            del model, head, backbone, optimizer, scheduler, scaler, checkpoint, loaders
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Directory containing original train/ and val/.')
    p.add_argument('--protocol-dir', type=Path, required=True, help='Frozen manifests; reuse for every condition/model.')
    p.add_argument('--output', type=Path, default=Path('results/full_baseline_v1'))
    p.add_argument('--models', nargs='+', choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--split-seed', type=int, default=20260918)
    p.add_argument('--dev-fraction', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--head-epochs', type=int, default=3)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--restart-incomplete', action='store_true',
                   help='Archive incomplete attempts and restart from epoch 1; completed runs are retained.')
    args = p.parse_args()
    if not 0 < args.dev_fraction < 1 or not 0 <= args.head_epochs < args.epochs:
        p.error('Require 0 < dev-fraction < 1 and 0 <= head-epochs < epochs.')
    if min(args.batch_size, args.patience, args.backbone_lr, args.head_lr) <= 0 or args.workers < 0 or args.weight_decay < 0:
        p.error('Invalid training settings.')
    args.dataset = args.dataset.resolve()
    args.protocol_dir = args.protocol_dir.resolve()
    args.output = args.output.resolve()
    for target in (args.protocol_dir, args.output):
        if target == args.dataset or args.dataset in target.parents:
            p.error('Outputs must be outside the dataset directory.')
    if not (args.dataset/'train').is_dir() or not (args.dataset/'val').is_dir():
        p.error('Dataset must contain original train/ and val/ folders.')
    rows, meta = prepare(args.dataset, args.protocol_dir, args.dev_fraction, args.split_seed)
    if not args.prepare_only:
        train(args, rows, meta)


if __name__ == '__main__':
    main()
