#!/usr/bin/env python3
"""Keras ResNet50 with pretrained-weight-compatible input preprocessing.

Controlled correction of train_original_recipe.py: retain architecture, training
schedule and frozen split, and replace /255 with ResNet preprocess_input.
Default runs development-only; --evaluate scores the frozen selected checkpoint.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import train_full_malevis as audit
from tune_and_compare_malevis import conditions, digest_rows, freeze_csv, freeze_json

MODELS = ('resnet50',)


def make_generators(reference, root, fitting, development, evaluation, classes, batch_size, model):
    import pandas as pd
    frames = [pd.DataFrame({'filename': [str(root/r['sample_id']) for r in rows],
                            'class': [r['class_label'] for r in rows]})
              for rows in (fitting, development, evaluation)]
    # ResNet50 pretrained weights expect 0..255 RGB, then BGR mean subtraction.
    # Do not also divide by 255. Use identically for every partition.
    from tensorflow.keras.applications.resnet50 import preprocess_input
    datagen = reference.ImageDataGenerator(preprocessing_function=preprocess_input)
    generators = tuple(datagen.flow_from_dataframe(frame, x_col='filename', y_col='class',
        target_size=(224,224), batch_size=batch_size, class_mode='categorical',
        shuffle=(i==0), classes=classes, interpolation='bicubic') for i, frame in enumerate(frames))
    expected = {c: i for i, c in enumerate(classes)}
    for gen, rows in zip(generators, (fitting, development, evaluation)):
        if gen.class_indices != expected or gen.samples != len(rows):
            raise ValueError('Generator changed class mapping or dropped samples.')
        if gen.interpolation != 'bicubic':
            raise ValueError('Unexpected interpolation.')
    return generators


def train_phases(tf, model, backbone, train, dev, args, run):
    """Same optimizers, trainable-layer rule, early stopping and LR callbacks."""
    cb = tf.keras.callbacks
    if args.model == 'custom_cnn':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train, validation_data=dev, epochs=args.custom_epochs,
            callbacks=[cb.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                       cb.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=4, min_lr=1e-6),
                       cb.CSVLogger(str(run/'custom_history.csv'))], verbose=2)
        selected = int(np.argmin(history.history['val_loss']))
        return dict(selected_phase='custom', selected_phase_epoch=selected+1,
                    development_loss=float(history.history['val_loss'][selected]),
                    development_accuracy=float(history.history['val_accuracy'][selected]),
                    epochs_completed=len(history.epoch))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print('Phase 1: frozen backbone; Adam learning rate 0.001.', flush=True)
    warm = model.fit(train, validation_data=dev, epochs=args.epochs_phase1,
        callbacks=[cb.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                   cb.CSVLogger(str(run/'warmup_history.csv'))], verbose=2)
    backbone.trainable = True
    split = int(len(backbone.layers) * .5)
    for layer in backbone.layers[:split]:
        layer.trainable = False
    audit.write_json(run/'finetuning_layers.json',
                     [{'name': layer.name, 'trainable': layer.trainable} for layer in backbone.layers])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print('Phase 2: upper half of backbone unfrozen; new Adam, learning rate 0.0001.', flush=True)
    fine = model.fit(train, validation_data=dev, epochs=args.epochs_phase2,
        callbacks=[cb.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                   cb.ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=3, min_lr=1e-6),
                   cb.CSVLogger(str(run/'finetuning_history.csv'))], verbose=2)
    selected = int(np.argmin(fine.history['val_loss']))
    # Matches the original: best phase-2 weights, not the best across both phases.
    return dict(selected_phase='finetuning', selected_phase_epoch=selected+1,
                development_loss=float(fine.history['val_loss'][selected]),
                development_accuracy=float(fine.history['val_accuracy'][selected]),
                warmup_epochs_completed=len(warm.epoch), finetuning_epochs_completed=len(fine.epoch))


def run_one(args, experiment, fitting, dev, evaluation, condition, seed, reference, tf):
    run = args.output/args.model/condition/f'seed_{seed}'
    config = dict(experiment_sha256=digest_rows(experiment), condition=condition, seed=seed,
                  fitting_sha256=digest_rows(fitting), development_sha256=digest_rows(dev),
                  evaluation_sha256=digest_rows(evaluation))
    if (run/'metrics.json').exists():
        freeze_json(run/'run.json', config)
        metrics = json.loads((run/'metrics.json').read_text())
        for name, digest in metrics['artifacts_sha256'].items():
            if audit.sha((run/name).read_bytes()) != digest:
                raise ValueError(f'Changed completed artifact: {run/name}')
        print(f'Reusing completed {condition}, seed {seed}.', flush=True)
        return metrics
    resume_evaluation = (run/'trained.json').exists()
    if not resume_evaluation and run.exists() and any(run.iterdir()):
        if not args.restart_incomplete:
            raise ValueError(f'Incomplete run at {run}; --restart-incomplete archives and restarts it.')
        audit.archive_attempt(run)
    run.mkdir(parents=True, exist_ok=True)
    freeze_json(run/'run.json', config)
    tf.keras.backend.clear_session()
    reference.set_seed(seed)
    # Keras 3 also has its own RNG; explicitly seed it for reproducible head initialization.
    tf.keras.utils.set_random_seed(seed)
    train_gen, dev_gen, eval_gen = make_generators(reference, args.dataset, fitting, dev, evaluation,
                                                 experiment['classes'], args.batch_size, args.model)
    start = time.time()
    if resume_evaluation:
        trained = json.loads((run/'trained.json').read_text())
        if audit.sha((run/'selected.keras').read_bytes()) != trained['checkpoint_sha256']:
            raise ValueError('Checkpoint changed.')
        model = tf.keras.models.load_model(run/'selected.keras', compile=False)
    else:
        model, backbone = reference.MODELS[args.model](len(experiment['classes']), (224, 224))
        state_hash = hashlib.sha256()
        for weight in model.get_weights():
            state_hash.update(audit.payload(weight))
        with (run/'model_summary.txt').open('w') as f:
            model.summary(print_fn=lambda line: f.write(line+'\n'))
        trained = train_phases(tf, model, backbone, train_gen, dev_gen, args, run)
        model.save(run/'selected.keras')
        trained.update(initial_state_sha256=state_hash.hexdigest(),
                       checkpoint_sha256=audit.sha((run/'selected.keras').read_bytes()),
                       training_seconds=time.time()-start)
        audit.write_json(run/'trained.json', trained)
    # Inspect the selected model on development before accessing evaluation scores.
    dev_gen.reset()
    dev_probs = np.asarray(model.predict(dev_gen, verbose=0))
    dev_scores, dev_per_class, _ = audit.classification_metrics(
        dev_gen.classes, dev_probs.argmax(1), experiment['classes'])
    audit.write_csv(run/'development_per_class.csv', dev_per_class)
    audit.write_csv(run/'development_predictions.csv', [
        dict(sample_id=r['sample_id'], true_label=experiment['classes'][int(y)],
             predicted_label=experiment['classes'][int(g)])
        for r,y,g in zip(dev,dev_gen.classes,dev_probs.argmax(1))])
    audit.write_json(run/'development_metrics.json',dev_scores)
    print('Selected checkpoint development metrics:', json.dumps(dev_scores),flush=True)
    if not args.evaluate:
        print('Development-only run complete. Checkpoint saved; evaluation was not scored.',flush=True)
        del model, train_gen, dev_gen, eval_gen
        tf.keras.backend.clear_session()
        return dict(status='development_complete',condition=condition,seed=seed,**dev_scores)
    # Evaluation happens only when explicitly requested, after checkpoint selection.
    eval_gen.reset()
    probabilities = np.asarray(model.predict(eval_gen, verbose=1))
    if probabilities.shape != (len(evaluation), len(experiment['classes'])) or not np.isfinite(probabilities).all():
        raise ValueError('Invalid prediction array.')
    actual, predicted = eval_gen.classes, probabilities.argmax(1)
    classes = experiment['classes']
    metrics, per_class, confusion = audit.classification_metrics(actual, predicted, classes)
    records = [dict(sample_id=r['sample_id'], true_label=classes[int(y)], predicted_label=classes[int(g)],
                    **{f'prob_{c}': float(p) for c, p in zip(classes, ps)})
               for r, y, g, ps in zip(evaluation, actual, predicted, probabilities)]
    audit.write_csv(run/'predictions.csv', records)
    audit.write_csv(run/'per_class.csv', per_class)
    audit.write_csv(run/'confusion_matrix.csv',
                    [dict(true_class=c, **dict(zip(classes, map(int, row)))) for c, row in zip(classes, confusion)])
    metrics.update(model=args.model, condition=condition, seed=seed, fitting_images=len(fitting),
                   development_images=len(dev), evaluation_images=len(evaluation), **trained,
                   artifacts_sha256={name: audit.sha((run/name).read_bytes()) for name in
                       ('run.json', 'trained.json', 'selected.keras', 'predictions.csv',
                        'per_class.csv', 'confusion_matrix.csv')})
    audit.write_json(run/'metrics.json', metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'artifacts_sha256'}, indent=2), flush=True)
    del model, train_gen, dev_gen, eval_gen
    tf.keras.backend.clear_session()
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--protocol-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='resnet50')
    p.add_argument('--conditions', nargs='+', choices=('full','clean_exact','random_control'), default=['full'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--removal-seed', type=int, default=20260918)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs-phase1', type=int, default=15)
    p.add_argument('--epochs-phase2', type=int, default=35)
    p.add_argument('--custom-epochs', type=int, default=50)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--evaluate', action='store_true', help='Score the selected checkpoint on fixed evaluation data.')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--restart-incomplete', action='store_true')
    args = p.parse_args()
    if min(args.batch_size,args.epochs_phase1,args.epochs_phase2,args.custom_epochs) < 1:
        p.error('Batch size and epoch limits must be positive.')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error('Seeds must be unique integers between 0 and 2**32-1.')
    args.dataset, args.protocol_dir, args.output = (x.resolve() for x in
                                                   (args.dataset, args.protocol_dir, args.output))
    if any(args.output == x or x in args.output.parents for x in (args.dataset,args.protocol_dir)):
        p.error('Output must be outside the dataset and frozen protocol.')
    if not (args.protocol_dir/'protocol.json').exists():
        p.error('Provide the existing frozen Full protocol directory.')
    meta = json.loads((args.protocol_dir/'protocol.json').read_text())
    settings = meta['split_settings']
    rows, meta = audit.prepare(args.dataset, args.protocol_dir,
                               settings['development_fraction'],settings['split_seed'])
    fitting, dev, evaluation, removed, table = conditions(rows,args.removal_seed)
    args.output.mkdir(parents=True,exist_ok=True)
    for name, subset in fitting.items():
        freeze_csv(args.output/(name+'_fitting.csv'),subset)
    freeze_csv(args.output/'development.csv',dev)
    freeze_csv(args.output/'evaluation.csv',evaluation)
    freeze_csv(args.output/'class_counts.csv',table)
    if removed:
        freeze_csv(args.output/'removed_exact.csv',removed)
    print('Verified bicubic-resized development disjointness; fitting sizes:',
          {k:len(v) for k,v in fitting.items()},flush=True)
    if args.prepare_only:
        return
    import tensorflow as tf
    import keras
    import original_training_reference as reference
    # Bounded host thread counts; the original optimizer/model settings are unchanged.
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus and not args.allow_cpu:
        raise RuntimeError('TensorFlow cannot see a GPU. Check its CUDA dependencies before training.')
    reference.setup_gpu()
    experiment = dict(model=args.model, classes=meta['classes'], protocol=meta,
        removal_seed=args.removal_seed, fitting_sha256={k:digest_rows(v) for k,v in fitting.items()},
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_phase1, epochs_phase2=args.epochs_phase2,custom_epochs=args.custom_epochs,
        pretrained='Keras ImageNet weights', normalization='keras.applications.resnet50.preprocess_input: RGB 0..255 to BGR minus ImageNet means; no /255',
        interpolation='Pillow bicubic, original article convention; differs from original code nearest default',
        augmentation=None, precision='float32',
        selection='restore best val_loss within warmup, then within fine-tuning; evaluate phase-2 selection',
        split_note='Frozen identity-group-aware split, not original per-seed random image split',
        python=platform.python_version(), numpy=np.__version__, Pillow=audit.PIL.__version__,
        tensorflow=tf.__version__, keras=keras.__version__, tensorflow_build=tf.sysconfig.get_build_info(),
        gpus=[tf.config.experimental.get_device_details(g) for g in gpus],
        source_sha256={Path(f).name:audit.sha(Path(f).read_bytes()) for f in
                       (__file__,reference.__file__,audit.__file__,Path(__file__).with_name('tune_and_compare_malevis.py'))})
    freeze_json(args.output/'experiment.json',experiment)
    for seed in args.seeds:
        for condition in args.conditions:
            print(f'Preprocessing-corrected recipe: {args.model}, {condition}, seed {seed}',flush=True)
            run_one(args,experiment,fitting[condition],dev,evaluation,condition,seed,reference,tf)
    completed=[json.loads(f.read_text()) for f in sorted((args.output/args.model).glob('*/seed_*/metrics.json'))]
    if not completed:
        print('Development results saved under model/condition/seed directories.',flush=True)
        return
    for seed in {r['seed'] for r in completed}:
        if len({r['initial_state_sha256'] for r in completed if r['seed']==seed})!=1:
            raise ValueError(f'Paired initialization differs for seed {seed}.')
    audit.write_csv(args.output/'results.csv',
                    [{k:v for k,v in r.items() if k!='artifacts_sha256'} for r in completed])
    contrasts=[]
    for seed in sorted({r['seed'] for r in completed}):
        by={r['condition']:r for r in completed if r['seed']==seed}
        if set(by)=={'full','clean_exact','random_control'}:
            row=dict(seed=seed)
            for metric in ('accuracy','macro_f1','macro_recall'):
                for a,b in [('full','clean_exact'),('full','random_control'),('random_control','clean_exact')]:
                    row[a+'_minus_'+b+'_'+metric+'_pp']=100*(by[a][metric]-by[b][metric])
            contrasts.append(row)
    if contrasts:
        audit.write_csv(args.output/'paired_differences.csv',contrasts)


if __name__=='__main__':
    main()
