"""
Unified Deep Learning Model Training for Malware Image Classification

This script trains and evaluates multiple CNN architectures on malware image datasets.
Supports: VGG16, ResNet50, DenseNet121, EfficientNetB0, and Custom CNN.

Training Strategy:
    - Uses 'val' folder as fixed test set
    - Splits 'train' folder into train (80%) / validation (20%)
    - Two-phase training for pretrained models (warmup + fine-tuning)
    - Multiple random seeds for statistical robustness

Usage:
    Train all models on all datasets:
        python train_models.py --data_root ./data
    
    Train specific model:
        python train_models.py --data_root ./data --model vgg16
    
    Train on specific dataset:
        python train_models.py --data_root ./data --dataset clean
    
    Custom configuration:
        python train_models.py --data_root ./data --epochs 100 --batch_size 32 --seeds 42,101,7

Requirements:
    pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
"""

import os
import sys
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

from tensorflow.keras.applications import VGG16, ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, 
    Dropout, BatchNormalization, Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# MODEL ARCHITECTURES
# ==========================================

def build_vgg16(num_classes, img_size):
    """Build VGG16 with custom classification head."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions), base_model


def build_resnet50(num_classes, img_size):
    """Build ResNet50 with custom classification head."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions), base_model


def build_densenet121(num_classes, img_size):
    """Build DenseNet121 with custom classification head."""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions), base_model


def build_efficientnetb0(num_classes, img_size):
    """Build EfficientNetB0 with custom classification head."""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions), base_model


def build_custom_cnn(num_classes, img_size):
    """Build custom CNN trained from scratch."""
    inputs = Input(shape=img_size + (3,))
    
    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    
    # Classification head
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Custom_CNN")
    return model, None


# Model registry
MODELS = {
    'vgg16': build_vgg16,
    'resnet50': build_resnet50,
    'densenet121': build_densenet121,
    'efficientnet': build_efficientnetb0,
    'custom_cnn': build_custom_cnn,
}


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            logger.warning(f"GPU setup failed: {e}")
    else:
        logger.info("No GPU detected, using CPU")


def get_dataframe_from_folder(folder_path):
    """
    Create DataFrame of image paths and labels from directory structure.
    
    Expected structure:
        folder_path/
            class1/
                image1.jpg
                image2.jpg
            class2/
                ...
    
    Args:
        folder_path (str): Path to folder containing class subdirectories
    
    Returns:
        pd.DataFrame: DataFrame with 'filename' and 'class' columns
    """
    filepaths = []
    labels = []
    
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return pd.DataFrame(columns=['filename', 'class'])

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                label = Path(root).name
                filepaths.append(str(Path(root) / file))
                labels.append(label)
    
    return pd.DataFrame({'filename': filepaths, 'class': labels})


def create_generators(train_df, val_df, test_df, img_size, batch_size, needs_rescale=True):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        img_size: Tuple of (height, width)
        batch_size: Batch size
        needs_rescale: Whether to rescale images (False for EfficientNet)
    
    Returns:
        Tuple of (train_gen, val_gen, test_gen)
    """
    datagen = ImageDataGenerator(rescale=1./255 if needs_rescale else None)
    
    train_gen = datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='class',
        target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=True
    )
    
    val_gen = datagen.flow_from_dataframe(
        val_df, x_col='filename', y_col='class',
        target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    
    test_gen = datagen.flow_from_dataframe(
        test_df, x_col='filename', y_col='class',
        target_size=img_size, batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    
    return train_gen, val_gen, test_gen


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_pretrained_model(model, base_model, train_gen, val_gen, epochs_phase1, epochs_phase2, lr_phase1, lr_phase2):
    """
    Two-phase training for pretrained models.
    
    Phase 1: Train classification head with frozen base
    Phase 2: Fine-tune top layers of base model
    """
    # Phase 1: Warmup (frozen base)
    logger.info("Phase 1: Training classification head (base frozen)")
    model.compile(
        optimizer=Adam(learning_rate=lr_phase1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs_phase1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)],
        verbose=0
    )
    
    # Phase 2: Fine-tuning (unfreeze top 50% of base)
    logger.info("Phase 2: Fine-tuning (top 50% of base unfrozen)")
    base_model.trainable = True
    split_layer = int(len(base_model.layers) * 0.5)
    for layer in base_model.layers[:split_layer]:
        layer.trainable = False
    
    model.compile(
        optimizer=Adam(learning_rate=lr_phase2),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs_phase2,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ],
        verbose=0
    )


def train_custom_model(model, train_gen, val_gen, total_epochs, init_lr):
    """Single-phase training for custom CNN (no pretrained weights)."""
    logger.info(f"Training custom CNN from scratch ({total_epochs} epochs)")
    
    model.compile(
        optimizer=Adam(learning_rate=init_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
        train_gen, validation_data=val_gen,
        epochs=total_epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ],
        verbose=0
    )


def evaluate_model(model, test_gen):
    """Evaluate model and return metrics."""
    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen, verbose=0), axis=1)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
    }


# ==========================================
# MAIN EXPERIMENT
# ==========================================

def run_experiment(args):
    """Run complete training experiment."""
    setup_gpu()
    
    # Parse dataset configurations
    data_root = Path(args.data_root)
    datasets = {
        'full': data_root / 'full_malevis',
        'clean': data_root / 'clean_malevis',
        'balanced': data_root / 'subsampled_malevis',
    }
    
    # Filter datasets if specified
    if args.dataset:
        if args.dataset not in datasets:
            logger.error(f"Unknown dataset: {args.dataset}. Choose from: {list(datasets.keys())}")
            sys.exit(1)
        datasets = {args.dataset: datasets[args.dataset]}
    
    # Filter models if specified
    models_to_train = list(MODELS.keys())
    if args.model:
        if args.model not in MODELS:
            logger.error(f"Unknown model: {args.model}. Choose from: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_train = [args.model]
    
    # Parse seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    
    # Results storage
    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logger.info(f"Starting experiment: {len(datasets)} dataset(s) × {len(models_to_train)} model(s) × {len(seeds)} seed(s)")
    
    for dataset_name, dataset_path in datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"DATASET: {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Load data
        test_df = get_dataframe_from_folder(dataset_path / 'val')
        train_pool_df = get_dataframe_from_folder(dataset_path / 'train')
        
        if len(test_df) == 0 or len(train_pool_df) == 0:
            logger.error(f"Missing data in {dataset_path}. Skipping.")
            continue
        
        num_classes = len(train_pool_df['class'].unique())
        logger.info(f"Train pool: {len(train_pool_df)} images | Test: {len(test_df)} images | Classes: {num_classes}")
        
        for model_name in models_to_train:
            logger.info(f"\n--- MODEL: {model_name.upper()} ---")
            
            for seed in seeds:
                logger.info(f"Seed: {seed}")
                set_seed(seed)
                
                # Split train pool into train/val
                try:
                    train_df, val_df = train_test_split(
                        train_pool_df, test_size=0.2,
                        stratify=train_pool_df['class'],
                        random_state=seed
                    )
                except ValueError:
                    logger.warning("Stratified split failed, using random split")
                    train_df, val_df = train_test_split(
                        train_pool_df, test_size=0.2, random_state=seed
                    )
                
                # Create generators
                needs_rescale = model_name != 'efficientnet'
                train_gen, val_gen, test_gen = create_generators(
                    train_df, val_df, test_df,
                    args.img_size, args.batch_size,
                    needs_rescale
                )
                
                # Build model
                build_fn = MODELS[model_name]
                model, base_model = build_fn(num_classes, args.img_size)
                
                # Train
                if model_name == 'custom_cnn':
                    train_custom_model(model, train_gen, val_gen, args.epochs, args.lr)
                else:
                    train_pretrained_model(
                        model, base_model, train_gen, val_gen,
                        args.epochs_phase1, args.epochs_phase2,
                        args.lr_phase1, args.lr_phase2
                    )
                
                # Evaluate
                metrics = evaluate_model(model, test_gen)
                logger.info(f"Results: Acc={metrics['accuracy']:.4f} | F1={metrics['f1_macro']:.4f}")
                
                # Store results
                all_results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'seed': seed,
                    'num_classes': num_classes,
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'test_samples': len(test_df),
                    **metrics
                })
                
                # Cleanup
                del model, train_gen, val_gen, test_gen
                if base_model:
                    del base_model
                tf.keras.backend.clear_session()
                gc.collect()
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = args.output or f"training_results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    summary = results_df.groupby(['dataset', 'model'])[['accuracy', 'f1_macro']].agg(['mean', 'std'])
    print(summary.to_string())


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train deep learning models for malware image classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data configuration
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing dataset folders')
    parser.add_argument('--dataset', type=str, choices=['full', 'clean', 'balanced'],
                        help='Specific dataset to train on (default: all)')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()),
                        help='Specific model to train (default: all)')
    
    # Training hyperparameters
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224],
                        help='Image size (height width) (default: 224 224)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total epochs for custom CNN (default: 50)')
    parser.add_argument('--epochs_phase1', type=int, default=15,
                        help='Epochs for phase 1 (pretrained models) (default: 15)')
    parser.add_argument('--epochs_phase2', type=int, default=35,
                        help='Epochs for phase 2 (pretrained models) (default: 35)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for custom CNN (default: 0.001)')
    parser.add_argument('--lr_phase1', type=float, default=1e-3,
                        help='Learning rate phase 1 (default: 0.001)')
    parser.add_argument('--lr_phase2', type=float, default=1e-4,
                        help='Learning rate phase 2 (default: 0.0001)')
    
    # Experiment configuration
    parser.add_argument('--seeds', type=str, default='42,101,7,2024,99',
                        help='Comma-separated random seeds (default: 42,101,7,2024,99)')
    parser.add_argument('--output', type=str,
                        help='Output CSV file path (default: training_results_TIMESTAMP.csv)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Convert img_size list to tuple
    args.img_size = tuple(args.img_size)
    
    run_experiment(args)
    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()