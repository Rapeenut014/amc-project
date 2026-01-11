"""
Training Pipeline for AMC Model
Supports mixed precision training and comprehensive logging
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import prepare_data_for_training, DatasetNotFoundError
from src.model import build_cnn_model, build_resnet_model, compile_model, get_callbacks
from src.utils import get_logger

logger = get_logger(__name__)


def setup_gpu(use_mixed_precision=False):
    """
    Configure GPU settings for optimal training
    
    Args:
        use_mixed_precision: Enable float16 mixed precision for faster training
    """
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            logger.info(f"  - {gpu.name}")
            # Enable memory growth to avoid OOM
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"Memory growth setting failed: {e}")
        
        # Mixed precision for faster training on compatible GPUs
        if use_mixed_precision:
            try:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy('mixed_float16')
                logger.info("✓ Mixed precision (float16) enabled")
            except Exception as e:
                logger.warning(f"Mixed precision not available: {e}")
    else:
        logger.warning("No GPU found, training will use CPU")


def train_model(data_path, classes_path=None, model_type='cnn', 
                epochs=50, batch_size=256, learning_rate=0.001,
                quick=False, output_dir='models', use_mixed_precision=False):
    """
    Complete training pipeline
    
    Args:
        data_path: Path to HDF5 dataset
        classes_path: Path to classes JSON
        model_type: 'cnn' or 'resnet'
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        quick: If True, use small subset for quick testing
        output_dir: Directory to save models and history
        use_mixed_precision: Enable mixed precision training
    
    Returns:
        Tuple of (model, history)
    """
    logger.info("=" * 60)
    logger.info("🚀 AMC Model Training Pipeline")
    logger.info("=" * 60)
    
    # Setup GPU
    setup_gpu(use_mixed_precision)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and prepare data
    logger.info("\n📂 Step 1: Loading data...")
    try:
        data = prepare_data_for_training(data_path, classes_path)
    except DatasetNotFoundError as e:
        logger.error(f"Dataset error: {e}")
        raise
    
    X_train = data['X_train']
    X_val = data['X_val']
    Y_train = data['Y_train']
    Y_val = data['Y_val']
    num_classes = data['num_classes']
    classes = data['classes']
    
    # Quick mode: use subset
    if quick:
        logger.info("⚡ Quick mode: using 10% of data")
        n_train = len(X_train) // 10
        n_val = len(X_val) // 10
        X_train = X_train[:n_train]
        Y_train = Y_train[:n_train]
        X_val = X_val[:n_val]
        Y_val = Y_val[:n_val]
        epochs = min(epochs, 5)
    
    logger.info(f"   Training samples: {len(X_train):,}")
    logger.info(f"   Validation samples: {len(X_val):,}")
    
    # Step 2: Build model
    logger.info(f"\n🔧 Step 2: Building {model_type.upper()} model...")
    input_shape = X_train.shape[1:]  # (1024, 2)
    
    if model_type == 'resnet':
        model = build_resnet_model(input_shape, num_classes)
    else:
        model = build_cnn_model(input_shape, num_classes)
    
    model = compile_model(model, learning_rate)
    model.summary(print_fn=logger.info)
    
    # Step 3: Setup callbacks
    logger.info("\n⚙️ Step 3: Setting up callbacks...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_path / f'amc_{model_type}_{timestamp}.keras'
    callbacks = get_callbacks(str(model_path), patience=10)
    
    # Step 4: Train
    logger.info(f"\n🎯 Step 4: Training for {epochs} epochs...")
    logger.info("-" * 60)
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 5: Save results
    logger.info("\n💾 Step 5: Saving results...")
    
    # Save training history
    history_path = output_path / f'history_{timestamp}.json'
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'epochs_trained': len(history.history['loss']),
        'epochs_requested': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'model_type': model_type,
        'classes': classes,
        'mixed_precision': use_mixed_precision,
        'quick_mode': quick,
        'timestamp': timestamp
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Print summary
    best_val_acc = max(history.history['val_accuracy'])
    final_val_acc = history.history['val_accuracy'][-1]
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training Complete!")
    logger.info("=" * 60)
    logger.info(f"   Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    logger.info(f"   Final Val Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    logger.info(f"   Model saved: {model_path}")
    logger.info(f"   History saved: {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Train AMC Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str, 
                        default='archive/GOLD_XYZ_OSC.0001_1024.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--classes', type=str,
                        default='archive/classes-fixed.json',
                        help='Path to classes JSON')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (10%% data, 5 epochs)')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training (faster on compatible GPUs)')
    
    args = parser.parse_args()
    
    try:
        train_model(
            data_path=args.data,
            classes_path=args.classes,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            quick=args.quick,
            output_dir=args.output,
            use_mixed_precision=args.mixed_precision
        )
    except DatasetNotFoundError:
        logger.error("Training aborted: dataset not found")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
