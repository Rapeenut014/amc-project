"""
Evaluation Module for AMC Model
Generates metrics, confusion matrix, and SNR analysis
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import prepare_data_for_training, DatasetNotFoundError
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, Y_test, classes) -> Dict:
    """
    Evaluate model on test set
    
    Args:
        model: Trained Keras model
        X_test: Test signal data
        Y_test: Test labels
        classes: List of class names
    
    Returns:
        Dictionary with predictions, accuracy, and confusion matrix
    """
    logger.info("🔍 Evaluating model...")
    
    # Predictions
    Y_pred_proba = model.predict(X_test, verbose=0)
    Y_pred = np.argmax(Y_pred_proba, axis=1)
    
    # Overall accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    
    # Classification report
    report = classification_report(Y_test, Y_pred, target_names=classes, output_dict=True)
    
    logger.info(f"✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        'predictions': Y_pred,
        'probabilities': Y_pred_proba,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }


def evaluate_per_snr(model, X_test, Y_test, Z_test, classes) -> Dict[int, float]:
    """
    Evaluate model accuracy per SNR level
    
    Args:
        model: Trained model
        X_test, Y_test, Z_test: Test data with SNR labels
        classes: Class names
    
    Returns:
        Dictionary mapping SNR -> accuracy
    """
    logger.info("📊 Evaluating per-SNR accuracy...")
    
    snr_values = np.unique(Z_test)
    snr_accuracy = {}
    
    for snr in sorted(snr_values):
        mask = Z_test == snr
        X_snr = X_test[mask]
        Y_snr = Y_test[mask]
        
        if len(X_snr) == 0:
            continue
            
        Y_pred = np.argmax(model.predict(X_snr, verbose=0), axis=1)
        acc = accuracy_score(Y_snr, Y_pred)
        snr_accuracy[int(snr)] = float(acc)
        
        logger.info(f"   SNR {snr:+3.0f} dB: {acc:.4f} ({acc*100:.1f}%)")
    
    return snr_accuracy


def plot_confusion_matrix(cm, classes, save_path=None, normalize=True):
    """
    Plot confusion matrix as heatmap
    
    Args:
        cm: Confusion matrix array
        classes: Class names
        save_path: Path to save figure
        normalize: Whether to normalize by row
    
    Returns:
        Normalized confusion matrix
    """
    plt.figure(figsize=(14, 12))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={'size': 8})
    
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Saved: {save_path}")
    
    plt.close()
    return cm


def plot_snr_accuracy(snr_accuracy: Dict[int, float], save_path: Optional[str] = None):
    """
    Plot accuracy vs SNR curve
    
    Args:
        snr_accuracy: Dict of SNR -> accuracy
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    snrs = sorted(snr_accuracy.keys())
    accs = [snr_accuracy[snr] for snr in snrs]
    
    plt.plot(snrs, accs, 'b-o', linewidth=2, markersize=8)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% baseline')
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Classification Accuracy vs SNR', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Saved: {save_path}")
    
    plt.close()


def plot_training_history(history_path: str, save_path: Optional[str] = None):
    """
    Plot training and validation curves
    
    Args:
        history_path: Path to history JSON file
        save_path: Path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['accuracy'], label='Train Accuracy')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"💾 Saved: {save_path}")
    
    plt.close()


def generate_full_report(model_path: str, data_path: str, classes_path: str, 
                         output_dir: str = 'results') -> Dict:
    """
    Generate complete evaluation report with all plots
    
    Args:
        model_path: Path to trained model
        data_path: Path to dataset
        classes_path: Path to classes JSON
        output_dir: Directory for output files
    
    Returns:
        Evaluation report dictionary
    """
    logger.info("=" * 60)
    logger.info("📋 Generating Full Evaluation Report")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("\n📥 Loading model...")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load data
    logger.info("📂 Loading test data...")
    try:
        data = prepare_data_for_training(data_path, classes_path)
    except DatasetNotFoundError as e:
        logger.error(f"Dataset error: {e}")
        raise
    
    X_test = data['X_test']
    Y_test = data['Y_test']
    Z_test = data['Z_test']
    classes = data['classes']
    
    # Evaluate
    results = evaluate_model(model, X_test, Y_test, classes)
    snr_accuracy = evaluate_per_snr(model, X_test, Y_test, Z_test, classes)
    
    # Generate plots
    logger.info("\n📊 Generating plots...")
    
    plot_confusion_matrix(
        results['confusion_matrix'], classes,
        save_path=str(output_path / 'confusion_matrix.png')
    )
    
    plot_snr_accuracy(
        snr_accuracy,
        save_path=str(output_path / 'snr_accuracy.png')
    )
    
    # Save results
    report = {
        'overall_accuracy': results['accuracy'],
        'snr_accuracy': snr_accuracy,
        'classification_report': results['classification_report'],
        'model_path': str(model_path),
        'num_test_samples': len(X_test)
    }
    
    report_path = output_path / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n✅ Report saved to: {output_dir}/")
    logger.info(f"   - confusion_matrix.png")
    logger.info(f"   - snr_accuracy.png")
    logger.info(f"   - evaluation_report.json")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate AMC Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data', type=str,
                        default='archive/GOLD_XYZ_OSC.0001_1024.hdf5',
                        help='Path to HDF5 dataset')
    parser.add_argument('--classes', type=str,
                        default='archive/classes-fixed.json',
                        help='Path to classes JSON')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    try:
        generate_full_report(
            model_path=args.model,
            data_path=args.data,
            classes_path=args.classes,
            output_dir=args.output
        )
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
