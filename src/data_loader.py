"""
Data Loader Module for RadioML 2018 Dataset
Handles loading HDF5 data and preprocessing for AMC
"""

import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when the dataset file is not found"""
    pass


class InvalidDataFormatError(Exception):
    """Raised when the dataset has unexpected format"""
    pass


def load_radioml_data(data_path, classes_path=None):
    """
    Load RadioML 2018 dataset from HDF5 file
    
    Args:
        data_path: Path to GOLD_XYZ_OSC.0001_1024.hdf5
        classes_path: Path to classes-fixed.json (optional)
    
    Returns:
        X: Signal data (samples, 1024, 2) - I/Q components
        Y: Labels (samples,)
        Z: SNR values (samples,)
        classes: List of modulation class names
    
    Raises:
        DatasetNotFoundError: If the dataset file doesn't exist
        InvalidDataFormatError: If the dataset has unexpected format
    """
    data_path = Path(data_path)
    
    # Check if file exists
    if not data_path.exists():
        error_msg = (
            f"Dataset not found: {data_path}\n"
            f"Please download the RadioML 2018.01A dataset from:\n"
            f"  https://www.deepsig.ai/datasets\n"
            f"And place it in the archive/ folder."
        )
        logger.error(error_msg)
        raise DatasetNotFoundError(error_msg)
    
    logger.info(f"Loading data from {data_path}...")
    
    try:
        with h5py.File(data_path, 'r') as f:
            # Validate expected keys
            expected_keys = {'X', 'Y', 'Z'}
            actual_keys = set(f.keys())
            if not expected_keys.issubset(actual_keys):
                missing = expected_keys - actual_keys
                raise InvalidDataFormatError(
                    f"Dataset missing required keys: {missing}. "
                    f"Found keys: {actual_keys}"
                )
            
            X = f['X'][:]  # Shape: (N, 1024, 2)
            Y = f['Y'][:]  # One-hot encoded labels
            Z = f['Z'][:]  # SNR values
    except OSError as e:
        logger.error(f"Failed to open HDF5 file: {e}")
        raise InvalidDataFormatError(f"Cannot read HDF5 file: {e}")
    
    # Validate shapes
    if len(X.shape) != 3 or X.shape[1:] != (1024, 2):
        logger.warning(
            f"Unexpected X shape: {X.shape}. Expected (N, 1024, 2)"
        )
    
    # Convert one-hot to class indices
    Y = np.argmax(Y, axis=1)
    Z = Z.flatten()
    
    # Load class names
    if classes_path and os.path.exists(classes_path):
        try:
            with open(classes_path, 'r') as f:
                classes = json.load(f)
            logger.info(f"Loaded class names from {classes_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse classes file: {e}. Using defaults.")
            classes = None
    else:
        classes = None
    
    if classes is None:
        classes = [
            'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK',
            '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
            '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC',
            'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
        ]
    
    logger.info(f"✓ Loaded {len(X):,} samples")
    logger.info(f"  Signal shape: {X.shape}")
    logger.info(f"  Classes: {len(classes)}")
    logger.info(f"  SNR range: {Z.min():.0f} to {Z.max():.0f} dB")
    
    return X, Y, Z, classes


def normalize_data(X):
    """
    Normalize I/Q data to zero mean and unit variance per sample
    
    Args:
        X: Signal data of shape (N, time_steps, 2)
    
    Returns:
        Normalized signal data
    """
    # Compute mean and std per sample
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True) + 1e-8
    X_norm = (X - mean) / std
    
    logger.debug(f"Normalized {len(X)} samples")
    return X_norm


def split_data(X, Y, Z, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train, validation, and test sets
    Stratified by class to maintain class distribution
    
    Args:
        X: Signal data
        Y: Labels
        Z: SNR values
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_test
    """
    logger.info(f"Splitting data: test={test_size}, val={val_size}")
    
    # First split: train+val vs test
    X_trainval, X_test, Y_trainval, Y_test, Z_trainval, Z_test = train_test_split(
        X, Y, Z, 
        test_size=test_size, 
        stratify=Y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval,
        test_size=val_ratio,
        stratify=Y_trainval,
        random_state=random_state
    )
    
    logger.info(f"✓ Split complete:")
    logger.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_test


def get_snr_subset(X, Y, Z, target_snr):
    """
    Get subset of data for a specific SNR value
    
    Args:
        X, Y, Z: Full dataset
        target_snr: SNR value to filter
    
    Returns:
        X_filtered, Y_filtered
    """
    mask = Z == target_snr
    count = mask.sum()
    logger.debug(f"SNR {target_snr} dB: {count} samples")
    return X[mask], Y[mask]


def prepare_data_for_training(data_path, classes_path=None, normalize=True, 
                               test_size=0.2, val_size=0.1):
    """
    Complete data preparation pipeline
    
    Args:
        data_path: Path to HDF5 dataset
        classes_path: Path to classes JSON file
        normalize: Whether to normalize signals
        test_size: Fraction for test set
        val_size: Fraction for validation set
    
    Returns:
        Dictionary with train/val/test splits and metadata
    """
    logger.info("=" * 50)
    logger.info("Starting data preparation pipeline")
    logger.info("=" * 50)
    
    # Load data
    X, Y, Z, classes = load_radioml_data(data_path, classes_path)
    
    # Normalize
    if normalize:
        logger.info("Normalizing signals...")
        X = normalize_data(X)
    
    # Split
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Z_test = split_data(
        X, Y, Z, test_size, val_size
    )
    
    logger.info("✓ Data preparation complete")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test,
        'Z_test': Z_test, 'classes': classes,
        'num_classes': len(classes)
    }


if __name__ == "__main__":
    # Test data loading
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "archive" / "GOLD_XYZ_OSC.0001_1024.hdf5"
    classes_path = base_dir / "archive" / "classes-fixed.json"
    
    try:
        X, Y, Z, classes = load_radioml_data(data_path, classes_path)
        print(f"\n✅ Data loaded successfully!")
        print(f"   X shape: {X.shape}")
        print(f"   Y shape: {Y.shape}")
        print(f"   Z shape: {Z.shape}")
    except DatasetNotFoundError as e:
        print(f"❌ {e}")
    except InvalidDataFormatError as e:
        print(f"❌ Invalid data format: {e}")
