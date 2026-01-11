"""
CNN Model for Automatic Modulation Classification
Based on proven architectures for RadioML dataset
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


def build_cnn_model(input_shape=(1024, 2), num_classes=24, dropout_rate=0.5):
    """
    Build CNN model for modulation classification
    
    Architecture:
    - 4 Conv1D blocks with increasing filters
    - MaxPooling after each conv block
    - 2 Dense layers with dropout
    - Softmax output
    
    Args:
        input_shape: Shape of input signal (time_steps, channels)
        num_classes: Number of modulation classes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras model
    """
    logger.info(f"Building CNN model: input={input_shape}, classes={num_classes}")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Conv Block 1
        layers.Conv1D(64, kernel_size=8, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv Block 2
        layers.Conv1D(128, kernel_size=4, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv Block 3
        layers.Conv1D(256, kernel_size=4, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),
        
        # Conv Block 4
        layers.Conv1D(256, kernel_size=4, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(256),
        layers.ReLU(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128),
        layers.ReLU(),
        layers.Dropout(dropout_rate),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    logger.info(f"✓ CNN model built: {model.count_params():,} parameters")
    return model


def build_resnet_model(input_shape=(1024, 2), num_classes=24):
    """
    Build ResNet-style model for better accuracy
    Uses residual connections for deeper architecture
    
    Args:
        input_shape: Shape of input signal
        num_classes: Number of modulation classes
    
    Returns:
        Keras Model instance
    """
    logger.info(f"Building ResNet model: input={input_shape}, classes={num_classes}")
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv1D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Residual blocks
    for filters in [64, 128, 256]:
        # Save for skip connection
        shortcut = layers.Conv1D(filters, 1, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
        # Conv path
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv1D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    
    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    logger.info(f"✓ ResNet model built: {model.count_params():,} parameters")
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile model with optimizer and loss
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate for Adam
    
    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"✓ Model compiled with lr={learning_rate}")
    return model


def get_callbacks(model_path='models/best_model.keras', patience=10):
    """
    Get training callbacks for checkpointing, early stopping, and LR scheduling
    
    Args:
        model_path: Path to save best model
        patience: Epochs to wait before early stopping
    
    Returns:
        List of Keras callbacks
    """
    # Ensure directory exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    logger.info(f"✓ Callbacks configured: checkpoint, early_stop, lr_schedule, tensorboard")
    return callbacks


def load_model(model_path: str):
    """
    Load a saved model from disk
    
    Args:
        model_path: Path to saved .keras model
    
    Returns:
        Loaded Keras model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    logger.info(f"✓ Model loaded: {model.count_params():,} parameters")
    
    return model


def list_available_models(models_dir: str = "models") -> list:
    """
    List all available trained models
    
    Args:
        models_dir: Directory containing saved models
    
    Returns:
        List of model file paths
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    models = list(models_path.glob("*.keras"))
    logger.info(f"Found {len(models)} trained models in {models_dir}")
    return models


def model_summary(input_shape=(1024, 2), num_classes=24):
    """
    Print model summary and statistics
    
    Args:
        input_shape: Input signal shape
        num_classes: Number of output classes
    
    Returns:
        Created model
    """
    model = build_cnn_model(input_shape, num_classes)
    model.summary()
    
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
    print(f"📦 Estimated size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("AMC CNN Model Architecture")
    print("=" * 60)
    model_summary()
