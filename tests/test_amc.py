"""
Unit tests for AMC project
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataLoader:
    """Tests for data_loader module"""
    
    def test_normalize_data(self):
        """Test that normalization produces zero mean and unit variance"""
        from src.data_loader import normalize_data
        
        # Create sample data
        X = np.random.randn(100, 1024, 2) * 5 + 10  # Non-normalized data
        X_norm = normalize_data(X)
        
        # Check shape preserved
        assert X_norm.shape == X.shape
        
        # Check approximate zero mean per sample
        means = np.mean(X_norm, axis=1)
        assert np.allclose(means, 0, atol=1e-6)
        
        # Check approximate unit variance per sample
        stds = np.std(X_norm, axis=1)
        assert np.allclose(stds, 1, atol=1e-6)
    
    def test_get_snr_subset(self):
        """Test SNR filtering"""
        from src.data_loader import get_snr_subset
        
        # Create sample data with known SNR values
        X = np.random.randn(100, 1024, 2)
        Y = np.random.randint(0, 24, 100)
        Z = np.array([10] * 50 + [20] * 50)
        
        # Filter for SNR=10
        X_filtered, Y_filtered = get_snr_subset(X, Y, Z, 10)
        
        assert len(X_filtered) == 50
        assert len(Y_filtered) == 50


class TestModel:
    """Tests for model module"""
    
    def test_build_cnn_model_shape(self):
        """Test CNN model has correct input/output shape"""
        from src.model import build_cnn_model
        
        model = build_cnn_model(input_shape=(1024, 2), num_classes=24)
        
        # Check input shape
        assert model.input_shape == (None, 1024, 2)
        
        # Check output shape
        assert model.output_shape == (None, 24)
    
    def test_build_cnn_model_prediction(self):
        """Test CNN model can make predictions"""
        from src.model import build_cnn_model, compile_model
        
        model = build_cnn_model(input_shape=(1024, 2), num_classes=24)
        model = compile_model(model)
        
        # Create sample input
        X = np.random.randn(5, 1024, 2).astype(np.float32)
        
        # Make prediction
        pred = model.predict(X, verbose=0)
        
        # Check output shape
        assert pred.shape == (5, 24)
        
        # Check probabilities sum to 1
        assert np.allclose(pred.sum(axis=1), 1, atol=1e-6)
    
    def test_build_resnet_model_shape(self):
        """Test ResNet model has correct shape"""
        from src.model import build_resnet_model
        
        model = build_resnet_model(input_shape=(1024, 2), num_classes=24)
        
        assert model.input_shape == (None, 1024, 2)
        assert model.output_shape == (None, 24)
    
    def test_list_available_models(self):
        """Test model listing function"""
        from src.model import list_available_models
        
        # This should not crash even if directory doesn't exist
        models = list_available_models("nonexistent_dir")
        assert models == []


class TestUtils:
    """Tests for utils module"""
    
    def test_get_logger(self):
        """Test logger creation"""
        from src.utils import get_logger
        
        logger = get_logger("test_logger")
        
        # Should have at least one handler
        assert len(logger.handlers) >= 1
        
        # Should be able to log without error
        logger.info("Test message")
        logger.warning("Test warning")
    
    def test_logger_no_duplicate_handlers(self):
        """Test that getting same logger twice doesn't duplicate handlers"""
        from src.utils import get_logger
        
        logger1 = get_logger("duplicate_test")
        handler_count = len(logger1.handlers)
        
        logger2 = get_logger("duplicate_test")
        
        assert len(logger2.handlers) == handler_count


class TestEvaluate:
    """Tests for evaluate module"""
    
    def test_evaluate_model(self):
        """Test model evaluation function"""
        from src.evaluate import evaluate_model
        from src.model import build_cnn_model, compile_model
        
        # Create simple model
        model = build_cnn_model(input_shape=(1024, 2), num_classes=5)
        model = compile_model(model)
        
        # Create sample data
        X_test = np.random.randn(20, 1024, 2).astype(np.float32)
        Y_test = np.random.randint(0, 5, 20)
        classes = ['A', 'B', 'C', 'D', 'E']
        
        # Evaluate
        results = evaluate_model(model, X_test, Y_test, classes)
        
        # Check results structure
        assert 'accuracy' in results
        assert 'predictions' in results
        assert 'confusion_matrix' in results
        assert 0 <= results['accuracy'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
