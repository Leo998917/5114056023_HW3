"""
Unit Tests for Model Training and Evaluation Module

Tests cover:
- Train/test splitting
- Model training for all three classifiers
- Evaluation metrics computation
- Model persistence (save/load)
- Comparison and reporting
"""

import pytest
import tempfile
import os
import numpy as np
import sys
from scipy.sparse import csr_matrix

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    train_test_split_data,
    train_logistic_regression,
    train_naive_bayes,
    train_svm,
    evaluate_model,
    train_models,
    save_model,
    load_model,
    save_all_models,
    compare_models
)


class TestDataSplitting:
    """Test train/test splitting functionality."""

    def test_split_creates_correct_sizes(self):
        """Test that split creates correct train/test sizes."""
        # Create sample data
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split_data(
            X, y, test_size=0.2, random_state=42
        )

        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_split_maintains_features(self):
        """Test that split maintains feature dimension."""
        X = np.random.rand(100, 15)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        assert X_train.shape[1] == 15
        assert X_test.shape[1] == 15

    def test_split_with_custom_test_size(self):
        """Test split with custom test size."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = train_test_split_data(
            X, y, test_size=0.3
        )

        assert X_train.shape[0] == 70
        assert X_test.shape[0] == 30

    def test_split_maintains_stratification(self):
        """Test that split maintains class balance."""
        # Create imbalanced data
        y = np.array([0] * 80 + [1] * 20)
        X = np.random.rand(100, 10)

        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        # Check approximate class distribution is maintained
        train_ham_ratio = sum(y_train == 0) / len(y_train)
        test_ham_ratio = sum(y_test == 0) / len(y_test)

        # Should be approximately 0.8
        assert 0.7 < train_ham_ratio < 0.9
        assert 0.7 < test_ham_ratio < 0.9


class TestModelTraining:
    """Test model training functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = csr_matrix(np.random.rand(50, 20))
        y = np.random.randint(0, 2, 50)
        return X, y

    def test_logistic_regression_trains(self, sample_data):
        """Test Logistic Regression training."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        model = train_logistic_regression(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_naive_bayes_trains(self, sample_data):
        """Test Multinomial Naïve Bayes training."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        model = train_naive_bayes(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_svm_trains(self, sample_data):
        """Test Linear SVM training."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        model = train_svm(X_train, y_train)

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'decision_function')

    def test_model_makes_predictions(self, sample_data):
        """Test that trained models can make predictions."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        # Train a model
        model = train_logistic_regression(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        assert len(predictions) == X_test.shape[0]
        assert all(p in [0, 1] for p in predictions)


class TestModelEvaluation:
    """Test model evaluation functionality."""

    @pytest.fixture
    def sample_evaluation_data(self):
        """Create sample data and trained model for evaluation."""
        X = csr_matrix(np.random.rand(50, 20))
        y = np.random.randint(0, 2, 50)

        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        model = train_logistic_regression(X_train, y_train)

        return model, X_test, y_test

    def test_evaluate_model_returns_dict(self, sample_evaluation_data):
        """Test evaluate_model returns expected dict."""
        model, X_test, y_test = sample_evaluation_data

        results = evaluate_model(model, X_test, y_test)

        assert isinstance(results, dict)
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'confusion_matrix' in results

    def test_evaluate_model_metrics_valid(self, sample_evaluation_data):
        """Test that evaluation metrics are valid."""
        model, X_test, y_test = sample_evaluation_data

        results = evaluate_model(model, X_test, y_test)

        # Check metric ranges
        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1_score'] <= 1

    def test_confusion_matrix_shape(self, sample_evaluation_data):
        """Test confusion matrix has correct shape."""
        model, X_test, y_test = sample_evaluation_data

        results = evaluate_model(model, X_test, y_test)

        conf_matrix = results['confusion_matrix']
        assert conf_matrix.shape == (2, 2)


class TestTrainAllModels:
    """Test full training pipeline."""

    def test_train_models_returns_dict(self):
        """Test train_models returns expected structure."""
        X = csr_matrix(np.random.rand(100, 20))
        y = np.random.randint(0, 2, 100)

        result = train_models(X, y)

        assert 'models' in result
        assert 'evaluations' in result
        assert 'train_test_split' in result
        assert 'data_info' in result

    def test_train_models_trains_all_three(self):
        """Test all three models are trained."""
        X = csr_matrix(np.random.rand(100, 20))
        y = np.random.randint(0, 2, 100)

        result = train_models(X, y)

        models = result['models']
        assert 'logistic_regression' in models
        assert 'naive_bayes' in models
        assert 'svm' in models

    def test_train_models_evaluates_all_three(self):
        """Test all three models are evaluated."""
        X = csr_matrix(np.random.rand(100, 20))
        y = np.random.randint(0, 2, 100)

        result = train_models(X, y)

        evals = result['evaluations']
        assert 'logistic_regression' in evals
        assert 'naive_bayes' in evals
        assert 'svm' in evals

    def test_data_info_is_correct(self):
        """Test data_info contains correct statistics."""
        X = csr_matrix(np.random.rand(100, 20))
        y = np.array([0] * 70 + [1] * 30)

        result = train_models(X, y, test_size=0.2)

        data_info = result['data_info']
        assert data_info['train_size'] == 80
        assert data_info['test_size'] == 20
        assert data_info['feature_dimension'] == 20


class TestModelPersistence:
    """Test saving and loading models."""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing."""
        X = csr_matrix(np.random.rand(50, 20))
        y = np.random.randint(0, 2, 50)

        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        model = train_logistic_regression(X_train, y_train)

        return model

    def test_save_model_creates_file(self, trained_model):
        """Test save_model creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_model(trained_model, 'test_model.pkl', tmpdir)

            assert os.path.exists(filepath)
            assert filepath.endswith('.pkl')

    def test_load_model_restores_model(self, trained_model):
        """Test load_model restores a saved model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            filepath = save_model(trained_model, 'test_model.pkl', tmpdir)

            # Load model
            loaded_model = load_model(filepath)

            # Verify it's a valid model
            assert hasattr(loaded_model, 'predict')

    def test_loaded_model_makes_same_predictions(self, trained_model):
        """Test loaded model makes same predictions as original."""
        X = csr_matrix(np.random.rand(20, 20))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Get original predictions
            original_pred = trained_model.predict(X)

            # Save and load
            filepath = save_model(trained_model, 'test_model.pkl', tmpdir)
            loaded_model = load_model(filepath)

            # Get predictions from loaded model
            loaded_pred = loaded_model.predict(X)

            # Should be identical
            assert np.array_equal(original_pred, loaded_pred)

    def test_save_all_models(self):
        """Test saving all models at once."""
        X = csr_matrix(np.random.rand(50, 20))
        y = np.random.randint(0, 2, 50)

        result = train_models(X, y)
        models = result['models']

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_paths = save_all_models(models, tmpdir)

            assert len(saved_paths) == 3
            assert all(os.path.exists(path) for path in saved_paths.values())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_load_nonexistent_model_raises_error(self):
        """Test loading nonexistent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model('/nonexistent/path/model.pkl')

    def test_split_with_small_dataset(self):
        """Test splitting with small dataset."""
        X = csr_matrix(np.random.rand(10, 5))
        y = np.random.randint(0, 2, 10)

        X_train, X_test, y_train, y_test = train_test_split_data(
            X, y, test_size=0.2
        )

        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2

    def test_train_models_with_imbalanced_data(self):
        """Test training with imbalanced data."""
        X = csr_matrix(np.random.rand(100, 20))
        # Highly imbalanced: 95% class 0, 5% class 1
        y = np.array([0] * 95 + [1] * 5)

        result = train_models(X, y)

        # Should still train and evaluate successfully
        assert result['models'] is not None
        assert result['evaluations'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
