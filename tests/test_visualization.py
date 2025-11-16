"""
Unit tests for visualization module

Tests for plot_confusion_matrix, plot_metrics_comparison, plot_top_features,
plot_roc_curve, and plot_probability_distribution functions.

Run with: pytest tests/test_visualization.py -v
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Dict

from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_top_features,
    plot_roc_curve,
    plot_probability_distribution,
    plot_feature_comparison
)


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_confusion_matrix():
    """Create sample confusion matrix"""
    return np.array([[45, 5], [10, 40]])


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary"""
    return {
        'Logistic Regression': {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1': 0.90,
            'roc_auc': 0.96
        },
        'Naïve Bayes': {
            'accuracy': 0.90,
            'precision': 0.85,
            'recall': 0.92,
            'f1': 0.88,
            'roc_auc': 0.93
        },
        'SVM': {
            'accuracy': 0.93,
            'precision': 0.91,
            'recall': 0.89,
            'f1': 0.90,
            'roc_auc': 0.94
        }
    }


@pytest.fixture
def sample_features():
    """Create sample feature names and coefficients"""
    return (
        ['buy', 'free', 'win', 'click', 'prize', 'limited', 'offer', 
         'now', 'cash', 'urgent', 'confirm', 'verify', 'update', 'act'],
        np.array([0.8, 0.9, 0.85, 0.6, 0.7, 0.75, 0.65, 0.5, 0.55, 0.6,
                  0.45, 0.4, 0.35, 0.3])
    )


@pytest.fixture
def sample_binary_data():
    """Create sample binary classification data"""
    y_test = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.15, 0.85, 0.95, 0.05, 0.1,
                        0.75, 0.88, 0.25, 0.92, 0.12])
    return y_test, y_proba


@pytest.fixture
def sample_probabilities_dict():
    """Create sample probabilities for multiple models"""
    return {
        'Logistic Regression': np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.15, 0.85]),
        'Naïve Bayes': np.array([0.15, 0.75, 0.95, 0.25, 0.65, 0.1, 0.9]),
        'SVM': np.array([0.12, 0.82, 0.88, 0.18, 0.72, 0.14, 0.84])
    }


# ============================================================================
# TESTS: plot_confusion_matrix
# ============================================================================

class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function"""

    def test_basic_confusion_matrix(self, sample_confusion_matrix):
        """Test basic confusion matrix plotting"""
        fig = plot_confusion_matrix(sample_confusion_matrix)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_confusion_matrix_with_custom_labels(self, sample_confusion_matrix):
        """Test confusion matrix with custom labels"""
        labels = ['Legitimate', 'Spam']
        fig = plot_confusion_matrix(sample_confusion_matrix, labels=labels)
        assert fig is not None
        plt.close(fig)

    def test_confusion_matrix_with_model_name(self, sample_confusion_matrix):
        """Test confusion matrix with model name"""
        fig = plot_confusion_matrix(sample_confusion_matrix, model_name='Test Model')
        assert fig is not None
        plt.close(fig)

    def test_confusion_matrix_normalized(self, sample_confusion_matrix):
        """Test normalized confusion matrix"""
        fig = plot_confusion_matrix(sample_confusion_matrix, normalize=True)
        assert fig is not None
        plt.close(fig)

    def test_confusion_matrix_3x3(self):
        """Test confusion matrix with 3x3 (multi-class)"""
        cm = np.array([[30, 5, 0], [2, 40, 8], [1, 3, 50]])
        fig = plot_confusion_matrix(cm)
        assert fig is not None
        plt.close(fig)

    def test_confusion_matrix_invalid_input(self):
        """Test invalid confusion matrix input"""
        # Should raise error when input is None
        with pytest.raises((TypeError, ValueError, AttributeError)):
            try:
                fig = plot_confusion_matrix(None)
                plt.close(fig)
            except TypeError:
                pass  # Expected


# ============================================================================
# TESTS: plot_metrics_comparison
# ============================================================================

class TestPlotMetricsComparison:
    """Tests for plot_metrics_comparison function"""

    def test_basic_metrics_comparison(self, sample_metrics):
        """Test basic metrics comparison"""
        fig = plot_metrics_comparison(sample_metrics)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_metrics_comparison_subset(self, sample_metrics):
        """Test metrics comparison with subset of metrics"""
        metrics = ['accuracy', 'precision', 'recall']
        fig = plot_metrics_comparison(sample_metrics, metrics=metrics)
        assert fig is not None
        plt.close(fig)

    def test_metrics_comparison_single_metric(self, sample_metrics):
        """Test metrics comparison with single metric"""
        fig = plot_metrics_comparison(sample_metrics, metrics=['accuracy'])
        assert fig is not None
        plt.close(fig)

    def test_metrics_comparison_empty_dict(self):
        """Test empty metrics dictionary"""
        with pytest.raises(ValueError):
            fig = plot_metrics_comparison({})

    def test_metrics_comparison_missing_metrics(self):
        """Test with missing metrics in some models"""
        metrics = {
            'Model A': {'accuracy': 0.9, 'precision': 0.85},
            'Model B': {'accuracy': 0.88}
        }
        fig = plot_metrics_comparison(metrics)
        assert fig is not None
        plt.close(fig)


# ============================================================================
# TESTS: plot_top_features
# ============================================================================

class TestPlotTopFeatures:
    """Tests for plot_top_features function"""

    def test_basic_top_features(self, sample_features):
        """Test basic top features plotting"""
        feature_names, coefs = sample_features
        fig = plot_top_features(feature_names, coefs, top_n=5)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_features_all(self, sample_features):
        """Test plotting all features"""
        feature_names, coefs = sample_features
        fig = plot_top_features(feature_names, coefs, top_n=len(feature_names))
        assert fig is not None
        plt.close(fig)

    def test_top_features_more_than_available(self, sample_features):
        """Test requesting more features than available"""
        feature_names, coefs = sample_features
        fig = plot_top_features(feature_names, coefs, top_n=100)
        assert fig is not None
        plt.close(fig)

    def test_top_features_with_model_name(self, sample_features):
        """Test with custom model name"""
        feature_names, coefs = sample_features
        fig = plot_top_features(feature_names, coefs, model_name='Logistic Regression')
        assert fig is not None
        plt.close(fig)

    def test_top_features_negative_coefficients(self):
        """Test with negative coefficients"""
        features = ['good', 'great', 'excellent', 'bad', 'spam']
        coefs = np.array([0.8, 0.7, 0.9, -0.6, -0.8])
        fig = plot_top_features(features, coefs, top_n=5)
        assert fig is not None
        plt.close(fig)

    def test_top_features_mismatched_lengths(self):
        """Test mismatched feature names and coefficients"""
        features = ['a', 'b', 'c']
        coefs = np.array([0.5, 0.6])
        with pytest.raises(ValueError):
            fig = plot_top_features(features, coefs)


# ============================================================================
# TESTS: plot_roc_curve
# ============================================================================

class TestPlotRocCurve:
    """Tests for plot_roc_curve function"""

    def test_basic_roc_curve(self, sample_binary_data):
        """Test basic ROC curve plotting"""
        y_test, y_proba = sample_binary_data
        fig = plot_roc_curve(y_test, y_proba)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roc_curve_with_model_name(self, sample_binary_data):
        """Test ROC curve with model name"""
        y_test, y_proba = sample_binary_data
        fig = plot_roc_curve(y_test, y_proba, model_name='Logistic Regression')
        assert fig is not None
        plt.close(fig)

    def test_roc_curve_perfect_predictions(self):
        """Test ROC curve with perfect predictions"""
        y_test = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.1, 0.9, 1.0])
        fig = plot_roc_curve(y_test, y_proba)
        assert fig is not None
        plt.close(fig)

    def test_roc_curve_random_predictions(self):
        """Test ROC curve with random-like predictions"""
        y_test = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        fig = plot_roc_curve(y_test, y_proba)
        assert fig is not None
        plt.close(fig)

    def test_roc_curve_mismatched_lengths(self):
        """Test mismatched y_test and y_proba"""
        y_test = np.array([0, 1, 0])
        y_proba = np.array([0.1, 0.8])
        with pytest.raises(ValueError):
            fig = plot_roc_curve(y_test, y_proba)


# ============================================================================
# TESTS: plot_probability_distribution
# ============================================================================

class TestPlotProbabilityDistribution:
    """Tests for plot_probability_distribution function"""

    def test_basic_probability_distribution(self, sample_probabilities_dict):
        """Test basic probability distribution"""
        fig = plot_probability_distribution(sample_probabilities_dict)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_model_probability_distribution(self):
        """Test with single model"""
        proba_dict = {'Model A': np.array([0.1, 0.8, 0.9, 0.2])}
        fig = plot_probability_distribution(proba_dict)
        assert fig is not None
        plt.close(fig)

    def test_probability_distribution_with_labels(self, sample_probabilities_dict):
        """Test probability distribution with true labels"""
        y_test = np.array([0, 1, 1, 0, 1, 0, 1])
        fig = plot_probability_distribution(sample_probabilities_dict, y_test=y_test)
        assert fig is not None
        plt.close(fig)

    def test_empty_probability_dict(self):
        """Test empty probability dictionary"""
        fig = plot_probability_distribution({})
        assert fig is not None
        plt.close(fig)


# ============================================================================
# TESTS: plot_feature_comparison
# ============================================================================

class TestPlotFeatureComparison:
    """Tests for plot_feature_comparison function"""

    def test_basic_feature_comparison(self):
        """Test basic feature comparison"""
        feature_dict = {
            'Model A': (
                ['buy', 'free', 'win', 'prize'],
                np.array([0.8, 0.7, 0.9, 0.6])
            ),
            'Model B': (
                ['buy', 'click', 'now', 'urgent'],
                np.array([0.75, 0.65, 0.7, 0.55])
            )
        }
        fig = plot_feature_comparison(feature_dict, top_n=3)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_feature_comparison_single_model(self):
        """Test feature comparison with single model"""
        feature_dict = {
            'Model A': (['a', 'b', 'c'], np.array([0.5, 0.6, 0.7]))
        }
        fig = plot_feature_comparison(feature_dict)
        assert fig is not None
        plt.close(fig)

    def test_feature_comparison_top_n_larger_than_features(self):
        """Test when top_n is larger than available features"""
        feature_dict = {
            'Model': (['a', 'b'], np.array([0.5, 0.6]))
        }
        fig = plot_feature_comparison(feature_dict, top_n=5)
        assert fig is not None
        plt.close(fig)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestVisualizationIntegration:
    """Integration tests for visualization module"""

    def test_all_visualizations_together(self, sample_confusion_matrix, 
                                         sample_metrics, sample_features,
                                         sample_binary_data):
        """Test creating all visualizations"""
        # Confusion matrix
        fig1 = plot_confusion_matrix(sample_confusion_matrix)
        
        # Metrics comparison
        fig2 = plot_metrics_comparison(sample_metrics)
        
        # Top features
        feature_names, coefs = sample_features
        fig3 = plot_top_features(feature_names, coefs, top_n=10)
        
        # ROC curve
        y_test, y_proba = sample_binary_data
        fig4 = plot_roc_curve(y_test, y_proba)
        
        # All should be valid figures
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
        assert fig4 is not None
        
        # Close all figures
        plt.close('all')

    def test_visualization_memory_cleanup(self, sample_confusion_matrix):
        """Test that visualizations don't leak memory"""
        initial_count = len(plt.get_fignums())
        
        for i in range(5):
            fig = plot_confusion_matrix(sample_confusion_matrix)
            plt.close(fig)
        
        final_count = len(plt.get_fignums())
        assert final_count == initial_count


# ============================================================================
# EDGE CASES
# ============================================================================

class TestVisualizationEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_array_handling(self):
        """Test handling of empty arrays"""
        with pytest.raises((ValueError, IndexError)):
            plot_top_features([], np.array([]))

    def test_single_feature(self):
        """Test with single feature"""
        fig = plot_top_features(['feature'], np.array([0.5]), top_n=1)
        assert fig is not None
        plt.close(fig)

    def test_all_zero_coefficients(self):
        """Test with all zero coefficients"""
        features = ['a', 'b', 'c']
        coefs = np.array([0.0, 0.0, 0.0])
        fig = plot_top_features(features, coefs)
        assert fig is not None
        plt.close(fig)

    def test_nan_handling(self):
        """Test handling of NaN values"""
        features = ['a', 'b', 'c']
        coefs = np.array([0.5, np.nan, 0.7])
        # Should either handle gracefully or raise error
        try:
            fig = plot_top_features(features, coefs)
            plt.close(fig)
        except (ValueError, RuntimeError):
            pass  # Expected for NaN values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
