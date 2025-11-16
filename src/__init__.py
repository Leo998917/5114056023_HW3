"""
HW3 Spam Email Classification Project
Source code package for data preprocessing, model training, and evaluation.
"""

__version__ = "0.1.0"
__author__ = "HW3 Student"

from .preprocessing import (
    load_sms_data,
    clean_text,
    tokenize_and_stem,
    vectorize_text,
    preprocess_pipeline
)

from .models import (
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

from .visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_top_features,
    plot_roc_curve,
    plot_probability_distribution,
    plot_feature_comparison
)

__all__ = [
    # Preprocessing
    'load_sms_data',
    'clean_text',
    'tokenize_and_stem',
    'vectorize_text',
    'preprocess_pipeline',
    # Model training
    'train_test_split_data',
    'train_logistic_regression',
    'train_naive_bayes',
    'train_svm',
    'evaluate_model',
    'train_models',
    'save_model',
    'load_model',
    'save_all_models',
    'compare_models',
    # Visualization
    'plot_confusion_matrix',
    'plot_metrics_comparison',
    'plot_top_features',
    'plot_roc_curve',
    'plot_probability_distribution',
    'plot_feature_comparison'
]
