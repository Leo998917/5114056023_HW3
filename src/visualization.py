"""
Visualization Module for Spam Classification Model Evaluation

This module provides functions for creating visualizations of model performance,
including confusion matrices, metrics comparisons, feature importance, and ROC curves.

Functions:
    - plot_confusion_matrix: Create heatmap of confusion matrix
    - plot_metrics_comparison: Bar chart comparing metrics across models
    - plot_top_features: Bar chart of top important features
    - plot_roc_curve: ROC curve with AUC score
    - plot_probability_distribution: Histogram of prediction probabilities

Dependencies:
    - numpy
    - matplotlib.pyplot
    - seaborn
    - scikit-learn (metrics, utils)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    model_name: str = 'Model',
    normalize: bool = False,
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Create and return a confusion matrix heatmap.

    Parameters:
        cm (np.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix
        labels (List[str], optional): Class labels. Defaults to ['Ham', 'Spam']
        model_name (str): Name of the model for title. Defaults to 'Model'
        normalize (bool): Whether to normalize by true condition. Defaults to False
        cmap (str): Colormap name. Defaults to 'Blues'

    Returns:
        plt.Figure: Matplotlib figure object containing the heatmap

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> y_true = [0, 1, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 1]
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> fig = plot_confusion_matrix(cm, model_name='Logistic Regression')
        >>> plt.show()
    """
    if labels is None:
        labels = ['Ham', 'Spam']

    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={'size': 14, 'weight': 'bold'},
        cbar_kws={'label': 'Count'}
    )

    # Set labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_metrics_comparison(
    evaluations: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> plt.Figure:
    """
    Create a bar chart comparing multiple metrics across models.

    Parameters:
        evaluations (Dict): Dictionary with model names as keys and metric dicts as values
                           Expected format: {'model_name': {'accuracy': 0.95, 'precision': 0.92, ...}}
        metrics (List[str], optional): List of metrics to compare. If None, uses all available.
                                      Defaults to None

    Returns:
        plt.Figure: Matplotlib figure object containing the bar chart

    Example:
        >>> evaluations = {
        ...     'Logistic Regression': {'accuracy': 0.95, 'precision': 0.92, 'recall': 0.88, 'f1': 0.90},
        ...     'Naïve Bayes': {'accuracy': 0.90, 'precision': 0.85, 'recall': 0.92, 'f1': 0.88}
        ... }
        >>> fig = plot_metrics_comparison(evaluations)
        >>> plt.show()
    """
    if not evaluations:
        raise ValueError("evaluations dictionary cannot be empty")

    # Determine metrics to plot
    if metrics is None:
        # Get all metrics from first model, excluding non-numeric values
        first_model = next(iter(evaluations.values()))
        metrics = [k for k, v in first_model.items() 
                   if isinstance(v, (int, float)) and k != 'roc_auc']
        # Standard order
        metric_order = ['accuracy', 'precision', 'recall', 'f1']
        metrics = [m for m in metric_order if m in metrics]

    # Prepare data for plotting
    model_names = list(evaluations.keys())
    metric_data = {metric: [] for metric in metrics}

    for model_name in model_names:
        for metric in metrics:
            value = evaluations[model_name].get(metric, 0)
            metric_data[metric].append(value)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.2
    multiplier = 0

    # Plot bars for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for metric, color in zip(metrics, colors):
        offset = width * multiplier
        ax.bar(x + offset, metric_data[metric], width, label=metric.capitalize(), color=color)
        multiplier += 1

    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_top_features(
    feature_names: List[str],
    coefficients: np.ndarray,
    top_n: int = 20,
    model_name: str = 'Model'
) -> plt.Figure:
    """
    Create a bar chart of the top N most important features.

    Parameters:
        feature_names (List[str]): List of feature names
        coefficients (np.ndarray): Feature coefficients/importance scores
        top_n (int): Number of top features to display. Defaults to 20
        model_name (str): Name of the model for title. Defaults to 'Model'

    Returns:
        plt.Figure: Matplotlib figure object containing the bar chart

    Example:
        >>> features = ['buy', 'free', 'win', 'click', 'prize']
        >>> coefs = np.array([0.8, 0.7, 0.9, 0.5, 0.6])
        >>> fig = plot_top_features(features, coefs, top_n=5, model_name='Logistic Regression')
        >>> plt.show()
    """
    if len(feature_names) != len(coefficients):
        raise ValueError("feature_names and coefficients must have same length")

    if top_n > len(feature_names):
        top_n = len(feature_names)

    # Get absolute values for importance and sort
    importance = np.abs(coefficients)
    sorted_idx = np.argsort(importance)[-top_n:]

    # Get top features
    top_features = [feature_names[i] for i in sorted_idx]
    top_importance = coefficients[sorted_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create horizontal bar chart
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in top_importance]
    bars = ax.barh(top_features, top_importance, color=colors, edgecolor='black')

    # Add value labels on bars
    for i, (feature, importance_val) in enumerate(zip(top_features, top_importance)):
        ax.text(importance_val, i, f' {importance_val:.3f}', 
                va='center', fontweight='bold')

    # Customize plot
    ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Important Features - {model_name}', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = 'Model'
) -> plt.Figure:
    """
    Create a ROC curve with AUC score.

    Parameters:
        y_test (np.ndarray): True binary labels
        y_proba (np.ndarray): Predicted probabilities for positive class
        model_name (str): Name of the model for title. Defaults to 'Model'

    Returns:
        plt.Figure: Matplotlib figure object containing the ROC curve

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.7])
        >>> fig = plot_roc_curve(y_true, y_pred_proba, model_name='Logistic Regression')
        >>> plt.show()
    """
    if len(y_test) != len(y_proba):
        raise ValueError("y_test and y_proba must have same length")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='#1f77b4', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')

    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_probability_distribution(
    y_proba_dict: Dict[str, np.ndarray],
    y_test: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Create histograms of prediction probabilities for multiple models.

    Parameters:
        y_proba_dict (Dict[str, np.ndarray]): Dictionary with model names as keys
                                             and probability arrays as values
        y_test (np.ndarray, optional): True labels for coloring. Defaults to None

    Returns:
        plt.Figure: Matplotlib figure object containing the histograms

    Example:
        >>> proba_dict = {
        ...     'Logistic Regression': np.array([0.1, 0.8, 0.9, 0.2, 0.7]),
        ...     'Naïve Bayes': np.array([0.15, 0.75, 0.95, 0.25, 0.65])
        ... }
        >>> fig = plot_probability_distribution(proba_dict)
        >>> plt.show()
    """
    n_models = len(y_proba_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, y_proba) in zip(axes, y_proba_dict.items()):
        ax.hist(y_proba, bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Spam Probability', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Prediction Probability Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_feature_comparison(
    feature_dict: Dict[str, Tuple[List[str], np.ndarray]],
    top_n: int = 10
) -> plt.Figure:
    """
    Compare top features across multiple models.

    Parameters:
        feature_dict (Dict): Dictionary with model names as keys and 
                            tuples of (feature_names, coefficients) as values
        top_n (int): Number of top features to compare. Defaults to 10

    Returns:
        plt.Figure: Matplotlib figure object containing the comparison

    Example:
        >>> feature_dict = {
        ...     'LogReg': (['buy', 'free', 'win'], np.array([0.8, 0.7, 0.9])),
        ...     'SVM': (['buy', 'free', 'click'], np.array([0.7, 0.6, 0.8]))
        ... }
        >>> fig = plot_feature_comparison(feature_dict, top_n=3)
        >>> plt.show()
    """
    n_models = len(feature_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, (features, coefs)) in zip(axes, feature_dict.items()):
        importance = np.abs(coefs)
        sorted_idx = np.argsort(importance)[-top_n:]
        top_features = [features[i] for i in sorted_idx]
        top_coefs = coefs[sorted_idx]

        colors = ['#2ca02c' if x > 0 else '#d62728' for x in top_coefs]
        ax.barh(top_features, top_coefs, color=colors, edgecolor='black')
        ax.set_xlabel('Coefficient', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle(f'Top {top_n} Features Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig
