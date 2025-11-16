"""
Model Training and Evaluation Module for SMS Spam Classification

This module provides functions for training multiple ML classifiers,
evaluating their performance, and persisting models to disk.
"""

import os
import pickle
import joblib
from typing import Dict, Tuple, List, Any
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)
import warnings

warnings.filterwarnings('ignore')


def train_test_split_data(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Split data into training and testing sets with stratification.

    Args:
        X: Feature matrix (sparse or dense)
        y: Label array
        test_size (float): Proportion of data for testing (default: 0.2)
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class balance
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train,
    y_train,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train Logistic Regression classifier.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        max_iter (int): Maximum iterations (default: 1000)
        random_state (int): Random seed

    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train, alpha: float = 1.0) -> MultinomialNB:
    """
    Train Multinomial Naïve Bayes classifier.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        alpha (float): Laplace smoothing parameter (default: 1.0)

    Returns:
        Trained MultinomialNB model
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_svm(
    X_train,
    y_train,
    max_iter: int = 2000,
    random_state: int = 42
) -> LinearSVC:
    """
    Train Linear SVM classifier.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        max_iter (int): Maximum iterations (default: 2000)
        random_state (int): Random seed

    Returns:
        Trained LinearSVC model
    """
    model = LinearSVC(
        random_state=random_state,
        max_iter=max_iter,
        dual=False,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained classifier
        X_test: Test feature matrix
        y_test: Test labels
        model_name (str): Name of the model for reporting

    Returns:
        Dict with metrics:
        - accuracy, precision, recall, f1_score
        - roc_auc (for binary classification)
        - confusion_matrix
        - classification_report
        - predictions
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Compute ROC-AUC for binary classification
    try:
        # Get probability estimates
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # For SVM, use decision function
            y_proba = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception as e:
        roc_auc = None

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Generate classification report
    class_report = classification_report(
        y_test, y_pred,
        target_names=['Ham', 'Spam'],
        zero_division=0
    )

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'predictions': y_pred,
        'y_test': y_test
    }


def train_models(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Train all three models and evaluate them.

    Args:
        X: Feature matrix
        y: Label array
        test_size (float): Proportion for testing
        random_state (int): Random seed

    Returns:
        Dict with structure:
        {
            'models': {
                'logistic_regression': model,
                'naive_bayes': model,
                'svm': model
            },
            'evaluations': {
                'logistic_regression': metrics_dict,
                'naive_bayes': metrics_dict,
                'svm': metrics_dict
            },
            'train_test_split': {
                'X_train', 'X_test', 'y_train', 'y_test'
            }
        }
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train models
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    print("Training Multinomial Naïve Bayes...")
    nb_model = train_naive_bayes(X_train, y_train)

    print("Training Linear SVM...")
    svm_model = train_svm(X_train, y_train)

    # Evaluate models
    print("\nEvaluating models...")
    lr_eval = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    nb_eval = evaluate_model(nb_model, X_test, y_test, "Multinomial Naïve Bayes")
    svm_eval = evaluate_model(svm_model, X_test, y_test, "Linear SVM")

    return {
        'models': {
            'logistic_regression': lr_model,
            'naive_bayes': nb_model,
            'svm': svm_model
        },
        'evaluations': {
            'logistic_regression': lr_eval,
            'naive_bayes': nb_eval,
            'svm': svm_eval
        },
        'train_test_split': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        },
        'data_info': {
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'feature_dimension': X_train.shape[1],
            'train_ham': sum(y_train == 0),
            'train_spam': sum(y_train == 1),
            'test_ham': sum(y_test == 0),
            'test_spam': sum(y_test == 1)
        }
    }


def save_model(
    model,
    filename: str,
    models_dir: str = 'models'
) -> str:
    """
    Save trained model to disk using joblib.

    Args:
        model: Trained model to save
        filename (str): Name of the file (e.g., 'logistic_regression.pkl')
        models_dir (str): Directory to save models (default: 'models')

    Returns:
        str: Path to saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Add timestamp to filename
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(models_dir, f"{name}_{timestamp}{ext}")

    # Save model
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

    return filepath


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.

    Args:
        filepath (str): Path to saved model

    Returns:
        Loaded model object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)
    return model


def save_all_models(
    models_dict: Dict,
    models_dir: str = 'models'
) -> Dict[str, str]:
    """
    Save all trained models to disk.

    Args:
        models_dict (Dict): Dict of models from train_models()
        models_dir (str): Directory to save models

    Returns:
        Dict mapping model names to saved file paths
    """
    saved_paths = {}

    for model_name, model in models_dict.items():
        filename = f"{model_name}.pkl"
        filepath = save_model(model, filename, models_dir)
        saved_paths[model_name] = filepath

    return saved_paths


def compare_models(evaluations: Dict) -> None:
    """
    Print comparison of all model evaluations.

    Args:
        evaluations (Dict): Evaluation results from train_models()
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    metrics_data = []
    for model_name, eval_dict in evaluations.items():
        metrics_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{eval_dict['accuracy']:.4f}",
            'Precision': f"{eval_dict['precision']:.4f}",
            'Recall': f"{eval_dict['recall']:.4f}",
            'F1-Score': f"{eval_dict['f1_score']:.4f}",
            'ROC-AUC': f"{eval_dict['roc_auc']:.4f}" if eval_dict['roc_auc'] else "N/A"
        })

    # Print table
    print("\nMetrics Comparison:")
    print("-" * 80)
    headers = metrics_data[0].keys()
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 80)

    for data in metrics_data:
        print(f"{data['Model']:<25} {data['Accuracy']:<12} {data['Precision']:<12} {data['Recall']:<12} {data['F1-Score']:<12} {data['ROC-AUC']:<12}")

    print("=" * 80)


if __name__ == '__main__':
    print("Model Training Module")
    print("-" * 50)
    print("Available functions:")
    print("  - train_test_split_data(X, y)")
    print("  - train_logistic_regression(X_train, y_train)")
    print("  - train_naive_bayes(X_train, y_train)")
    print("  - train_svm(X_train, y_train)")
    print("  - evaluate_model(model, X_test, y_test)")
    print("  - train_models(X, y)")
    print("  - save_model(model, filename)")
    print("  - load_model(filepath)")
    print("  - compare_models(evaluations)")
