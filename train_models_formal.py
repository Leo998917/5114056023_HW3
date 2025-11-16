"""
Model Training Script with Formal Dataset

This script trains all three models (Logistic Regression, Naïve Bayes, SVM)
using the formal SMS spam dataset: hw3_dataset/sms_spam_no_header.csv

The script:
1. Loads and preprocesses the formal dataset
2. Trains all three classifiers
3. Evaluates models and generates metrics
4. Saves trained models with standardized names
5. Creates visualization of results

Usage:
    python train_models_formal.py

Output:
    - models/model_logreg.pkl
    - models/model_nb.pkl
    - models/model_svm.pkl
    - Console output with metrics and statistics
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import preprocess_pipeline
from src.models import (
    train_test_split_data,
    train_logistic_regression,
    train_naive_bayes,
    train_svm,
    evaluate_model,
    compare_models
)


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_section(title):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print("-" * len(title))


def main():
    """Main training function"""
    
    print_header("SMS SPAM CLASSIFICATION - MODEL TRAINING WITH FORMAL DATASET")
    
    # ========================================================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ========================================================================
    
    print_section("STEP 1: Data Loading & Preprocessing")
    
    formal_dataset_path = 'hw3_dataset/sms_spam_no_header.csv'
    
    if not Path(formal_dataset_path).exists():
        print(f"❌ Error: Dataset not found at {formal_dataset_path}")
        return
    
    print(f"📂 Loading dataset from: {formal_dataset_path}")
    
    try:
        preprocess_result = preprocess_pipeline(formal_dataset_path, method='tfidf')
        X = preprocess_result['X']
        y = preprocess_result['y']
        feature_names = preprocess_result['feature_names']
        metadata = preprocess_result['metadata']
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total samples: {metadata['total_samples']}")
        print(f"   Ham messages: {metadata['label_distribution']['ham']}")
        print(f"   Spam messages: {metadata['label_distribution']['spam']}")
        print(f"   Features extracted: {len(feature_names)}")
        print(f"   Sparsity: {metadata['sparsity']*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 2: TRAIN-TEST SPLIT
    # ========================================================================
    
    print_section("STEP 2: Train-Test Split")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2)
        
        print(f"✅ Data split completed!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Train/Test ratio: 80/20")
        print(f"   Training spam ratio: {(y_train == 1).sum() / len(y_train):.2%}")
        print(f"   Test spam ratio: {(y_test == 1).sum() / len(y_test):.2%}")
        
    except Exception as e:
        print(f"❌ Error during train-test split: {e}")
        return
    
    # ========================================================================
    # STEP 3: TRAIN MODELS
    # ========================================================================
    
    print_section("STEP 3: Model Training")
    
    models = {}
    evaluations = {}
    
    # Train Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    try:
        model_lr = train_logistic_regression(X_train, y_train)
        models['Logistic Regression'] = model_lr
        eval_lr = evaluate_model(model_lr, X_test, y_test)
        evaluations['Logistic Regression'] = eval_lr
        print(f"      ✅ Accuracy: {eval_lr['accuracy']:.4f}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return
    
    # Train Naïve Bayes
    print("[2/3] Training Multinomial Naïve Bayes...")
    try:
        model_nb = train_naive_bayes(X_train, y_train)
        models['Naïve Bayes'] = model_nb
        eval_nb = evaluate_model(model_nb, X_test, y_test)
        evaluations['Naïve Bayes'] = eval_nb
        print(f"      ✅ Accuracy: {eval_nb['accuracy']:.4f}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return
    
    # Train SVM
    print("[3/3] Training Linear SVM...")
    try:
        model_svm = train_svm(X_train, y_train)
        models['SVM'] = model_svm
        eval_svm = evaluate_model(model_svm, X_test, y_test)
        evaluations['SVM'] = eval_svm
        print(f"      ✅ Accuracy: {eval_svm['accuracy']:.4f}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return
    
    # ========================================================================
    # STEP 4: MODEL EVALUATION
    # ========================================================================
    
    print_section("STEP 4: Model Evaluation & Metrics")
    
    # Create metrics table
    print("\nDetailed Metrics Comparison:")
    print("-" * 100)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 100)
    
    for model_name, metrics in evaluations.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['roc_auc']:<12.4f}")
    
    print("-" * 100)
    
    # Find best model
    best_model_name = max(evaluations.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = evaluations[best_model_name]['accuracy']
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    # ========================================================================
    # STEP 5: SAVE MODELS
    # ========================================================================
    
    print_section("STEP 5: Saving Trained Models")
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_mapping = {
        'Logistic Regression': 'model_logreg.pkl',
        'Naïve Bayes': 'model_nb.pkl',
        'SVM': 'model_svm.pkl'
    }
    
    saved_files = {}
    
    for model_name, filename in model_mapping.items():
        filepath = models_dir / filename
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(models[model_name], f)
            saved_files[model_name] = str(filepath)
            file_size = filepath.stat().st_size
            print(f"✅ {model_name:<25} → {filepath.name:<25} ({file_size} bytes)")
        except Exception as e:
            print(f"❌ {model_name:<25} → Error: {e}")
            return
    
    # ========================================================================
    # STEP 6: SAVE METADATA
    # ========================================================================
    
    print_section("STEP 6: Saving Training Metadata")
    
    metadata_file = models_dir / 'training_metadata.pkl'
    
    training_metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset_path': formal_dataset_path,
        'dataset_size': metadata['total_samples'],
        'ham_count': metadata['label_distribution']['ham'],
        'spam_count': metadata['label_distribution']['spam'],
        'features': len(feature_names),
        'feature_names': feature_names,
        'models_trained': list(model_mapping.keys()),
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'test_indices': None,  # Could store for reproducibility
        'evaluations': {
            name: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in evals.items() 
                   if k != 'confusion_matrix' and k != 'predictions' and k != 'probabilities'}
            for name, evals in evaluations.items()
        },
        'best_model': best_model_name
    }
    
    try:
        with open(metadata_file, 'wb') as f:
            pickle.dump(training_metadata, f)
        print(f"✅ Metadata saved to: {metadata_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save metadata: {e}")
    
    # ========================================================================
    # STEP 7: SUMMARY
    # ========================================================================
    
    print_header("TRAINING COMPLETED SUCCESSFULLY ✅")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Dataset: {formal_dataset_path}")
    print(f"   Total Messages: {metadata['total_samples']:,}")
    print(f"   Ham: {metadata['label_distribution']['ham']:,} ({metadata['label_distribution']['ham']/metadata['total_samples']*100:.1f}%)")
    print(f"   Spam: {metadata['label_distribution']['spam']:,} ({metadata['label_distribution']['spam']/metadata['total_samples']*100:.1f}%)")
    
    print(f"\n🤖 Models Trained:")
    print(f"   1. Logistic Regression  → Accuracy: {evaluations['Logistic Regression']['accuracy']:.4f}")
    print(f"   2. Naïve Bayes          → Accuracy: {evaluations['Naïve Bayes']['accuracy']:.4f}")
    print(f"   3. SVM                  → Accuracy: {evaluations['SVM']['accuracy']:.4f}")
    
    print(f"\n💾 Model Files:")
    for model_name, filepath in saved_files.items():
        print(f"   ✅ {filepath}")
    
    print(f"\n🎯 Performance Ranking:")
    sorted_models = sorted(evaluations.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_models, 1):
        print(f"   {rank}. {name:<25} Accuracy: {metrics['accuracy']:.4f}")
    
    print(f"\n📅 Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
