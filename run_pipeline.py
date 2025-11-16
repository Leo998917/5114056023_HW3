#!/usr/bin/env python
"""
HW3 SMS Spam Classification - Complete Pipeline Demo

This script demonstrates the complete workflow:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Model persistence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing import preprocess_pipeline
from src.models import train_models, save_all_models, compare_models


def main():
    """Run the complete SMS spam classification pipeline."""

    print("=" * 80)
    print("HW3 SMS SPAM CLASSIFICATION - COMPLETE PIPELINE")
    print("=" * 80)

    # Step 1: Preprocess Data
    print("\n[STEP 1] Data Preprocessing")
    print("-" * 80)
    csv_path = 'data/sample_sms_spam.csv'

    try:
        result = preprocess_pipeline(csv_path, method='tfidf', min_df=1)
        X = result['X']
        y = result['y']
        feature_names = result['feature_names']
        metadata = result['metadata']

        print(f"✓ Data loaded and preprocessed")
        print(f"  - Messages: {metadata['total_samples']}")
        print(f"  - Features: {metadata['feature_dimension']}")
        print(f"  - Ham: {metadata['label_distribution']['ham']}")
        print(f"  - Spam: {metadata['label_distribution']['spam']}")
        print(f"  - Sparsity: {metadata['sparsity']:.2%}")

    except Exception as e:
        print(f"✗ Error during preprocessing: {e}")
        return

    # Step 2: Train Models
    print("\n[STEP 2] Model Training")
    print("-" * 80)

    try:
        training_result = train_models(X, y, test_size=0.2, random_state=42)
        models = training_result['models']
        evaluations = training_result['evaluations']
        data_info = training_result['data_info']

        print(f"✓ All models trained successfully")
        print(f"  - Training samples: {data_info['train_size']}")
        print(f"  - Test samples: {data_info['test_size']}")

    except Exception as e:
        print(f"✗ Error during model training: {e}")
        return

    # Step 3: Evaluate Models
    print("\n[STEP 3] Model Evaluation")
    print("-" * 80)

    try:
        compare_models(evaluations)

    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        return

    # Step 4: Save Models
    print("\n[STEP 4] Model Persistence")
    print("-" * 80)

    try:
        saved_paths = save_all_models(models, models_dir='models')
        print(f"✓ All models saved successfully")
        for model_name, path in saved_paths.items():
            print(f"  - {model_name}: {os.path.basename(path)}")

    except Exception as e:
        print(f"✗ Error during model saving: {e}")
        return

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Open notebooks/01_data_exploration.ipynb for detailed preprocessing analysis")
    print("2. Open notebooks/02_model_training.ipynb for detailed training results")
    print("3. Deploy the best model to Streamlit for interactive predictions")
    print("4. Monitor model performance on real data")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
