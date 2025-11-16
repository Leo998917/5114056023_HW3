"""
Streamlit Web Application for Email Spam Classification

This application provides a user-friendly interface for:
- Real-time spam classification of SMS messages
- Interactive visualizations of model performance
- Feature importance analysis
- Model comparison and metrics display

Run with: streamlit run app.py

Author: HW3 Project
Date: 2025-11-16
"""

import streamlit as st

# ============================================================================
# vvvvvvvv 強制 NLTK 資源下載 (請加在這裡) vvvvvvvv
# ============================================================================
import nltk

@st.cache_resource
def download_nltk_resources():
    """Ensures NLTK resources are available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    
    # 從您舊的 app.py 中保留，以防萬一
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass # 如果 'punkt_tab' 下載失敗，就忽略
    
    # 這是最重要的：下載 stopwords
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

# 應用程式啟動時立刻執行下載
download_nltk_resources()
# ============================================================================
# ^^^^^^^^^^ NLTK 資源下載區塊結束 ^^^^^^^^^^
# ============================================================================


import pandas as pd
import numpy as np
from pathlib import Path
import glob
import pickle
import warnings
from typing import Dict, Tuple, Optional
import nltk
import matplotlib.pyplot as plt

from src.preprocessing import preprocess_pipeline, clean_text, tokenize_and_stem, vectorize_text
from src.models import load_model, evaluate_model
from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_top_features,
    plot_roc_curve,
    plot_probability_distribution
)

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .spam-result {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .ham-result {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHED FUNCTIONS FOR PERFORMANCE
# ============================================================================

@st.cache_resource
def load_all_trained_models(models_dir: str = 'models') -> Dict:
    """
    Load all trained models from the models directory.
    
    Parameters:
        models_dir (str): Directory containing model files
        
    Returns:
        Dict: Dictionary with model names and loaded model objects
    """
    models = {}
    model_files = glob.glob(f"{models_dir}/*.pkl")
    
    if not model_files:
        st.error(f"❌ No model files found in {models_dir}/ directory")
        return models
    
    for model_file in sorted(model_files):
        try:
            model_name = Path(model_file).stem.split('_')[0]
            model = load_model(model_file)
            
            # Normalize model names
            if model_name == 'logistic':
                display_name = 'Logistic Regression'
            elif model_name == 'naive':
                display_name = 'Naïve Bayes'
            elif model_name == 'svm':
                display_name = 'SVM'
            else:
                display_name = model_name.replace('_', ' ').title()
            
            models[display_name] = model
        except Exception as e:
            st.warning(f"⚠️  Failed to load {model_file}: {e}")
    
    return models


@st.cache_resource
def load_preprocessing_data(data_path: str = 'data/sample_sms_spam.csv') -> Dict:
    """
    Load and preprocess sample data for vectorizer and evaluation.
    
    Parameters:
        data_path (str): Path to CSV file
        
    Returns:
        Dict: Preprocessing results including vectorizer and feature names
    """
    try:
        result = preprocess_pipeline(data_path, method='tfidf')
        return result
    except Exception as e:
        st.error(f"❌ Failed to preprocess data: {e}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def predict_spam(
    text: str,
    model,
    vectorizer,
    feature_names: list
) -> Tuple[str, float]:
    """
    Predict spam/ham for input text.
    
    Parameters:
        text (str): Input SMS message
        model: Trained model object
        vectorizer: Fitted vectorizer (CountVectorizer or TfidfVectorizer)
        feature_names (list): List of feature names from vectorizer
        
    Returns:
        Tuple[str, float]: (prediction, probability/confidence)
    """
    # Preprocess text
    cleaned_text = clean_text(text)
    tokens = tokenize_and_stem(cleaned_text, remove_stopwords=True)
    
    # Vectorize (note: manually vectorize using sklearn's transform)
    from sklearn.feature_extraction.text import TfidfVectorizer
    temp_vectorizer = TfidfVectorizer(vocabulary={f: i for i, f in enumerate(feature_names)})
    X_vec = temp_vectorizer.fit_transform([' '.join(tokens)])
    
    # Make prediction
    prediction = model.predict(X_vec)[0]
    
    # Get confidence/probability
    confidence = 0.5
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_vec)[0]
            confidence = max(proba)
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X_vec)[0]
            confidence = 1 / (1 + np.exp(-scores))
    except Exception:
        pass
    
    label = 'SPAM' if prediction == 1 else 'HAM'
    return label, confidence


def create_metrics_table(evaluations: Dict) -> pd.DataFrame:
    """
    Create a dataframe from model evaluations for display.
    
    Parameters:
        evaluations (Dict): Model evaluation results
        
    Returns:
        pd.DataFrame: Formatted metrics table
    """
    data = []
    for model_name, metrics in evaluations.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'Precision': f"{metrics.get('precision', 0):.3f}",
            'Recall': f"{metrics.get('recall', 0):.3f}",
            'F1-Score': f"{metrics.get('f1', 0):.3f}",
            'ROC-AUC': f"{metrics.get('roc_auc', 0):.3f}"
        }
        data.append(row)
    
    return pd.DataFrame(data)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Title and description
    st.title("📧 Email Spam Classification System")
    st.markdown("""
    Welcome to the Email Spam Classifier! This application helps identify whether 
    an email or SMS message is spam or legitimate.
    
    **Features:**
    - 🔮 Real-time spam prediction
    - 📊 Model performance visualizations
    - 📈 Feature importance analysis
    - 🎯 Multi-model comparison
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Prediction", "📊 Model Performance", "🔍 Feature Analysis", "ℹ️ About"]
    )
    
    # Load models and data
    models = load_all_trained_models()
    preprocess_data = load_preprocessing_data()
    
    if not models:
        st.error("❌ No models available. Please ensure model files are in the models/ directory.")
        return
    
    if not preprocess_data:
        st.error("❌ Could not load preprocessing data.")
        return
    
    # ========================================================================
    # PAGE 1: PREDICTION
    # ========================================================================
    
    if page == "🏠 Prediction":
        st.header("Spam Prediction")
        st.markdown("Enter an SMS message and select a model to classify it.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            user_input = st.text_area(
                "Enter SMS message:",
                placeholder="Type your message here (e.g., 'Click here to win FREE prizes NOW!')",
                height=120
            )
        
        with col2:
            # Model selection
            model_choice = st.selectbox(
                "Select Model:",
                list(models.keys()),
                help="Choose which trained model to use"
            )
        
        # Prediction button
        if st.button("🔮 Predict", use_container_width=True):
            if user_input.strip():
                with st.spinner("Analyzing message..."):
                    try:
                        selected_model = models[model_choice]
                        feature_names = preprocess_data['feature_names']
                        
                        label, confidence = predict_spam(
                            user_input,
                            selected_model,
                            None,
                            feature_names
                        )
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            if label == 'SPAM':
                                st.markdown(f"<p class='spam-result'>{label}</p>", 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p class='ham-result'>{label}</p>", 
                                          unsafe_allow_html=True)
                        
                        with result_col2:
                            st.metric("Confidence", f"{confidence*100:.1f}%")
                        
                        with result_col3:
                            st.metric("Model Used", model_choice)
                        
                        # Display processed message
                        with st.expander("📝 Processed Message"):
                            processed = clean_text(user_input)
                            tokens = tokenize_and_stem(processed, remove_stopwords=True)
                            st.write(f"**Original:** {user_input}")
                            st.write(f"**Cleaned:** {processed}")
                            st.write(f"**Tokens:** {', '.join(tokens[:20])}...")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
            else:
                st.warning("⚠️  Please enter a message to classify.")
    
    # ========================================================================
    # PAGE 2: MODEL PERFORMANCE
    # ========================================================================
    
    elif page == "📊 Model Performance":
        st.header("Model Performance Dashboard")
        
        # Load evaluation data
        try:
            X = preprocess_data['X']
            y = preprocess_data['y']
            
            # Train-test split
            from src.models import train_test_split_data
            X_train, X_test, y_train, y_test = train_test_split_data(X, y)
            
            # Evaluate all models
            evaluations = {}
            confusion_matrices = {}
            probabilities = {}
            
            with st.spinner("Evaluating models..."):
                for model_name, model in models.items():
                    eval_results = evaluate_model(model, X_test, y_test)
                    evaluations[model_name] = eval_results
                    
                    cm = eval_results.get('confusion_matrix')
                    if cm is not None:
                        confusion_matrices[model_name] = cm
                    
                    predictions = eval_results.get('predictions')
                    proba = eval_results.get('probabilities', None)
                    if proba is not None:
                        probabilities[model_name] = proba
            
            # Metrics comparison table
            st.subheader("📈 Metrics Summary")
            metrics_df = create_metrics_table(evaluations)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Confusion matrices
            st.subheader("🎯 Confusion Matrices")
            cols = st.columns(len(confusion_matrices))
            
            for col, (model_name, cm) in zip(cols, confusion_matrices.items()):
                with col:
                    fig = plot_confusion_matrix(cm, model_name=model_name)
                    st.pyplot(fig)
                    plt.close(fig)
            
            # Metrics comparison chart
            st.subheader("📊 Metrics Comparison")
            fig = plot_metrics_comparison(evaluations)
            st.pyplot(fig)
            plt.close(fig)
            
            # Probability distributions
            if probabilities:
                st.subheader("📉 Probability Distributions")
                fig = plot_probability_distribution(probabilities)
                st.pyplot(fig)
                plt.close(fig)
        
        except Exception as e:
            st.error(f"❌ Error loading evaluation data: {e}")
    
    # ========================================================================
    # PAGE 3: FEATURE ANALYSIS
    # ========================================================================
    
    elif page == "🔍 Feature Analysis":
        st.header("Feature Importance Analysis")
        
        st.markdown("""
        This section shows the most important features (words) that the models 
        rely on for spam classification.
        
        **Positive coefficients** = Indicate spam  
        **Negative coefficients** = Indicate ham
        """)
        
        try:
            feature_names = preprocess_data['feature_names']
            
            # Feature importance for Logistic Regression (has coefficients)
            for model_name, model in models.items():
                if hasattr(model, 'coef_'):
                    st.subheader(f"Top Features - {model_name}")
                    
                    coefs = model.coef_[0]
                    fig = plot_top_features(
                        feature_names,
                        coefs,
                        top_n=20,
                        model_name=model_name
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    st.divider()
        
        except Exception as e:
            st.warning(f"⚠️  Could not display feature analysis: {e}")
    
    # ========================================================================
    # PAGE 4: ABOUT
    # ========================================================================
    
    elif page == "ℹ️ About":
        st.header("About This Application")
        
        st.markdown("""
        ### Email Spam Classification System
        
        **Project:** HW3 - Email Spam Classification with OpenSpec Workflow
        
        **Date:** 2025-11-16
        
        **Description:**
        This application demonstrates a complete machine learning pipeline for email 
        spam classification. It includes data preprocessing, model training, evaluation, 
        and interactive visualizations.
        
        ### Models Included
        
        The system uses three different classifiers:
        
        1. **Logistic Regression**
           - Fast, interpretable linear classifier
           - Good baseline performance
        
        2. **Multinomial Naïve Bayes**
           - Probabilistic classifier
           - Good for text classification
        
        3. **Linear SVM**
           - Powerful non-linear classifier
           - Handles sparse data well
        
        ### Technology Stack
        
        - **Language:** Python 3.10+
        - **ML Framework:** scikit-learn
        - **Text Processing:** NLTK
        - **Web Framework:** Streamlit
        - **Visualization:** matplotlib, seaborn
        
        ### Data Preprocessing
        
        Input text goes through:
        1. Lowercase normalization
        2. Special character removal
        3. URL/email removal
        4. NLTK tokenization
        5. Porter Stemming
        6. TF-IDF Vectorization (5000 features)
        
        ### Project Structure
        
        ```
        hw3/
        ├── src/
        │   ├── preprocessing.py    # Text preprocessing
        │   ├── models.py           # Model training
        │   └── visualization.py    # Visualization functions
        ├── models/                 # Trained model files
        ├── data/                   # Sample dataset
        ├── notebooks/              # Jupyter notebooks
        ├── app.py                  # This Streamlit app
        └── README.md               # Documentation
        ```
        
        ### Features
        
        ✅ Real-time spam prediction  
        ✅ Multi-model comparison  
        ✅ Performance visualization  
        ✅ Feature importance analysis  
        ✅ Interactive interface  
        ✅ Confidence scores  
        
        ### How to Use
        
        1. Go to the **Prediction** page
        2. Enter an SMS message
        3. Select a model
        4. Click "Predict"
        5. View results and processed message
        
        For more details, visit the other pages!
        
        ---
        
        **OpenSpec Methodology:** This project follows the OpenSpec (Spec-Driven Development) 
        workflow with complete proposals, specifications, and agent implementation traces.
        """)
        
        # Display model information
        st.subheader("📦 Loaded Models")
        for model_name in models.keys():
            st.success(f"✅ {model_name}")
        
        # Display data statistics
        st.subheader("📊 Dataset Statistics")
        if preprocess_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Features", len(preprocess_data['feature_names']))
            with col2:
                st.metric("Total Samples", preprocess_data['metadata']['total_samples'])
            with col3:
                st.metric("Ham Messages", preprocess_data['metadata']['ham_count'])
            with col4:
                st.metric("Spam Messages", preprocess_data['metadata']['spam_count'])


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
