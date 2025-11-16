# Proposal 004: Add Streamlit Visualization & Interactive Interface

**Status**: Pending  
**Created**: 2025-11-16  
**Type**: Feature Addition  

---

## Title

Implement Streamlit Web Application with Visualization Dashboard for Email Spam Classification

---

## Summary

This proposal adds a comprehensive web-based interface for the spam classification system using Streamlit. The application includes real-time spam prediction, interactive visualizations (confusion matrices, performance metrics, feature importance), and model selection capabilities. Users can input SMS messages and receive instant spam/ham classification with confidence scores.

---

## Problem Statement

Currently, the spam classification system consists of backend modules for preprocessing and model training, but lacks a user-friendly interface for:

1. **End-user Interaction**: No way for non-technical users to test the classifier
2. **Model Transparency**: No visualization of model performance metrics
3. **Feature Insights**: No display of important features/words the models rely on
4. **Model Comparison**: No easy way to see differences between Logistic Regression, Naïve Bayes, and SVM
5. **Deployment Gap**: System not accessible to stakeholders without technical setup

This limits usability for demonstration, evaluation, and practical application.

---

## Proposed Solution

### 1. Visualization Module (`src/visualization.py`)

Create a dedicated visualization module with functions for:

- **Confusion Matrix Visualization**: Heatmap display of TP/TN/FP/FN for each model
- **Metrics Comparison Chart**: Bar chart comparing Accuracy, Precision, Recall, F1-Score across models
- **Feature Importance**: Bar chart of top 20 most important features (highest TF-IDF weights)
- **ROC Curve**: ROC-AUC curves for binary classification evaluation
- **Probability Distribution**: Histogram of spam probability scores

Functions:
```python
def plot_confusion_matrix(confusion_matrix, labels=['Ham', 'Spam'], model_name='Model')
def plot_metrics_comparison(evaluations_dict)
def plot_top_features(feature_names, model, top_n=20)
def plot_roc_curve(y_test, y_proba, model_name='Model')
def plot_probability_distribution(y_proba_list, model_names)
```

### 2. Streamlit Application (`app.py`)

Create a comprehensive web application with sections:

#### 2.1 Layout & Navigation
- Sidebar with navigation between sections
- Consistent styling with custom CSS theme
- Model selection dropdown

#### 2.2 Main Prediction Interface
- Text area for SMS input
- "Predict" button
- Results display showing:
  - Classification result (SPAM / HAM)
  - Confidence/Probability score
  - Selected model name
  - Formatted message display

#### 2.3 Model Performance Dashboard
- Tabs for each model (Logistic Regression, Naïve Bayes, SVM)
- For each model:
  - Confusion matrix heatmap
  - Metrics bar chart (Accuracy, Precision, Recall, F1)
  - ROC curve (if applicable)

#### 2.4 Feature Analysis
- Top 20 important features visualization
- Feature frequency/importance score
- Display as bar chart

#### 2.5 Data Statistics
- Dataset overview (total samples, ham/spam distribution)
- Feature statistics (vocabulary size, sparsity)
- Sample messages display

#### 2.6 Model Information
- Model names and algorithms
- Training parameters
- Accuracy summary table

### 3. Backend Integration

Data Flow:
```
User Input (SMS text)
    ↓
Text Preprocessing (clean_text, tokenize_and_stem)
    ↓
TF-IDF Vectorization (using saved vectorizer)
    ↓
Load Selected Model
    ↓
Make Prediction (predict + predict_proba)
    ↓
Display Results
```

Model Loading Strategy:
- Load all three trained models on app startup
- Cache models using Streamlit's `@st.cache_resource`
- Handle missing models gracefully with error messages
- Support model reload functionality

### 4. Implementation Details

#### 4.1 File: `src/visualization.py`

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple, Optional

def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None, 
                         model_name: str = 'Model') -> plt.Figure:
    """Plot confusion matrix as heatmap"""
    # Implementation with matplotlib/seaborn

def plot_metrics_comparison(evaluations: Dict[str, Dict]) -> plt.Figure:
    """Create bar chart comparing metrics across models"""
    # Implementation with matplotlib

def plot_top_features(feature_names: List[str], coefficients: np.ndarray, 
                      top_n: int = 20) -> plt.Figure:
    """Plot top N most important features"""
    # Implementation with matplotlib

def plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, 
                   model_name: str = 'Model') -> plt.Figure:
    """Plot ROC curve with AUC score"""
    # Implementation with sklearn utilities
```

#### 4.2 File: `app.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
import glob

# Import project modules
from src.preprocessing import preprocess_pipeline, clean_text, tokenize_and_stem
from src.models import load_model, evaluate_model
from src.visualization import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_top_features,
    plot_roc_curve
)

# App configuration
st.set_page_config(page_title="Spam Email Classifier", layout="wide")

# Cached functions for model loading
@st.cache_resource
def load_all_models(models_dir='models'):
    """Load all trained models with caching"""
    # Implementation

@st.cache_resource
def load_vectorizer(data_path='data/sample_sms_spam.csv'):
    """Load TF-IDF vectorizer from training data"""
    # Implementation

# Main app structure
if __name__ == "__main__":
    st.title("📧 Email Spam Classifier")
    # Sidebar setup
    # Prediction interface
    # Visualization sections
    # Performance dashboard
```

---

## Acceptance Criteria

### Functionality

- ✅ **Prediction Interface**
  - Users can input SMS text in text area
  - "Predict" button triggers classification
  - Results display spam/ham classification correctly
  - Shows selected model name

- ✅ **Probability Score**
  - For models supporting predict_proba, display confidence
  - Format as percentage (e.g., "85% spam")
  - Handle models without probability scores

- ✅ **Visualizations**
  - Confusion matrix heatmap displays correctly for each model
  - Metrics bar chart (Accuracy, Precision, Recall, F1) renders properly
  - Top 20 features visualization shows feature importance
  - ROC curves plot correctly with AUC scores

- ✅ **Model Management**
  - All three models load successfully at startup
  - Model selection dropdown works
  - Handle missing model files gracefully
  - Display which model is currently selected

### Code Quality

- ✅ Type hints on all function signatures
- ✅ Comprehensive docstrings for all functions
- ✅ Error handling for missing files/invalid input
- ✅ Code follows PEP 8 conventions
- ✅ Proper modularization (separate visualization.py)

### Testing & Validation

- ✅ Streamlit app runs without errors: `streamlit run app.py`
- ✅ Test predictions on sample messages work correctly
- ✅ All visualizations render without errors
- ✅ Handle edge cases (empty input, special characters)
- ✅ No unhandled exceptions

### User Experience

- ✅ Clean, intuitive layout
- ✅ Clear navigation between sections
- ✅ Helpful instructions for users
- ✅ Fast response time (caching of models)
- ✅ Professional appearance

### Documentation

- ✅ README updated with Streamlit app instructions
- ✅ docstrings for all visualization functions
- ✅ Comments explaining key logic
- ✅ Usage examples in docstrings

---

## Affected Files

### Files to Create

1. **`src/visualization.py`** (350+ lines)
   - Visualization functions for confusion matrix, metrics, features, ROC curves
   - Integration with matplotlib and seaborn
   - Type hints and comprehensive docstrings

2. **`app.py`** (400+ lines)
   - Main Streamlit application
   - Prediction interface and results display
   - Dashboard sections for visualizations
   - Model management and caching

3. **`tests/test_visualization.py`** (200+ lines)
   - Unit tests for visualization functions
   - Mock data generation for plot testing
   - Error handling validation

### Files to Modify

1. **`src/__init__.py`**
   - Add visualization function exports

2. **`README.md`**
   - Add Streamlit app section with usage instructions
   - Add screenshots/description of features

3. **`requirements.txt`**
   - Add streamlit dependency

---

## Agent Workflow

### Task Breakdown

1. **Create Visualization Module**
   - Implement plot_confusion_matrix function
   - Implement plot_metrics_comparison function
   - Implement plot_top_features function
   - Implement plot_roc_curve function
   - Add comprehensive tests

2. **Create Streamlit Application**
   - Setup Streamlit page configuration
   - Create sidebar with navigation
   - Implement prediction interface
   - Add visualizations and dashboards
   - Integrate with backend modules

3. **Model Integration**
   - Load trained models with proper error handling
   - Load/cache vectorizer for inference
   - Create prediction pipeline
   - Handle different model types

4. **Testing & Validation**
   - Write unit tests for visualization functions
   - Test Streamlit app locally
   - Verify all features work end-to-end
   - Test edge cases and error conditions

5. **Documentation**
   - Update README with Streamlit section
   - Add docstrings and comments
   - Create usage examples

---

## Success Metrics

- **Code**: All visualization functions working, 100% of acceptance criteria met
- **Testing**: All unit tests passing for visualization module
- **Deployment**: Streamlit app runs successfully without errors
- **User Testing**: All sample predictions work correctly, all visualizations display properly
- **Documentation**: Complete with examples and deployment instructions

---

## Timeline

- **Phase 1**: Create visualization module (src/visualization.py) + tests
- **Phase 2**: Create Streamlit app (app.py) with prediction interface
- **Phase 3**: Integrate visualizations into app
- **Phase 4**: Testing and refinement
- **Phase 5**: Documentation updates

---

## Dependencies

- streamlit >= 1.28.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0 (for metrics utilities)

---

## Notes

- Models are assumed to be saved in `models/` directory with timestamp-based naming
- TF-IDF vectorizer will be regenerated from training data or loaded from pickle
- Streamlit's caching will improve performance on repeated predictions
- App will be designed for local development initially, with Streamlit Cloud deployment as future step

---

## References

- Streamlit Documentation: https://docs.streamlit.io/
- Matplotlib Visualization: https://matplotlib.org/
- Seaborn Heatmaps: https://seaborn.pydata.org/
- OpenSpec Methodology: See project.md for full context
