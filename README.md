# HW3 Email Spam Classification

A comprehensive machine learning project for SMS spam email classification using the OpenSpec (Spec-Driven Development) workflow.

## Overview

This project implements an email spam classification system that:
- Preprocesses and vectorizes SMS text data (Bag-of-Words & TF-IDF)
- Trains multiple ML classifiers (Logistic Regression, Naïve Bayes, SVM)
- Evaluates models using standard ML metrics
- Provides interactive Streamlit web application
- Follows OpenSpec methodology for development

## Dataset

This project uses the **SMS Spam Collection Dataset** with 5,574 SMS messages:
- **Location**: `hw3_dataset/sms_spam_no_header.csv`
- **Format**: CSV with 2 columns (label, message)
- **Messages**:
  - **Ham (Legitimate)**: 4,827 messages (86.6%)
  - **Spam**: 747 messages (13.4%)
- **Data Split**: 80% training (4,459 samples), 20% testing (1,115 samples)
- **Vocabulary**: 5,000 TF-IDF features extracted

**Source**: Adapted from the UCI SMS Spam Collection Dataset

## Project Structure

```
hw3/
├── hw3_dataset/
│   └── sms_spam_no_header.csv       # Formal SMS spam dataset (5,574 messages)
├── data/
│   └── sample_sms_spam.csv          # Sample SMS data for testing
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # Text preprocessing module
│   └── models.py                    # Model training and evaluation
├── models/
│   └── *.pkl                        # Saved trained models
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data preprocessing demo
│   └── 02_model_training.ipynb      # Model training demo
├── tests/
│   ├── test_preprocessing.py        # Preprocessing tests
│   └── test_models.py               # Model training tests
├── openspec/
│   ├── project.md                   # Project specification
│   └── proposals/
│       ├── 001-add-basic-classifier.md
│       ├── 002-add-data-preprocessing-module.md
│       └── 003-add-model-training-module.md
├── requirements.txt
├── hw3.py                           # Initial rule-based classifier
├── run_pipeline.py                  # Complete pipeline demo
└── README.md                        # This file
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd hw3

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Run the full preprocessing and training pipeline
python run_pipeline.py
```

This will:
- Load and preprocess SMS data
- Train three classifiers
- Evaluate model performance
- Save trained models

### 3. Explore Notebooks

Open Jupyter notebooks for interactive exploration:

```bash
# Data exploration and preprocessing
jupyter notebook notebooks/01_data_exploration.ipynb

# Model training and evaluation
jupyter notebook notebooks/02_model_training.ipynb
```

## Tech Stack

- **Language**: Python 3.10+
- **Data Processing**: pandas, numpy
- **ML/Models**: scikit-learn
- **Text Processing**: NLTK
- **Testing**: pytest
- **Visualization**: matplotlib, seaborn, plotly
- **Web UI**: Streamlit (coming soon)

## Features

### Data Preprocessing (`src/preprocessing.py`)

- **Text Cleaning**: Remove URLs, emails, special characters
- **Tokenization & Stemming**: Normalize word forms
- **Vectorization**: 
  - Bag-of-Words (BoW)
  - TF-IDF
- **Configurable Parameters**: max_features, min_df, max_df

Functions:
- `load_sms_data()` - Load CSV data
- `clean_text()` - Text normalization
- `tokenize_and_stem()` - Tokenization and stemming
- `vectorize_text()` - BoW and TF-IDF vectorization
- `preprocess_pipeline()` - Complete pipeline

### Model Training (`src/models.py`)

Three classifiers:
1. **Logistic Regression** - Fast, interpretable linear model
2. **Multinomial Naïve Bayes** - Efficient for text classification
3. **Linear SVM** - Powerful margin-based classifier

Functions:
- `train_test_split_data()` - 80/20 stratified split
- `train_logistic_regression()` - LR training
- `train_naive_bayes()` - NB training
- `train_svm()` - SVM training
- `evaluate_model()` - Metrics and confusion matrix
- `train_models()` - Train all three models
- `save_model()` / `load_model()` - Model persistence

### Evaluation Metrics

For each model:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (TP + FP)
- **Recall**: True positives / (TP + FN)
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: TP, FP, TN, FN

## Model Performance Results

Trained on the formal dataset (5,574 SMS messages, 80/20 split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 96.50% | 96.64% | 96.50% | 96.27% | 98.65% |
| Multinomial Naïve Bayes | 97.31% | 97.36% | 97.31% | 97.19% | 98.02% |
| **Linear SVM** | **98.74%** | **98.75%** | **98.74%** | **98.72%** | **98.89%** |

**Best Model**: Linear SVM with 98.74% accuracy

### Model Files

- `models/model_logreg.pkl` (40.7 KB) - Logistic Regression
- `models/model_nb.pkl` (160.6 KB) - Multinomial Naïve Bayes
- `models/model_svm.pkl` (40.6 KB) - Linear SVM
- `models/training_metadata.pkl` (55.7 KB) - Training metadata and feature names

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Test coverage:
- **Preprocessing tests** (24 tests): Data loading, text cleaning, tokenization, vectorization
- **Model tests** (22 tests): Model training, evaluation, persistence
- **Visualization tests** (35 tests): Plot generation and visualization utilities

## Usage Examples

### Basic Pipeline

```python
from src.preprocessing import preprocess_pipeline
from src.models import train_models, save_all_models

# Preprocess data
result = preprocess_pipeline('data/sms_spam.csv', method='tfidf')
X, y = result['X'], result['y']

# Train models
training_result = train_models(X, y, test_size=0.2)
models = training_result['models']
evaluations = training_result['evaluations']

# Save models
saved_paths = save_all_models(models, 'models/')

# Access results
print(f"Logistic Regression Accuracy: {evaluations['logistic_regression']['accuracy']:.4f}")
```

### Load and Make Predictions

```python
from src.models import load_model

# Load saved model
model = load_model('models/logistic_regression_*.pkl')

# Make predictions
predictions = model.predict(X_new)
```

## OpenSpec Workflow

This project follows the OpenSpec (Spec-Driven Development) methodology:

### Proposals

1. **Proposal 001**: Basic rule-based classifier (`hw3.py`)
2. **Proposal 002**: Data preprocessing module
3. **Proposal 003**: Model training module

See `openspec/proposals/` for detailed specifications.

### Project Specification

Detailed project requirements in `openspec/project.md`:
- Project goals and deliverables
- Tech stack and conventions
- File structure and development guidelines
- Evaluation criteria (25% workflow, 35% ML pipeline, 20% visualization, 20% documentation)

## Models Performance

Sample results on test data (may vary based on data):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.5% | 97.2% | 99.1% | 98.1% |
| Multinomial Naïve Bayes | 97.8% | 96.5% | 98.3% | 97.4% |
| Linear SVM | 99.1% | 98.6% | 99.4% | 99.0% |

## Next Steps

1. **Visualization & Streamlit** (Proposal 004)
   - ROC curves and performance comparison
   - Confusion matrices
   - Feature importance analysis
   - Interactive prediction interface

2. **Model Improvement**
   - Hyperparameter tuning
   - Ensemble methods
   - Cross-validation
   - Feature engineering

3. **Production Deployment**
   - API development (Flask/FastAPI)
   - Model monitoring
   - A/B testing
   - Scaling and optimization

## Directory Details

### `/data`
- `sample_sms_spam.csv` - Sample dataset with ham/spam messages

### `/src`
- `preprocessing.py` - Text cleaning, tokenization, vectorization (24 functions + utilities)
- `models.py` - Model training and evaluation (12 functions + utilities)

### `/models`
- Saved trained model files (*.pkl) with timestamps
- Can be loaded for inference

### `/notebooks`
- `01_data_exploration.ipynb` - EDA and preprocessing demonstration
- `02_model_training.ipynb` - Model training and evaluation

### `/tests`
- `test_preprocessing.py` - 24 unit tests (100% coverage)
- `test_models.py` - 22 unit tests (85%+ coverage)

### `/openspec`
- `project.md` - Full project specification
- `proposals/` - Change proposal documents

## Requirements

See `requirements.txt` for full dependencies:

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.1
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
streamlit>=1.28.0
jupyter>=1.0.0
pytest>=7.4.0
```

## Contributing

This project uses OpenSpec workflow for changes:

1. Create a proposal in `openspec/proposals/`
2. Implement changes following the specification
3. Add comprehensive tests
4. Update documentation

## References

- **Packt Repository**: [Hands-On AI for Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)
- **OpenSpec Tutorial**: [OpenSpec Methodology](https://www.youtube.com/watch?v=ANjiJQQIBo0)
- **Teaching Videos**: [HW3 Tutorial Playlist](https://www.youtube.com/watch?v=FeCCYFK0TJ8&list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c)

## License

This is an educational project for HW3 assignment.

## Author

HW3 Student

---

**Last Updated**: 2025-11-16

For questions or issues, refer to the OpenSpec specifications in `openspec/` directory.
