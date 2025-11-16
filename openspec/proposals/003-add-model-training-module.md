# Proposal 003 — Add Model Training Module

## Summary
Build a comprehensive model training module that implements three machine learning classifiers (Logistic Regression, Multinomial Naïve Bayes, and Linear SVM), trains them on preprocessed SMS data, evaluates their performance with comprehensive metrics, and persists trained models for inference.

## Problem
After preprocessing the SMS data, we need to:
1. Split data into training and testing sets
2. Train and evaluate multiple classification models
3. Compare model performance using standard ML metrics
4. Persist trained models for future inference
5. Generate confusion matrices and detailed evaluation reports

## Motivation
Multiple models provide different perspectives on the data:
- **Logistic Regression**: Interpretable linear model, fast training/inference
- **Multinomial Naïve Bayes**: Probabilistic model, efficient for text classification
- **Linear SVM**: Powerful margin-based classifier, good generalization

This module enables:
- Systematic model comparison
- Production-ready model persistence
- Reproducible evaluation pipelines
- Foundation for ensemble methods

## Technical Plan

### 1. Data Splitting
- Use `sklearn.model_selection.train_test_split`
- 80% training, 20% testing
- `random_state=42` for reproducibility
- Stratified split to maintain class balance

### 2. Model Training
- **LogisticRegression**: 
  - `max_iter=1000`
  - `random_state=42`
  - `solver='lbfgs'`
  
- **MultinomialNB**:
  - `alpha=1.0` (Laplace smoothing)
  
- **LinearSVC**:
  - `random_state=42`
  - `max_iter=2000`
  - `dual=False` (for sparse data)

### 3. Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives

### 4. Model Persistence
- Save trained models using `joblib`
- Directory: `models/` with format `{model_name}_{timestamp}.pkl`
- Also save vectorizers for feature consistency

### 5. File Structure
```
src/
  models.py          # Model training and evaluation functions
  
models/
  logistic_*.pkl     # Saved models
  naive_bayes_*.pkl
  svm_*.pkl
  vectorizer_*.pkl
  
tests/
  test_models.py     # Unit tests for model training
  
notebooks/
  02_model_training.ipynb  # Training and evaluation demo
```

## Acceptance Criteria
- [ ] `src/models.py` created with core training functions
- [ ] `train_test_split_data()` function splits data stratified 80/20
- [ ] `train_models()` function trains all three classifiers
- [ ] `evaluate_model()` function computes metrics and confusion matrix
- [ ] `evaluate_all_models()` function compares all models
- [ ] `save_model()` function persists models and vectorizers to disk
- [ ] `load_model()` function loads saved models for inference
- [ ] All functions have docstrings and type hints
- [ ] Unit tests in `tests/test_models.py` achieve >85% coverage
- [ ] Demo notebook shows training process and results
- [ ] Models folder created and models saved successfully
- [ ] All evaluation metrics are computed and displayed
- [ ] Confusion matrices are generated for all models

## Agent Workflow

### Phase 1: Planning & Review
1. **Architect** reviews proposal and approves ML approach
2. **Developer** clarifies model selection and evaluation metrics
3. **QA** defines evaluation criteria and test scenarios

### Phase 2: Implementation
1. **Developer** implements `src/models.py`:
   - Data splitting functions
   - Model training and evaluation
   - Model persistence/loading
   - Metrics computation

2. **Developer** creates `tests/test_models.py`:
   - Unit tests for each model
   - Tests for train/test splitting
   - Tests for model persistence
   - Tests for metrics computation

3. **Developer** creates demo notebook `notebooks/02_model_training.ipynb`:
   - Load preprocessed data
   - Train models on training set
   - Evaluate on test set
   - Compare model performance
   - Display confusion matrices

### Phase 3: Code Review & Testing
1. **Reviewer** validates code quality and test coverage
2. **QA** runs test suite and verifies all models train successfully
3. **Developer** addresses feedback and optimizations

### Phase 4: Integration
1. Merge to main branch
2. Verify models folder and saved models
3. Create next proposal for visualization/Streamlit

## Affected Files

### New Files
- `src/models.py` - Model training and evaluation module
- `tests/test_models.py` - Unit tests
- `notebooks/02_model_training.ipynb` - Training demo notebook
- `models/` directory - Saved model storage

### Modified Files
- `requirements.txt` - Add joblib if needed
- `src/__init__.py` - Import model functions

## Code Tasks

### Task 1: Create src/models.py
Implement core model training functions:

```python
def train_test_split_data(X, y, test_size=0.2, random_state=42) -> tuple:
    """Split data into train/test sets."""
    
def train_logistic_regression(X_train, y_train) -> object:
    """Train Logistic Regression model."""
    
def train_naive_bayes(X_train, y_train) -> object:
    """Train Multinomial Naïve Bayes model."""
    
def train_svm(X_train, y_train) -> object:
    """Train Linear SVM model."""
    
def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    
def train_models(X, y) -> dict:
    """Train all three models."""
    
def save_model(model, filename: str) -> str:
    """Save trained model to disk."""
    
def load_model(filename: str) -> object:
    """Load saved model from disk."""
```

### Task 2: Create tests/test_models.py
Write comprehensive tests:
- Test train/test splitting
- Test each model trains successfully
- Test metrics computation
- Test model persistence/loading
- Test evaluation on known datasets

### Task 3: Create demo notebook
Demonstrate model training:
- Load preprocessed data
- Show data split statistics
- Train each model
- Display evaluation metrics
- Show confusion matrices and ROC curves
- Compare model performance

### Task 4: Create models directory
```bash
mkdir -p models
```

## Experimental Results

### Dataset Statistics
- **Dataset**: `hw3_dataset/sms_spam_no_header.csv`
- **Total Messages**: 5,574
- **Ham (Legitimate)**: 4,827 (86.6%)
- **Spam**: 747 (13.4%)
- **Features**: 5,000 TF-IDF features
- **Sparsity**: 99.86%

### Train-Test Split
- **Training Samples**: 4,459 (80%)
- **Test Samples**: 1,115 (20%)
- **Class Balance**: Maintained through stratified split

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9650 | 0.9664 | 0.9650 | 0.9627 | 0.9865 |
| Multinomial Naïve Bayes | 0.9731 | 0.9736 | 0.9731 | 0.9719 | 0.9802 |
| **Linear SVM** | **0.9874** | **0.9875** | **0.9874** | **0.9872** | **0.9889** |

### Key Findings
1. **Best Performer**: Linear SVM achieved 98.74% accuracy
2. **Strong Performance**: All three models performed well (>96% accuracy)
3. **Ranking**: SVM > Naïve Bayes > Logistic Regression
4. **ROC-AUC**: All models showed excellent discrimination (>98%)

### Model Files Generated
- `models/model_logreg.pkl` (40.7 KB)
- `models/model_nb.pkl` (160.6 KB)
- `models/model_svm.pkl` (40.6 KB)
- `models/training_metadata.pkl` (55.7 KB)

## Success Metrics
- All unit tests pass (>85% coverage)
- All three models train successfully
- Evaluation metrics computed correctly
- Models saved and loaded without error
- Confusion matrices display correctly
- Training time <30 seconds on sample data
- Model performance metrics are reasonable

## Related Issues
- Depends on: Proposal 002 (Data Preprocessing)
- Links to: Proposal 004 (Visualization & Streamlit)
- Related to: HW3 Project Specification

## Timeline
- Implementation: 3-4 hours
- Testing: 1-2 hours
- Code review: 30 minutes
- Total: ~5 hours

## Deliverables
1. **src/models.py**: Production-ready model training module
2. **tests/test_models.py**: Comprehensive test suite
3. **notebooks/02_model_training.ipynb**: Interactive training demo
4. **models/**: Directory with example trained models
5. **Evaluation Report**: Metrics and comparison of all models
