# Proposal 002 — Add Data Preprocessing Module

## Summary
Build a comprehensive data preprocessing module that reads SMS spam CSV data, cleans text, performs tokenization, stemming, and vectorization (Bag-of-Words + TF-IDF), and outputs processed features with corresponding labels.

## Problem
The current classifier in hw3.py is rule-based and cannot scale to machine learning models. To train ML classifiers, we need:
1. A standardized way to load and preprocess raw SMS data
2. Text cleaning (removing noise, special characters, etc.)
3. Tokenization and stemming to normalize word forms
4. Vectorization using both Bag-of-Words and TF-IDF representations
5. Structured outputs (X feature matrix, y labels, feature names) for model training

## Motivation
Proper data preprocessing is foundational for any ML pipeline. This module will:
- Enable reproducible data processing workflows
- Support multiple vectorization methods (BoW and TF-IDF)
- Provide a clean interface for downstream model training
- Improve code organization and maintainability

## Technical Plan

### 1. Data Loading
- Read SMS spam CSV from `data/sms_spam_no_header.csv`
- Handle missing values and data validation
- Split into message text and labels (spam/ham)

### 2. Text Cleaning
- Convert to lowercase
- Remove special characters, numbers, URLs
- Strip extra whitespace
- Remove common English stopwords

### 3. Tokenization & Stemming
- Split text into individual tokens
- Apply Porter Stemmer to reduce words to base forms
- Maintain token frequency information

### 4. Vectorization
- **Bag-of-Words (BoW)**: CountVectorizer with configurable parameters
  - Maximum features: 5000
  - Minimum document frequency: 2
  - Maximum document frequency: 0.8
- **TF-IDF**: TfidfVectorizer with same parameters
  - Returns normalized sparse matrices
  - Includes feature names for interpretability

### 5. Output Format
- Return structured dictionary with:
  - `X_bow`: Sparse matrix of BoW features
  - `X_tfidf`: Sparse matrix of TF-IDF features
  - `y`: Label array (0=ham, 1=spam)
  - `feature_names`: List of feature vocabulary
  - `metadata`: Data shape and statistics

## Acceptance Criteria
- [ ] `src/preprocessing.py` module created with all required functions
- [ ] `load_sms_data(filepath)` function reads and validates CSV data
- [ ] `clean_text(text)` function handles text normalization
- [ ] `tokenize_and_stem(text)` function performs tokenization and stemming
- [ ] `vectorize_text(texts, method='tfidf', **kwargs)` returns X, feature_names
- [ ] `preprocess_pipeline(csv_path)` orchestrates full pipeline
- [ ] All functions have docstrings and type hints
- [ ] Unit tests in `tests/test_preprocessing.py` achieve >90% coverage
- [ ] Demo script shows loaded data statistics
- [ ] Handles edge cases (empty text, special characters, etc.)
- [ ] Data can be successfully loaded from CSV and vectorized

## Agent Workflow

### Phase 1: Planning & Review
1. **Architect** reviews proposal and approves technical approach
2. **Developer** clarifies data format and preprocessing requirements
3. **QA** defines test scenarios and acceptance criteria

### Phase 2: Implementation
1. **Developer** implements `src/preprocessing.py`:
   - Data loading functions
   - Text cleaning pipeline
   - Tokenization and stemming
   - Vectorization wrappers

2. **Developer** creates `tests/test_preprocessing.py`:
   - Unit tests for each function
   - Integration tests for full pipeline
   - Edge case handling

3. **Developer** creates demo script `notebooks/01_data_exploration.ipynb`:
   - Load and display data samples
   - Show preprocessing step outputs
   - Display vectorization results and statistics

### Phase 3: Code Review & Testing
1. **Reviewer** validates code quality and test coverage
2. **QA** runs test suite and verifies acceptance criteria
3. **Developer** addresses feedback and optimizations

### Phase 4: Integration
1. Merge to main branch
2. Update documentation (README.md)
3. Create next proposal for model training

## Affected Files

### New Files
- `src/preprocessing.py` - Main preprocessing module
- `src/__init__.py` - Package initialization
- `tests/test_preprocessing.py` - Unit tests
- `notebooks/01_data_exploration.ipynb` - Exploration and demo

### Modified Files
- `requirements.txt` - Add scikit-learn, nltk, pandas dependencies

### Reference Files
- `openspec/project.md` - Update file structure confirmation
- `data/sms_spam_no_header.csv` - Dataset (should be downloaded separately)

## Code Tasks

### Task 1: Create src/preprocessing.py
Implement core preprocessing functions:
```python
def load_sms_data(filepath: str) -> tuple:
    """Load SMS spam CSV data."""
    
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    
def tokenize_and_stem(text: str) -> list:
    """Tokenize and apply stemming."""
    
def vectorize_text(texts: list, method: str = 'tfidf', max_features: int = 5000) -> dict:
    """Vectorize texts using BoW or TF-IDF."""
    
def preprocess_pipeline(csv_path: str, method: str = 'tfidf') -> dict:
    """Full preprocessing pipeline."""
```

### Task 2: Create tests/test_preprocessing.py
Write comprehensive tests:
- Test data loading with valid/invalid files
- Test text cleaning on various inputs
- Test tokenization and stemming
- Test vectorization output shapes and types
- Test full pipeline integration
- Test with sample data

### Task 3: Create demo notebook
Demonstrate preprocessing functionality:
- Load sample data
- Show raw vs cleaned text examples
- Display BoW and TF-IDF outputs
- Generate statistics (vocabulary size, sparsity, etc.)

### Task 4: Update requirements.txt
Add dependencies:
```
scikit-learn>=1.3.0
nltk>=3.8.1
pandas>=2.0.0
numpy>=1.24.0
```

## Success Metrics
- All unit tests pass
- Code coverage >90%
- Preprocessing handles 5000+ messages without errors
- Generated features are compatible with sklearn estimators
- Documentation is clear and complete
- Performance: preprocess CSV in <5 seconds

## Related Issues
- Links to: HW3 Project Specification
- Related to: Proposal 003 (Model Training)
- Depends on: Dataset download from Packt

## Timeline
- Implementation: 2-3 hours
- Testing: 1 hour
- Code review: 30 minutes
- Total: ~4 hours
