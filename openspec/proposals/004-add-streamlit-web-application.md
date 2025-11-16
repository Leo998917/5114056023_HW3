# Proposal 004 — Add Streamlit Web Application for Spam Classification

**Status**: Pending Implementation  
**Created**: 2025-11-16  
**Type**: Feature Addition  
**Priority**: High  

---

## Summary

Build a comprehensive interactive web application using Streamlit that provides real-time email/SMS spam classification, model comparison, visualization of performance metrics, feature importance analysis, and an intuitive user interface for stakeholder demonstration.

---

## Problem Statement

Current system has:
- ✅ Functional preprocessing pipeline
- ✅ Trained ML models (LogReg, Naïve Bayes, SVM)
- ✅ Complete evaluation metrics
- ❌ No user-friendly interface
- ❌ No interactive model demonstration
- ❌ No visualization of results
- ❌ No real-time prediction capability
- ❌ Not accessible to non-technical users

This limits:
- Model validation by stakeholders
- Real-world demonstration capability
- Understanding of model behavior
- Accessibility to end users

---

## Motivation

A web interface enables:
1. **Stakeholder Engagement**: Non-technical users can test models
2. **Model Transparency**: Visual explanation of predictions
3. **Feature Understanding**: See which words drive spam detection
4. **Performance Validation**: Compare models side-by-side
5. **Production-Ready**: Deploy to Streamlit Cloud
6. **OpenSpec Completion**: Final deliverable for HW3

---

## Technical Plan

### 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Streamlit Web Application              │
├─────────────────────────────────────────────────────┤
│  Frontend (Streamlit UI)                            │
│  ├─ Text Input                                      │
│  ├─ Model Selection Dropdown                        │
│  ├─ Prediction Display                              │
│  └─ Performance Dashboard                           │
├─────────────────────────────────────────────────────┤
│  Backend Logic (Python)                             │
│  ├─ Text Preprocessing (clean_text, tokenize)       │
│  ├─ TF-IDF Vectorization                            │
│  ├─ Model Loading & Caching                         │
│  └─ Prediction & Probability Scoring                │
├─────────────────────────────────────────────────────┤
│  Data & Models                                      │
│  ├─ models/model_logreg.pkl                         │
│  ├─ models/model_nb.pkl                             │
│  ├─ models/model_svm.pkl                            │
│  ├─ models/training_metadata.pkl                    │
│  └─ hw3_dataset/sms_spam_no_header.csv              │
└─────────────────────────────────────────────────────┘
```

### 2. Page Structure

#### Page 1: Home / Prediction Interface
```
📧 Email Spam Classifier
─────────────────────────────────────
Welcome message
Quick start instructions

┌─────────────────────────────────┐
│ Select Model:  [Dropdown ▼]     │
├─────────────────────────────────┤
│ Enter your message:             │
│ ┌─────────────────────────────┐ │
│ │                             │ │
│ │ [Text input box]            │ │
│ │                             │ │
│ └─────────────────────────────┘ │
│ [🔮 Predict Button]             │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ PREDICTION RESULTS              │
├─────────────────────────────────┤
│ Result: [SPAM / HAM]            │
│ Confidence: [XX.X%]             │
│ Model: [Model Name]             │
│ Processed Text: [Expandable]    │
└─────────────────────────────────┘
```

#### Page 2: Model Performance Dashboard
```
📊 Model Performance
─────────────────────────────────────

Tab 1: Logistic Regression
├─ Confusion Matrix Heatmap
├─ Metrics Table (Accuracy, Precision, Recall, F1, ROC-AUC)
├─ Metrics Bar Chart Comparison
└─ ROC Curve (if applicable)

Tab 2: Naïve Bayes
├─ Confusion Matrix Heatmap
├─ Metrics Table
├─ Metrics Bar Chart Comparison
└─ Probability Distribution Histogram

Tab 3: Linear SVM
├─ Confusion Matrix Heatmap
├─ Metrics Table
├─ Metrics Bar Chart Comparison
└─ ROC Curve
```

#### Page 3: Feature Analysis
```
🔍 Feature Importance
─────────────────────────────────────

Model Selection: [Dropdown ▼]

Top 20 Important Features
[Horizontal Bar Chart]
- Each bar shows coefficient strength
- Color indicates direction (+ spam, - ham)

Feature Statistics
- Total vocabulary: 5,000
- Features used: 5,000
- Average feature weight: X.XXX
```

#### Page 4: Dataset Information
```
ℹ️ About & Statistics
─────────────────────────────────────

Dataset Information
├─ Name: SMS Spam Collection
├─ Location: hw3_dataset/sms_spam_no_header.csv
├─ Total Messages: 5,574
├─ Ham: 4,827 (86.6%)
├─ Spam: 747 (13.4%)
└─ Training/Test Split: 80/20 (4,459 / 1,115)

Models
├─ 1. Logistic Regression (96.50% accuracy)
├─ 2. Multinomial Naïve Bayes (97.31% accuracy)
└─ 3. Linear SVM (98.74% accuracy ⭐)

Technology Stack
├─ Python 3.10+
├─ scikit-learn (ML)
├─ NLTK (NLP)
├─ Streamlit (Web UI)
├─ matplotlib/seaborn (Visualization)
└─ plotly (Interactive charts)
```

---

## 3. API Specifications

### Core Functions (in app.py)

```python
@st.cache_resource
def load_all_trained_models(models_dir: str = 'models') -> Dict[str, Any]:
    """
    Load all three trained models from disk with caching.
    
    Args:
        models_dir (str): Directory containing model files
        
    Returns:
        Dict with structure:
        {
            'Logistic Regression': LogisticRegression model,
            'Naïve Bayes': MultinomialNB model,
            'SVM': LinearSVC model
        }
        
    Caching: Streamlit caches loaded models to avoid reloading
    """
```

```python
@st.cache_resource
def load_vectorizer_and_features(data_path: str) -> Tuple[object, List[str]]:
    """
    Load or recreate TF-IDF vectorizer from preprocessing.
    
    Args:
        data_path (str): Path to training data
        
    Returns:
        Tuple of (fitted_vectorizer, feature_names_list)
    """
```

```python
def predict_spam(
    user_input: str,
    model: object,
    vectorizer: object,
    feature_names: List[str]
) -> Tuple[str, float]:
    """
    Preprocess input text and make spam prediction.
    
    Args:
        user_input (str): Raw SMS/email text
        model: Trained classifier (LogReg, NB, or SVM)
        vectorizer: Fitted TF-IDF or BoW vectorizer
        feature_names (List[str]): Vocabulary from training
        
    Returns:
        Tuple of (prediction_label, confidence_score)
        - prediction_label: 'SPAM' or 'HAM'
        - confidence_score: float [0.0, 1.0]
        
    Process:
        1. Clean text (remove URLs, special chars, normalize)
        2. Tokenize and stem
        3. Vectorize using fitted vectorizer
        4. Make prediction
        5. Get confidence/probability
    """
```

```python
def create_metrics_comparison_table(evaluations: Dict) -> pd.DataFrame:
    """
    Format evaluation metrics into displayable table.
    
    Args:
        evaluations (Dict): Model evaluation results from train_models()
        
    Returns:
        DataFrame with columns:
        [Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC]
    """
```

```python
def generate_report(
    model_name: str,
    user_input: str,
    prediction: str,
    confidence: float
) -> Dict[str, Any]:
    """
    Generate comprehensive prediction report.
    
    Returns:
        {
            'prediction': 'SPAM' | 'HAM',
            'confidence': float,
            'model': str,
            'processed_text': str,
            'tokens': List[str],
            'timestamp': str
        }
    """
```

### Visualization Functions (src/visualization.py)

Already implemented in previous work:
- `plot_confusion_matrix(cm, model_name)` → matplotlib.figure
- `plot_metrics_comparison(evaluations)` → matplotlib.figure
- `plot_top_features(feature_names, coefficients, top_n)` → matplotlib.figure
- `plot_roc_curve(y_test, y_proba, model_name)` → matplotlib.figure
- `plot_probability_distribution(proba_dict)` → matplotlib.figure

---

## 4. UI/UX Design Specifications

### Color Scheme
```
Primary:    #1f77b4 (Blue)    - Main accent
Success:    #2ca02c (Green)   - Ham/Legitimate
Danger:     #d62728 (Red)     - Spam/Warning
Background: #f0f2f6 (Light)   - Card backgrounds
Text:       #262730 (Dark)    - Primary text
```

### Typography
```
Main Title:     Bold, 24pt, Blue
Section Headers: Bold, 16pt, Dark
Body Text:      Regular, 12pt, Dark
Code/Metrics:   Monospace, 11pt
```

### Interactive Elements
```
Buttons:        Rounded corners, hover effect, 14pt bold
Dropdowns:      Clean, blue accent on focus
Text Inputs:     120px height, large font, clear placeholder
Tables:         Striped rows, center-aligned metrics
Charts:         Interactive (Plotly preferred), tooltips
```

### Responsive Design
```
Desktop (≥1200px):  Full width, side-by-side layouts
Tablet (768-1200px): Stacked 2-column layouts
Mobile (<768px):    Single column, touch-friendly buttons
```

---

## 5. File Structure

```
hw3/
├── app.py (NEW - 500+ lines)
│   ├── Page setup & configuration
│   ├── Sidebar navigation
│   ├── Model loading functions
│   ├── Prediction functions
│   ├── Page implementations
│   │   ├── page_prediction()
│   │   ├── page_performance()
│   │   ├── page_features()
│   │   └── page_about()
│   └── Main app flow
│
├── src/
│   ├── preprocessing.py (EXISTING - use as-is)
│   ├── models.py (EXISTING - use as-is)
│   └── visualization.py (EXISTING - use as-is)
│
├── models/ (EXISTING)
│   ├── model_logreg.pkl
│   ├── model_nb.pkl
│   ├── model_svm.pkl
│   └── training_metadata.pkl
│
├── hw3_dataset/ (EXISTING)
│   └── sms_spam_no_header.csv
│
├── .streamlit/ (NEW - configuration)
│   └── config.toml
│
└── requirements.txt (UPDATE - add streamlit)
```

---

## 6. Implementation Steps

### Step 1: Environment Setup
```bash
# Ensure Streamlit is in requirements.txt
pip install streamlit>=1.28.0

# Test Streamlit installation
streamlit --version
```

### Step 2: Create .streamlit/config.toml
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
toolbarMode = "auto"

[logger]
level = "info"
```

### Step 3: Implement app.py
```
1. Page configuration (title, icon, layout)
2. Import all dependencies
3. Implement caching functions for models/vectorizers
4. Implement utility functions (prediction, formatting)
5. Implement page functions
6. Implement sidebar navigation
7. Main execution block
```

### Step 4: Create Visualization Integration
- Use existing visualization.py functions
- Integrate with st.pyplot() for matplotlib plots
- Use st.metric() for key statistics
- Use st.dataframe() for tables

### Step 5: Add Interactive Components
- Text input for SMS/email
- Model selection dropdown
- Tabs for different visualizations
- Expanders for detailed information
- Buttons for actions

### Step 6: Testing & Validation
```bash
# Run Streamlit app locally
streamlit run app.py

# Test each page/feature
# - Prediction on sample texts
# - Model switching
# - All visualizations load
# - No errors in console
```

### Step 7: Deployment (Optional)
```bash
# Push to GitHub
git add app.py .streamlit/
git commit -m "Add Streamlit web application"
git push origin main

# Deploy to Streamlit Cloud
# - Connect GitHub repo
# - Set secrets (if needed)
# - Deploy app
```

---

## 7. Acceptance Criteria

### Functional Requirements
- ✅ **Text Input**: Users can input SMS/email text (min 1 char, max 1000 chars)
- ✅ **Model Selection**: Dropdown allows selecting 3 models
- ✅ **Prediction**: Button triggers prediction workflow
- ✅ **Results Display**: Shows prediction (SPAM/HAM) with confidence score
- ✅ **Model Info**: Displays which model was used
- ✅ **Text Processing**: Shows cleaned/processed text in expandable section

### Visualization Requirements
- ✅ **Confusion Matrices**: Heatmap for each model showing TP/TN/FP/FN
- ✅ **Metrics Table**: All models' metrics in formatted table
- ✅ **Metrics Chart**: Bar chart comparing accuracy across models
- ✅ **Feature Importance**: Horizontal bar chart of top 20 features
- ✅ **ROC Curves**: ROC curve visualization (if model supports)
- ✅ **Label Distribution**: Pie and bar charts of ham vs spam

### Performance Requirements
- ✅ **Load Time**: Models load in <5 seconds on first run
- ✅ **Prediction Time**: Prediction returns in <1 second
- ✅ **Caching**: Repeated predictions don't reload models
- ✅ **Memory**: App uses <500MB RAM with all models loaded

### Code Quality Requirements
- ✅ **Type Hints**: All function signatures have type hints
- ✅ **Docstrings**: All functions have comprehensive docstrings
- ✅ **Error Handling**: Graceful errors for missing files, invalid input
- ✅ **Code Style**: PEP 8 compliant, well-organized
- ✅ **Comments**: Inline comments for complex logic

### User Experience Requirements
- ✅ **Navigation**: Clear sidebar navigation between pages
- ✅ **Layout**: Consistent, professional appearance
- ✅ **Instructions**: Clear guidance for new users
- ✅ **Responsiveness**: Works on desktop, tablet, mobile
- ✅ **Accessibility**: Readable fonts, good contrast, proper spacing

### Testing Requirements
- ✅ **Manual Testing**: All features work without errors
- ✅ **Sample Data**: Pre-loaded sample messages for quick testing
- ✅ **Edge Cases**: Handles empty input, special characters, very long text
- ✅ **Model Consistency**: All models make reasonable predictions

---

## 8. Dependencies

### New Dependencies
```
streamlit>=1.28.0          # Web framework
plotly>=5.14.0             # Interactive visualizations
```

### Existing Dependencies (Already Available)
```
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
scikit-learn>=1.3.0        # ML models
nltk>=3.8.1                # NLP preprocessing
matplotlib>=3.7.0          # Static plotting
seaborn>=0.12.0            # Enhanced visualization
```

---

## 9. Configuration

### Environment Variables
```python
# Optional: API keys, secrets (not needed for basic version)
# Set in .streamlit/secrets.toml if needed
```

### Streamlit Settings
```toml
# In .streamlit/config.toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
showMenuItems = true
```

---

## 10. Deliverables

### Code Files
1. **`app.py`** (500+ lines)
   - Main Streamlit application
   - All page implementations
   - Model loading and caching
   - Prediction and visualization logic

2. **`.streamlit/config.toml`** (NEW)
   - Theme configuration
   - Streamlit settings

### Documentation
1. **README.md** (UPDATE)
   - Add Streamlit section with usage instructions
   - Add screenshot descriptions (if deployed)
   - Add deployment instructions

2. **IMPLEMENTATION_SUMMARY.md** (UPDATE)
   - Note completion of Proposal 004

### Testing Checklist
- [ ] All pages load without errors
- [ ] Prediction works on test messages
- [ ] All visualizations render correctly
- [ ] Model switching works
- [ ] Text input validation works
- [ ] Edge cases handled gracefully
- [ ] Performance meets requirements
- [ ] Code passes linting

---

## 11. Success Metrics

**Development Success**
- All acceptance criteria met
- Code passes quality checks
- All tests pass
- Zero unhandled exceptions

**User Success**
- Users can make predictions easily
- Visualizations are clear and informative
- Model selection impacts results
- App responds quickly (<2 seconds)
- Professional appearance

**Business Success**
- Stakeholders can validate models
- Feature importance explained
- Model comparison visible
- Ready for demonstration/deployment

---

## 12. Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Setup & Architecture | 1 hour |
| 2 | Core Prediction Logic | 1 hour |
| 3 | UI Pages Implementation | 2 hours |
| 4 | Visualization Integration | 1 hour |
| 5 | Testing & Validation | 1 hour |
| 6 | Documentation | 1 hour |
| **Total** | | **7 hours** |

---

## 13. Related Items

### Dependencies
- **Proposal 002**: Data Preprocessing Module (provides preprocessing functions)
- **Proposal 003**: Model Training Module (provides trained models)
- **src/visualization.py**: Visualization functions (provides plotting)

### Blockers
- None identified

### Future Enhancements
- [ ] Real-time model retraining
- [ ] A/B testing framework
- [ ] User feedback collection
- [ ] Batch prediction upload
- [ ] API endpoint creation
- [ ] Mobile app version
- [ ] Multi-language support

---

## 14. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Models fail to load | Low | High | Test model paths, add error handling |
| Vectorizer mismatch | Low | High | Load from metadata, validate dimensions |
| Slow performance | Low | Medium | Cache models, vectorize efficiently |
| UI layout issues | Medium | Low | Test on multiple screen sizes |
| Missing dependencies | Low | High | Ensure requirements.txt is complete |

---

## 15. Questions & Clarifications

- Should the app support batch prediction upload? → Future enhancement
- Should users be able to retrain models? → Not in Phase 1
- What about model explanations (LIME/SHAP)? → Future enhancement
- Should the app track prediction history? → Future enhancement
- Mobile app needed? → Streamlit handles responsively

---

## Approval & Sign-off

This proposal defines the complete specification for the Streamlit Web Application component of HW3 Email Spam Classification project.

**Expected Completion**: 2025-11-16  
**Estimated Effort**: 7 hours  
**Status**: Ready for Implementation ✅

