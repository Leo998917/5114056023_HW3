# Project Specification — HW3 Email Spam Classification# Project Specification — HW3 Spam Demo



## Overview## Overview

This project implements an email spam classification system using the OpenSpec (Spec-Driven Development) workflow. We reproduce and extend Chapter 3 of "Hands-On Artificial Intelligence for Cybersecurity" by Packt, adding data preprocessing steps, expanding visualizations and metrics, and integrating CLI/Streamlit front-ends.This is a simple demo project to practice OpenSpec workflow.  

The goal is to build a minimal spam classifier inside `hw3.py`.  

## Project Goals

1. **Data Preprocessing**: Clean and preprocess email/SMS text data (tokenization, vectorization)## Project Goals

2. **Model Training**: Implement multiple classification models (Logistic Regression, Naïve Bayes, SVM)1. Have a simple function `classify_message(text)`  

3. **Evaluation & Metrics**: Generate comprehensive evaluation metrics and confusion matrices2. Rule-based spam detection (e.g., detect "buy now", "free", "win")  

4. **Visualization**: Create interactive visualizations and metrics plots3. Later can extend to ML-based detection.

5. **Streamlit Demo**: Deploy a functional Streamlit web application for the spam classifier

6. **OpenSpec Workflow**: Demonstrate complete OpenSpec methodology with proposals and agent traces## Tech Stack

- Python 3.10

## Tech Stack- No external dependencies for the initial version

- **Language**: Python 3.10+

- **ML/Data**: scikit-learn, pandas, numpy## File Structure

- **Preprocessing**: NLTK, string utilities- hw3.py : main script

- **Visualization**: matplotlib, seaborn, plotly- openspec/project.md : top-level specification

- **Web Framework**: Streamlit- openspec/proposals/ : change proposals

- **Development**: OpenSpec + AI Coding CLI (GitHub Copilot)

- **Deployment**: Streamlit Community Cloud## Conventions

- All functions named in snake_case

## Project Deliverables- All new features must be added through proposals

1. **GitHub Repository**: https://github.com/huanchen1107/2025ML-spamEmail
2. **Streamlit Demo Site**: https://2025spamemail.streamlit.app/
3. **Dataset**: Chapter03/datasets/sms_spam_no_header.csv (from Packt repo)

## File Structure
```
.
├── hw3.py                          # Initial spam classifier (from proposal 001)
├── data/
│   └── sms_spam_no_header.csv     # SMS spam dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA and preprocessing
│   ├── 02_model_training.ipynb    # Model training and evaluation
│   └── 03_metrics_analysis.ipynb  # Detailed metrics and visualization
├── src/
│   ├── preprocessing.py           # Data cleaning and vectorization
│   ├── models.py                  # Model training and evaluation
│   └── visualization.py           # Metric plots and charts
├── app.py                         # Streamlit application
├── openspec/
│   ├── project.md                 # This specification
│   ├── AGENTS.md                  # Agent workflow guidelines
│   └── proposals/                 # Change proposals
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Tech Stack Details
- **Data Processing**: pandas, numpy
- **ML Models**: scikit-learn (LogisticRegression, MultinomialNB, SVC)
- **Text Processing**: NLTK, string module
- **Visualization**: matplotlib, seaborn, plotly
- **Web UI**: Streamlit
- **Evaluation**: sklearn.metrics (accuracy, precision, recall, F1, confusion matrix)

## Development Conventions
1. **Naming**: All functions and variables use `snake_case`
2. **Code Organization**: 
   - Preprocessing logic in `src/preprocessing.py`
   - Model training in `src/models.py`
   - Visualization in `src/visualization.py`
3. **New Features**: All enhancements must be proposed through OpenSpec proposals
4. **Documentation**: 
   - Docstrings for all functions
   - Inline comments for complex logic
   - README with setup and usage instructions
5. **Notebooks**: Jupyter notebooks for exploration and analysis
6. **Streamlit App**: Interactive UI for model demonstration

## References
- **Packt Repository**: https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity
- **Teaching Videos**: https://www.youtube.com/watch?v=FeCCYFK0TJ8&list=PLYlM4-ln5HcCoM_TcLKGL5NcOpNVJ3g7c
- **OpenSpec Tutorial**: https://www.youtube.com/watch?v=ANjiJQQIBo0

## Evaluation Criteria
| Aspect | Weight |
|--------|--------|
| OpenSpec workflow completeness (project.md + proposals + agent traces) | 25% |
| ML Pipeline Implementation (data prep + training + evaluation) | 35% |
| Visualization & Interpretability (metrics, confusion matrix, Streamlit views) | 20% |
| Documentation & Presentation (README, structure, clarity) | 20% |

## Key Features to Implement
1. **Data Pipeline**: Load, clean, tokenize, and vectorize email/SMS texts
2. **Model Selection**: Train and compare multiple classifiers
3. **Evaluation**: Generate metrics (accuracy, precision, recall, F1-score, ROC-AUC)
4. **Visualization**: Confusion matrices, ROC curves, feature importance, prediction examples
5. **Interactive UI**: Streamlit app for real-time spam classification predictions
6. **CLI Interface**: Command-line tools for training and evaluation
