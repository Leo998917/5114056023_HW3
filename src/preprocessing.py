"""
Data Preprocessing Module for SMS Spam Classification

This module provides functions for loading, cleaning, and vectorizing SMS text data.
It includes text normalization, tokenization, stemming, and multiple vectorization methods.
"""

import re
import string
from typing import Tuple, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')


def load_sms_data(filepath: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load SMS spam CSV data from file.
    
    Expected CSV format: label, message (tab or comma separated)
    Labels: 'spam', 'ham' or 1, 0
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        Tuple containing:
        - texts (List[str]): SMS messages
        - labels (List[int]): Binary labels (0=ham, 1=spam)
        - label_names (List[str]): ['ham', 'spam']
        
    Raises:
        FileNotFoundError: If file not found
        ValueError: If data format is invalid
    """
    try:
        # Try reading CSV with proper settings for quoted fields
        try:
            # First try: CSV with quoted fields (standard format)
            df = pd.read_csv(filepath, sep=',', header=None, quotechar='"', 
                           escapechar=None, encoding='utf-8')
            if len(df.columns) < 2:
                raise ValueError("Need at least 2 columns")
        except:
            try:
                # Second try: Tab-separated (for test data)
                df = pd.read_csv(filepath, sep='\t', header=None, encoding='utf-8')
                if len(df.columns) < 2:
                    raise ValueError("Need at least 2 columns")
            except:
                # Third try: Basic comma-separated
                df = pd.read_csv(filepath, sep=',', header=None, encoding='utf-8')
                if len(df.columns) < 2:
                    raise ValueError("Need at least 2 columns")
        
        # Handle different column arrangements
        if len(df.columns) >= 2:
            # Assume format: label, message (take first 2 columns)
            label_col, text_col = df.iloc[:, 0], df.iloc[:, 1]
        else:
            raise ValueError(f"Expected at least 2 columns, got {len(df.columns)}")
        
        # Convert labels to binary (0=ham, 1=spam)
        labels = []
        for label in label_col:
            label_str = str(label).strip().lower()
            if label_str in ['ham', '0']:
                labels.append(0)
            elif label_str in ['spam', '1']:
                labels.append(1)
            else:
                labels.append(0)  # Default to ham if unclear
        
        texts = [str(text).strip() for text in text_col]
        label_names = ['ham', 'spam']
        
        return texts, labels, label_names
        
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing noise and special characters.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove special characters and numbers
    5. Remove extra whitespace
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def tokenize_and_stem(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text and apply Porter Stemming.
    
    Args:
        text (str): Text to tokenize and stem
        remove_stopwords (bool): Whether to remove English stopwords
        
    Returns:
        List[str]: List of stemmed tokens
    """
    # Initialize stemmer
    stemmer = PorterStemmer()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    return stemmed_tokens


def vectorize_text(
    texts: List[str],
    method: str = 'tfidf',
    max_features: int = 5000,
    min_df: int = 1,
    max_df: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 1)
) -> Dict:
    """
    Vectorize text using Bag-of-Words or TF-IDF.
    
    Args:
        texts (List[str]): List of text documents
        method (str): 'bow' for Bag-of-Words or 'tfidf' for TF-IDF (default)
        max_features (int): Maximum number of features (default: 5000)
        min_df (int): Minimum document frequency (default: 2)
        max_df (float): Maximum document frequency ratio (default: 0.8)
        ngram_range (Tuple[int, int]): N-gram range (default: (1,1) unigrams)
        
    Returns:
        Dict with keys:
        - 'X': Sparse matrix of vectorized features
        - 'feature_names': Array of feature vocabulary
        - 'vectorizer': The fitted vectorizer object
        - 'method': The method used ('bow' or 'tfidf')
    """
    if method.lower() == 'bow':
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=True
        )
    elif method.lower() == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english'
        )
    else:
        raise ValueError(f"Unknown vectorization method: {method}")
    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return {
        'X': X,
        'feature_names': feature_names,
        'vectorizer': vectorizer,
        'method': method.lower()
    }


def preprocess_pipeline(
    csv_path: str,
    method: str = 'tfidf',
    remove_stopwords: bool = True,
    max_features: int = 5000,
    min_df: int = 1,
    max_df: float = 1.0
) -> Dict:
    """
    Complete preprocessing pipeline: load, clean, tokenize, and vectorize SMS data.
    
    Args:
        csv_path (str): Path to CSV file with SMS data
        method (str): Vectorization method ('bow' or 'tfidf')
        remove_stopwords (bool): Whether to remove stopwords
        max_features (int): Maximum features for vectorization
        min_df (int): Minimum document frequency
        max_df (float): Maximum document frequency ratio
        
    Returns:
        Dict with keys:
        - 'X': Feature matrix (sparse)
        - 'y': Label array
        - 'feature_names': Feature vocabulary
        - 'vectorizer': Fitted vectorizer
        - 'texts_original': Original text samples
        - 'texts_cleaned': Cleaned text samples
        - 'metadata': Statistics and info
    """
    # Step 1: Load data
    texts_original, labels, label_names = load_sms_data(csv_path)
    
    # Step 2: Clean text
    texts_cleaned = [clean_text(text) for text in texts_original]
    
    # Step 3: Tokenize and stem (optional preprocessing before vectorization)
    # Note: TF-IDF vectorizer will do its own tokenization, so we mainly use this for analysis
    texts_tokens = [tokenize_and_stem(text, remove_stopwords) for text in texts_cleaned]
    
    # Step 4: Vectorize
    vectorization_result = vectorize_text(
        texts_cleaned,
        method=method,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    
    # Step 5: Compile results
    X = vectorization_result['X']
    y = np.array(labels)
    
    metadata = {
        'total_samples': len(texts_original),
        'feature_dimension': X.shape[1],
        'sparsity': 1 - (X.nnz / (X.shape[0] * X.shape[1])),
        'label_distribution': {
            'ham': sum(1 for label in labels if label == 0),
            'spam': sum(1 for label in labels if label == 1)
        },
        'method': method.lower(),
        'max_features': max_features
    }
    
    return {
        'X': X,
        'y': y,
        'feature_names': vectorization_result['feature_names'],
        'vectorizer': vectorization_result['vectorizer'],
        'texts_original': texts_original,
        'texts_cleaned': texts_cleaned,
        'texts_tokens': texts_tokens,
        'label_names': label_names,
        'metadata': metadata
    }


if __name__ == '__main__':
    # Demo: Show usage
    print("SMS Spam Preprocessing Module")
    print("-" * 50)
    print("Available functions:")
    print("  - load_sms_data(filepath)")
    print("  - clean_text(text)")
    print("  - tokenize_and_stem(text)")
    print("  - vectorize_text(texts, method='tfidf')")
    print("  - preprocess_pipeline(csv_path)")
