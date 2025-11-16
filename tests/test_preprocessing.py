"""
Unit Tests for Data Preprocessing Module

Tests cover:
- Data loading from CSV
- Text cleaning functionality
- Tokenization and stemming
- Vectorization methods (BoW and TF-IDF)
- Complete pipeline integration
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
from scipy.sparse import issparse
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import (
    load_sms_data,
    clean_text,
    tokenize_and_stem,
    vectorize_text,
    preprocess_pipeline
)


class TestLoadSmsData:
    """Test data loading functionality."""

    def test_load_valid_csv(self):
        """Test loading a valid CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ham\tThis is a normal message\n")
            f.write("spam\tBuy now! Click here\n")
            f.write("ham\tHello how are you\n")
            temp_path = f.name

        try:
            texts, labels, label_names = load_sms_data(temp_path)

            assert len(texts) == 3
            assert len(labels) == 3
            assert label_names == ['ham', 'spam']
            assert labels[0] == 0  # ham
            assert labels[1] == 1  # spam

        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_sms_data('/nonexistent/path/file.csv')

    def test_load_csv_with_binary_labels(self):
        """Test loading CSV with numeric labels (0, 1)."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("0\tNormal message\n")
            f.write("1\tSpam message\n")
            temp_path = f.name

        try:
            texts, labels, label_names = load_sms_data(temp_path)

            assert len(texts) == 2
            assert labels == [0, 1]

        finally:
            os.unlink(temp_path)


class TestCleanText:
    """Test text cleaning functionality."""

    def test_clean_basic(self):
        """Test basic text cleaning."""
        text = "Hello World!"
        cleaned = clean_text(text)
        assert cleaned == "hello world"

    def test_clean_removes_numbers(self):
        """Test removal of numbers."""
        text = "Message 123 with numbers 456"
        cleaned = clean_text(text)
        assert "123" not in cleaned
        assert "456" not in cleaned

    def test_clean_removes_special_chars(self):
        """Test removal of special characters."""
        text = "Hello@World#$%^&*()"
        cleaned = clean_text(text)
        assert "@" not in cleaned
        assert "#" not in cleaned

    def test_clean_removes_urls(self):
        """Test removal of URLs."""
        text = "Check this http://example.com and www.test.com"
        cleaned = clean_text(text)
        assert "http" not in cleaned
        assert "www" not in cleaned

    def test_clean_removes_email(self):
        """Test removal of email addresses."""
        text = "Contact me at user@example.com"
        cleaned = clean_text(text)
        assert "@" not in cleaned

    def test_clean_lowercase(self):
        """Test text is converted to lowercase."""
        text = "HELLO World"
        cleaned = clean_text(text)
        assert cleaned == "hello world"


class TestTokenizeAndStem:
    """Test tokenization and stemming."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Hello world test"
        tokens = tokenize_and_stem(text, remove_stopwords=False)
        assert len(tokens) >= 1

    def test_tokenize_remove_stopwords(self):
        """Test stopword removal."""
        text = "the quick brown fox"
        tokens = tokenize_and_stem(text, remove_stopwords=True)
        # 'the' is a stopword and should be removed
        assert 'the' not in tokens

    def test_stemming(self):
        """Test stemming functionality."""
        text = "running runner runs"
        tokens = tokenize_and_stem(text, remove_stopwords=False)
        # After stemming, these should have the same root
        assert len(tokens) > 0


class TestVectorizeText:
    """Test text vectorization."""

    def test_vectorize_bow(self):
        """Test Bag-of-Words vectorization."""
        texts = ["hello world", "hello test", "world test"]
        result = vectorize_text(texts, method='bow', max_features=10)

        assert issparse(result['X'])
        assert result['X'].shape[0] == 3
        assert result['method'] == 'bow'
        assert len(result['feature_names']) > 0
        assert result['vectorizer'] is not None

    def test_vectorize_tfidf(self):
        """Test TF-IDF vectorization."""
        texts = ["hello world", "hello test", "world test"]
        result = vectorize_text(texts, method='tfidf', max_features=10)

        assert issparse(result['X'])
        assert result['X'].shape[0] == 3
        assert result['method'] == 'tfidf'
        assert len(result['feature_names']) > 0

    def test_vectorize_invalid_method(self):
        """Test error handling for invalid method."""
        texts = ["hello world"]
        with pytest.raises(ValueError):
            vectorize_text(texts, method='invalid')

    def test_vectorize_max_features(self):
        """Test max_features parameter."""
        texts = ["hello world test"] * 20
        result = vectorize_text(texts, method='tfidf', max_features=5, min_df=1, max_df=1.0)

        assert result['X'].shape[1] <= 5


class TestPreprocessPipeline:
    """Test complete preprocessing pipeline."""

    def test_pipeline_basic(self):
        """Test basic pipeline execution."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ham\tThis is a normal message\n")
            f.write("spam\tBuy now click here\n")
            f.write("ham\tHow are you today\n")
            f.write("spam\tFree money win now\n")
            temp_path = f.name

        try:
            result = preprocess_pipeline(temp_path, method='tfidf', min_df=1)

            assert 'X' in result
            assert 'y' in result
            assert 'feature_names' in result
            assert 'vectorizer' in result
            assert 'metadata' in result

            assert result['X'].shape[0] == 4  # 4 messages
            assert result['y'].shape[0] == 4
            assert result['metadata']['total_samples'] == 4

        finally:
            os.unlink(temp_path)

    def test_pipeline_bag_of_words(self):
        """Test pipeline with Bag-of-Words."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ham\ttest message\n")
            f.write("spam\ttest spam\n")
            temp_path = f.name

        try:
            result = preprocess_pipeline(temp_path, method='bow', min_df=1)

            assert result['metadata']['method'] == 'bow'
            assert issparse(result['X'])

        finally:
            os.unlink(temp_path)

    def test_pipeline_labels(self):
        """Test pipeline generates correct labels."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ham\tMessage 1\n")
            f.write("spam\tMessage 2\n")
            f.write("ham\tMessage 3\n")
            temp_path = f.name

        try:
            result = preprocess_pipeline(temp_path)

            assert result['y'][0] == 0  # ham
            assert result['y'][1] == 1  # spam
            assert result['y'][2] == 0  # ham

        finally:
            os.unlink(temp_path)

    def test_pipeline_metadata(self):
        """Test pipeline metadata is correct."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            f.write("ham\ttest\n")
            f.write("spam\ttest spam\n")
            f.write("ham\ttest ham\n")
            temp_path = f.name

        try:
            result = preprocess_pipeline(temp_path)
            metadata = result['metadata']

            assert metadata['total_samples'] == 3
            assert metadata['feature_dimension'] > 0
            assert 'sparsity' in metadata
            assert metadata['label_distribution']['ham'] == 2
            assert metadata['label_distribution']['spam'] == 1

        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_clean_empty_string(self):
        """Test cleaning empty string."""
        result = clean_text("")
        assert result == ""

    def test_clean_only_special_chars(self):
        """Test cleaning string with only special characters."""
        result = clean_text("@#$%^&*()")
        assert result == ""

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        result = tokenize_and_stem("", remove_stopwords=True)
        assert result == []

    def test_vectorize_single_text(self):
        """Test vectorizing single text."""
        result = vectorize_text(["hello world"], method='tfidf', min_df=1, max_df=1.0)
        assert result['X'].shape[0] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
