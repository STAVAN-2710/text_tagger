"""Tests for ML prediction and scoring."""

import pytest
import numpy as np
from ml.predict import normalize_scores, combine_scores
from ml.config import TRAINING_THRESHOLD, DEFAULT_ALPHA, FEATURE_COLUMNS


def test_normalize_scores():
    """Test score normalization to 0-1 range."""
    scores = [0.1, 0.5, 0.3, 0.9, 0.2]
    normalized = normalize_scores(scores)

    # Check normalization bounds
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    assert len(normalized) == 5

    # Check relative ordering is preserved
    assert normalized[0] < normalized[1]  # 0.1 < 0.5
    assert normalized[1] > normalized[2]  # 0.5 > 0.3


def test_combine_scores():
    """Test hybrid score combination formula."""
    yake_scores = [0.1, 0.2, 0.3]
    ml_probs = [0.9, 0.7, 0.5]
    alpha = 0.7

    final = combine_scores(yake_scores, ml_probs, alpha)

    # Check output properties
    assert len(final) == 3
    assert all(isinstance(score, (float, np.floating)) for score in final)
    assert all(0 <= score <= 1 for score in final)

    # Test with different alpha values
    final_yake_heavy = combine_scores(yake_scores, ml_probs, alpha=0.9)
    final_ml_heavy = combine_scores(yake_scores, ml_probs, alpha=0.3)

    # Scores should differ based on alpha
    assert not np.array_equal(final_yake_heavy, final_ml_heavy)


def test_config_values():
    """Test ML configuration values are reasonable."""
    assert TRAINING_THRESHOLD >= 10
    assert TRAINING_THRESHOLD <= 100
    assert 0.0 <= DEFAULT_ALPHA <= 1.0
    assert 'yake_score' in FEATURE_COLUMNS
    assert 'size' in FEATURE_COLUMNS
    assert 'wfreq' in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 7
