"""Prediction and scoring utilities for keyword classification."""

import numpy as np
import joblib
import os
import glob
from ml.config import FEATURE_COLUMNS, MODEL_DIR, DEFAULT_ALPHA


def load_latest_model():
    """Load most recently trained model or None if no model exists."""
    if not os.path.exists(MODEL_DIR):
        return None
    model_files = glob.glob(os.path.join(MODEL_DIR, 'rf_model_*.pkl'))
    if not model_files:
        return None
    return joblib.load(max(model_files, key=os.path.getmtime))


def prepare_features(keywords_data):
    """Convert keyword dictionaries to feature matrix."""
    return np.array([[kw['size'], kw['yake_score'], kw['wfreq'], kw['wcase'],
                      kw['wpos'], kw['wrel'], kw['wspread']] for kw in keywords_data])


def predict_keywords(keywords_data, model=None):
    """Predict relevance probabilities for keywords (0-1)."""
    if model is None:
        model = load_latest_model()
    if model is None:
        return None
    return model.predict_proba(prepare_features(keywords_data))[:, 1]


def normalize_scores(scores):
    """Normalize scores to 0-1 range using min-max normalization."""
    scores = np.array(scores)
    min_score, max_score = scores.min(), scores.max()
    if max_score - min_score < 1e-10:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def combine_scores(yake_scores, model_probs, alpha=DEFAULT_ALPHA):
    """Combine YAKE and model scores using weighted sum.
    Formula: final = alpha * yake_norm + (1-alpha) * (1-model_prob)
    Lower final score = better keyword."""
    yake_norm = normalize_scores(yake_scores)
    model_inv = 1 - np.array(model_probs)
    return alpha * yake_norm + (1 - alpha) * model_inv


def get_predictions_with_scores(keywords_data, alpha=DEFAULT_ALPHA):
    """Get predictions and combined scores for keywords.
    Returns dict with model_probs, final_scores, and model_available flag."""
    model = load_latest_model()
    if model is None:
        return {'model_available': False, 'model_probs': None, 'final_scores': None}

    model_probs = predict_keywords(keywords_data, model)
    yake_scores = [kw['yake_score'] for kw in keywords_data]
    final_scores = combine_scores(yake_scores, model_probs, alpha)

    return {'model_available': True, 'model_probs': model_probs, 'final_scores': final_scores}
