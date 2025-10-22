"""
Prediction and scoring utilities for keyword classification.
Simple, clean prediction logic following CLAUDE.md principles.
"""

import numpy as np
import joblib
import os
import glob
from ml.config import FEATURE_COLUMNS, MODEL_DIR, DEFAULT_ALPHA


def load_latest_model():
    """
    Load the most recently trained model.

    Returns:
        sklearn model or None if no model exists
    """
    if not os.path.exists(MODEL_DIR):
        return None

    # Find all model files
    model_files = glob.glob(os.path.join(MODEL_DIR, 'rf_model_*.pkl'))

    if not model_files:
        return None

    # Get most recent model
    latest_model = max(model_files, key=os.path.getmtime)
    return joblib.load(latest_model)


def prepare_features(keywords_data):
    """
    Prepare feature matrix from keyword data.

    Args:
        keywords_data: List of keyword dictionaries with features

    Returns:
        numpy array of features
    """
    features = []
    for kw in keywords_data:
        features.append([
            kw['size'],
            kw['yake_score'],
            kw['wfreq'],
            kw['wcase'],
            kw['wpos'],
            kw['wrel'],
            kw['wspread']
        ])
    return np.array(features)


def predict_keywords(keywords_data, model=None):
    """
    Predict probabilities for keywords being relevant.

    Args:
        keywords_data: List of keyword dictionaries
        model: Trained model (if None, loads latest)

    Returns:
        numpy array of probabilities (0-1), or None if no model
    """
    if model is None:
        model = load_latest_model()

    if model is None:
        return None

    X = prepare_features(keywords_data)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (relevant)

    return probabilities


def normalize_scores(scores):
    """
    Normalize scores to 0-1 range using min-max normalization.

    Args:
        scores: List or array of scores

    Returns:
        numpy array of normalized scores
    """
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()

    # Avoid division by zero
    if max_score - min_score < 1e-10:
        return np.zeros_like(scores)

    return (scores - min_score) / (max_score - min_score)


def combine_scores(yake_scores, model_probs, alpha=DEFAULT_ALPHA):
    """
    Combine YAKE and model scores using weighted sum.

    Formula: final = alpha * yake_norm + (1-alpha) * (1-model_prob)
    Lower final score = better keyword

    Args:
        yake_scores: List of YAKE scores (lower is better)
        model_probs: List of model probabilities (higher is better)
        alpha: Weight for YAKE (0-1), model gets (1-alpha)

    Returns:
        numpy array of combined scores (lower is better)
    """
    # Normalize YAKE scores to 0-1
    yake_norm = normalize_scores(yake_scores)

    # Invert model probabilities (lower is better)
    model_inv = 1 - np.array(model_probs)

    # Weighted combination
    final_scores = alpha * yake_norm + (1 - alpha) * model_inv

    return final_scores


def get_predictions_with_scores(keywords_data, alpha=DEFAULT_ALPHA):
    """
    Get predictions and combined scores for keywords.

    Args:
        keywords_data: List of keyword dictionaries
        alpha: Weight for YAKE score combination

    Returns:
        dict with model_probs, final_scores, and model_available flag
    """
    model = load_latest_model()

    if model is None:
        return {
            'model_available': False,
            'model_probs': None,
            'final_scores': None
        }

    # Get model predictions
    model_probs = predict_keywords(keywords_data, model)

    # Get YAKE scores
    yake_scores = [kw['yake_score'] for kw in keywords_data]

    # Combine scores
    final_scores = combine_scores(yake_scores, model_probs, alpha)

    return {
        'model_available': True,
        'model_probs': model_probs,
        'final_scores': final_scores
    }
