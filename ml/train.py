"""Model training module for keyword classification."""

import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from ml.config import FEATURE_COLUMNS, MODEL_PARAMS, FEEDBACK_CSV, MODEL_DIR, TRAINING_THRESHOLD


def load_feedback_data():
    """
    Load all feedback data from CSV.

    Returns:
        tuple: (X, y) features and labels, or (None, None) if insufficient data
    """
    if not os.path.exists(FEEDBACK_CSV):
        return None, None

    df = pd.read_csv(FEEDBACK_CSV)

    if len(df) < TRAINING_THRESHOLD:
        return None, None

    # Extract features and labels
    X = df[FEATURE_COLUMNS].values
    y = df['label'].values

    return X, y


def train_model():
    """
    Train Random Forest on all available feedback data.

    Returns:
        dict: Training results with model_path, num_samples, cv_accuracy
              Returns None if insufficient data
    """
    # Load data
    X, y = load_feedback_data()

    if X is None:
        return None

    # Train Random Forest
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='accuracy')
    avg_cv_score = cv_scores.mean()

    # Save model with timestamp
    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'rf_model_{timestamp}.pkl')
    joblib.dump(model, model_path)

    # Return training stats
    return {
        'model_path': model_path,
        'num_samples': len(X),
        'cv_accuracy': avg_cv_score,
        'timestamp': timestamp
    }


def get_feature_importance(model_path):
    """
    Get feature importance from trained model.

    Args:
        model_path: Path to saved model

    Returns:
        dict: Feature names mapped to importance scores
    """
    model = joblib.load(model_path)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
