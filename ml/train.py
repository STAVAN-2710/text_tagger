"""Model training module for keyword classification."""

import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from ml.config import FEATURE_COLUMNS, MODEL_PARAMS, FEEDBACK_CSV, MODEL_DIR, TRAINING_THRESHOLD


def load_feedback_data():
    """Load feedback data and remove duplicate keywords.
    Drops keywords with conflicting labels entirely."""
    if not os.path.exists(FEEDBACK_CSV):
        return None, None, None

    df = pd.read_csv(FEEDBACK_CSV)
    stats = {'original_samples': len(df), 'duplicates_removed': 0,
             'conflicts_dropped': 0, 'final_samples': len(df)}

    if len(df) < TRAINING_THRESHOLD:
        return None, None, None

    original_count = len(df)
    conflicting_keywords = df.groupby('keyword')['label'].apply(lambda x: x.nunique() > 1)
    conflicting_keywords = conflicting_keywords[conflicting_keywords].index
    df = df[~df['keyword'].isin(conflicting_keywords)]
    stats['conflicts_dropped'] = len(conflicting_keywords)

    df = df.drop_duplicates(subset=['keyword'], keep='first')
    stats['duplicates_removed'] = original_count - len(df)
    stats['final_samples'] = len(df)

    if len(df) < TRAINING_THRESHOLD:
        return None, None, stats

    X = df[FEATURE_COLUMNS].values
    y = df['label'].values
    return X, y, stats


def train_model():
    """Train Random Forest classifier on feedback data.
    Returns dict with model_path, num_samples, cv_accuracy, dedup_stats."""
    X, y, dedup_stats = load_feedback_data()
    if X is None:
        return None

    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='accuracy')

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f'rf_model_{timestamp}.pkl')
    joblib.dump(model, model_path)

    return {'model_path': model_path, 'num_samples': len(X),
            'cv_accuracy': cv_scores.mean(), 'timestamp': timestamp,
            'dedup_stats': dedup_stats}


def get_feature_importance(model_path):
    """Get feature importance from trained model."""
    model = joblib.load(model_path)
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
