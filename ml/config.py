"""
ML configuration for keyword classification.
Simple, clean configuration following CLAUDE.md principles.
"""

# Feature columns from feedback CSV
FEATURE_COLUMNS = [
    'length',
    'yake_score',
    'f1_wfreq',
    'f2_wcase',
    'f3_wpos',
    'f4_wrel',
    'f5_wspread'
]

# Random Forest hyperparameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Scoring configuration
DEFAULT_ALPHA = 0.7  # 70% YAKE, 30% ML
TRAINING_THRESHOLD = 50  # Minimum feedbacks needed to train

# Paths
FEEDBACK_CSV = 'data/feedback.csv'
MODEL_DIR = 'ml/models'
