"""ML configuration for keyword classification."""

MODEL_DIR = 'ml/models'
FEEDBACK_CSV = 'data/feedback.csv'
TRAINING_THRESHOLD = 10
DEFAULT_ALPHA = 0.5

FEATURE_COLUMNS = [
    'length',
    'yake_score',
    'f1_wfreq',
    'f2_wcase',
    'f3_wpos',
    'f4_wrel',
    'f5_wspread'
]

MODEL_PARAMS = {
    'n_estimators': 50,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 8,
    'random_state': 42
}
