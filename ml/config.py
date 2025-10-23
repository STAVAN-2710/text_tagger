"""ML configuration for keyword classification."""

MODEL_DIR = 'ml/models'
FEEDBACK_CSV = 'data/feedback.csv'
TRAINING_THRESHOLD = 10
DEFAULT_ALPHA = 0.5

FEATURE_COLUMNS = [
    'size',
    'yake_score',
    'wfreq',
    'wcase',
    'wpos',
    'wrel',
    'wspread'
]

MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}
