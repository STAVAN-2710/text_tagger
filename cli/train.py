"""
Model training command for CLI.
Simple, clean training logic following CLAUDE.md principles.
"""

from ml.train import train_model, get_feature_importance
from cli.utils import print_success, print_error, print_info, print_table


def train_command(args):
    """
    Execute train command.

    Args:
        args: Parsed command-line arguments
    """
    print_info("Training Random Forest model on feedback data...")

    # Train model
    try:
        result = train_model()
    except Exception as e:
        print_error(f"Training failed: {e}")
        return 1

    if result is None:
        print_error("Insufficient feedback data. Need at least 50 samples.")
        return 1

    # Print training results
    print_success("Model trained successfully!")
    print(f"\nTraining Statistics:")
    print(f"  Samples: {result['num_samples']}")
    print(f"  Cross-validation accuracy: {result['cv_accuracy']:.1%}")
    print(f"  Model saved: {result['model_path']}")

    # Show feature importance if requested
    if args.show_importance:
        try:
            importance = get_feature_importance(result['model_path'])
            print(f"\nFeature Importance:")

            headers = ["Feature", "Importance"]
            rows = [[feat, f"{imp:.4f}"] for feat, imp in importance.items()]
            print_table(headers, rows)
        except Exception as e:
            print_error(f"Failed to get feature importance: {e}")

    return 0
