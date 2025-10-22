#!/usr/bin/env python3
"""
Intelligent Text Tagger - CLI Interface

Command-line tool for keyword extraction, model training, and batch tagging.
Complements the Streamlit UI with automation capabilities.

Usage:
    python main.py extract --input file.txt --top 15
    python main.py train
    python main.py tag --input file.txt --mode hybrid
    python main.py stats

Simple, clean CLI following CLAUDE.md principles.
"""

import argparse
import sys
import os
from pathlib import Path
from feedback_manager import FeedbackManager
from ml.predict import load_latest_model
from ml.config import TRAINING_THRESHOLD, DEFAULT_ALPHA
from cli.utils import print_table, print_success, print_error, print_info


def stats_command(args):
    """Show system statistics."""
    print_info("System Statistics\n")

    # Feedback statistics
    fm = FeedbackManager()
    feedback_count = fm.get_feedback_count()

    if feedback_count > 0:
        all_feedback = fm.load_all_feedback()
        approved = sum(1 for f in all_feedback if int(f['label']) == 1)
        rejected = feedback_count - approved

        print("Feedback:")
        print(f"  Total: {feedback_count}")
        print(f"  Approved: {approved} ({approved/feedback_count:.1%})")
        print(f"  Rejected: {rejected} ({rejected/feedback_count:.1%})")
        print(f"  Training threshold: {TRAINING_THRESHOLD}")

        if feedback_count >= TRAINING_THRESHOLD:
            print_success(f"Ready for training ({feedback_count} >= {TRAINING_THRESHOLD})")
        else:
            print_info(f"Need {TRAINING_THRESHOLD - feedback_count} more samples for training")
    else:
        print("  No feedback data collected yet")

    # Model statistics
    print("\nModels:")
    model_dir = Path('ml/models')
    if model_dir.exists():
        model_files = list(model_dir.glob('rf_model_*.pkl'))
        if model_files:
            latest_model = max(model_files, key=os.path.getmtime)
            print(f"  Latest model: {latest_model.name}")
            print(f"  Total models: {len(model_files)}")

            # Try to load and get info
            try:
                model = load_latest_model()
                if model:
                    print_success("Model loaded successfully")
            except Exception as e:
                print_error(f"Failed to load model: {e}")
        else:
            print("  No trained models found")
    else:
        print("  Model directory not found")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Intelligent Text Tagger - Keyword extraction with YAKE and ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract keywords from a document
  python main.py extract --input document.txt --top 15

  # Process entire folder
  python main.py extract --input docs/ --output results.json

  # Train model from feedback
  python main.py train

  # Tag documents with hybrid mode
  python main.py tag --input document.txt --mode hybrid

  # Show system statistics
  python main.py stats
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract keywords using YAKE')
    extract_parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    extract_parser.add_argument('--top', '-t', type=int, default=15, help='Number of keywords (default: 15)')
    extract_parser.add_argument('--language', '-l', default='en', choices=['en', 'hi'], help='Language (default: en)')
    extract_parser.add_argument('--ngrams', '-n', type=int, default=3, help='Max n-gram size (default: 3)')
    extract_parser.add_argument('--dedup', '-d', type=float, default=0.9, help='Deduplication threshold (default: 0.9)')
    extract_parser.add_argument('--output', '-o', help='Save results to JSON file')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model from feedback')
    train_parser.add_argument('--show-importance', action='store_true', help='Show feature importance')

    # Tag command
    tag_parser = subparsers.add_parser('tag', help='Tag documents with YAKE or hybrid mode')
    tag_parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    tag_parser.add_argument('--mode', '-m', default='yake', choices=['yake', 'hybrid'], help='Tagging mode (default: yake)')
    tag_parser.add_argument('--top', '-t', type=int, default=15, help='Number of keywords (default: 15)')
    tag_parser.add_argument('--language', '-l', default='en', choices=['en', 'hi'], help='Language (default: en)')
    tag_parser.add_argument('--ngrams', '-n', type=int, default=3, help='Max n-gram size (default: 3)')
    tag_parser.add_argument('--dedup', '-d', type=float, default=0.5, help='Deduplication threshold (default: 0.5)')
    tag_parser.add_argument('--alpha', '-a', type=float, default=DEFAULT_ALPHA, help=f'YAKE weight for hybrid mode (default: {DEFAULT_ALPHA})')
    tag_parser.add_argument('--output', '-o', help='Save results to JSON file')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    if args.command == 'extract':
        from cli.extract import extract_command
        return extract_command(args)
    elif args.command == 'train':
        from cli.train import train_command
        return train_command(args)
    elif args.command == 'tag':
        from cli.tag import tag_command
        return tag_command(args)
    elif args.command == 'stats':
        return stats_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
