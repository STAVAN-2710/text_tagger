#!/usr/bin/env python3
"""
Intelligent Text Tagger - Simple CLI Interface
Minimal, easy-to-understand command-line tool for keyword extraction.
"""

import sys
import argparse
from pathlib import Path
from yake import KeywordExtractor
from ml.train import train_model, get_feature_importance
from ml.predict import load_latest_model, predict_keywords

# =============================================================================
# Helper Functions
# =============================================================================

def extract_keywords(text, n=3, top=15, dedup_lim=0.9):
    """Extract keywords using YAKE."""
    extractor = KeywordExtractor(n=n, top=top, dedup_threshold=dedup_lim)
    return extractor.extract_keywords(text)

def read_file(filepath):
    """Read text from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

def print_keywords(keywords, title="Keywords"):
    """Print keywords as numbered list."""
    print(f"\n{title}:")
    for i, (kw, score) in enumerate(keywords, 1):
        print(f"{i}. {kw} ({score:.4f})")

# =============================================================================
# Command Handlers
# =============================================================================

def extract_command(args):
    """Extract keywords from text file."""
    text = read_file(args.input)
    keywords = extract_keywords(text, args.ngrams, args.top, args.dedup)
    print_keywords(keywords)

def tag_command(args):
    """Tag text with keywords."""
    text = read_file(args.input)
    keywords = extract_keywords(text, args.ngrams, args.top, args.dedup)
    print_keywords(keywords, "Keywords")

def train_command(args):
    """Train ML model on feedback data."""
    print("🔄 Training model...")
    result = train_model()

    if result is None:
        print("❌ Insufficient feedback data. Use Streamlit UI to provide feedback.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Model trained! Accuracy: {result['cv_accuracy']:.2%}")

def stats_command(args):
    """Show system statistics."""
    import json
    from ml.config import FEEDBACK_CSV

    print("\n📊 System Statistics")

    # Check feedback
    if Path(FEEDBACK_CSV).exists():
        import pandas as pd
        df = pd.read_csv(FEEDBACK_CSV)
        total = len(df)
        accepted = df['label'].sum()
        print(f"Feedback: {total} entries ({accepted} accepted, {total-accepted} rejected)")
    else:
        print("Feedback: No data")

    # Check model
    model = load_latest_model()
    if model:
        print("Model: Trained ✓")
    else:
        print("Model: Not trained")

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Intelligent Text Tagger")
    subs = parser.add_subparsers(dest='command')

    # Extract command
    ext = subs.add_parser('extract', help='Extract keywords')
    ext.add_argument('-i', '--input', required=True)
    ext.add_argument('-t', '--top', type=int, default=15)
    ext.add_argument('-n', '--ngrams', type=int, default=3)
    ext.add_argument('-d', '--dedup', type=float, default=0.9)

    # Tag command (same as extract)
    tag = subs.add_parser('tag', help='Tag keywords')
    tag.add_argument('-i', '--input', required=True)
    tag.add_argument('-t', '--top', type=int, default=10)
    tag.add_argument('-n', '--ngrams', type=int, default=3)
    tag.add_argument('-d', '--dedup', type=float, default=0.9)

    # Train & stats commands
    subs.add_parser('train', help='Train ML model')
    subs.add_parser('stats', help='Show statistics')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route commands
    if args.command == 'extract':
        extract_command(args)
    elif args.command == 'tag':
        tag_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'stats':
        stats_command(args)

if __name__ == '__main__':
    main()
