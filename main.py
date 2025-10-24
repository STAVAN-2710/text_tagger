#!/usr/bin/env python3
"""Intelligent Text Tagger - CLI for keyword extraction."""

import sys
import argparse
from pathlib import Path
from yake import KeywordExtractor
from yake.scorer import extract_keywords_with_scores
from yake.utils import extract_features_for_keyword
from ml.train import train_model
from ml.predict import load_latest_model, get_predictions_with_scores
from ml.config import DEFAULT_ALPHA


def extract_keywords(text, n=3, top=15, dedup_lim=0.9, use_hybrid=False, alpha=DEFAULT_ALPHA):
    """Extract keywords with optional ML enhancement."""
    extractor = KeywordExtractor(n=n, top=top, dedup_threshold=dedup_lim)

    if not use_hybrid:
        return extractor.extract_keywords(text)

    # Hybrid mode: extract larger pool for ML filtering
    extractor_large = KeywordExtractor(n=n, top=top*3, dedup_threshold=dedup_lim)
    keywords = extractor_large.extract_keywords(text)

    result = extract_keywords_with_scores(text.replace("\n", " "), extractor.stopwords, n_grams=n, window_size=1)
    keywords_data = [extract_features_for_keyword(kw, score, result['terms']) for kw, score in keywords]

    predictions = get_predictions_with_scores(keywords_data, alpha=alpha)

    if not predictions['model_available']:
        print("WARNING: No trained model found. Using YAKE only. Train a model with 'python main.py train'", file=sys.stderr)
        return keywords[:top]

    for i, kw_data in enumerate(keywords_data):
        kw_data.update({'ml_prob': predictions['model_probs'][i], 'final_score': predictions['final_scores'][i]})

    keywords_data.sort(key=lambda x: x['final_score'])
    return [(kw['keyword'], kw['final_score']) for kw in keywords_data[:top]]

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"ERROR: Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

def print_keywords(keywords, title="Keywords"):
    print(f"\n{title}:")
    for i, (kw, score) in enumerate(keywords, 1):
        print(f"{i}. {kw} ({score:.4f})")


def extract_command(args):
    text = read_file(args.input)
    keywords = extract_keywords(text, args.ngrams, args.top, args.dedup, args.hybrid, args.alpha)

    if args.feedback:
        collect_feedback(keywords, text, args.ngrams)
    else:
        print_keywords(keywords, f"Keywords ({'Hybrid (YAKE + ML)' if args.hybrid else 'YAKE Only'})")

def collect_feedback(keywords, text, n_grams):
    """Interactive feedback collection for keywords."""
    from feedback_manager import FeedbackManager

    print("\nReview Keywords (y=accept, n=reject, s=skip, q=quit):\n")

    fb = FeedbackManager()
    feedback_data, labels = [], []

    extractor = KeywordExtractor(n=n_grams, dedup_threshold=0.9)
    result = extract_keywords_with_scores(text.replace("\n", " "), extractor.stopwords, n_grams=n_grams, window_size=1)
    terms = result['terms']

    for i, (kw, score) in enumerate(keywords, 1):
        while True:
            response = input(f"{i}. {kw} ({score:.4f}) - [y/n/s/q]: ").lower().strip()

            if response == 'q':
                if feedback_data:
                    fb.save_feedback('cli_input', feedback_data, labels)
                print(f"\nSaved {sum(labels)} accepted, {len(labels) - sum(labels)} rejected")
                return

            if response == 's':
                break

            if response in ['y', 'n']:
                feedback_data.append(extract_features_for_keyword(kw, score, terms))
                labels.append(1 if response == 'y' else 0)
                print(f"   {'✓ Accepted' if response == 'y' else '✗ Rejected'}")
                break
            else:
                print("   Invalid input. Use y/n/s/q")

    if feedback_data:
        fb.save_feedback('cli_input', feedback_data, labels)
    print(f"\nSaved {sum(labels)} accepted, {len(labels) - sum(labels)} rejected")

def train_command(args):
    print("Training model...")
    result = train_model()

    if result is None:
        print("ERROR: Insufficient feedback data. Use Streamlit UI to provide feedback.", file=sys.stderr)
        sys.exit(1)

    print(f"Model trained! Accuracy: {result['cv_accuracy']:.2%}")

def stats_command(args):
    from ml.config import FEEDBACK_CSV
    import pandas as pd

    print("\nSystem Statistics")

    if Path(FEEDBACK_CSV).exists():
        df = pd.read_csv(FEEDBACK_CSV)
        total = len(df)
        accepted = df['label'].sum()
        print(f"Feedback: {total} entries ({accepted} accepted, {total-accepted} rejected)")
    else:
        print("Feedback: No data")

    model = load_latest_model()
    print(f"Model: {'Trained' if model else 'Not trained'}")


def main():
    parser = argparse.ArgumentParser(description="Intelligent Text Tagger")
    subs = parser.add_subparsers(dest='command')

    ext = subs.add_parser('extract', help='Extract keywords')
    ext.add_argument('-i', '--input', required=True)
    ext.add_argument('-t', '--top', type=int, default=15)
    ext.add_argument('-n', '--ngrams', type=int, default=3)
    ext.add_argument('-d', '--dedup', type=float, default=0.6)
    ext.add_argument('-f', '--feedback', action='store_true', help='Interactive feedback mode')
    ext.add_argument('--hybrid', action='store_true', help='Use hybrid mode (YAKE + ML)')
    ext.add_argument('--alpha', type=float, default=DEFAULT_ALPHA, help=f'YAKE weight in hybrid mode (default: {DEFAULT_ALPHA})')

    subs.add_parser('train', help='Train ML model')
    subs.add_parser('stats', help='Show statistics')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {'extract': extract_command, 'train': train_command, 'stats': stats_command}[args.command](args)

if __name__ == '__main__':
    main()
