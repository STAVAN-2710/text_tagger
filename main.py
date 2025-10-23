#!/usr/bin/env python3
"""Intelligent Text Tagger - CLI for keyword extraction."""

import sys
import argparse
from pathlib import Path
from yake import KeywordExtractor
from ml.train import train_model
from ml.predict import load_latest_model


def extract_keywords(text, n=3, top=15, dedup_lim=0.9):
    extractor = KeywordExtractor(n=n, top=top, dedup_threshold=dedup_lim)
    return extractor.extract_keywords(text)

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

def print_keywords(keywords, title="Keywords"):
    print(f"\n{title}:")
    for i, (kw, score) in enumerate(keywords, 1):
        print(f"{i}. {kw} ({score:.4f})")


def extract_command(args):
    text = read_file(args.input)
    keywords = extract_keywords(text, args.ngrams, args.top, args.dedup)

    if args.feedback:
        collect_feedback(keywords, text, args.ngrams)
    else:
        print_keywords(keywords)

def collect_feedback(keywords, text, n_grams):
    """Interactive feedback collection for keywords."""
    from feedback_manager import FeedbackManager
    from yake.scorer import extract_keywords_with_scores

    print("\n📝 Review Keywords (y=accept, n=reject, s=skip, q=quit):\n")

    fb = FeedbackManager()
    feedback_data = []
    labels = []

    extractor = KeywordExtractor(n=n_grams, dedup_threshold=0.9)
    result = extract_keywords_with_scores(
        text.replace("\n", " "),
        extractor.stopwords,
        n_grams=n_grams,
        window_size=1
    )
    terms = result['terms']

    for i, (kw, score) in enumerate(keywords, 1):
        while True:
            response = input(f"{i}. {kw} ({score:.4f}) - [y/n/s/q]: ").lower().strip()

            if response == 'q':
                if feedback_data:
                    fb.save_feedback('cli_input', feedback_data, labels)
                accepted = sum(labels)
                rejected = len(labels) - accepted
                print(f"\n✅ Saved {accepted} accepted, {rejected} rejected")
                return

            if response == 's':
                break

            if response in ['y', 'n']:
                words = kw.lower().split()
                features = {
                    'keyword': kw,
                    'yake_score': score,
                    'size': len(words),
                    'wfreq': 0,
                    'wcase': 0,
                    'wpos': 0,
                    'wrel': 0,
                    'wspread': 0
                }

                if len(words) == 1:
                    word = words[0]
                    word_normalized = word[:-1] if word.endswith('s') and len(word) > 3 else word
                    if word_normalized in terms:
                        term = terms[word_normalized]
                        features['wfreq'] = term.get('wfreq', 0)
                        features['wcase'] = term.get('wcase', 0)
                        features['wpos'] = term.get('wpos', 0)
                        features['wrel'] = term.get('wrel', 0)
                        features['wspread'] = term.get('wspread', 0)
                else:
                    wfreq_vals, wcase_vals, wpos_vals, wrel_vals, wspread_vals = [], [], [], [], []
                    for word in words:
                        word_normalized = word[:-1] if word.endswith('s') and len(word) > 3 else word
                        if word_normalized in terms and not terms[word_normalized].get('is_stopword', False):
                            term = terms[word_normalized]
                            wfreq_vals.append(term.get('wfreq', 0))
                            wcase_vals.append(term.get('wcase', 0))
                            wpos_vals.append(term.get('wpos', 0))
                            wrel_vals.append(term.get('wrel', 0))
                            wspread_vals.append(term.get('wspread', 0))

                    if wfreq_vals:
                        features['wfreq'] = sum(wfreq_vals) / len(wfreq_vals)
                        features['wcase'] = sum(wcase_vals) / len(wcase_vals)
                        features['wpos'] = sum(wpos_vals) / len(wpos_vals)
                        features['wrel'] = sum(wrel_vals) / len(wrel_vals)
                        features['wspread'] = sum(wspread_vals) / len(wspread_vals)

                feedback_data.append(features)
                labels.append(1 if response == 'y' else 0)

                if response == 'y':
                    print(f"   ✓ Accepted")
                else:
                    print(f"   ✗ Rejected")
                break
            else:
                print("   Invalid input. Use y/n/s/q")

    if feedback_data:
        fb.save_feedback('cli_input', feedback_data, labels)
    accepted = sum(labels)
    rejected = len(labels) - accepted
    print(f"\n✅ Saved {accepted} accepted, {rejected} rejected")

def train_command(args):
    print("🔄 Training model...")
    result = train_model()

    if result is None:
        print("❌ Insufficient feedback data. Use Streamlit UI to provide feedback.", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Model trained! Accuracy: {result['cv_accuracy']:.2%}")

def stats_command(args):
    from ml.config import FEEDBACK_CSV
    import pandas as pd

    print("\n📊 System Statistics")

    if Path(FEEDBACK_CSV).exists():
        df = pd.read_csv(FEEDBACK_CSV)
        total = len(df)
        accepted = df['label'].sum()
        print(f"Feedback: {total} entries ({accepted} accepted, {total-accepted} rejected)")
    else:
        print("Feedback: No data")

    model = load_latest_model()
    print(f"Model: {'Trained ✓' if model else 'Not trained'}")


def main():
    parser = argparse.ArgumentParser(description="Intelligent Text Tagger")
    subs = parser.add_subparsers(dest='command')

    ext = subs.add_parser('extract', help='Extract keywords')
    ext.add_argument('-i', '--input', required=True)
    ext.add_argument('-t', '--top', type=int, default=15)
    ext.add_argument('-n', '--ngrams', type=int, default=3)
    ext.add_argument('-d', '--dedup', type=float, default=0.9)
    ext.add_argument('-f', '--feedback', action='store_true', help='Interactive feedback mode')

    subs.add_parser('train', help='Train ML model')
    subs.add_parser('stats', help='Show statistics')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'extract':
        extract_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'stats':
        stats_command(args)

if __name__ == '__main__':
    main()
