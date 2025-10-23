"""
Test script to extract and display all YAKE intermediate features (SIMPLIFIED version).
Run with: poetry run python test_features_simplified.py
"""

from yake_simplified import KeywordExtractor
from yake_simplified.scorer import extract_keywords_with_scores


def extract_keywords_with_features(text, n=3, top=10):
    """
    Extract keywords with all intermediate YAKE features.

    Returns:
        List of dicts with keyword and all features
    """
    # Initialize extractor
    kw_extractor = KeywordExtractor(n=n, top=top, dedup_threshold=0.5)

    # Get detailed results from scorer
    result = extract_keywords_with_scores(
        text.replace("\n", " "),
        kw_extractor.stopwords,
        n_grams=n,
        window_size=1
    )

    candidates = result['candidates']
    terms = result['terms']

    # Sort candidates by score
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])[:top]

    # Extract features for each candidate
    results = []
    for keyword, score in sorted_candidates:
        # Get the words in this keyword
        words = keyword.lower().split()

        keyword_data = {
            'keyword': keyword,
            'yake_score': score,
            'size': len(words),
        }

        # For single-word keywords, get features directly
        if len(words) == 1:
            word = words[0]
            # Handle plural normalization
            if word.endswith('s') and len(word) > 3:
                word_normalized = word[:-1]
            else:
                word_normalized = word

            if word_normalized in terms:
                term = terms[word_normalized]
                keyword_data['tf'] = term.get('tf', 0)
                keyword_data['wfreq'] = term.get('wfreq', 0)
                keyword_data['wcase'] = term.get('wcase', 0)
                keyword_data['wpos'] = term.get('wpos', 0)
                keyword_data['wrel'] = term.get('wrel', 0)
                keyword_data['wspread'] = term.get('wspread', 0)
            else:
                # Fallback values
                keyword_data['tf'] = 0
                keyword_data['wfreq'] = 0
                keyword_data['wcase'] = 0
                keyword_data['wpos'] = 0
                keyword_data['wrel'] = 0
                keyword_data['wspread'] = 0

        # For multi-word keywords, aggregate features from constituent terms
        else:
            wfreq_vals = []
            wcase_vals = []
            wpos_vals = []
            wrel_vals = []
            wspread_vals = []
            tf_sum = 0

            for word in words:
                # Handle plural normalization
                if word.endswith('s') and len(word) > 3:
                    word_normalized = word[:-1]
                else:
                    word_normalized = word

                if word_normalized in terms and not terms[word_normalized].get('is_stopword', False):
                    term = terms[word_normalized]
                    wfreq_vals.append(term.get('wfreq', 0))
                    wcase_vals.append(term.get('wcase', 0))
                    wpos_vals.append(term.get('wpos', 0))
                    wrel_vals.append(term.get('wrel', 0))
                    wspread_vals.append(term.get('wspread', 0))
                    tf_sum += term.get('tf', 0)

            keyword_data['tf'] = tf_sum
            keyword_data['wfreq'] = sum(wfreq_vals) / len(wfreq_vals) if wfreq_vals else 0
            keyword_data['wcase'] = sum(wcase_vals) / len(wcase_vals) if wcase_vals else 0
            keyword_data['wpos'] = sum(wpos_vals) / len(wpos_vals) if wpos_vals else 0
            keyword_data['wrel'] = sum(wrel_vals) / len(wrel_vals) if wrel_vals else 0
            keyword_data['wspread'] = sum(wspread_vals) / len(wspread_vals) if wspread_vals else 0

        results.append(keyword_data)

    return results


def main():
    print("=" * 110)
    print("YAKE KEYWORD EXTRACTION WITH INTERMEDIATE FEATURES (SIMPLIFIED VERSION)")
    print("=" * 110)

    text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    Deep learning, a type of machine learning, uses neural networks with
    multiple layers to analyze complex patterns in data.
    Natural language processing is another important area of AI that helps
    computers understand and process human language.
    """

    print("\nInput Text:")
    print("-" * 110)
    print(text.strip())
    print()

    # Extract keywords with features
    print("\nExtracting keywords with features...\n")
    results = extract_keywords_with_features(text, n=2, top=10)

    # Display results with all features
    print("Keywords with Intermediate Features:")
    print("=" * 110)
    print(f"{'#':<3} {'Keyword':<30} {'Len':<5} {'YAKE':<8} {'F1:Freq':<10} {'F2:Case':<10} {'F3:Pos':<10} {'F4:Rel':<10} {'F5:Spread':<10}")
    print("-" * 110)

    for i, kw_data in enumerate(results, 1):
        print(f"{i:<3} {kw_data['keyword']:<30} "
              f"{kw_data['size']:<5} "
              f"{kw_data['yake_score']:<8.4f} "
              f"{kw_data.get('wfreq', 0):<10.4f} "
              f"{kw_data.get('wcase', 0):<10.4f} "
              f"{kw_data.get('wpos', 0):<10.4f} "
              f"{kw_data.get('wrel', 0):<10.4f} "
              f"{kw_data.get('wspread', 0):<10.4f}")

    print("=" * 110)

    # Explain features
    print("\nFeature Explanations:")
    print("-" * 110)
    print("F1 (wfreq):   Word Frequency - TF normalized by (mean + std)")
    print("F2 (wcase):   Word Casing - Uppercase/proper noun preference")
    print("F3 (wpos):    Word Position - Earlier words score better")
    print("F4 (wrel):    Word Relatedness - Co-occurrence with other terms")
    print("F5 (wspread): Word Spread - Distribution across sentences")
    print("Len:          Candidate Length - Number of words in the keyword")
    print("\nYAKE Score = (F3 * F4) / (F2 + F1/F4 + F5/F4)")
    print("             Lower score = more relevant keyword")
    print("=" * 110)


if __name__ == "__main__":
    main()
