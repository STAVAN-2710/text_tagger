"""
Helper functions to extract and display YAKE features.

Makes it easy to inspect intermediate feature values for debugging or analysis.
"""

from .scorer import extract_keywords_with_scores


def get_keyword_features(text, stopwords, keyword, n_grams=3, window_size=1):
    """
    Get detailed feature breakdown for a specific keyword.

    Args:
        text: Input text
        stopwords: Set of stopwords
        keyword: The keyword to analyze (e.g., "machine learning")
        n_grams: Maximum n-gram length
        window_size: Co-occurrence window size

    Returns:
        Dict with all features for the keyword, or None if not found
    """
    result = extract_keywords_with_scores(
        text.replace("\n", " "),
        stopwords,
        n_grams=n_grams,
        window_size=window_size
    )

    # Check if keyword exists
    keyword_lower = keyword.lower()
    if keyword_lower not in result['candidates']:
        return None

    score = result['candidates'][keyword_lower]
    terms = result['terms']
    words = keyword_lower.split()

    feature_data = {
        'keyword': keyword,
        'score': score,
        'length': len(words),
        'words': []
    }

    # Get features for each word
    for word in words:
        # Handle plural normalization
        if word.endswith('s') and len(word) > 3:
            word_normalized = word[:-1]
        else:
            word_normalized = word

        if word_normalized in terms:
            term = terms[word_normalized]
            word_features = {
                'word': word,
                'normalized': word_normalized,
                'is_stopword': term.get('is_stopword', False),
                'tf': term.get('tf', 0),
                'wfreq': term.get('wfreq', 0),
                'wcase': term.get('wcase', 0),
                'wpos': term.get('wpos', 0),
                'wrel': term.get('wrel', 0),
                'wspread': term.get('wspread', 0),
                'score': term.get('score', 0),
            }
            feature_data['words'].append(word_features)

    return feature_data


def print_keyword_features(feature_data):
    """
    Pretty-print feature data for a keyword.

    Args:
        feature_data: Dict returned from get_keyword_features()
    """
    if not feature_data:
        print("Keyword not found!")
        return

    print("=" * 80)
    print(f"KEYWORD: '{feature_data['keyword']}'")
    print("=" * 80)
    print(f"Final YAKE Score: {feature_data['score']:.6f} (lower = better)")
    print(f"Length: {feature_data['length']} word(s)")
    print()

    if feature_data['words']:
        print("Constituent Word Features:")
        print("-" * 80)

        for i, word_data in enumerate(feature_data['words'], 1):
            print(f"\n{i}. '{word_data['word']}' (normalized: '{word_data['normalized']}')")
            print(f"   Stopword: {word_data['is_stopword']}")
            print(f"   TF:       {word_data['tf']}")
            print(f"   Score:    {word_data['score']:.6f}")
            print()
            print(f"   Features:")
            print(f"     F1 (wfreq):   {word_data['wfreq']:.6f}  - Frequency (normalized by mean+std)")
            print(f"     F2 (wcase):   {word_data['wcase']:.6f}  - Casing (uppercase/proper noun)")
            print(f"     F3 (wpos):    {word_data['wpos']:.6f}  - Position (earlier is better)")
            print(f"     F4 (wrel):    {word_data['wrel']:.6f}  - Relatedness (co-occurrence)")
            print(f"     F5 (wspread): {word_data['wspread']:.6f}  - Spread (sentence distribution)")

    print()
    print("=" * 80)


def compare_keywords(text, stopwords, keyword1, keyword2, n_grams=3, window_size=1):
    """
    Compare features of two keywords side-by-side.

    Args:
        text: Input text
        stopwords: Set of stopwords
        keyword1: First keyword
        keyword2: Second keyword
        n_grams: Maximum n-gram length
        window_size: Co-occurrence window size
    """
    features1 = get_keyword_features(text, stopwords, keyword1, n_grams, window_size)
    features2 = get_keyword_features(text, stopwords, keyword2, n_grams, window_size)

    print("=" * 80)
    print("KEYWORD COMPARISON")
    print("=" * 80)

    if not features1:
        print(f"Keyword 1 ('{keyword1}') not found!")
    if not features2:
        print(f"Keyword 2 ('{keyword2}') not found!")

    if not features1 or not features2:
        return

    print(f"\n{'Feature':<20} {keyword1:<25} {keyword2:<25}")
    print("-" * 80)
    print(f"{'YAKE Score':<20} {features1['score']:<25.6f} {features2['score']:<25.6f}")
    print(f"{'Length':<20} {features1['length']:<25} {features2['length']:<25}")
    print()


# Example usage
if __name__ == "__main__":
    from .extractor import KeywordExtractor

    text = """
    Machine learning is a subset of artificial intelligence.
    Deep learning uses neural networks.
    """

    extractor = KeywordExtractor(n=2, top=10)

    # Get features for a specific keyword
    features = get_keyword_features(
        text,
        extractor.stopwords,
        "machine learning",
        n_grams=2
    )

    # Print them nicely
    print_keyword_features(features)
