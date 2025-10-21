"""
Test script to extract and display all YAKE intermediate features.
Run with: poetry run python test_features.py
"""

from yake.core.yake import KeywordExtractor


def extract_keywords_with_features(text, lan="en", n=3, top=10):
    """
    Extract keywords with all intermediate YAKE features.

    Returns:
        List of dicts with keyword and all features
    """
    # Initialize extractor
    kw_extractor = KeywordExtractor(lan=lan, n=n, top=top, dedup_lim=0.9)

    # Get the DataCore object by running extraction
    from yake.data import DataCore

    # Create data core
    core_config = {
        "windows_size": 1,
        "n": n,
    }
    dc = DataCore(text=text.replace("\n", " "), stopword_set=kw_extractor.stopword_set, config=core_config)

    # Build features
    dc.build_single_terms_features(features=None)
    dc.build_mult_terms_features(features=None)

    # Get valid candidates
    candidates = [cc for cc in dc.candidates.values() if cc.is_valid()]
    candidates_sorted = sorted(candidates, key=lambda c: c.h)[:top]

    # Extract features for each candidate
    results = []
    for cand in candidates_sorted:
        keyword_data = {
            'keyword': cand.kw,
            'yake_score': cand.h,
            'tf': cand.tf,
            'size': cand.size,
        }

        # For single-word keywords, get features directly
        if cand.size == 1 and len(cand.terms) > 0:
            term = cand.terms[0]
            keyword_data['wfreq'] = term.wfreq  # F1: Frequency
            keyword_data['wcase'] = term.wcase  # F2: Casing
            keyword_data['wpos'] = term.wpos    # F3: Position
            keyword_data['wrel'] = term.wrel    # F4: Relatedness
            keyword_data['wspread'] = term.data['wspread']  # F5: Spread

        # For multi-word keywords, aggregate features
        else:
            # Get aggregated features from constituent terms
            wfreq_vals = [t.wfreq for t in cand.terms if not t.stopword]
            wcase_vals = [t.wcase for t in cand.terms if not t.stopword]
            wpos_vals = [t.wpos for t in cand.terms if not t.stopword]
            wrel_vals = [t.wrel for t in cand.terms if not t.stopword]
            wspread_vals = [t.data['wspread'] for t in cand.terms if not t.stopword]

            keyword_data['wfreq'] = sum(wfreq_vals) / len(wfreq_vals) if wfreq_vals else 0
            keyword_data['wcase'] = sum(wcase_vals) / len(wcase_vals) if wcase_vals else 0
            keyword_data['wpos'] = sum(wpos_vals) / len(wpos_vals) if wpos_vals else 0
            keyword_data['wrel'] = sum(wrel_vals) / len(wrel_vals) if wrel_vals else 0
            keyword_data['wspread'] = sum(wspread_vals) / len(wspread_vals) if wspread_vals else 0

        results.append(keyword_data)

    return results


def main():
    print("=" * 100)
    print("YAKE KEYWORD EXTRACTION WITH INTERMEDIATE FEATURES")
    print("=" * 100)

    text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    Deep learning, a type of machine learning, uses neural networks with
    multiple layers to analyze complex patterns in data.
    Natural language processing is another important area of AI that helps
    computers understand and process human language.
    """

    print("\nInput Text:")
    print("-" * 100)
    print(text.strip())
    print()

    # Extract keywords with features
    print("\nExtracting keywords with features...\n")
    results = extract_keywords_with_features(text, lan="en", n=2, top=10)

    # Display results with all features
    print("Keywords with Intermediate Features:")
    print("=" * 110)
    print(f"{'#':<3} {'Keyword':<30} {'Len':<5} {'YAKE':<8} {'F1:Freq':<10} {'F2:Case':<10} {'F3:Pos':<10} {'F4:Rel':<10} {'F5:Spread':<10}")
    print("-" * 110)

    for i, kw_data in enumerate(results, 1):
        print(f"{i:<3} {kw_data['keyword']:<30} "
              f"{kw_data['size']:<5} "
              f"{kw_data['yake_score']:<8.4f} "
              f"{kw_data['wfreq']:<10.4f} "
              f"{kw_data['wcase']:<10.4f} "
              f"{kw_data['wpos']:<10.4f} "
              f"{kw_data['wrel']:<10.4f} "
              f"{kw_data['wspread']:<10.4f}")

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
