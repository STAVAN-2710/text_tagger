"""
Simple test script to verify YAKE implementation is working.
Run with: poetry run python test_yake_simple.py
"""

from yake.core.yake import KeywordExtractor


def main():
    print("=" * 80)
    print("YAKE KEYWORD EXTRACTION TEST")
    print("=" * 80)

    # Sample text
    text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    Deep learning, a type of machine learning, uses neural networks with
    multiple layers to analyze complex patterns in data.
    Natural language processing is another important area of AI that helps
    computers understand and process human language.
    """

    print("\nInput Text:")
    print("-" * 80)
    print(text.strip())
    print()

    # Initialize YAKE extractor
    print("\nInitializing YAKE extractor...")
    kw_extractor = KeywordExtractor(
        lan="en",           # English language
        n=2,                # Max n-gram size (1, 2 words)
        dedup_lim=0.9,      # Deduplication threshold
        top=10              # Top 10 keywords
    )

    # Extract keywords
    print("Extracting keywords...\n")
    keywords = kw_extractor.extract_keywords(text)

    # Display results
    print("Extracted Keywords (lower score = more relevant):")
    print("-" * 80)

    if not keywords:
        print("No keywords found!")
    else:
        for i, (keyword, score) in enumerate(keywords, 1):
            print(f"{i:2}. {keyword:35} (score: {score:.4f})")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
