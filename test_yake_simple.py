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
    Architecture Review Meeting - Database Selection
Date: October 21, 2025

Discussion Summary:
Team evaluated PostgreSQL versus MongoDB for the new analytics platform. Key debate centered on data model flexibility versus transactional integrity.

PostgreSQL Advantages:
- Strong ACID compliance ensures data consistency
- Complex SQL queries and joins for reporting
- Mature tooling and wide team expertise
- Better suited for structured, relational data

MongoDB Considerations:
- Document-oriented model offers schema flexibility
- Horizontal scaling through sharding
- Faster for high-volume writes
- JSON-native storage aligns with API responses

Decision Rationale:
Team selected PostgreSQL based on:
1. Application requires complex multi-table joins for analytics
2. Data relationships are well-defined and stable
3. ACID guarantees critical for financial reporting compliance
4. Team has stronger PostgreSQL experience

MongoDB deferred for future microservices where schema evolution and horizontal scaling are priorities.

Next Steps:
- David to draft database schema design by Oct 28
- Sarah to configure PostgreSQL cluster with replication
- Marcus to update ORM mappings for PostgreSQL compatibility
- Proof-of-concept deployment scheduled for Nov 5
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
