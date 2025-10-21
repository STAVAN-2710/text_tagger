"""
Test YAKE with both English and Hindi languages.
Run with: poetry run python test_multilang.py
"""

from yake.core.yake import KeywordExtractor


def test_english():
    print("=" * 80)
    print("ENGLISH KEYWORD EXTRACTION")
    print("=" * 80)

    text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    Deep learning uses neural networks with multiple layers.
    """

    print("\nInput Text:")
    print("-" * 80)
    print(text.strip())
    print()

    kw_extractor = KeywordExtractor(lan="en", n=2, top=10)
    keywords = kw_extractor.extract_keywords(text)

    print("Extracted Keywords:")
    print("-" * 80)
    for i, (keyword, score) in enumerate(keywords, 1):
        print(f"{i:2}. {keyword:30} (score: {score:.4f})")
    print()


def test_hindi():
    print("=" * 80)
    print("HINDI KEYWORD EXTRACTION")
    print("=" * 80)

    text = """
    मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक महत्वपूर्ण हिस्सा है।
    डीप लर्निंग एक प्रकार की मशीन लर्निंग है जो न्यूरल नेटवर्क का उपयोग करती है।
    प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानव भाषा समझने में मदद करता है।
    """

    print("\nInput Text:")
    print("-" * 80)
    print(text.strip())
    print()

    kw_extractor = KeywordExtractor(lan="hi", n=2, top=10)
    keywords = kw_extractor.extract_keywords(text)

    print("Extracted Keywords:")
    print("-" * 80)
    for i, (keyword, score) in enumerate(keywords, 1):
        print(f"{i:2}. {keyword:30} (score: {score:.4f})")
    print()


def main():
    print("\n")
    test_english()
    print("\n")
    test_hindi()
    print("=" * 80)
    print("✓ Multilingual test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
