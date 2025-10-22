"""
Tests for YAKE keyword extraction.
Validates core extraction functionality.
"""

from cli.extract import extract_keywords_from_text


def test_extract_keywords_basic():
    """Test YAKE extracts keywords from simple text."""
    text = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks."
    keywords = extract_keywords_from_text(text, top=5)

    assert len(keywords) <= 5
    assert len(keywords) > 0
    assert all('keyword' in kw for kw in keywords)
    assert all('yake_score' in kw for kw in keywords)


def test_keywords_have_features():
    """Test each keyword has all required YAKE features."""
    text = "Artificial intelligence is transforming technology and business."
    keywords = extract_keywords_from_text(text, top=3)

    assert len(keywords) > 0

    for kw in keywords:
        # Verify all required features exist
        assert 'keyword' in kw
        assert 'yake_score' in kw
        assert 'size' in kw
        assert 'wfreq' in kw
        assert 'wcase' in kw
        assert 'wpos' in kw
        assert 'wrel' in kw
        assert 'wspread' in kw

        # Verify feature types
        assert isinstance(kw['keyword'], str)
        assert isinstance(kw['yake_score'], float)
        assert isinstance(kw['size'], int)


def test_stopwords_removed():
    """Test stopwords don't appear in extracted keywords."""
    text = "The machine learning algorithm is very good at classification tasks."
    keywords = extract_keywords_from_text(text, top=10)

    # Common stopwords that should not appear as single-word keywords
    stopwords = {'the', 'is', 'at', 'very', 'a', 'an', 'and', 'or'}

    for kw in keywords:
        kw_lower = kw['keyword'].lower()
        # Single words shouldn't be stopwords
        if kw['size'] == 1:
            assert kw_lower not in stopwords, f"Stopword '{kw_lower}' found in results"


def test_top_k_limit():
    """Test extraction respects top_k parameter."""
    text = "Machine learning and deep learning are subfields of artificial intelligence and computer science."

    # Test different top_k values
    for top_k in [3, 5, 10]:
        keywords = extract_keywords_from_text(text, top=top_k)
        assert len(keywords) <= top_k
        assert len(keywords) > 0
