"""
Main YAKE keyword extractor.

Simple entry point for extracting keywords from text using the YAKE algorithm.
"""

import os
from .scorer import extract_keywords_with_scores
from .utils import clean_text, load_stopwords


class KeywordExtractor:
    """
    Extract keywords from text using YAKE (Yet Another Keyword Extractor).

    YAKE uses statistical features to identify important keywords without
    requiring external corpora or training data.
    """

    def __init__(self, stopwords_file=None, n=3, window_size=1, top=20, dedup_threshold=0.6):
        """
        Initialize the keyword extractor.

        Args:
            stopwords_file: Path to stopwords file (defaults to built-in English)
            n: Maximum n-gram length (default: 3)
            window_size: Co-occurrence window size (default: 1)
            top: Number of top keywords to return (default: 20)
            dedup_threshold: Similarity threshold for deduplication (default: 0.9)
                           Set to 1.0 to disable deduplication
        """
        self.n = n
        self.window_size = window_size
        self.top = top
        self.dedup_threshold = dedup_threshold

        # Load stopwords
        if stopwords_file is None:
            # Use built-in English stopwords
            dir_path = os.path.dirname(os.path.abspath(__file__))
            stopwords_file = os.path.join(dir_path, 'stopwords_en.txt')

        self.stopwords = load_stopwords(stopwords_file)

    def extract_keywords(self, text):
        """
        Extract keywords from text.

        Args:
            text: Input text to analyze

        Returns:
            List of (keyword, score) tuples, sorted by score (lower = better)
        """
        if not text or not text.strip():
            return []

        # Clean text
        cleaned_text = clean_text(text)

        # Extract candidates with scores
        result = extract_keywords_with_scores(
            cleaned_text,
            self.stopwords,
            n_grams=self.n,
            window_size=self.window_size
        )

        candidates = result['candidates']

        # Sort by score (lower is better)
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])

        # Apply deduplication if threshold < 1.0
        if self.dedup_threshold < 1.0:
            sorted_candidates = self._deduplicate(sorted_candidates)

        # Return top N
        return sorted_candidates[:self.top]

    def _deduplicate(self, candidates):
        """
        Remove duplicate/similar keywords.

        Args:
            candidates: List of (keyword, score) tuples

        Returns:
            Deduplicated list of (keyword, score) tuples
        """
        result = []

        for keyword, score in candidates:
            # Check if similar to any already-selected keyword
            is_duplicate = False

            for selected_kw, _ in result:
                similarity = self._string_similarity(keyword, selected_kw)
                if similarity > self.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                result.append((keyword, score))

        return result

    def _string_similarity(self, str1, str2):
        """
        Calculate string similarity using simple character-level comparison.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        
        if str1 == str2:
            return 1.0

        # Check substring containment
        shorter = min(str1, str2, key=len)
        longer = max(str1, str2, key=len)

        if shorter in longer:
            return len(shorter) / len(longer)

        # Character-level overlap
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0
