"""
Text preprocessing utilities for YAKE keyword extraction.

Simple, readable helper functions for text cleaning and tokenization.
"""

import re
from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions


def clean_text(text):
    """
    Clean and normalize input text.

    Args:
        text: Raw input text

    Returns:
        Cleaned text with normalized spacing
    """
    # Detect paragraph breaks (lines starting with capital letters)
    lines = text.split("\n")
    cleaned_parts = []

    for line in lines:
        line = line.replace("\t", " ")
        # Add double newline for new paragraphs (start with capital)
        if line.strip() and line.strip()[0].isupper():
            cleaned_parts.append("\n\n" + line)
        elif line.strip():
            cleaned_parts.append(" " + line)

    return "".join(cleaned_parts).strip()


def tokenize_sentences(text):
    """
    Split text into sentences and tokenize each into words.

    Args:
        text: Input text

    Returns:
        List of sentences, where each sentence is a list of word tokens
    """
    sentences = []

    for sentence_text in split_multi(text):
        if not sentence_text.strip():
            continue

        # Tokenize sentence into words
        tokens = []
        for word in split_contractions(web_tokenizer(sentence_text)):
            # Skip apostrophes and empty tokens
            if word.startswith("'") and len(word) > 1:
                continue
            if len(word) > 0:
                tokens.append(word)

        if tokens:
            sentences.append(tokens)

    return sentences


def get_word_type(word, position_in_sentence, punctuation):
    """
    Classify a word based on its characteristics.

    Args:
        word: The word to classify
        position_in_sentence: Position (0 = first word)
        punctuation: Set of punctuation characters

    Returns:
        Single character tag:
        - 'd': digit/number
        - 'u': unusual (mixed chars, special chars)
        - 'a': acronym (all uppercase)
        - 'n': proper noun (capitalized, not at start)
        - 'p': plain word
    """
    # Check if it's a number
    if word.replace(",", "").isdigit() or word.replace(",", "").replace(".", "", 1).isdigit():
        return "d"

    # Count character types
    n_digits = sum(c.isdigit() for c in word)
    n_alpha = sum(c.isalpha() for c in word)
    n_punct = sum(c in punctuation for c in word)

    # Unusual: mixed alphanumeric, no alphanumeric, or multiple punctuation
    if (n_digits > 0 and n_alpha > 0) or (n_digits == 0 and n_alpha == 0) or n_punct > 1:
        return "u"

    # Acronym: all uppercase
    if word.isupper() and len(word) > 0:
        return "a"

    # Proper noun: capitalized, not at sentence start, only first letter uppercase
    if len(word) > 1 and word[0].isupper() and position_in_sentence > 0:
        if sum(c.isupper() for c in word) == 1:
            return "n"

    return "p"


def load_stopwords(filepath):
    """
    Load stopwords from a text file.

    Args:
        filepath: Path to stopwords file

    Returns:
        Set of stopword strings (lowercase)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(f.read().lower().split('\n'))
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='ISO-8859-1') as f:
            return set(f.read().lower().split('\n'))
