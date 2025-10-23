"""
YAKE keyword scoring logic.

This module contains all the statistical feature calculations for YAKE.
It processes text and scores keyword candidates based on 5 core features.
"""

import math
import string
import numpy as np
import networkx as nx
from .utils import tokenize_sentences, get_word_type


def extract_keywords_with_scores(text, stopwords, n_grams=3, window_size=1):
    """
    Extract keywords from text with YAKE scoring.

    This is the main function that orchestrates the entire YAKE pipeline:
    1. Tokenize text into sentences and words
    2. Build vocabulary and track word occurrences
    3. Build co-occurrence graph for context analysis
    4. Calculate 5 YAKE features for each word
    5. Generate n-gram candidates
    6. Score all candidates (lower = better)

    Args:
        text: Input text to extract keywords from
        stopwords: Set of stopword strings (lowercase)
        n_grams: Maximum n-gram length (1 to n words)
        window_size: Window for co-occurrence analysis

    Returns:
        Dict with:
        - 'candidates': dict mapping keyword string -> score
        - 'terms': dict mapping word -> term stats (for debugging)
    """
    punctuation = set(string.punctuation)

    # Step 1: Tokenize text
    sentences = tokenize_sentences(text)
    total_sentences = len(sentences)

    # Step 2: Build vocabulary and track occurrences
    vocab = {}  # word -> term_data dict
    global_position = 0  # Track position across entire document

    for sent_id, sentence in enumerate(sentences):
        for pos_in_sent, word in enumerate(sentence):
            # Skip pure punctuation
            if all(c in punctuation for c in word):
                continue

            # Get word type tag
            word_type = get_word_type(word, pos_in_sent, punctuation)

            # Normalize word (lowercase, remove trailing 's' for plurals)
            normalized = word.lower()
            if normalized.endswith('s') and len(normalized) > 3:
                normalized = normalized[:-1]

            # Check if stopword
            is_stopword = normalized in stopwords or len(normalized) < 3

            # Initialize term data if new
            if normalized not in vocab:
                vocab[normalized] = {
                    'word': normalized,
                    'is_stopword': is_stopword,
                    'tf': 0,  # Total frequency
                    'tf_upper': 0,  # Frequency as uppercase/acronym
                    'tf_proper': 0,  # Frequency as proper noun
                    'occurrences': [],  # List of (sent_id, pos_in_sent, global_pos)
                    'sentence_ids': set(),  # Which sentences contain this word
                }

            # Record this occurrence
            term = vocab[normalized]
            term['tf'] += 1
            term['occurrences'].append((sent_id, pos_in_sent, global_position))
            term['sentence_ids'].add(sent_id)

            # Track special cases
            if word_type == 'a':  # Acronym/uppercase
                term['tf_upper'] += 1
            if word_type == 'n':  # Proper noun
                term['tf_proper'] += 1

            global_position += 1

    total_words = global_position

    # Step 3: Build co-occurrence graph (for F4: relatedness)
    # IMPORTANT: Add ALL words (including stopwords) as nodes
    # Stopwords affect in/out-degree counts of content words
    graph = nx.DiGraph()
    for word in vocab:
        graph.add_node(word)

    # Process sentences again to capture co-occurrences
    for sentence in sentences:
        normalized_sentence = []
        for pos, word in enumerate(sentence):
            if all(c in punctuation for c in word):
                continue
            word_type = get_word_type(word, pos, punctuation)
            normalized = word.lower()
            if normalized.endswith('s') and len(normalized) > 3:
                normalized = normalized[:-1]
            if word_type not in {'u', 'd'}:  # Skip unusual and digit tags
                normalized_sentence.append(normalized)

        # Add edges for co-occurring words (only left-to-right direction)
        # IMPORTANT: Include stopwords in window for correct graph metrics
        for i in range(len(normalized_sentence)):
            current_word = normalized_sentence[i]
            if current_word not in vocab:
                continue
            # Only track edges TO non-stopwords (for their in-degree)
            if vocab[current_word]['is_stopword']:
                continue

            # Look at previous words within window (only backward, not forward)
            # This creates directed edges: previous_word -> current_word
            for j in range(max(0, i - window_size), i):
                previous_word = normalized_sentence[j]
                if previous_word not in vocab:
                    continue
                # Include edges FROM stopwords (they count for in-degree)
                # But only if both nodes exist in graph
                if not graph.has_node(previous_word) or not graph.has_node(current_word):
                    continue

                # Add/update edge from previous to current
                if not graph.has_edge(previous_word, current_word):
                    graph.add_edge(previous_word, current_word, weight=0)
                graph[previous_word][current_word]['weight'] += 1

    # Step 4: Calculate YAKE features for each term
    # First, get statistics for normalization
    non_stopword_tfs = [t['tf'] for t in vocab.values() if not t['is_stopword']]
    if non_stopword_tfs:
        mean_tf = np.mean(non_stopword_tfs)
        std_tf = np.std(non_stopword_tfs)
        max_tf = max(t['tf'] for t in vocab.values())
    else:
        mean_tf = std_tf = max_tf = 1

    # Calculate features for each term
    for word, term in vocab.items():
        if term['is_stopword']:
            term['score'] = 999  # High score for stopwords
            continue

        # F1: Frequency (wfreq) - normalized term frequency
        term['wfreq'] = term['tf'] / (mean_tf + std_tf)

        # F2: Casing (wcase) - preference for uppercase/proper nouns
        term['wcase'] = max(term['tf_upper'], term['tf_proper']) / (1.0 + math.log(term['tf']))

        # F3: Position (wpos) - preference for words appearing early
        # Use median of first occurrence positions
        first_positions = [sent_id for sent_id, _, _ in term['occurrences']]
        median_pos = np.median(first_positions)
        term['wpos'] = math.log(math.log(3.0 + median_pos))

        # F4: Relatedness (wrel) - based on co-occurrence graph
        # Count distinct neighbors and total edge weights
        out_edges = list(graph.out_edges(word, data=True)) if graph.has_node(word) else []
        in_edges = list(graph.in_edges(word, data=True)) if graph.has_node(word) else []

        n_out = len(out_edges)  # wdr: word different right
        weight_out = sum(d['weight'] for _, _, d in out_edges)  # wir: word importance right
        pwr = n_out / weight_out if weight_out > 0 else 0  # probability weight right

        n_in = len(in_edges)  # wdl: word different left
        weight_in = sum(d['weight'] for _, _, d in in_edges)  # wil: word importance left
        pwl = n_in / weight_in if weight_in > 0 else 0  # probability weight left

        term['pl'] = n_in / max_tf
        term['pr'] = n_out / max_tf
        term['wrel'] = (0.5 + (pwl * (term['tf'] / max_tf))) + (0.5 + (pwr * (term['tf'] / max_tf)))

        # F5: Spread (wspread) - distribution across sentences
        term['wspread'] = len(term['sentence_ids']) / total_sentences

        # Final YAKE score: lower is better
        # Score = (Position * Relatedness) / (Casing + Frequency/Relatedness + Spread/Relatedness)
        term['score'] = (term['wpos'] * term['wrel']) / (
            term['wcase'] + (term['wfreq'] / term['wrel']) + (term['wspread'] / term['wrel'])
        )

    # Step 5: Generate n-gram candidates
    candidates = {}  # normalized_phrase -> {'phrase': original, 'score': float, 'tf': int}

    for sentence in sentences:
        # Build list of (word, normalized, term) for this sentence
        sentence_terms = []
        for pos, word in enumerate(sentence):
            if all(c in punctuation for c in word):
                continue

            word_type = get_word_type(word, pos, punctuation)
            if word_type in {'u', 'd'}:  # Skip unusual and digits
                continue

            normalized = word.lower()
            if normalized.endswith('s') and len(normalized) > 3:
                normalized = normalized[:-1]

            if normalized in vocab:
                sentence_terms.append((word, normalized, vocab[normalized]))

        # Extract n-grams
        for i in range(len(sentence_terms)):
            for ngram_len in range(1, min(n_grams + 1, len(sentence_terms) - i + 1)):
                ngram_terms = sentence_terms[i:i + ngram_len]

                # Build phrase
                phrase = ' '.join(t[0] for t in ngram_terms)
                normalized_phrase = ' '.join(t[1] for t in ngram_terms)

                # Validate: no stopwords anywhere in the phrase
                if any(t[2]['is_stopword'] for t in ngram_terms):
                    continue

                # Initialize or update candidate
                if normalized_phrase not in candidates:
                    candidates[normalized_phrase] = {
                        'phrase': phrase,
                        'score': 0,
                        'tf': 0,
                        'terms': [t[2] for t in ngram_terms]
                    }
                candidates[normalized_phrase]['tf'] += 1

    # Step 6: Score n-gram candidates
    for norm_phrase, cand in candidates.items():
        terms = cand['terms']

        if len(terms) == 1:
            # Single word: use term score directly, multiplied by TF
            cand['score'] = terms[0]['score']
        else:
            # Multi-word: aggregate term scores
            # Use product / (sum + 1) * TF formula
            scores = [t['score'] for t in terms if not t['is_stopword']]
            if scores:
                score_product = np.prod(scores)
                score_sum = sum(scores)
                cand['score'] = score_product / ((score_sum + 1) * cand['tf'])
            else:
                cand['score'] = 999  # Invalid

    # Return candidates dict (normalized_phrase -> score)
    result_candidates = {
        cand['phrase']: cand['score']
        for cand in candidates.values()
        if cand['score'] < 999
    }

    return {
        'candidates': result_candidates,
        'terms': vocab  # For debugging
    }
