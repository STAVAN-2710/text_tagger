"""
Multi-word term representation module for YAKE keyword extraction.

This module contains the ComposedWord class which represents multi-word terms
(potential keyword phrases) in a document. It handles the aggregation of features
from individual terms, scoring of candidate phrases, and validation to determine
which phrases make good keyword candidates.
"""

import numpy as np
from .utils import STOPWORD_WEIGHT


class ComposedWord:
    """
    Representation of a multi-word term in the document.

    This class stores and aggregates information about multi-word keyword candidates,
    calculating combined scores from the properties of their constituent terms.
    It tracks statistics like term frequency and provides methods to
    validate whether a phrase is likely to be a good keyword.

    Attributes:
        See property accessors below for available attributes.
    """

    def __init__(self, terms):
        """
        Initialize a ComposedWord object representing a multi-word term.

        Args:
            terms (list): List of tuples (tag, word, term_obj) representing
                          the individual words in this phrase. Can be None to
                          initialize an invalid candidate.
        """
        # If terms is None, initialize an invalid candidate
        if terms is None:
            self.data = {
                "start_or_end_stopwords": True,
                "tags": set(),
                "h": 0.0,
                "tf": 0.0,
                "kw": "",
                "unique_kw": "",
                "size": 0,
                "terms": [],
            }
            return

        # Basic initialization from terms
        self.data = {}

        # Calculate derived properties
        self.data["tags"] = set(["".join([w[0] for w in terms])])
        self.data["kw"] = " ".join([w[1] for w in terms])
        self.data["unique_kw"] = self.data["kw"].lower()
        self.data["size"] = len(terms)
        self.data["terms"] = [w[2] for w in terms if w[2] is not None]
        self.data["tf"] = 0.0
        self.data["h"] = 1.0

        # Check if the candidate starts or ends with stopwords
        if len(self.data["terms"]) > 0:
            self.data["start_or_end_stopwords"] = (
                self.data["terms"][0].stopword or self.data["terms"][-1].stopword
            )
        else:
            self.data["start_or_end_stopwords"] = True

    # Property accessors for backward compatibility
    @property
    def tags(self):
        """Get the set of part-of-speech tag sequences for this phrase."""
        return self.data["tags"]

    @property
    def kw(self):
        """Get the original form of the keyword phrase."""
        return self.data["kw"]

    @property
    def unique_kw(self):
        """Get the normalized (lowercase) form of the keyword phrase."""
        return self.data["unique_kw"]

    @property
    def size(self):
        """Get the number of words in this phrase."""
        return self.data["size"]

    @property
    def terms(self):
        """Get the list of SingleWord objects for each constituent term."""
        return self.data["terms"]

    @property
    def tf(self):
        """Get the term frequency (number of occurrences) in the document."""
        return self.data["tf"]

    @tf.setter
    def tf(self, value):
        """
        Set the term frequency value.

        Args:
            value (float): The new term frequency value
        """
        self.data["tf"] = value

    @property
    def h(self):
        """Get the final relevance score of this phrase (lower is better)."""
        return self.data["h"]

    @h.setter
    def h(self, value):
        """
        Set the final relevance score of this phrase.

        Args:
            value (float): The new score value
        """
        self.data["h"] = value

    @property
    def start_or_end_stopwords(self):
        """Get whether this phrase starts or ends with stopwords."""
        return self.data["start_or_end_stopwords"]

    def uptade_cand(self, cand):
        """
        Update this candidate with data from another candidate.

        Merges tag information from another candidate representing
        the same keyword phrase.

        Args:
            cand (ComposedWord): Another instance of the same keyword to merge with
        """
        # Add all tags from the other candidate to this one's tags
        for tag in cand.tags:
            self.tags.add(tag)

    def is_valid(self):
        """
        Check if this candidate is a valid keyword phrase.

        A valid keyword phrase doesn't contain unusual characters or digits,
        and doesn't start or end with stopwords.

        Returns:
            bool: True if this is a valid keyword candidate, False otherwise
        """
        is_valid = False
        # Check that at least one tag sequence has no unusual characters or digits
        for tag in self.tags:
            is_valid = is_valid or ("u" not in tag and "d" not in tag)

        # A valid keyword cannot start or end with a stopword
        return is_valid and not self.start_or_end_stopwords

    def get_composed_feature(self, feature_name, discart_stopword=True):
        """
               Get composed feature values for the n-gram.

        This function aggregates a specific feature across all terms in the n-gram.
        It computes the sum, product, and ratio of the feature values, optionally
        excluding stopwords from the calculation.

        Args:
            feature_name: Name of feature to get (must be an attribute of the term objects)
            discard_stopword: Whether to exclude stopwords from calculation (True by default)

        Returns:
            Tuple of (sum, product, ratio) for the feature where:
            - sum: Sum of the feature values across all relevant terms
            - product: Product of the feature values across all relevant terms
            - ratio: Product divided by (sum + 1), a measure of feature consistency

        """
        # Get feature values from each term, filtering stopwords if requested
        list_of_features = [
            getattr(term, feature_name)
            for term in self.terms
            if (discart_stopword and not term.stopword) or not discart_stopword
        ]

        # Calculate aggregate statistics
        sum_f = sum(list_of_features)
        prod_f = np.prod(list_of_features)

        # Return the three aggregated values
        return (sum_f, prod_f, prod_f / (sum_f + 1))

    def update_h(self, features=None, is_virtual=False):
        """
        Update the term's score based on its constituent terms.

        Calculates a combined relevance score for the multi-word term by
        aggregating scores of its constituent words, with special handling for
        stopwords to improve keyword quality.

        Args:
            features (list, optional): Specific features to use for scoring
            is_virtual (bool): Whether this is a virtual candidate not in text
        """
        sum_h = 0.0
        prod_h = 1.0

        # Process each term in the phrase
        for t, term_base in enumerate(self.terms):
            # Handle non-stopwords directly
            if not term_base.stopword:
                sum_h += term_base.h
                prod_h *= term_base.h

            # Handle stopwords according to configured weight method
            else:
                if STOPWORD_WEIGHT == "bi":
                    # BiWeight: use probabilities of adjacent term connections
                    prob_t1 = 0.0
                    # Check connection with previous term
                    if t > 0 and term_base.g.has_edge(
                        self.terms[t - 1].id, self.terms[t].id
                    ):
                        prob_t1 = (
                            term_base.g[self.terms[t - 1].id][self.terms[t].id]["tf"]
                            / self.terms[t - 1].tf
                        )

                    prob_t2 = 0.0
                    # Check connection with next term
                    if t < len(self.terms) - 1 and term_base.g.has_edge(
                        self.terms[t].id, self.terms[t + 1].id
                    ):
                        prob_t2 = (
                            term_base.g[self.terms[t].id][self.terms[t + 1].id]["tf"]
                            / self.terms[t + 1].tf
                        )

                    # Calculate combined probability and update scores
                    prob = prob_t1 * prob_t2
                    prod_h *= 1 + (1 - prob)
                    sum_h -= 1 - prob
                elif STOPWORD_WEIGHT == "h":
                    # HWeight: treat stopwords like normal words
                    sum_h += term_base.h
                    prod_h *= term_base.h
                elif STOPWORD_WEIGHT == "none":
                    # None: ignore stopwords entirely
                    pass

        # Determine term frequency to use in scoring
        tf_used = 1.0
        if features is None or "KPF" in features:
            tf_used = self.tf

        # For virtual candidates, use mean frequency of constituent terms
        if is_virtual:
            tf_used = np.mean([term_obj.tf for term_obj in self.terms])

        # Calculate final score (lower is better)
        self.h = prod_h / ((sum_h + 1) * tf_used)