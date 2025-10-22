"""
Tests for feedback storage and retrieval.
Validates FeedbackManager functionality.
"""

import pytest
from feedback_manager import FeedbackManager


def test_save_feedback(tmp_path):
    """Test saving feedback to CSV."""
    feedback_file = tmp_path / "test_feedback.csv"
    fm = FeedbackManager(str(feedback_file))

    keywords_data = [
        {
            'keyword': 'test keyword',
            'size': 2,
            'yake_score': 0.1,
            'wfreq': 0.5,
            'wcase': 0.2,
            'wpos': 0.3,
            'wrel': 0.4,
            'wspread': 0.6
        }
    ]
    labels = [1]

    fm.save_feedback('test_doc.txt', keywords_data, labels)

    # Verify feedback was saved
    assert fm.get_feedback_count() == 1


def test_load_feedback(tmp_path):
    """Test loading feedback from CSV."""
    feedback_file = tmp_path / "test_feedback.csv"
    fm = FeedbackManager(str(feedback_file))

    keywords_data = [
        {
            'keyword': 'machine learning',
            'size': 2,
            'yake_score': 0.05,
            'wfreq': 0.5,
            'wcase': 0.2,
            'wpos': 0.3,
            'wrel': 0.4,
            'wspread': 0.6
        }
    ]
    labels = [1]

    fm.save_feedback('test.txt', keywords_data, labels)
    data = fm.load_all_feedback()

    # Verify data was loaded correctly
    assert len(data) == 1
    assert data[0]['keyword'] == 'machine learning'
    assert data[0]['label'] == '1'
    assert data[0]['doc_name'] == 'test.txt'


def test_feedback_count_increments(tmp_path):
    """Test feedback count increases correctly."""
    feedback_file = tmp_path / "test_feedback.csv"
    fm = FeedbackManager(str(feedback_file))

    # Initial count should be 0
    assert fm.get_feedback_count() == 0

    keywords_data = [
        {
            'keyword': 'test1',
            'size': 1,
            'yake_score': 0.1,
            'wfreq': 0.5,
            'wcase': 0.2,
            'wpos': 0.3,
            'wrel': 0.4,
            'wspread': 0.6
        }
    ]

    # Add first feedback
    fm.save_feedback('doc1.txt', keywords_data, [1])
    assert fm.get_feedback_count() == 1

    # Add second feedback
    fm.save_feedback('doc2.txt', keywords_data, [0])
    assert fm.get_feedback_count() == 2

    # Add multiple feedbacks at once
    keywords_data_multi = [
        {
            'keyword': 'test2',
            'size': 1,
            'yake_score': 0.2,
            'wfreq': 0.4,
            'wcase': 0.3,
            'wpos': 0.2,
            'wrel': 0.5,
            'wspread': 0.7
        },
        {
            'keyword': 'test3',
            'size': 1,
            'yake_score': 0.3,
            'wfreq': 0.3,
            'wcase': 0.4,
            'wpos': 0.1,
            'wrel': 0.6,
            'wspread': 0.8
        }
    ]
    fm.save_feedback('doc3.txt', keywords_data_multi, [1, 0])
    assert fm.get_feedback_count() == 4
