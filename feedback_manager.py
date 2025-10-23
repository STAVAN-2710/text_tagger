"""Feedback manager for storing keyword annotations."""

import csv
import os
from datetime import datetime
from typing import List, Dict


class FeedbackManager:
    """Manages feedback storage in CSV format."""

    def __init__(self, feedback_file='data/feedback.csv'):
        """
        Initialize feedback manager.

        Args:
            feedback_file: Path to CSV file for storing feedback
        """
        self.feedback_file = feedback_file
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create feedback file with headers if it doesn't exist."""
        # Create directory if needed
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)

        # Create file with headers if it doesn't exist
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'doc_name', 'keyword', 'length', 'yake_score',
                    'f1_wfreq', 'f2_wcase', 'f3_wpos', 'f4_wrel', 'f5_wspread',
                    'label'
                ])

    def save_feedback(self, doc_name: str, keywords_data: List[Dict], labels: List[int]):
        """
        Save feedback to CSV.

        Args:
            doc_name: Name of the document
            keywords_data: List of keyword dictionaries with features
            labels: List of labels (1 for approved, 0 for rejected)
        """
        timestamp = datetime.now().isoformat()

        with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for kw_data, label in zip(keywords_data, labels):
                writer.writerow([
                    timestamp,
                    doc_name,
                    kw_data['keyword'],
                    kw_data['size'],
                    kw_data['yake_score'],
                    kw_data['wfreq'],
                    kw_data['wcase'],
                    kw_data['wpos'],
                    kw_data['wrel'],
                    kw_data['wspread'],
                    label
                ])

    def get_feedback_count(self) -> int:
        """Get total number of feedback entries."""
        if not os.path.exists(self.feedback_file):
            return 0

        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            # Subtract 1 for header row
            return sum(1 for line in f) - 1

    def load_all_feedback(self) -> List[Dict]:
        """Load all feedback as list of dictionaries."""
        if not os.path.exists(self.feedback_file):
            return []

        feedback = []
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                feedback.append(row)

        return feedback
