"""Streamlit UI for keyword extraction and feedback collection."""

import streamlit as st
import os
from yake import KeywordExtractor
from yake.scorer import extract_keywords_with_scores
from yake.utils import extract_features_for_keyword
from feedback_manager import FeedbackManager
from ml.predict import get_predictions_with_scores
from ml.train import train_model
from ml.config import TRAINING_THRESHOLD, DEFAULT_ALPHA


def extract_keywords_with_features(text, lan="en", n=3, top=15, dedup_lim=0.9):
    """Extract keywords with YAKE features using simplified implementation."""
    stopwords_file = os.path.join(os.path.dirname(__file__), 'yake', 'stopwords_hi.txt') if lan == "hi" else None

    kw_extractor = KeywordExtractor(stopwords_file=stopwords_file, n=n, top=top, dedup_threshold=dedup_lim)
    extracted_keywords = kw_extractor.extract_keywords(text)

    result = extract_keywords_with_scores(text.replace("\n", " "), kw_extractor.stopwords, n_grams=n, window_size=1)

    return [extract_features_for_keyword(kw, score, result['terms']) for kw, score in extracted_keywords]


def main():
    st.set_page_config(page_title="Keyword Tagger", page_icon="🏷️", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
        .stCheckbox {
            padding: 0.5rem 0;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem;
            border-radius: 0.8rem;
            color: white;
        }
        div[data-testid="stMetric"] label {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize feedback manager
    if 'feedback_mgr' not in st.session_state:
        st.session_state.feedback_mgr = FeedbackManager()

    # Header
    st.title("Intelligent Text Tagger")
    st.caption("Extract keywords and train ML models with your feedback")

    # Top dashboard - Stats and Settings
    col1, col2, col3 = st.columns(3)

    with col1:
        feedback_count = st.session_state.feedback_mgr.get_feedback_count()
        st.metric("Feedbacks", feedback_count)
        if feedback_count >= TRAINING_THRESHOLD:
            st.caption("Ready to train")
        else:
            st.caption(f"{TRAINING_THRESHOLD - feedback_count} more needed")

    with col2:
        mode = st.radio(
            "Mode",
            ["YAKE Only", "Hybrid"],
            horizontal=True,
            help="YAKE Only: Pure algorithm | Hybrid: ML-enhanced"
        )
        use_ml = (mode == "Hybrid")

    with col3:
        with st.expander("Settings"):
            n_grams = st.slider("Max n-gram size:", 1, 5, 3)
            top_k = st.slider("Keywords to extract:", 5, 30, 15)
            dedup_lim = st.slider("Deduplication threshold:", 0.0, 1.0, 0.9, 0.1)
            if use_ml:
                alpha = st.slider("YAKE Weight (α):", 0.0, 1.0, DEFAULT_ALPHA, 0.1,
                                help="1.0 = Pure YAKE, 0.0 = Pure ML")
            else:
                alpha = DEFAULT_ALPHA

    st.markdown("---")

    # Input Section
    st.subheader("Input Document")

    col_text, col_upload = st.columns([3, 1])

    with col_text:
        text_input = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Enter or paste your text here...",
            label_visibility="collapsed"
        )

    with col_upload:
        uploaded_file = st.file_uploader("Or upload a .txt file:", type=['txt'])

    # Determine text source
    text, doc_name = (uploaded_file.read().decode('utf-8'), uploaded_file.name) if uploaded_file else (text_input, "pasted_text") if text_input else (None, None)
    language = "en"

    # Extract button
    if text:
        st.caption(f"{len(text.split())} words, {len(text.split('.'))} sentences")

        extract_col, train_col = st.columns([3, 1])

        with extract_col:
            if st.button("Extract Keywords", type="primary", use_container_width=True):
                with st.spinner("Extracting keywords..."):
                    keywords_data = extract_keywords_with_features(text, lan=language, n=n_grams, top=top_k*3 if use_ml else top_k, dedup_lim=dedup_lim)

                    if use_ml:
                        predictions = get_predictions_with_scores(keywords_data, alpha=alpha)
                        if predictions['model_available']:
                            for i, kw_data in enumerate(keywords_data):
                                kw_data.update({'ml_prob': predictions['model_probs'][i], 'final_score': predictions['final_scores'][i]})
                            keywords_data.sort(key=lambda x: x['final_score'])
                            keywords_data = keywords_data[:top_k]
                        else:
                            st.warning("No trained model found. Using YAKE only.")
                            use_ml = False
                            keywords_data = keywords_data[:top_k]

                    st.session_state.update({'keywords_data': keywords_data, 'doc_name': doc_name, 'use_ml': use_ml})

        with train_col:
            if feedback_count >= TRAINING_THRESHOLD:
                if st.button("Train Model", use_container_width=True):
                    with st.spinner("Training model..."):
                        result = train_model()
                        if result:
                            st.success(f"Trained! Accuracy: {result['cv_accuracy']:.1%}")
                        else:
                            st.error("Training failed")

    # Display keywords and collect feedback
    if 'keywords_data' in st.session_state:
        st.markdown("---")
        st.subheader("Extracted Keywords")
        st.caption("Select the keywords that are relevant")

        # Create checkboxes for each keyword
        selected_keywords = []

        for i, kw_data in enumerate(st.session_state.keywords_data):
            if st.session_state.get('use_ml', False) and 'ml_prob' in kw_data:
                col1, col2, col3, col4 = st.columns([5, 2, 2, 2])
            else:
                col1, col2, col3 = st.columns([6, 2, 2])

            with col1:
                is_selected = st.checkbox(
                    f"{kw_data['keyword']}",
                    key=f"kw_{i}",
                    value=False
                )
                if is_selected:
                    selected_keywords.append(i)

            with col2:
                st.caption(f"YAKE: {kw_data['yake_score']:.4f}")

            with col3:
                if st.session_state.get('use_ml', False) and 'ml_prob' in kw_data:
                    st.caption(f"ML: {kw_data['ml_prob']:.0%}")
                else:
                    st.caption(f"Words: {kw_data['size']}")

            if st.session_state.get('use_ml', False) and 'ml_prob' in kw_data:
                with col4:
                    st.caption(f"Final: {kw_data['final_score']:.4f}")

        # Feedback submission
        st.markdown("---")
        feedback_col1, feedback_col2 = st.columns([1, 4])

        with feedback_col1:
            if st.button(f"Save Feedback ({len(selected_keywords)} selected)", type="primary", use_container_width=True):
                labels = [1 if i in selected_keywords else 0 for i in range(len(st.session_state.keywords_data))]
                st.session_state.feedback_mgr.save_feedback(st.session_state.doc_name, st.session_state.keywords_data, labels)
                st.success(f"Saved {len(selected_keywords)} approved keywords!")
                del st.session_state.keywords_data, st.session_state.doc_name
                st.rerun()

    elif not text:
        # Empty state - show example
        st.info("Paste text or upload a file to get started")
        st.markdown("""
        **How it works:**
        1. Input your text (paste or upload)
        2. Extract keywords using YAKE algorithm
        3. Select relevant keywords and save feedback
        4. After 10+ feedbacks, train an ML model
        5. Switch to Hybrid mode for ML-enhanced extraction
        """)


if __name__ == "__main__":
    main()
