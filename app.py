"""Streamlit UI for keyword extraction and feedback collection."""

import streamlit as st
import os
from pathlib import Path
from yake import KeywordExtractor
from yake.scorer import extract_keywords_with_scores
from feedback_manager import FeedbackManager
from ml.predict import get_predictions_with_scores
from ml.train import train_model
from ml.config import TRAINING_THRESHOLD, DEFAULT_ALPHA


def extract_keywords_with_features(text, lan="en", n=3, top=15, dedup_lim=0.9):
    """Extract keywords with YAKE features using simplified implementation."""
    # Determine stopwords file based on language
    if lan == "hi":
        import os
        dir_path = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(dir_path, 'yake', 'stopwords_hi.txt')
    else:
        stopwords_file = None  # Uses default English

    # Initialize extractor
    kw_extractor = KeywordExtractor(
        stopwords_file=stopwords_file,
        n=n,
        top=top,
        dedup_threshold=dedup_lim
    )

    # Extract keywords using the extractor (applies deduplication)
    extracted_keywords = kw_extractor.extract_keywords(text)

    # Get detailed results from scorer for feature extraction
    result = extract_keywords_with_scores(
        text.replace("\n", " "),
        kw_extractor.stopwords,
        n_grams=n,
        window_size=1
    )

    terms = result['terms']

    # Extract features for each candidate
    results = []
    for keyword, score in extracted_keywords:
        # Get the words in this keyword
        words = keyword.lower().split()

        keyword_data = {
            'keyword': keyword,
            'yake_score': score,
            'size': len(words),
        }

        # For single-word keywords, get features directly
        if len(words) == 1:
            word = words[0]
            # Handle plural normalization
            if word.endswith('s') and len(word) > 3:
                word_normalized = word[:-1]
            else:
                word_normalized = word

            if word_normalized in terms:
                term = terms[word_normalized]
                keyword_data['wfreq'] = term.get('wfreq', 0)
                keyword_data['wcase'] = term.get('wcase', 0)
                keyword_data['wpos'] = term.get('wpos', 0)
                keyword_data['wrel'] = term.get('wrel', 0)
                keyword_data['wspread'] = term.get('wspread', 0)
            else:
                # Fallback values
                keyword_data['wfreq'] = 0
                keyword_data['wcase'] = 0
                keyword_data['wpos'] = 0
                keyword_data['wrel'] = 0
                keyword_data['wspread'] = 0

        # For multi-word keywords, aggregate features from constituent terms
        else:
            wfreq_vals = []
            wcase_vals = []
            wpos_vals = []
            wrel_vals = []
            wspread_vals = []

            for word in words:
                # Handle plural normalization
                if word.endswith('s') and len(word) > 3:
                    word_normalized = word[:-1]
                else:
                    word_normalized = word

                if word_normalized in terms and not terms[word_normalized].get('is_stopword', False):
                    term = terms[word_normalized]
                    wfreq_vals.append(term.get('wfreq', 0))
                    wcase_vals.append(term.get('wcase', 0))
                    wpos_vals.append(term.get('wpos', 0))
                    wrel_vals.append(term.get('wrel', 0))
                    wspread_vals.append(term.get('wspread', 0))

            keyword_data['wfreq'] = sum(wfreq_vals) / len(wfreq_vals) if wfreq_vals else 0
            keyword_data['wcase'] = sum(wcase_vals) / len(wcase_vals) if wcase_vals else 0
            keyword_data['wpos'] = sum(wpos_vals) / len(wpos_vals) if wpos_vals else 0
            keyword_data['wrel'] = sum(wrel_vals) / len(wrel_vals) if wrel_vals else 0
            keyword_data['wspread'] = sum(wspread_vals) / len(wspread_vals) if wspread_vals else 0

        results.append(keyword_data)

    return results


def main():
    st.set_page_config(page_title="Keyword Tagger", page_icon="🏷️", layout="wide")

    # Header
    st.title("🏷️ Intelligent Text Tagger")
    st.markdown("Extract keywords and provide feedback to improve the system")

    # Initialize feedback manager
    if 'feedback_mgr' not in st.session_state:
        st.session_state.feedback_mgr = FeedbackManager()

    # Sidebar - File selection
    st.sidebar.header("📂 Select Document")

    # Drag and drop file upload
    uploaded_file = st.sidebar.file_uploader("Upload a text file:", type=['txt'])

    # Get text files from documents folder
    docs_folder = "documents"
    if os.path.exists(docs_folder):
        txt_files = [f for f in os.listdir(docs_folder) if f.endswith('.txt')]
        if txt_files and not uploaded_file:
            selected_file = st.sidebar.selectbox("Or choose from documents/:", txt_files)
            file_path = os.path.join(docs_folder, selected_file)
        elif not uploaded_file:
            st.sidebar.warning("No .txt files found in documents/ folder")
            selected_file = None
        else:
            selected_file = uploaded_file.name
            file_path = None
    else:
        if not uploaded_file:
            st.sidebar.info(f"Upload a file or create '{docs_folder}' folder")
        selected_file = uploaded_file.name if uploaded_file else None
        file_path = None

    # Configuration
    st.sidebar.header("⚙️ Settings")

    # Mode selection
    mode = st.sidebar.radio(
        "Extraction Mode:",
        ["YAKE Only", "Hybrid (YAKE + Feedback)"],
        help="YAKE Only: Pure algorithm | Hybrid: Combines YAKE with ML model trained on your feedback"
    )
    use_ml = (mode == "Hybrid (YAKE + Feedback)")

    # Alpha slider (only show if hybrid mode)
    if use_ml:
        alpha = st.sidebar.slider("YAKE Weight (α):", 0.0, 1.0, DEFAULT_ALPHA, 0.1,
                                   help="1.0 = Pure YAKE, 0.0 = Pure ML")
    else:
        alpha = DEFAULT_ALPHA

    st.sidebar.markdown("---")
    language = st.sidebar.selectbox("Language:", ["en", "hi"], index=0)
    n_grams = st.sidebar.slider("Max n-gram size:", 1, 5, 3)
    top_k = st.sidebar.slider("Keywords to extract:", 5, 30, 15)
    dedup_lim = st.sidebar.slider("Deduplication threshold:", 0.0, 1.0, 0.9, 0.1)

    # Feedback stats
    st.sidebar.header("📊 Statistics")
    feedback_count = st.session_state.feedback_mgr.get_feedback_count()
    st.sidebar.metric("Total Feedbacks", feedback_count)

    if feedback_count >= TRAINING_THRESHOLD:
        st.sidebar.success("✅ Ready for model training!")
        if st.sidebar.button("🎯 Train Model"):
            with st.spinner("Training model..."):
                result = train_model()
                if result:
                    st.sidebar.success(f"Model trained! ({result['num_samples']} samples, {result['cv_accuracy']:.1%} accuracy)")
                else:
                    st.sidebar.error("Training failed")
    else:
        st.sidebar.info(f"📝 {TRAINING_THRESHOLD - feedback_count} more needed for training")

    # Main content
    if selected_file:
        # Read file
        if uploaded_file:
            text = uploaded_file.read().decode('utf-8')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # Show text preview
        st.subheader(f"📄 Document: {selected_file}")
        with st.expander("View document text"):
            st.text(text[:1000] + ("..." if len(text) > 1000 else ""))

        # Extract keywords button
        if st.button("🔍 Extract Keywords", type="primary"):
            with st.spinner("Extracting keywords..."):
                # Extract larger pool if using ML, otherwise just top_k
                candidate_pool_size = top_k * 3 if use_ml else top_k  # 3x larger pool for ML

                keywords_data = extract_keywords_with_features(
                    text, lan=language, n=n_grams, top=candidate_pool_size, dedup_lim=dedup_lim
                )

                # Apply ML filtering if hybrid mode
                if use_ml:
                    predictions = get_predictions_with_scores(keywords_data, alpha=alpha)
                    if predictions['model_available']:
                        # Add ML scores to all candidates
                        for i, kw_data in enumerate(keywords_data):
                            kw_data['ml_prob'] = predictions['model_probs'][i]
                            kw_data['final_score'] = predictions['final_scores'][i]

                        # Sort by final score and take top_k
                        keywords_data.sort(key=lambda x: x['final_score'])
                        keywords_data = keywords_data[:top_k]
                    else:
                        st.warning("⚠️ No trained model found. Using YAKE only. Train a model first!")
                        use_ml = False
                        keywords_data = keywords_data[:top_k]  # Trim to top_k

                st.session_state.keywords_data = keywords_data
                st.session_state.doc_name = selected_file
                st.session_state.use_ml = use_ml

        # Display keywords and collect feedback
        if 'keywords_data' in st.session_state:
            st.markdown("---")
            st.subheader("🏷️ Extracted Keywords")

            # Show mode indicator
            if st.session_state.get('use_ml', False):
                st.info("🤖 Using Hybrid Mode (YAKE + ML Feedback)")
            else:
                st.info("📊 Using YAKE Only")

            st.markdown("**Select the keywords that are relevant:**")

            # Create checkboxes for each keyword
            selected_keywords = []

            # Display in a nice format
            for i, kw_data in enumerate(st.session_state.keywords_data):
                # Adjust columns based on ML usage
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

                # Fourth column for final score if ML is used
                if st.session_state.get('use_ml', False) and 'ml_prob' in kw_data:
                    with col4:
                        st.caption(f"Final: {kw_data['final_score']:.4f}")

            # Feedback submission
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 2, 6])

            with col1:
                if st.button("💾 Save Feedback", type="primary"):
                    # Create labels
                    labels = [1 if i in selected_keywords else 0
                              for i in range(len(st.session_state.keywords_data))]

                    # Save feedback
                    st.session_state.feedback_mgr.save_feedback(
                        st.session_state.doc_name,
                        st.session_state.keywords_data,
                        labels
                    )

                    st.success(f"✅ Saved! {len(selected_keywords)} keywords approved")

                    # Clear session
                    del st.session_state.keywords_data
                    del st.session_state.doc_name
                    st.rerun()

            with col2:
                st.caption(f"Selected: {len(selected_keywords)} keywords")

    else:
        # Instructions when no file selected
        st.info("👈 Select a document from the sidebar to begin")
        st.markdown("""
        ### How to use:
        1. Create a `documents/` folder in the project root
        2. Add `.txt` files to the folder
        3. Select a document from the sidebar
        4. Extract keywords and provide feedback
        5. After 50 feedbacks, the system can train an ML model
        """)


if __name__ == "__main__":
    main()
