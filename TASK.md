# Intelligent Text Tagger with Custom YAKE Implementation

## Objective

Develop a reproducible, Python-based intelligent tagger for internal documents using a custom implementation of the YAKE algorithm. The system must be efficient, interpretable, and adaptable through user feedback. Design it with simplicity in mind—modular, local, and easy to extend via Streamlit.

## Core Requirements

- Implement YAKE manually (no third‑party YAKE pip module)
- Build a Streamlit app for document upload, keyword visualization, and user feedback collection
- Implement the feedback loop:
  - Users approve/reject suggested keywords
  - Save labels in SQLite or CSV
  - Train/update a lightweight logistic regression model using YAKE's features (F₁–F₅) + user feedback
  - Re‑score new documents using both the raw YAKE score and learned model outputs
- Keep design modular and dependency‑light (`numpy`, `pandas`, `scikit‑learn`, `streamlit`)
- Run locally on standard Python ≥3.9

## System Overview

### Components

| File | Responsibility |
|------|---------------|
| `main.py` | Driver for batch or CLI processing |
| `app.py` | Streamlit interface for upload, review, and re‑ranking |
| `yake_custom.py` | Core YAKE algorithm implementation (feature extraction, scoring) |
| `feedback.py` | Handles approval/rejection logging and persistence |
| `model.py` | Manages incremental logistic regression model for learned feedback |
| `config.py` | Stores configuration (N‑gram size, deduplication limit, paths) |
| `tests/` | Unit and integration testing folder |

## Implementation Steps

### 1. Document Ingestion

Accept `.txt`, `.md`, or `.docx` input.

**Preprocess:**
- Lowercase normalization
- Tokenization (retain sentence boundaries)
- Stopword filtering (use nltk stopword list or custom domain list)

### 2. Keyword Extraction (Custom YAKE)

Implement each YAKE component internally:

#### Candidate Generation
- Extract contiguous word sequences up to the max n-gram (default 3)
- Eliminate phrases that start/end with stopwords or punctuation

#### Feature Calculation

- **F₁ — Casing**: Ratio of uppercase forms to total occurrences
- **F₂ — Position**: Mean first occurrence index / total tokens
- **F₃ — Frequency**: Word frequency, normalized per document
- **F₄ — Context Relatedness**: Co‑occurrence entropy of word neighbors
- **F₅ — Dispersion**: How many sentences contain the word

#### Keyword Score

For single word \( w \):

\[
S(w) = F_1(w) \times F_2(w) \times F_3(w) \times F_4(w) \times F_5(w)
\]

For multi‑word phrases \( k = (w_1, \ldots, w_n) \):

\[
\text{YAKE}(k) = \left( \prod_{i=1}^{n} S(w_i) \right)^{1/n}
\]

Rank by ascending score (lower = better).

#### Deduplication
- Remove overlapping phrases by Jaccard similarity on token sets (>0.8 default)

### 3. Streamlit Interface

Create an intuitive, compact UI:

- **Upload Section**: Upload a text file or paste raw text
- **Extract Section**: Show extracted keywords and scores in a table
- **Feedback Section**: Users click ✅ or ❌ next to each keyword
- **Model Section**: Optional button to retrain classifier with latest feedback
- **Re‑rank Section**: Display updated keyword order after model re‑weighting

**Example layout:**

st.title("Intelligent Text Tagger (Custom YAKE)")
st.file_uploader("Upload a text document")
st.dataframe(keyword_table)
st.button("Approve Selected / Retrain Model")


### 4. Adaptive Learning (Feedback → Ranking)

Persist feedback as:

| doc_id | keyword | yake_score | f1 | f2 | f3 | f4 | f5 | label | timestamp |
|--------|---------|------------|----|----|----|----|-------|-----------|-----------|

Train a `scikit‑learn` `LogisticRegression` model incrementally:
final_score = 0.7 * yake_score + 0.3 * (1 - model_probability)

Update ranking live in Streamlit with weighted score.

### 5. Configuration and Data Storage

`config.py` defines:
NGRAM_MAX = 3
TOP_N = 20
DEDUP_THRESHOLD = 0.9
FEEDBACK_PATH = "data/feedback.csv"
MODEL_PATH = "models/classifier.pkl"


Use `pandas` for feedback persistence; switchable to SQLite via `sqlite3`.

### 6. Testing & Validation

- Test extraction speed (3k–5k words <2s)
- Validate deduplication logic and feature calculations

**Include tests:**
- `test_feature_extraction.py`
- `test_ranking.py`
- `test_feedback_loop.py`

## Example Project Flow

Step 1: Extract keywords manually
python main.py extract --input docs/sample.txt --top 15
Step 2: Launch the Streamlit UI
streamlit run app.py
Step 3: Review and approve keywords interactively
Step 4: Retrain model from feedback
python main.py train
Step 5: Adapt future document tagging
streamlit run app.py

## Dependencies

streamlit
numpy
pandas
scikit-learn
nltk

(Optional) `spacy` if advanced tokenization or POS filtering is required.

## Deliverables

- Source code (`main.py`, `app.py`, modules under `/src`)
- `requirements.txt`
- `README.md`
- `Task.md` (this document)
- `/tests` with pytest-compatible scripts
- Example feedback log under `/data/feedback.csv`
- Trained model snapshot in `/models/classifier.pkl`
