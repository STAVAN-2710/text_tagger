# Feature Inspection Guide

Yes! The simplified implementation provides **full access to all YAKE features**, just like the original.

## Quick Answer

You can extract features in **3 ways**:

### 1. Run the Test Script (Easiest)
```bash
poetry run python test_features_simplified.py
```

This displays a table with all 5 features (F1-F5) for the top keywords.

### 2. Use the Helper Functions (Most Flexible)

```python
from yake_simplified import KeywordExtractor
from yake_simplified.feature_extractor import get_keyword_features, print_keyword_features

text = "Machine learning is a subset of artificial intelligence."

extractor = KeywordExtractor(n=2, top=10)

# Get detailed features for a specific keyword
features = get_keyword_features(
    text,
    extractor.stopwords,
    "artificial intelligence",
    n_grams=2
)

# Pretty print them
print_keyword_features(features)
```

**Output:**
```
================================================================================
KEYWORD: 'artificial intelligence'
================================================================================
Final YAKE Score: 0.049404 (lower = better)
Length: 2 word(s)

Constituent Word Features:
--------------------------------------------------------------------------------

1. 'artificial' (normalized: 'artificial')
   Stopword: False
   TF:       1
   Score:    0.423215

   Features:
     F1 (wfreq):   1.000000  - Frequency (normalized by mean+std)
     F2 (wcase):   0.000000  - Casing (uppercase/proper noun)
     F3 (wpos):    0.094048  - Position (earlier is better)
     F4 (wrel):    3.000000  - Relatedness (co-occurrence)
     F5 (wspread): 1.000000  - Spread (sentence distribution)

2. 'intelligence' (normalized: 'intelligence')
   ...
```

### 3. Access Raw Data Directly (For Custom Analysis)

```python
from yake_simplified.scorer import extract_keywords_with_scores
from yake_simplified import KeywordExtractor

text = "Your text here..."
extractor = KeywordExtractor()

# Get full results with all internal data
result = extract_keywords_with_scores(
    text,
    extractor.stopwords,
    n_grams=3,
    window_size=1
)

# Access vocabulary with all term features
vocab = result['terms']

# Check features for a specific word
if 'machine' in vocab:
    term = vocab['machine']
    print(f"TF: {term['tf']}")
    print(f"F1 (wfreq): {term['wfreq']}")
    print(f"F2 (wcase): {term['wcase']}")
    print(f"F3 (wpos): {term['wpos']}")
    print(f"F4 (wrel): {term['wrel']}")
    print(f"F5 (wspread): {term['wspread']}")
    print(f"Final score: {term['score']}")

# Access all candidates with scores
candidates = result['candidates']
for keyword, score in sorted(candidates.items(), key=lambda x: x[1])[:5]:
    print(f"{keyword}: {score:.4f}")
```

## Feature Meanings

All 5 YAKE features are available:

| Feature | Name | What It Measures | Formula |
|---------|------|------------------|---------|
| **F1** | `wfreq` | Word Frequency | `tf / (mean_tf + std_tf)` |
| **F2** | `wcase` | Word Casing | `max(tf_upper, tf_proper) / (1 + log(tf))` |
| **F3** | `wpos` | Word Position | `log(log(3 + median_position))` |
| **F4** | `wrel` | Word Relatedness | Based on co-occurrence graph in/out-degree |
| **F5** | `wspread` | Word Spread | `num_sentences_with_word / total_sentences` |

**Final score:** `(wpos * wrel) / (wcase + wfreq/wrel + wspread/wrel)`

Lower scores = more important keywords!

## Comparison with Original

Both implementations provide **identical feature values**:

```bash
# Original
poetry run python test_features.py

# Simplified
poetry run python test_features_simplified.py
```

The features match exactly because the simplified version uses the **same formulas** and **same graph structure**.

## Full Example

See [`example_feature_inspection.py`](../example_feature_inspection.py) for a complete working example showing:
- How to extract keywords
- How to inspect features for specific keywords
- How to compare keywords
- How to access features programmatically

## API Reference

### `get_keyword_features(text, stopwords, keyword, n_grams=3, window_size=1)`

Get detailed feature breakdown for a specific keyword.

**Returns:**
```python
{
    'keyword': 'artificial intelligence',
    'score': 0.049404,
    'length': 2,
    'words': [
        {
            'word': 'artificial',
            'normalized': 'artificial',
            'is_stopword': False,
            'tf': 1,
            'wfreq': 1.0,
            'wcase': 0.0,
            'wpos': 0.094048,
            'wrel': 3.0,
            'wspread': 1.0,
            'score': 0.423215
        },
        ...
    ]
}
```

### `print_keyword_features(feature_data)`

Pretty-print the feature data returned by `get_keyword_features()`.

### `compare_keywords(text, stopwords, keyword1, keyword2, ...)`

Compare features of two keywords side-by-side.

## Why This Matters for Interviews

Being able to inspect features demonstrates:

1. **Understanding**: You know what each feature represents
2. **Debugging**: You can trace why a keyword scored high/low
3. **Explainability**: You can explain the algorithm's decisions
4. **Customization**: You can adjust features based on insights

Example interview exchange:
```
Interviewer: "Why did 'artificial intelligence' score 0.049?"
You: "Let me show you the features..."
     [runs get_keyword_features()]
     "It scored well because:
      - wrel is 2.5 (high co-occurrence with other words)
      - wpos is 0.094 (appears early in text)
      - wspread is 0.33 (focused, not scattered)

      The formula combines these: (wpos * wrel) / (wcase + wfreq/wrel + wspread/wrel)"
```

Much better than: "Uh... it's in the paper somewhere..." 😅
