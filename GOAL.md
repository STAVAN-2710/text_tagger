# AI Engineer Challenge: Intelligent Text Tagger

## Scenario

Your company wants to automatically tag internal documents (e.g., meeting notes, design docs, support tickets) with relevant keywords to improve search and categorization. You want to build a simple AI-powered tagging system that can learn from examples and improve over time.

## Task Overview

Build a Python-based system that:

1. Ingests a set of text documents (can be `.txt` or `.md` files)
2. Generates tags for each document using basic NLP techniques (e.g., keyword extraction, TF-IDF, or simple rule-based heuristics)
3. Implements a feedback mechanism where a simulated user can approve or reject tags
4. Learns from feedback to improve future tag suggestions (e.g., by adjusting weights or filtering logic)
5. Includes a CLI or script-based interface to run the tagging process and view results

## Constraints

- Use only Python standard libraries or free packages like `scikit-learn`, `nltk`, or `spaCy`
- No paid APIs or external services
- No need for deep learning or model training unless the candidate chooses to
- GitHub Copilot may assist, but the candidate should demonstrate understanding of the logic and design

## Bonus Points

- Add a simple web interface using Flask or Streamlit
- Include basic unit tests
- Use GitHub Actions for CI (if submitting via GitHub)
- Document how the system could be extended to use LLMs or vector search in the future

## Deliverables

- Source code (GitHub repo or zip)
- README with setup instructions and design decisions
- Sample input/output
- Short write-up on how feedback improves the system
