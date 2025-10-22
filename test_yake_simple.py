"""
Simple test script to verify YAKE implementation is working.
Run with: poetry run python test_yake_simple.py
"""

from yake.core.yake import KeywordExtractor


def main():
    print("=" * 80)
    print("YAKE KEYWORD EXTRACTION TEST")
    print("=" * 80)

    # Sample text
    text = """
    Artificial Intelligence and Machine Learning Revolution

Artificial intelligence has transformed the technology landscape in unprecedented ways. Machine learning algorithms now power recommendation systems, natural language processing applications, and computer vision tasks across industries.

Deep learning, a subset of machine learning, uses neural networks with multiple layers to extract hierarchical features from data. Convolutional neural networks excel at image recognition, while recurrent neural networks handle sequential data processing effectively.

The transformer architecture revolutionized natural language processing. Models like GPT and BERT demonstrate remarkable performance in text generation, sentiment analysis, and question answering tasks. These models use attention mechanisms to capture contextual relationships between words.

Training deep neural networks requires substantial computational resources. Graphics processing units accelerate the matrix operations essential for backpropagation. Distributed training across multiple GPUs enables researchers to train larger models on massive datasets.

Reinforcement learning presents another paradigm where agents learn optimal policies through interaction with environments. Deep reinforcement learning combines neural networks with reinforcement learning principles, achieving superhuman performance in games like chess and Go.

Transfer learning leverages pre-trained models to solve new tasks with limited data. Fine-tuning allows practitioners to adapt large language models to domain-specific applications without training from scratch.

Ethical considerations surrounding artificial intelligence include algorithmic bias, privacy concerns, and transparency requirements. Researchers emphasize responsible AI development to ensure fairness and accountability in automated decision-making systems.

    """

    print("\nInput Text:")
    print("-" * 80)
    print(text.strip())
    print()

    # Initialize YAKE extractor
    print("\nInitializing YAKE extractor...")
    kw_extractor = KeywordExtractor(
        lan="en",           # English language
        n=5,                # Max n-gram size (1, 2 words)
        dedup_lim=0.6,      # Deduplication threshold
        top=15               # Top 15 keywords
    )

    # Extract keywords
    print("Extracting keywords...\n")
    keywords = kw_extractor.extract_keywords(text)

    # Display results
    print("Extracted Keywords (lower score = more relevant):")
    print("-" * 80)

    if not keywords:
        print("No keywords found!")
    else:
        for i, (keyword, score) in enumerate(keywords, 1):
            print(f"{i:2}. {keyword:35} (score: {score:.4f})")

    print("\n" + "=" * 80)
    print("✓ Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
