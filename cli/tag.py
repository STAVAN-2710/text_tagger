"""
Batch tagging command for CLI.
Simple, clean tagging logic following CLAUDE.md principles.
"""

import json
from pathlib import Path
from cli.extract import extract_keywords_from_text
from ml.predict import get_predictions_with_scores
from cli.utils import print_keywords_table, print_success, print_error, print_info


def tag_command(args):
    """
    Execute tag command with optional ML model.

    Args:
        args: Parsed command-line arguments
    """
    input_path = Path(args.input)

    # Check if input exists
    if not input_path.exists():
        print_error(f"Input path does not exist: {input_path}")
        return 1

    # Collect files to process
    files_to_process = []
    if input_path.is_file():
        if input_path.suffix == '.txt':
            files_to_process.append(input_path)
        else:
            print_error(f"Only .txt files are supported. Got: {input_path.suffix}")
            return 1
    elif input_path.is_dir():
        files_to_process = list(input_path.glob('*.txt'))
        if not files_to_process:
            print_error(f"No .txt files found in: {input_path}")
            return 1

    # Determine mode
    use_ml = (args.mode == "hybrid")

    if use_ml:
        print_info("Mode: Hybrid (YAKE + ML)")
    else:
        print_info("Mode: YAKE Only")

    # Process each file
    all_results = {}

    for file_path in files_to_process:
        print_info(f"Processing: {file_path.name}")

        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print_error(f"Failed to read {file_path.name}: {e}")
            continue

        # Extract larger pool if using ML
        try:
            candidate_pool_size = args.top * 3 if use_ml else args.top

            keywords = extract_keywords_from_text(
                text,
                language=args.language,
                n=args.ngrams,
                top=candidate_pool_size,
                dedup_lim=args.dedup
            )
        except Exception as e:
            print_error(f"Extraction failed for {file_path.name}: {e}")
            continue

        # Apply ML if hybrid mode
        if use_ml:
            try:
                predictions = get_predictions_with_scores(keywords, alpha=args.alpha)

                if predictions['model_available']:
                    # Add ML scores
                    for i, kw_data in enumerate(keywords):
                        kw_data['ml_prob'] = predictions['model_probs'][i]
                        kw_data['final_score'] = predictions['final_scores'][i]

                    # Sort by final score and take top_k
                    keywords.sort(key=lambda x: x['final_score'])
                    keywords = keywords[:args.top]
                else:
                    print_error(f"No trained model found. Using YAKE only for {file_path.name}")
                    use_ml = False
                    keywords = keywords[:args.top]
            except Exception as e:
                print_error(f"ML prediction failed for {file_path.name}: {e}")
                use_ml = False
                keywords = keywords[:args.top]

        all_results[str(file_path.name)] = keywords

        # Print results
        print(f"\nDocument: {file_path.name}")
        print_keywords_table(keywords, mode="hybrid" if use_ml else "yake")
        print()

    # Save to file if requested
    if args.output:
        try:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            print_success(f"Results saved to: {output_path}")
        except Exception as e:
            print_error(f"Failed to save results: {e}")
            return 1

    return 0
