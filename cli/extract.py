"""
Keyword extraction command for CLI.
Simple, clean extraction logic following CLAUDE.md principles.
"""

import os
import json
from pathlib import Path
from yake.core.yake import KeywordExtractor
from yake.data import DataCore
from cli.utils import print_keywords_table, print_success, print_error, print_info


def extract_keywords_from_text(text, language="en", n=3, top=15, dedup_lim=0.9):
    """
    Extract keywords from text using YAKE.

    Args:
        text: Input text
        language: Language code ('en' or 'hi')
        n: Max n-gram size
        top: Number of keywords to extract
        dedup_lim: Deduplication threshold

    Returns:
        List of keyword dictionaries with features
    """
    kw_extractor = KeywordExtractor(lan=language, n=n, top=top, dedup_lim=dedup_lim)

    # Create data core
    core_config = {"windows_size": 1, "n": n}
    dc = DataCore(
        text=text.replace("\n", " "),
        stopword_set=kw_extractor.stopword_set,
        config=core_config
    )

    # Build features
    dc.build_single_terms_features(features=None)
    dc.build_mult_terms_features(features=None)

    # Get valid candidates and sort
    candidates = [cc for cc in dc.candidates.values() if cc.is_valid()]
    candidates_sorted = sorted(candidates, key=lambda c: c.h)

    # Deduplication
    deduplicated_cands = []
    if dedup_lim >= 1.0:
        deduplicated_cands = candidates_sorted[:top]
    else:
        for cand in candidates_sorted:
            should_add = True
            for existing_cand in deduplicated_cands:
                similarity = kw_extractor.dedup_function(cand.unique_kw, existing_cand.unique_kw)
                if similarity > dedup_lim:
                    should_add = False
                    break
            if should_add:
                deduplicated_cands.append(cand)
            if len(deduplicated_cands) == top:
                break

    # Extract features
    results = []
    for cand in deduplicated_cands:
        keyword_data = {
            'keyword': cand.kw,
            'yake_score': cand.h,
            'size': cand.size,
        }

        # Aggregate features from terms
        if cand.size == 1 and len(cand.terms) > 0:
            term = cand.terms[0]
            keyword_data['wfreq'] = term.wfreq
            keyword_data['wcase'] = term.wcase
            keyword_data['wpos'] = term.wpos
            keyword_data['wrel'] = term.wrel
            keyword_data['wspread'] = term.data['wspread']
        else:
            # Multi-word: average non-stopword features
            wfreq_vals = [t.wfreq for t in cand.terms if not t.stopword]
            wcase_vals = [t.wcase for t in cand.terms if not t.stopword]
            wpos_vals = [t.wpos for t in cand.terms if not t.stopword]
            wrel_vals = [t.wrel for t in cand.terms if not t.stopword]
            wspread_vals = [t.data['wspread'] for t in cand.terms if not t.stopword]

            keyword_data['wfreq'] = sum(wfreq_vals) / len(wfreq_vals) if wfreq_vals else 0
            keyword_data['wcase'] = sum(wcase_vals) / len(wcase_vals) if wcase_vals else 0
            keyword_data['wpos'] = sum(wpos_vals) / len(wpos_vals) if wpos_vals else 0
            keyword_data['wrel'] = sum(wrel_vals) / len(wrel_vals) if wrel_vals else 0
            keyword_data['wspread'] = sum(wspread_vals) / len(wspread_vals) if wspread_vals else 0

        results.append(keyword_data)

    return results


def extract_command(args):
    """
    Execute extract command.

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

        # Extract keywords
        try:
            keywords = extract_keywords_from_text(
                text,
                language=args.language,
                n=args.ngrams,
                top=args.top,
                dedup_lim=args.dedup
            )
        except Exception as e:
            print_error(f"Extraction failed for {file_path.name}: {e}")
            continue

        all_results[str(file_path.name)] = keywords

        # Print results
        print(f"\nDocument: {file_path.name}")
        print_keywords_table(keywords, mode="yake")
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
