"""
Utility functions for CLI output formatting.
Simple, clean formatting following CLAUDE.md principles.
"""


def print_table(headers, rows):
    """
    Print a simple table to stdout.

    Args:
        headers: List of column headers
        rows: List of row tuples/lists
    """
    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Print header
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


def print_keywords_table(keywords_data, mode="yake"):
    """
    Print keywords in table format.

    Args:
        keywords_data: List of keyword dictionaries
        mode: 'yake' or 'hybrid'
    """
    if mode == "yake":
        headers = ["Rank", "Keyword", "YAKE Score", "Words"]
        rows = []
        for i, kw in enumerate(keywords_data, 1):
            rows.append([
                i,
                kw['keyword'],
                f"{kw['yake_score']:.4f}",
                kw['size']
            ])
    else:  # hybrid
        headers = ["Rank", "Keyword", "YAKE", "ML Conf", "Final"]
        rows = []
        for i, kw in enumerate(keywords_data, 1):
            rows.append([
                i,
                kw['keyword'],
                f"{kw['yake_score']:.4f}",
                f"{kw.get('ml_prob', 0):.0%}",
                f"{kw.get('final_score', kw['yake_score']):.4f}"
            ])

    print_table(headers, rows)


def print_success(message):
    """Print success message."""
    print(f"✓ {message}")


def print_error(message):
    """Print error message."""
    print(f"✗ Error: {message}")


def print_info(message):
    """Print info message."""
    print(f"ℹ {message}")
