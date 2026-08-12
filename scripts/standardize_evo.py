import argparse
from pathlib import Path

from dms.evo import OUTPUT_DIR, SOURCE_DIR, STANDARDIZERS, standardize_studies


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and optional study names."""
    parser = argparse.ArgumentParser(description="Standardize the Evo 1 DMS studies.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--study", action="append", choices=STANDARDIZERS)
    return parser.parse_args()


def main() -> None:
    """Run the selected study converters and print their row counts."""
    args = parse_args()
    row_counts = standardize_studies(
        args.source_dir,
        args.output_dir,
        args.study,
    )

    for assay_id, row_count in row_counts.items():
        print(f"{assay_id}: {row_count:,}")

    print(f"Total: {sum(row_counts.values()):,}")


if __name__ == "__main__":
    main()
