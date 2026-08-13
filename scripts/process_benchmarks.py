import argparse
from pathlib import Path

from dms.evo import (
    OUTPUT_DIR as EVO_OUTPUT_DIR,
    SOURCE_DIR as EVO_SOURCE_DIR,
    standardize_studies,
)
from dms.mavedb import (
    MAVEDB_ARCHIVE_PATH,
    OUTPUT_DIR as MAVEDB_OUTPUT_DIR,
    standardize_mavedb,
)
from dms.ncrna import (
    ARCHIVE_PATH as NCRNA_ARCHIVE_PATH,
    OUTPUT_DIR as NCRNA_OUTPUT_DIR,
    standardize_ncrna,
)


def parse_args() -> argparse.Namespace:
    """Parse the benchmark name and its input paths."""
    parser = argparse.ArgumentParser(
        description="Create the standardized DMS benchmark datasets."
    )
    benchmarks = parser.add_subparsers(dest="benchmark", required=True)

    evo_parser = benchmarks.add_parser("evo1")
    evo_parser.add_argument("--source-dir", type=Path, default=EVO_SOURCE_DIR)
    evo_parser.add_argument("--output-dir", type=Path, default=EVO_OUTPUT_DIR)

    ncrna_parser = benchmarks.add_parser("evo1-ncrna")
    ncrna_parser.add_argument("--archive", type=Path, default=NCRNA_ARCHIVE_PATH)
    ncrna_parser.add_argument("--output-dir", type=Path, default=NCRNA_OUTPUT_DIR)

    mavedb_parser = benchmarks.add_parser("mavedb")
    mavedb_parser.add_argument("--archive", type=Path, default=MAVEDB_ARCHIVE_PATH)
    mavedb_parser.add_argument("--output-dir", type=Path, default=MAVEDB_OUTPUT_DIR)

    return parser.parse_args()


def main() -> None:
    """Process the selected benchmark dataset."""
    args = parse_args()

    if args.benchmark == "evo1":
        row_counts = standardize_studies(
            args.source_dir,
            args.output_dir,
        )

        for assay_id, row_count in row_counts.items():
            print(f"{assay_id}: {row_count:,}")

        print(f"Total: {sum(row_counts.values()):,}")
    elif args.benchmark == "evo1-ncrna":
        row_counts = standardize_ncrna(
            args.archive,
            args.output_dir,
        )

        for study_id, row_count in row_counts.items():
            print(f"{study_id}: {row_count:,}")

        print(f"Total: {sum(row_counts.values()):,}")
    else:
        standardize_mavedb(
            archive_path=args.archive,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
