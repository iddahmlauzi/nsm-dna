import argparse
from pathlib import Path

from nsm_dna.data.variant_effects.evo import (
    OUTPUT_DIR as EVO_OUTPUT_DIR,
    SOURCE_DIR as EVO_SOURCE_DIR,
    standardize_studies,
)
from nsm_dna.data.variant_effects.mavedb import (
    MAVEDB_ARCHIVE_PATH,
    OUTPUT_DIR as MAVEDB_OUTPUT_DIR,
    standardize_mavedb,
)
from nsm_dna.data.variant_effects.ncrna import (
    ARCHIVE_PATH as NCRNA_ARCHIVE_PATH,
    OUTPUT_DIR as NCRNA_OUTPUT_DIR,
    standardize_ncrna,
)


def parse_args() -> argparse.Namespace:
    """Parse the dataset name and its input paths."""
    parser = argparse.ArgumentParser(
        description="Create standardized variant datasets for evaluation."
    )
    datasets = parser.add_subparsers(dest="dataset", required=True)

    evo_parser = datasets.add_parser("evo1")
    evo_parser.add_argument("--source-dir", type=Path, default=EVO_SOURCE_DIR)
    evo_parser.add_argument("--output-dir", type=Path, default=EVO_OUTPUT_DIR)

    ncrna_parser = datasets.add_parser("evo1-ncrna")
    ncrna_parser.add_argument("--archive", type=Path, default=NCRNA_ARCHIVE_PATH)
    ncrna_parser.add_argument("--output-dir", type=Path, default=NCRNA_OUTPUT_DIR)

    mavedb_parser = datasets.add_parser("mavedb")
    mavedb_parser.add_argument("--archive", type=Path, default=MAVEDB_ARCHIVE_PATH)
    mavedb_parser.add_argument("--output-dir", type=Path, default=MAVEDB_OUTPUT_DIR)

    return parser.parse_args()


def main() -> None:
    """Standardize the selected variant dataset."""
    args = parse_args()

    if args.dataset == "evo1":
        row_counts = standardize_studies(
            args.source_dir,
            args.output_dir,
        )

        for assay_id, row_count in row_counts.items():
            print(f"{assay_id}: {row_count:,}")

        print(f"Total: {sum(row_counts.values()):,}")
    elif args.dataset == "evo1-ncrna":
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
