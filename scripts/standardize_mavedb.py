import argparse
from pathlib import Path

from dms.mavedb import (
    MAVEDB_ARCHIVE_PATH,
    MAVEDB_SCORE_SETS_DIR,
    OUTPUT_DIR,
    standardize_mavedb,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line paths for the MaveDB conversion."""
    parser = argparse.ArgumentParser(description="Standardize the MaveDB DMS assays.")
    parser.add_argument("--archive", type=Path, default=MAVEDB_ARCHIVE_PATH)
    parser.add_argument("--score-sets-dir", type=Path, default=MAVEDB_SCORE_SETS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    standardize_mavedb(
        archive_path=arguments.archive,
        score_sets_dir=arguments.score_sets_dir,
        output_dir=arguments.output_dir,
    )
