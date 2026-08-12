from pathlib import Path

from dms import (
    adkar,
    chen,
    firnberg,
    jacquier,
    kelsic,
    melnikov,
    rockah_shmuel,
    tsuboyama,
    weeks,
)

SOURCE_DIR = Path("/home/iddah/datasets/benchmark_sources/evo1")
OUTPUT_DIR = Path("/home/iddah/datasets/benchmarks/evo1_prokaryotic")

STANDARDIZERS = {
    "firnberg_2014": firnberg.standardize,
    "jacquier_2013": jacquier.standardize,
    "adkar_2012": adkar.standardize,
    "tsuboyama_2023": tsuboyama.standardize,
    "kelsic_2016": kelsic.standardize,
    "weeks_2023": weeks.standardize,
    "rockah_shmuel_2015": rockah_shmuel.standardize,
    "chen_2020": chen.standardize,
    "melnikov_2014": melnikov.standardize,
}


def standardize_studies(
    source_dir: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Run all Evo study converters and return assay row counts."""
    row_counts = {}

    for study_name, standardize in STANDARDIZERS.items():
        row_counts.update(
            standardize(
                source_dir / study_name,
                output_dir,
            )
        )

    return row_counts
