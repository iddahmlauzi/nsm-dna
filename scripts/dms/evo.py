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

SOURCE_DIR = Path("/home/iddah/datasets/evo_panel")
OUTPUT_DIR = SOURCE_DIR / "standardized"

STUDY_NAMES = (
    "firnberg_2014",
    "jacquier_2013",
    "adkar_2012",
    "tsuboyama_2023",
    "kelsic_2016",
    "weeks_2023",
    "rockah_shmuel_2015",
    "chen_2020",
    "melnikov_2014",
)

SINGLE_ASSAY_STANDARDIZERS = {
    "firnberg_2014": (firnberg.ASSAY_ID, firnberg.standardize),
    "jacquier_2013": (jacquier.ASSAY_ID, jacquier.standardize),
    "adkar_2012": (adkar.ASSAY_ID, adkar.standardize),
    "kelsic_2016": (kelsic.ASSAY_ID, kelsic.standardize),
    "weeks_2023": (weeks.ASSAY_ID, weeks.standardize),
    "rockah_shmuel_2015": (
        rockah_shmuel.ASSAY_ID,
        rockah_shmuel.standardize,
    ),
    "chen_2020": (chen.ASSAY_ID, chen.standardize),
    "melnikov_2014": (melnikov.ASSAY_ID, melnikov.standardize),
}


def standardize_studies(
    source_dir: Path,
    output_dir: Path,
    study_names: tuple[str, ...] = STUDY_NAMES,
) -> dict[str, int]:
    """Run the selected Evo study converters and return assay row counts."""
    row_counts = {}

    for study_name in study_names:
        if study_name == "tsuboyama_2023":
            row_counts.update(
                tsuboyama.standardize(
                    source_dir / study_name,
                    output_dir,
                )
            )
            continue

        assay_id, standardize = SINGLE_ASSAY_STANDARDIZERS[study_name]
        row_count = standardize(
            source_dir / study_name,
            output_dir,
        )
        row_counts[assay_id] = row_count

    return row_counts
