import csv
import io
import tarfile
from pathlib import Path

from dms.shared import describe_coding_edit

ARCHIVE_PATH = Path(
    "/home/iddah/datasets/benchmark_sources/evo1/arvind_dms.tar.gz"
)
OUTPUT_DIR = Path("/home/iddah/datasets/benchmarks/evo1_ncrna")

NCRNA_COLUMNS = (
    "panel",
    "study_id",
    "assay_id",
    "wt_nt",
    "mutant_nt",
    "nt_edit",
    "experimental_score",
    "directionality",
)

SOURCE_MEMBERS = {
    "arvind_dms/processed_dms_nt_andreasson_2020.tsv": "andreasson_2020",
    "arvind_dms/processed_dms_nt_domingo_2018.tsv": "domingo_2018",
    "arvind_dms/processed_dms_nt_guy_2014.tsv": "guy_2014",
    "arvind_dms/processed_dms_nt_hayden_2011.tsv": "hayden_2011",
    "arvind_dms/processed_dms_nt_kobori_2016.tsv": "kobori_2016",
    "arvind_dms/processed_dms_nt_pitt_2010.tsv": "pitt_2010",
    "arvind_dms/processed_dms_nt_zhang_2009.tsv": "zhang_2009",
}


def write_study(
    archive: tarfile.TarFile,
    member_name: str,
    study_id: str,
    output_dir: Path,
) -> int:
    """Copy one Evo 1 ncRNA assay into the standardized schema."""
    output_path = output_dir / f"{study_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with archive.extractfile(member_name) as binary_file:
        source_file = io.TextIOWrapper(binary_file, encoding="utf-8")
        source_rows = csv.DictReader(source_file, delimiter="\t")

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=NCRNA_COLUMNS)
            writer.writeheader()
            row_count = 0

            for row in source_rows:
                wt_nt = row["wt_seq_nt"]
                mutant_nt = row["mt_seq_nt"]
                writer.writerow(
                    {
                        "panel": "evo1_ncrna",
                        "study_id": study_id,
                        "assay_id": study_id,
                        "wt_nt": wt_nt,
                        "mutant_nt": mutant_nt,
                        "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
                        "experimental_score": row["fitness"],
                        "directionality": 1,
                    }
                )
                row_count += 1

    return row_count


def standardize_ncrna(
    archive_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    """Write the seven Evo 1 ncRNA assays."""
    row_counts = {}

    with tarfile.open(archive_path, "r:gz") as archive:
        for member_name, study_id in SOURCE_MEMBERS.items():
            row_counts[study_id] = write_study(
                archive,
                member_name,
                study_id,
                output_dir,
            )

    return row_counts
