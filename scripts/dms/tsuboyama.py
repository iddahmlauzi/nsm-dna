import csv
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    VariantRecord,
    describe_amino_acid_changes,
    describe_coding_edit,
    read_fasta,
    translate_dna,
    write_variants,
)

SOURCE_MEMBER = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"
WT_FASTA_NAME = "evo1_tsuboyama_wt.fasta"

ASSAYS = {
    "1AOY.pdb": (
        "ARGR_ECOLI_Tsuboyama_2023_1AOY",
        "Arginine repressor",
    ),
    "2D1U.pdb": (
        "FECA_ECOLI_Tsuboyama_2023_2D1U",
        "Fe(3+) dicitrate transport protein FecA",
    ),
    "1WCL.pdb": (
        "NUSA_ECOLI_Tsuboyama_2023_1WCL",
        "Transcription termination/antitermination protein NusA",
    ),
    "2LCL.pdb": (
        "RFAH_ECOLI_Tsuboyama_2023_2LCL",
        "Transcription antitermination protein RfaH",
    ),
    "2KVT.pdb": (
        "YAIA_ECOLI_Tsuboyama_2023_2KVT",
        "Uncharacterized protein YaiA",
    ),
}


def read_selected_rows(
    source_dir: Path,
    wt_sequences: dict[str, str],
) -> dict[str, list[dict[str, str]]]:
    """Stream Dataset 1 once and retain Evo's five E. coli targets."""
    variant_rows: dict[str, list[dict[str, str]]] = {
        wt_name: [] for wt_name in ASSAYS
    }
    archive_path = source_dir / "Processed_K50_dG_datasets.zip"

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(SOURCE_MEMBER) as binary_file:
            text_file = io.TextIOWrapper(binary_file, encoding="utf-8")

            for row in csv.DictReader(text_file):
                # Names begin with the PDB target. For example, 1AOY.pdb is
                # the WT row and 1AOY.pdb_D4K is a measured mutant row.
                wt_name = row["name"].split("_", 1)[0]

                if wt_name not in ASSAYS:
                    continue

                # Some source rows lack the combined stability estimate used
                # by Evo and therefore cannot be included in this benchmark.
                if not row["deltaG"]:
                    continue

                mutant_nt = row["dna_seq"].upper()

                # Exclude only the exact reference DNA. Other nucleotide
                # constructs can encode the same protein and remain distinct
                # measured sequences in the author data.
                if mutant_nt == wt_sequences[wt_name]:
                    continue

                variant_rows[wt_name].append(
                    {
                        # Preserve the complete nucleotide sequence reported
                        # by the authors, including its exact codon background.
                        "mutant_nt": mutant_nt,
                        "experimental_score": row["deltaG"],
                    }
                )

    return variant_rows


def iter_assay_variants(
    wt_name: str,
    wt_nt: str,
    source_rows: list[dict[str, str]],
) -> Iterator[VariantRecord]:
    """Convert one Tsuboyama Dataset 1 target."""
    assay_id, target = ASSAYS[wt_name]
    wt_aa = translate_dna(wt_nt)

    for row in source_rows:
        mutant_nt = row["mutant_nt"]
        mutant_aa = translate_dna(mutant_nt)

        yield {
            "panel": "evo1",
            "study_id": "tsuboyama_2023",
            "assay_id": assay_id,
            "organism": "Escherichia coli",
            "target": target,
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": describe_amino_acid_changes(wt_aa, mutant_aa),
            "experimental_score": row["experimental_score"],
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the five standardized Tsuboyama assays."""
    wt_sequences = read_fasta(source_dir / WT_FASTA_NAME)
    variant_rows = read_selected_rows(source_dir, wt_sequences)
    row_counts = {}

    for wt_name, (assay_id, _) in ASSAYS.items():
        row_counts[assay_id] = write_variants(
            iter_assay_variants(
                wt_name,
                wt_sequences[wt_name],
                variant_rows[wt_name],
            ),
            output_dir / f"{assay_id}.csv",
        )

    return row_counts
