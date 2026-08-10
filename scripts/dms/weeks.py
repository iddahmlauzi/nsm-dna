import csv
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    CODON_TABLE,
    VariantRecord,
    describe_coding_edit,
    replace_codon,
    translate_dna,
    write_variants,
)

ASSAY_ID = "RNC_ECOLI_Weeks_2023"
SOURCE_NAME = "Supplementary Data S8.csv"


def read_weighted_mean_rows(
    source_dir: Path,
) -> tuple[list[str], list[str], list[list[str]]]:
    """Read the WT residues, positions, and weighted-mean score matrix."""
    with (source_dir / SOURCE_NAME).open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.reader(source_file))

    wt_residues = rows[0][4:]
    positions = rows[1][4:]
    score_rows = [
        row
        for row in rows[2:]
        if row[2] == "Functional Score" and row[3] == "Weighted Mean"
    ]
    return wt_residues, positions, score_rows


def reconstruct_wt(
    wt_residues: list[str],
    score_rows: list[list[str]],
) -> str:
    """Recover each WT codon from its blank weighted-mean matrix entry."""
    wt_codons = []

    for column_offset, wt_residue in enumerate(wt_residues, start=4):
        candidates = [
            row[1]
            for row in score_rows
            if CODON_TABLE[row[1]] == wt_residue and not row[column_offset]
        ]

        if len(candidates) != 1:
            raise ValueError(f"Expected one WT codon at matrix column {column_offset}")

        wt_codons.append(candidates[0])

    return "".join(wt_codons)


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert the non-WT rnc codons with reported weighted-mean scores."""
    wt_residues, positions, score_rows = read_weighted_mean_rows(source_dir)
    wt_nt = reconstruct_wt(wt_residues, score_rows)
    wt_aa = translate_dna(wt_nt)

    for score_row in score_rows:
        mutant_codon = score_row[1]

        for column_offset, score in enumerate(score_row[4:], start=1):
            if not score:
                continue

            position = int(positions[column_offset - 1])
            mutant_nt = replace_codon(wt_nt, position, mutant_codon)
            mutant_aa = translate_dna(mutant_nt)

            yield {
                "panel": "evo1",
                "study_id": "weeks_2023",
                "assay_id": ASSAY_ID,
                "variant_id": f"position_{position}_{mutant_codon}",
                "organism": "Escherichia coli",
                "target": "RNase III",
                "wt_nt": wt_nt,
                "mutant_nt": mutant_nt,
                "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
                "wt_aa": wt_aa,
                "mutant_aa": mutant_aa,
                "aa_change": (
                    f"{wt_aa[position - 1]}{position}{mutant_aa[position - 1]}"
                ),
                "experimental_score": score,
                "directionality": 1,
            }


def standardize(source_dir: Path, output_dir: Path) -> int:
    """Write the standardized Weeks assay."""
    return write_variants(
        iter_variants(source_dir),
        output_dir / f"{ASSAY_ID}.csv",
    )
