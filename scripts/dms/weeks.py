import csv
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    CODON_TABLE,
    VariantRecord,
    describe_amino_acid_changes,
    describe_coding_edit,
    read_worksheet,
    replace_codon,
    translate_dna,
    write_variants,
)

ASSAY_ID = "RNC_ECOLI_Weeks_2023"
FITNESS_SOURCE = "Supplementary Data S1.xlsx"
SEQUENCE_SOURCE = "Supplementary Data S8.csv"
TERMINAL_STOP_CODON = "TGA"


def reconstruct_wt(source_dir: Path) -> str:
    """Recover the rnc WT codons from the functional-score matrix."""
    with (source_dir / SEQUENCE_SOURCE).open(
        encoding="utf-8",
        newline="",
    ) as source_file:
        rows = list(csv.reader(source_file))

    wt_residues = rows[0][4:]
    score_rows = [
        row
        for row in rows[2:]
        if row[2] == "Functional Score" and row[3] == "Weighted Mean"
    ]

    # Each protein position is a column and each possible codon is a row. The
    # WT cell is blank because it is the reference rather than a mutation. For
    # example, position 1 is M and the blank ATG cell identifies its WT codon.
    wt_codons = [
        next(
            row[1]
            for row in score_rows
            if CODON_TABLE[row[1]] == wt_residue and not row[column]
        )
        for column, wt_residue in enumerate(wt_residues, start=4)
    ]
    return "".join(wt_codons) + TERMINAL_STOP_CODON


def read_fitness_rows(
    source_dir: Path,
) -> tuple[list[int], list[tuple[object, ...]]]:
    """Read the codon-level weighted-mean fitness matrix."""
    rows = read_worksheet(
        source_dir / FITNESS_SOURCE,
        "Fitness (weighted mean)",
    )

    # The first two columns describe each mutant codon. The remaining columns
    # are protein positions, with one fitness score per codon and position.
    # Ignore the trailing blank columns retained by Excel.
    positions = [
        int(position) for position in rows[1][2:] if position is not None
    ]
    # Summary rows below the matrix are labeled Min, Max, and Mean rather than
    # with a DNA codon.
    score_rows = [row for row in rows[2:] if row[1] in CODON_TABLE]
    return positions, score_rows


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert the scored rnc codons from the author fitness table."""
    wt_nt = reconstruct_wt(source_dir)
    wt_aa = translate_dna(wt_nt)
    positions, score_rows = read_fitness_rows(source_dir)

    for score_row in score_rows:
        mutant_codon = str(score_row[1])

        for position, score in zip(positions, score_row[2:]):
            if not isinstance(score, (int, float)):
                continue

            mutant_nt = replace_codon(wt_nt, position, mutant_codon)
            mutant_aa = translate_dna(mutant_nt)

            yield {
                "panel": "evo1",
                "study_id": "weeks_2023",
                "assay_id": ASSAY_ID,
                "organism": "Escherichia coli",
                "target": "RNase III",
                "wt_nt": wt_nt,
                "mutant_nt": mutant_nt,
                "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
                "wt_aa": wt_aa,
                "mutant_aa": mutant_aa,
                "aa_change": describe_amino_acid_changes(wt_aa, mutant_aa),
                # Evo stores the author workbook values to nine decimal places.
                "experimental_score": str(round(float(score), 9)),
                "directionality": 1,
            }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Weeks assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
