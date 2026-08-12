from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    VariantRecord,
    describe_coding_edit,
    read_worksheet,
    replace_codon,
    translate_dna,
    write_variants,
)

ASSAY_ID = "A4GRB6_PSEAI_Chen_2020"
WORKBOOK_NAME = "elife-56707-supp2-v2.xlsx"
POSITION_COLUMN = "codon position (G2 stands for Glycine 2 from the inhouse sequence)"


def read_source_rows(source_dir: Path) -> list[dict[str, object]]:
    """Read codon fitness rows from the author supplement."""
    rows = read_worksheet(
        source_dir / WORKBOOK_NAME,
        "SF2D Codon Fitness scores",
        min_row=2,
    )
    headers = [str(value) for value in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:]]


def reconstruct_wt(rows: list[dict[str, object]]) -> str:
    """Reconstruct the VIM-2 WT coding sequence."""
    # A position has many rows for different variant codons, but every row at
    # that position repeats the same WT codon. G2 is an extra glycine in the
    # experimental construct between standard VIM-2 positions 1 and 2.
    wt_codons: dict[object, str] = {}

    for row in rows:
        position = row[POSITION_COLUMN]

        if position is None:
            continue

        if position not in wt_codons:
            wt_codons[position] = str(row["wt codon"]).upper()

    return "".join(wt_codons.values())


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert VIM-2 codons scored under 128 micrograms/mL ampicillin."""
    rows = read_source_rows(source_dir)
    wt_nt = reconstruct_wt(rows)
    wt_aa = translate_dna(wt_nt)

    # The source order is 1, G2, 2, 3, ...; map these labels to positions
    # 1, 2, 3, 4, ... in the experimental WT sequence.
    sequence_positions: dict[object, int] = {}

    for row in rows:
        position_label = row[POSITION_COLUMN]

        if position_label is not None and position_label not in sequence_positions:
            sequence_positions[position_label] = len(sequence_positions) + 1

    for row in rows:
        position_label = row[POSITION_COLUMN]
        score = row["fitness score"]

        if position_label is None or score is None:
            continue

        position = sequence_positions[position_label]
        mutant_codon = str(row["variant codon"]).upper()
        mutant_nt = replace_codon(wt_nt, position, mutant_codon)
        mutant_aa = translate_dna(mutant_nt)

        yield {
            "panel": "evo1",
            "study_id": "chen_2020",
            "assay_id": ASSAY_ID,
            "organism": "Pseudomonas aeruginosa",
            "target": "VIM-2 beta-lactamase",
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": (
                f"{wt_aa[position - 1]}{position}"
                f"{mutant_aa[position - 1]}"
            ),
            "experimental_score": str(score),
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Chen assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
