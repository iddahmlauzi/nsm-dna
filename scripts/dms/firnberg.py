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

ASSAY_ID = "BLAT_ECOLX_Firnberg_2014"
WORKBOOK_NAME = "supp_msu081_Data_S1-S4.xlsx"


def read_source_rows(source_dir: Path) -> list[tuple[object, ...]]:
    """Read codon fitness rows from the author supplement."""
    return read_worksheet(
        source_dir / WORKBOOK_NAME,
        "S1 Codon fitnesses",
        min_row=3,
    )


def reconstruct_wt(rows: list[tuple[object, ...]]) -> str:
    """Reconstruct the WT coding sequence from repeated WT codons."""
    # A position can have many rows for different mutant codons. For example,
    # every H24 row has WT codon CAT but a different mutant codon. Store the
    # single WT codon associated with each position.
    wt_codons: dict[object, str] = {}

    for row in rows:
        position_label = row[0]
        wt_codon = str(row[1]).upper()

        if position_label not in wt_codons:
            wt_codons[position_label] = wt_codon
        elif wt_codons[position_label] != wt_codon:
            raise ValueError(f"Inconsistent WT codon at {position_label}")

    return "".join(wt_codons.values())


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert scored TEM-1 codons from the author supplement."""
    rows = read_source_rows(source_dir)
    wt_nt = reconstruct_wt(rows)
    wt_aa = translate_dna(wt_nt)

    # The spreadsheet labels positions using TEM-1 numbering, such as H24 and
    # P25. replace_codon() needs positions within our reconstructed sequence,
    # so map H24 to 1, P25 to 2, and so on.
    sequence_positions: dict[object, int] = {}

    for row in rows:
        # The first three columns give the position, WT codon, and mutant codon.
        position_label = row[0]
        sequence_position = sequence_positions.setdefault(
            position_label,
            len(sequence_positions) + 1,
        )
        wt_codon = str(row[1]).upper()
        mutant_codon = str(row[2]).upper()

        # Spreadsheet column U contains the author-reported codon fitness.
        score = row[20]

        if mutant_codon == wt_codon or score is None:
            continue

        mutant_nt = replace_codon(wt_nt, sequence_position, mutant_codon)
        mutant_aa = translate_dna(mutant_nt)
        wt_residue = wt_aa[sequence_position - 1]
        mutant_residue = mutant_aa[sequence_position - 1]

        yield {
            "panel": "evo1",
            "study_id": "firnberg_2014",
            "assay_id": ASSAY_ID,
            "organism": "Escherichia coli",
            "target": "TEM-1 beta-lactamase",
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": (f"{wt_residue}{sequence_position}{mutant_residue}"),
            "experimental_score": str(round(float(score), 4)),
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Firnberg assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
