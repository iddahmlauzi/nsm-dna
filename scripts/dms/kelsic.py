import csv
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    VariantRecord,
    describe_amino_acid_changes,
    describe_coding_edit,
    replace_codon,
    translate_dna,
    write_variants,
)

ASSAY_ID = "IF1_ECOLI_Kelsic_2016"
SINGLE_CODON_SOURCE = "NIHMS836960-supplement-1.csv"
CODON_PAIR_SOURCE = "NIHMS836960-supplement-2.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read one Kelsic supplementary table."""
    with path.open(encoding="utf-8", newline="") as source_file:
        return list(csv.DictReader(source_file, skipinitialspace=True))


def reconstruct_wt(single_codon_rows: list[dict[str, str]]) -> str:
    """Reconstruct the infA WT sequence from rows marked as WT."""
    # Data S1 includes one row marked is_wt=1 for the original codon at each
    # protein position. Map each position to that codon to recover the WT DNA.
    wt_codons = {
        int(row["pos"]): row["codon"].upper()
        for row in single_codon_rows
        if row["is_wt"] == "1"
    }
    return "".join(wt_codons[position] for position in sorted(wt_codons))


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert the single- and paired-codon infA measurements."""
    single_codon_rows = read_rows(source_dir / SINGLE_CODON_SOURCE)
    codon_pair_rows = read_rows(source_dir / CODON_PAIR_SOURCE)
    wt_nt = reconstruct_wt(single_codon_rows)
    wt_aa = translate_dna(wt_nt)
    seen_sequences: set[str] = set()

    # Data S1 describes one codon position per row. Data S2 describes two
    # positions per row, so each source identifies which position and codon
    # columns must be applied to the WT sequence.
    variant_sources = (
        (single_codon_rows[1:], (("pos", "codon"),)),
        (codon_pair_rows[1:], (("pos1", "codon1"), ("pos2", "codon2"))),
    )

    # Evo omits the first data row from each table. If both tables produce the
    # same nucleotide sequence, Evo retains the score encountered first.
    for rows, codon_columns in variant_sources:
        for row in rows:
            mutant_nt = wt_nt

            for position_column, codon_column in codon_columns:
                position = int(row[position_column])
                mutant_codon = row[codon_column].upper()
                codon_start = (position - 1) * 3

                if mutant_codon == wt_nt[codon_start : codon_start + 3]:
                    continue

                mutant_nt = replace_codon(mutant_nt, position, mutant_codon)

            if mutant_nt == wt_nt or mutant_nt in seen_sequences:
                continue

            seen_sequences.add(mutant_nt)
            mutant_aa = translate_dna(mutant_nt)
            yield {
                "panel": "evo1",
                "study_id": "kelsic_2016",
                "assay_id": ASSAY_ID,
                "organism": "Escherichia coli",
                "target": "translation initiation factor IF-1",
                "wt_nt": wt_nt,
                "mutant_nt": mutant_nt,
                "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
                "wt_aa": wt_aa,
                "mutant_aa": mutant_aa,
                "aa_change": describe_amino_acid_changes(wt_aa, mutant_aa),
                "experimental_score": str(float(row["fitness_rich"])),
                "directionality": 1,
            }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Kelsic assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
