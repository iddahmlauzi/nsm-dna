import re
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    VariantRecord,
    describe_coding_edit,
    read_fasta,
    read_worksheet,
    translate_dna,
    write_variants,
)

ASSAY_ID = "BLAT_ECOLX_Jacquier_2013"

# Parse nucleotide substitutions such as A71T: WT base A at coding position
# 71 is replaced by T. A source row can join multiple substitutions with `_`.
SUBSTITUTION_PATTERN = re.compile(r"([ACGT])(\d+)([ACGT])")


def apply_substitutions(wt_nt: str, mutation: str) -> str:
    """Apply the underscore-separated nucleotide changes in one source row."""
    mutant_bases = list(wt_nt)

    for substitution in mutation.split("_"):
        match = SUBSTITUTION_PATTERN.fullmatch(substitution)

        if match is None:
            raise ValueError(f"Unsupported Jacquier mutation: {mutation}")

        wt_base, position_text, mutant_base = match.groups()
        position = int(position_text)

        if mutant_bases[position - 1] != wt_base:
            raise ValueError(f"WT base mismatch for {substitution}")

        mutant_bases[position - 1] = mutant_base

    return "".join(mutant_bases)


def read_source_rows(source_dir: Path) -> list[dict[str, object]]:
    """Read TEM-1 mutation and MIC score rows from the author supplement."""
    rows = read_worksheet(
        source_dir / "1215206110_sd01.xlsx",
        "TEM-1 DB",
    )
    headers = [str(value) for value in rows[0]]
    return [dict(zip(headers, row)) for row in rows[1:]]


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert the reported TEM-1 nucleotide variants and MIC scores."""
    wt_nt = next(iter(read_fasta(source_dir / "J01749.1_TEM-1_CDS.fasta").values()))
    wt_aa = translate_dna(wt_nt)
    rows = read_source_rows(source_dir)

    for row in rows:
        mutation = str(row["Mutant_nt"])
        mutant_nt = apply_substitutions(wt_nt, mutation)
        mutant_aa = translate_dna(mutant_nt)
        residue = int(row["Residue"])
        mutant_residue = str(row["Mutant_AA"])

        # Confirm that the nucleotide edit produces the amino acid and protein
        # position reported in the author spreadsheet.
        if mutant_aa[residue - 1] != mutant_residue:
            raise ValueError(f"Mutant residue mismatch for {mutation}")

        yield {
            "panel": "evo1",
            "study_id": "jacquier_2013",
            "assay_id": ASSAY_ID,
            "organism": "Escherichia coli",
            "target": "TEM-1 beta-lactamase",
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": str(row["Mutation"]),
            "experimental_score": str(row["MIC_Score"]),
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Jacquier assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
