import re
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

ASSAY_ID = "MTH3_HAEAE_RockahShmuel_2015"

# Parse missense mutations such as N2D, synonymous entries such as N2, and
# nonsense mutations such as L3*. Positions use the authors' zero-based system.
MUTATION_PATTERN = re.compile(r"([A-Z])(\d+)([A-Z*])?")


def read_wt_sequence(source_dir: Path) -> str:
    """Read positions 0 through 331 of the author HaeIII construct."""
    rows = read_worksheet(
        source_dir / "pcbi.1004421.s002.xlsx",
        "raw G0",
        min_row=3,
    )
    # raw G0 begins with an upstream tag at positions -20 through -1. The
    # experimental HaeIII construct starts with ATG at position 0 and ends with
    # TAG at position 331, so retain only positions 0 through 331.
    wt_codons = {
        int(row[0]): str(row[2]).upper()
        for row in rows
        if isinstance(row[0], int) and 0 <= row[0] <= 331
    }

    return "".join(wt_codons[position] for position in range(332))


def read_single_nt_codons(source_dir: Path) -> dict[int, list[str]]:
    """Read the exact single-nucleotide codons listed by the authors."""
    rows = read_worksheet(
        source_dir / "pcbi.1004421.s003.xlsx",
        "# of nt exchanges",
    )
    codons = [str(codon) for codon in rows[0][3:]]

    # Each row gives the number of nucleotide changes from the WT codon to
    # every possible codon. Keep columns marked 1 because the authors defined
    # the analyzed dataset as single-nucleotide mutations.
    return {
        int(values[0]): [
            codon
            for codon, num_changes in zip(codons, values[3:])
            if num_changes == 1
        ]
        for values in rows[2:]
        if values[0] is not None
    }


def iter_source_scores(source_dir: Path) -> Iterator[tuple[str, float]]:
    """Read the G17 scores for missense, synonymous, and nonsense variants."""
    workbook_path = source_dir / "pcbi.1004421.s003.xlsx"

    for sheet_name in ("Single nt nonSyn", "Syn", "nonSense"):
        rows = read_worksheet(workbook_path, sheet_name)
        headers = [str(value) for value in rows[0]]

        for values in rows[1:]:
            row = dict(zip(headers, values))

            if row["Mutation"] is not None and row["Wrel G17"] is not None:
                yield str(row["Mutation"]), row["Wrel G17"]


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Pair each reported amino-acid effect with its measured source codons."""
    wt_nt = read_wt_sequence(source_dir)
    wt_aa = translate_dna(wt_nt)
    single_nt_codons = read_single_nt_codons(source_dir)

    for mutation, score in iter_source_scores(source_dir):
        match = MUTATION_PATTERN.fullmatch(mutation)

        if match is None:
            raise ValueError(f"Unsupported Rockah-Shmuel mutation: {mutation}")

        source_residue, source_position_text, reported_mutant = match.groups()
        source_position = int(source_position_text)

        # The source calls its first codon position 0. Our sequence functions
        # use one-based positions, so source position 2 becomes position 3.
        sequence_position = source_position + 1

        if wt_aa[sequence_position - 1] != source_residue:
            raise ValueError(f"WT residue mismatch for {mutation}")

        # Synonymous rows omit the mutant residue, for example N2. In that case
        # the target amino acid remains the WT residue N.
        target_residue = reported_mutant or source_residue

        # S3 reports one Wrel score after combining codons that produce the
        # same amino-acid mutation. Pair that score with each corresponding
        # single-nucleotide codon in the authors' analyzed dataset.
        for mutant_codon in single_nt_codons[source_position]:
            if translate_dna(mutant_codon) != target_residue:
                continue

            mutant_nt = replace_codon(wt_nt, sequence_position, mutant_codon)
            mutant_aa = translate_dna(mutant_nt)

            yield {
                "panel": "evo1",
                "study_id": "rockah_shmuel_2015",
                "assay_id": ASSAY_ID,
                "organism": "Haemophilus aegyptius",
                "target": "HaeIII DNA methyltransferase",
                "wt_nt": wt_nt,
                "mutant_nt": mutant_nt,
                "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
                "wt_aa": wt_aa,
                "mutant_aa": mutant_aa,
                "aa_change": (f"{source_residue}{sequence_position}{target_residue}"),
                "experimental_score": str(score),
                "directionality": 1,
            }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Rockah-Shmuel assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
