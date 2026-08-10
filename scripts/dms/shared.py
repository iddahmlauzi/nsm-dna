import csv
from collections.abc import Iterable
from pathlib import Path
from typing import TypedDict

STANDARD_COLUMNS = (
    "panel",
    "study_id",
    "assay_id",
    "variant_id",
    "organism",
    "target",
    "wt_nt",
    "mutant_nt",
    "nt_edit",
    "wt_aa",
    "mutant_aa",
    "aa_change",
    "experimental_score",
    "directionality",
)

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


class VariantRecord(TypedDict):
    """One variant in the shared NSM-DNA DMS table."""

    panel: str
    study_id: str
    assay_id: str
    variant_id: str
    organism: str
    target: str
    wt_nt: str
    mutant_nt: str
    nt_edit: str
    wt_aa: str
    mutant_aa: str
    aa_change: str
    experimental_score: str
    directionality: int


def translate_dna(sequence: str) -> str:
    """Translate a coding DNA sequence without truncating at stop codons.

    Args:
        sequence: Coding DNA sequence containing A, C, G, and T.

    Returns:
        Amino-acid sequence using `*` for stop codons.

    Raises:
        ValueError: If the sequence is not a complete, unambiguous set of
            codons.
    """
    normalized_sequence = sequence.upper()

    if len(normalized_sequence) % 3 != 0:
        raise ValueError(
            f"Coding sequence length is not divisible by three: "
            f"{len(normalized_sequence)}"
        )

    amino_acids = []

    for start in range(0, len(normalized_sequence), 3):
        codon = normalized_sequence[start : start + 3]

        if codon not in CODON_TABLE:
            raise ValueError(f"Unsupported codon: {codon}")

        amino_acids.append(CODON_TABLE[codon])

    return "".join(amino_acids)


def read_fasta(fasta_path: Path) -> str:
    """Read one DNA sequence from a FASTA file."""
    sequence_parts = []
    num_records = 0

    with fasta_path.open(encoding="utf-8") as fasta_file:
        for line in fasta_file:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                num_records += 1
                continue

            sequence_parts.append(line)

    if num_records != 1:
        raise ValueError(
            f"Expected one FASTA record in {fasta_path}, found {num_records}"
        )

    sequence = "".join(sequence_parts).upper()
    translate_dna(sequence)
    return sequence


def replace_codon(
    sequence: str,
    one_based_position: int,
    mutant_codon: str,
) -> str:
    """Replace one codon in a coding DNA sequence."""
    codon_start = (one_based_position - 1) * 3
    codon_end = codon_start + 3
    normalized_codon = mutant_codon.upper()

    if one_based_position < 1 or codon_end > len(sequence):
        raise ValueError(f"Codon position {one_based_position} is outside the sequence")

    if normalized_codon not in CODON_TABLE:
        raise ValueError(f"Unsupported codon: {mutant_codon}")

    return sequence[:codon_start] + normalized_codon + sequence[codon_end:]


def describe_coding_edit(wt_nt: str, mutant_nt: str) -> str:
    """Describe equal-length coding substitutions using HGVS-like notation."""
    if len(wt_nt) != len(mutant_nt):
        raise ValueError("Coding edit requires equal-length sequences")

    substitutions = [
        f"{position}{wt_base}>{mutant_base}"
        for position, (wt_base, mutant_base) in enumerate(
            zip(wt_nt.upper(), mutant_nt.upper()),
            start=1,
        )
        if wt_base != mutant_base
    ]

    if not substitutions:
        return "c.="

    if len(substitutions) == 1:
        return f"c.{substitutions[0]}"

    return f"c.[{';'.join(substitutions)}]"


def write_variants(
    variants: Iterable[VariantRecord],
    output_path: Path,
) -> int:
    """Write standardized variants to a CSV file.

    Args:
        variants: Standardized variant records.
        output_path: Destination CSV path.

    Returns:
        Number of variants written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_variants = 0

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()

        for variant in variants:
            writer.writerow(variant)
            num_variants += 1

    return num_variants
