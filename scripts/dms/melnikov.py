import io
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    CODON_TABLE,
    VariantRecord,
    describe_coding_edit,
    read_worksheet,
    replace_codon,
    translate_dna,
    write_variants,
)

ASSAY_ID = "KKA2_KLEPN_Melnikov_2014"

# Each oligo is a short synthesized piece of the gene. An ID such as
# KKA2_KLEPN_A1_2_F means tile KKA2_KLEPN_A1 was designed to change protein
# position 2 to F. The matching unmutated tile ends in `_wt`.
OLIGO_PATTERN = re.compile(
    r"^(KKA2_KLEPN_[AB]\d+)_(\d+)_(wt|[ACDEFGHIKLMNPQRSTVWY])$"
)
SCORE_FILES = (
    "Supplementary Data 2 - Unpacked/KKA2_S1_Kan18_L1.aadiff.txt",
    "Supplementary Data 2 - Unpacked/KKA2_S3_Kan18_L1.aadiff.txt",
)


def read_wt_sequence(source_dir: Path) -> str:
    """Read the exact synthetic APH(3')II construct."""
    rows = read_worksheet(
        source_dir / "supp_gku511_nar-01049-met-h-2014-File006.xlsx",
        "Primers_etc",
    )

    # Primers_etc contains many named sequences. KKA2_KLEPN_opt is the complete
    # WT coding sequence used to design this study's KKA2 mutant library.
    for row in rows:
        if row[0] == "KKA2_KLEPN_opt":
            return str(row[2]).upper()

    raise ValueError("KKA2_KLEPN_opt was not found in Primers_etc")


def read_oligos(source_dir: Path) -> dict[str, str]:
    """Read the six KKA2 WT tiles and 4,997 designed mutant tiles."""
    workbook_path = source_dir / "supp_gku511_nar-01049-met-h-2014-File006.xlsx"
    oligos = {}

    # OLS_A and OLS_B list all DNA oligos synthesized on two microarrays. Keep
    # only IDs belonging to the KKA2 experiment.
    for sheet_name in ("OLS_A", "OLS_B"):
        rows = read_worksheet(workbook_path, sheet_name, min_row=6, max_col=2)

        for oligo_id, sequence in rows:
            if isinstance(oligo_id, str) and OLIGO_PATTERN.fullmatch(oligo_id):
                oligos[oligo_id] = str(sequence).upper()

    return oligos


def read_score_matrix(
    source_dir: Path,
) -> dict[tuple[int, str], float]:
    """Average S1 and S3 scores for each amino-acid substitution."""
    archive_path = source_dir / "supp_gku511_nar-01049-met-h-2014-File007.zip"
    replicate_scores: list[dict[tuple[int, str], float]] = []

    with zipfile.ZipFile(archive_path) as archive:
        for score_filename in SCORE_FILES:
            with archive.open(score_filename) as binary_file:
                lines = io.TextIOWrapper(binary_file, encoding="utf-8")

                # Each file is one selection round under Kan18: kanamycin at
                # 1/8 of the WT minimum inhibitory concentration. The first
                # line names the experiment, and the second lists protein
                # positions across the matrix columns.
                next(lines)
                positions = [int(value) for value in next(lines).split("\t")[1:]]

                # The third line lists the WT amino acid at every position.
                next(lines)
                scores = {}

                for line in lines:
                    values = line.strip().split("\t")

                    if not values[0].startswith("Delta-"):
                        continue

                    # A Delta-F row gives the change in frequency after
                    # selection for variants producing F at every position.
                    mutant_residue = values[0].removeprefix("Delta-")

                    for position, value in zip(positions, values[1:]):
                        scores[(position, mutant_residue)] = float(value)

                replicate_scores.append(scores)

    # S1 and S3 are independent selection rounds under the same condition.
    # The authors marked S2 as a failed sequencing library, so it is excluded.
    # Average the two measurements for each (position, mutant amino acid).
    return {
        key: (replicate_scores[0][key] + replicate_scores[1][key]) / 2
        for key in replicate_scores[0]
    }


def find_mutant_codon(
    wt_tile: str,
    mutant_tile: str,
    expected_wt_codon: str,
    mutant_residue: str,
) -> str:
    """Locate the designed mutant codon within a matched oligo tile."""
    candidates: set[str] = set()

    # A mutant oligo and its WT tile differ only at the designed codon. Some
    # tiles are written in the coding direction and others in the opposite
    # direction, so compare both the supplied sequences and their reverse
    # complements.
    for oriented_wt, oriented_mutant in (
        (wt_tile, mutant_tile),
        (reverse_complement(wt_tile), reverse_complement(mutant_tile)),
    ):
        changed_indices = [
            index
            for index, (wt_base, mutant_base) in enumerate(
                zip(oriented_wt, oriented_mutant)
            )
            if wt_base != mutant_base
        ]

        # Find the three-base window containing all differences. Accept it only
        # if the WT window matches the full WT gene and the mutant window
        # translates to the amino acid named in the oligo ID.
        for codon_start in range(
            max(0, min(changed_indices) - 2),
            min(changed_indices) + 1,
        ):
            codon_end = codon_start + 3
            wt_codon = oriented_wt[codon_start:codon_end]
            mutant_codon = oriented_mutant[codon_start:codon_end]

            if any(
                index < codon_start or index >= codon_end for index in changed_indices
            ):
                continue

            if wt_codon != expected_wt_codon:
                continue

            if CODON_TABLE.get(mutant_codon) == mutant_residue:
                candidates.add(mutant_codon)

    if len(candidates) != 1:
        raise ValueError("Could not identify one designed mutant codon")

    return next(iter(candidates))


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA oligo."""
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert the author-designed APH(3')II nucleotide variants."""
    wt_nt = read_wt_sequence(source_dir)
    wt_aa = translate_dna(wt_nt)
    oligos = read_oligos(source_dir)
    scores = read_score_matrix(source_dir)
    wt_tiles = {}

    # Build one WT sequence for each of the six tiles. For example,
    # KKA2_KLEPN_A1_2_F is compared with the KKA2_KLEPN_A1 WT tile to recover
    # the exact mutant codon synthesized by the authors.
    for oligo_id, sequence in oligos.items():
        match = OLIGO_PATTERN.fullmatch(oligo_id)

        if match is not None and match.group(3) == "wt":
            wt_tiles[match.group(1)] = sequence

    for oligo_id, mutant_tile in oligos.items():
        match = OLIGO_PATTERN.fullmatch(oligo_id)

        if match is None or match.group(3) == "wt":
            continue

        tile_id, position_text, mutant_residue = match.groups()
        position = int(position_text)
        wt_tile = wt_tiles[tile_id]
        wt_residue = wt_aa[position - 1]

        # The oligo ID provides the protein position and resulting amino acid,
        # while comparison with the WT oligo provides the exact mutant codon.
        codon_start = (position - 1) * 3
        wt_codon = wt_nt[codon_start : codon_start + 3]
        mutant_codon = find_mutant_codon(
            wt_tile,
            mutant_tile,
            wt_codon,
            mutant_residue,
        )
        mutant_nt = replace_codon(wt_nt, position, mutant_codon)
        mutant_aa = translate_dna(mutant_nt)

        yield {
            "panel": "evo1",
            "study_id": "melnikov_2014",
            "assay_id": ASSAY_ID,
            "organism": "Klebsiella pneumoniae",
            "target": "APH(3')II aminoglycoside phosphotransferase",
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": f"{wt_residue}{position}{mutant_residue}",
            "experimental_score": str(scores[(position, mutant_residue)]),
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the standardized Melnikov assay."""
    row_count = write_variants(
        iter_variants(source_dir), output_dir / f"{ASSAY_ID}.csv"
    )
    return {ASSAY_ID: row_count}
