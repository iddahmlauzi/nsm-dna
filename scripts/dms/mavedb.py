import csv
import json
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dms.shared import VariantRecord, translate_dna, write_variants

MAVEDB_ARCHIVE_PATH = Path(
    "/home/iddah/datasets/mavedb/raw/2026-06-24/mavedb-dump.2026062418131.tar.gz"
)
MAVEDB_SCORE_SETS_DIR = Path(
    "/home/iddah/datasets/mavedb/ecoli_k12_combined_score_sets"
)
OUTPUT_DIR = Path("/home/iddah/datasets/mavedb/standardized")

PANEL = "dnahnet"
STUDY_ID = "tsuboyama_2023"
DIRECTIONALITY = 1
EXPECTED_TOTAL_VARIANTS = 21_250

SCORE_SET_URNS = (
    "urn:mavedb:00000122-0-1",
    "urn:mavedb:00000160-0-1",
    "urn:mavedb:00000161-0-1",
    "urn:mavedb:00000162-0-1",
    "urn:mavedb:00000213-0-1",
    "urn:mavedb:00000214-0-1",
    "urn:mavedb:00000230-0-1",
    "urn:mavedb:00000233-0-1",
    "urn:mavedb:00000282-0-1",
    "urn:mavedb:00000369-0-1",
    "urn:mavedb:00000383-0-1",
    "urn:mavedb:00000630-0-1",
)

REQUIRED_SCORE_COLUMNS = {
    "accession",
    "hgvs_nt",
    "hgvs_pro",
    "score",
}

SUBSTITUTION_PATTERN = re.compile(r"(\d+)([ACGT])>([ACGT])")
DELETION_PATTERN = re.compile(r"(\d+)(?:_(\d+))?del")
INSERTION_PATTERN = re.compile(r"(\d+)_(\d+)ins([ACGT]+)")
DELINS_PATTERN = re.compile(r"(\d+)(?:_(\d+))?delins([ACGT]+)")


@dataclass(frozen=True)
class SequenceEdit:
    """One zero-based sequence replacement parsed from an HGVS edit."""

    start: int
    end: int
    replacement: str
    reference: str | None = None


def load_public_dump(archive_path: Path) -> dict[str, Any]:
    """Read `main.json` directly from a compressed MaveDB archive.

    Args:
        archive_path: Path to the MaveDB public dump archive.

    Returns:
        Parsed public-dump object.

    Raises:
        FileNotFoundError: If the archive does not contain `main.json`.
    """
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/main.json"):
                continue

            main_file = archive.extractfile(member)

            if main_file is None:
                break

            with main_file:
                return json.load(main_file)

    raise FileNotFoundError(f"main.json was not found in {archive_path}")


def find_score_sets(
    public_dump: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Find the twelve approved score sets in the public dump.

    Args:
        public_dump: Parsed MaveDB `main.json` object.

    Returns:
        Score-set metadata keyed by URN.

    Raises:
        ValueError: If any approved score-set URN is missing.
    """
    selected_score_sets: dict[str, dict[str, Any]] = {}
    expected_urns = set(SCORE_SET_URNS)

    for experiment_set in public_dump["experimentSets"]:
        for experiment in experiment_set["experiments"]:
            for score_set in experiment["scoreSets"]:
                score_set_urn = score_set["urn"]

                if score_set_urn in expected_urns:
                    selected_score_sets[score_set_urn] = score_set

    missing_urns = expected_urns - selected_score_sets.keys()

    if missing_urns:
        missing_list = ", ".join(sorted(missing_urns))
        raise ValueError(f"Missing approved MaveDB score sets: {missing_list}")

    return selected_score_sets


def get_target(score_set: dict[str, Any]) -> tuple[str, str, str]:
    """Read the target name, organism, and coding sequence.

    Args:
        score_set: MaveDB score-set metadata.

    Returns:
        Target name, organism name, and uppercase coding DNA sequence.

    Raises:
        ValueError: If the score set does not contain exactly one target.
    """
    target_genes = score_set["targetGenes"]

    if len(target_genes) != 1:
        raise ValueError(
            f"Expected one target for {score_set['urn']}, found {len(target_genes)}"
        )

    target_gene = target_genes[0]
    target_sequence = target_gene["targetSequence"]

    return (
        target_gene["name"],
        target_sequence["taxonomy"]["organismName"],
        target_sequence["sequence"].upper(),
    )


def parse_nucleotide_edit(hgvs_nt: str) -> list[SequenceEdit]:
    """Parse the coding HGVS operations present in the selected score sets.

    Args:
        hgvs_nt: MaveDB coding HGVS expression.

    Returns:
        Zero-based sequence replacements for the expression.

    Raises:
        ValueError: If the expression contains an unsupported operation.
    """
    if hgvs_nt == "c.=":
        return []

    edit_text = hgvs_nt.removeprefix("c.")

    if edit_text.startswith("[") and edit_text.endswith("]"):
        edit_parts = edit_text[1:-1].split(";")
    else:
        edit_parts = [edit_text]

    sequence_edits = []

    for edit_part in edit_parts:
        substitution_match = SUBSTITUTION_PATTERN.fullmatch(edit_part)

        if substitution_match is not None:
            position, reference_base, alternate_base = substitution_match.groups()
            start = int(position) - 1
            sequence_edits.append(
                SequenceEdit(start, start + 1, alternate_base, reference_base)
            )
            continue

        deletion_match = DELETION_PATTERN.fullmatch(edit_part)

        if deletion_match is not None:
            first_position, last_position = deletion_match.groups()
            start = int(first_position) - 1
            end = int(last_position or first_position)
            sequence_edits.append(SequenceEdit(start, end, ""))
            continue

        insertion_match = INSERTION_PATTERN.fullmatch(edit_part)

        if insertion_match is not None:
            left_position, _, inserted_sequence = insertion_match.groups()
            insertion_index = int(left_position)
            sequence_edits.append(
                SequenceEdit(insertion_index, insertion_index, inserted_sequence)
            )
            continue

        delins_match = DELINS_PATTERN.fullmatch(edit_part)

        if delins_match is not None:
            first_position, last_position, replacement = delins_match.groups()
            start = int(first_position) - 1
            end = int(last_position or first_position)
            sequence_edits.append(SequenceEdit(start, end, replacement))
            continue

        raise ValueError(f"Unsupported nucleotide edit: {hgvs_nt}")

    return sequence_edits


def apply_nucleotide_edit(wt_nt: str, hgvs_nt: str) -> str:
    """Apply a MaveDB coding HGVS edit to its wild-type sequence.

    Args:
        wt_nt: Wild-type coding DNA sequence.
        hgvs_nt: Coding HGVS expression from MaveDB.

    Returns:
        Mutant coding DNA sequence.

    Raises:
        ValueError: If an edit falls outside the target or its reference base
            disagrees with the target sequence.
    """
    mutant_bases = list(wt_nt.upper())

    sequence_edits = parse_nucleotide_edit(hgvs_nt)

    # Applying edits from right to left preserves their original HGVS positions.
    for edit in sorted(
        sequence_edits, key=lambda item: (item.start, item.end), reverse=True
    ):
        if edit.start < 0 or edit.end > len(mutant_bases):
            raise ValueError(
                f"Edit range is outside a {len(mutant_bases)} nt target: {hgvs_nt}"
            )

        if edit.reference is not None and mutant_bases[edit.start] != edit.reference:
            raise ValueError(
                f"Reference base mismatch at c.{edit.start + 1}: expected "
                f"{mutant_bases[edit.start]}, found {edit.reference}"
            )

        mutant_bases[edit.start : edit.end] = edit.replacement

    return "".join(mutant_bases)


def iter_standardized_variants(
    score_set: dict[str, Any],
    score_path: Path,
) -> Iterator[VariantRecord]:
    """Convert the variants from one MaveDB score CSV.

    Args:
        score_set: Metadata for the score set.
        score_path: Path to its source score CSV.

    Yields:
        Standardized variants in source order.

    Raises:
        ValueError: If required source columns are missing, a variant
            accession repeats, or a reported stop does not translate to `*`.
    """
    target, organism, wt_nt = get_target(score_set)
    wt_aa = translate_dna(wt_nt)
    seen_variant_ids: set[str] = set()

    with score_path.open(encoding="utf-8", newline="") as score_file:
        reader = csv.DictReader(score_file)
        source_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_SCORE_COLUMNS - source_columns

        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing columns in {score_path}: {missing_list}")

        for source_variant in reader:
            variant_id = source_variant["accession"]

            if variant_id in seen_variant_ids:
                raise ValueError(
                    f"Duplicate variant accession in {score_set['urn']}: {variant_id}"
                )

            seen_variant_ids.add(variant_id)
            mutant_nt = apply_nucleotide_edit(
                wt_nt,
                source_variant["hgvs_nt"],
            )
            mutant_aa = translate_dna(mutant_nt)

            if "Ter" in source_variant["hgvs_pro"] and "*" not in mutant_aa:
                raise ValueError(
                    f"Reported stop does not translate to `*`: {variant_id}"
                )

            yield {
                "panel": PANEL,
                "study_id": STUDY_ID,
                "assay_id": score_set["urn"],
                "variant_id": variant_id,
                "organism": organism,
                "target": target,
                "wt_nt": wt_nt,
                "mutant_nt": mutant_nt,
                "nt_edit": source_variant["hgvs_nt"],
                "wt_aa": wt_aa,
                "mutant_aa": mutant_aa,
                "aa_change": source_variant["hgvs_pro"],
                "experimental_score": source_variant["score"],
                "directionality": DIRECTIONALITY,
            }


def score_filename(score_set_urn: str) -> str:
    """Return the source CSV filename for a score-set URN."""
    return f"{score_set_urn.replace(':', '-')}.scores.csv"


def output_filename(score_set_urn: str) -> str:
    """Return the standardized CSV filename for a score-set URN."""
    return f"{score_set_urn.replace(':', '-')}.csv"


def standardize_mavedb(
    archive_path: Path = MAVEDB_ARCHIVE_PATH,
    score_sets_dir: Path = MAVEDB_SCORE_SETS_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> int:
    """Standardize the twelve approved MaveDB score sets.

    Args:
        archive_path: MaveDB public dump containing target metadata.
        score_sets_dir: Directory containing the exact source score CSVs.
        output_dir: Destination for standardized score-set CSVs.

    Returns:
        Total number of variants written.

    Raises:
        ValueError: If a source or total row count differs from its expected
            value.
    """
    public_dump = load_public_dump(archive_path)
    score_sets = find_score_sets(public_dump)
    total_variants = 0

    for score_set_urn in SCORE_SET_URNS:
        score_set = score_sets[score_set_urn]
        score_path = score_sets_dir / score_filename(score_set_urn)
        output_path = output_dir / output_filename(score_set_urn)
        num_variants = write_variants(
            iter_standardized_variants(score_set, score_path),
            output_path,
        )

        if num_variants != score_set["numVariants"]:
            raise ValueError(
                f"Row count differs for {score_set_urn}: wrote "
                f"{num_variants}, metadata reports {score_set['numVariants']}"
            )

        total_variants += num_variants
        print(f"{score_set_urn}: {num_variants:,} variants")

    if total_variants != EXPECTED_TOTAL_VARIANTS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_VARIANTS:,} total variants, wrote "
            f"{total_variants:,}"
        )

    print(f"Total: {total_variants:,} variants")
    return total_variants
