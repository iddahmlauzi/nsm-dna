import csv
import json
import re
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, TextIO

from dms.shared import VariantRecord, translate_dna, write_variants

MAVEDB_ARCHIVE_PATH = Path(
    "/home/iddah/datasets/mavedb/raw/2026-06-24/mavedb-dump.2026062418131.tar.gz"
)
OUTPUT_DIR = Path("/home/iddah/datasets/mavedb/standardized")

PANEL = "dnahnet"
STUDY_ID = "tsuboyama_2023"
TARGET_ORGANISM = "Escherichia coli K-12"
DIRECTIONALITY = 1

SUBSTITUTION_PATTERN = re.compile(r"(\d+)([ACGT])>([ACGT])")
DELETION_PATTERN = re.compile(r"(\d+)(?:_(\d+))?del")
INSERTION_PATTERN = re.compile(r"(\d+)_(\d+)ins([ACGT]+)")
DELINS_PATTERN = re.compile(r"(\d+)(?:_(\d+))?delins([ACGT]+)")


@dataclass(frozen=True)
class SequenceEdit:
    start: int
    end: int
    replacement: str


def load_public_dump(
    archive: tarfile.TarFile,
    archive_dir: str,
) -> dict[str, Any]:
    """Read `main.json` from an open MaveDB archive."""
    main_file = archive.extractfile(f"{archive_dir}/main.json")
    return json.load(main_file)


def find_score_sets(public_dump: dict[str, Any]) -> list[dict[str, Any]]:
    """Find combined E. coli K-12 score sets."""
    selected_score_sets = []

    for experiment_set in public_dump["experimentSets"]:
        for experiment in experiment_set["experiments"]:
            for score_set in experiment["scoreSets"]:
                target_gene = score_set["targetGenes"][0]
                target_sequence = target_gene["targetSequence"]

                if target_sequence is None:
                    continue

                organism = target_sequence["taxonomy"]["organismName"]

                if (
                    organism == TARGET_ORGANISM
                    and score_set["metaAnalyzesScoreSetUrns"]
                ):
                    selected_score_sets.append(score_set)

    return sorted(selected_score_sets, key=lambda score_set: score_set["urn"])


def get_target(score_set: dict[str, Any]) -> tuple[str, str, str]:
    """Read the target name, organism, and coding sequence."""
    target_gene = score_set["targetGenes"][0]
    target_sequence = target_gene["targetSequence"]

    return (
        target_gene["name"],
        target_sequence["taxonomy"]["organismName"],
        target_sequence["sequence"].upper(),
    )


def parse_nucleotide_edit(hgvs_nt: str) -> list[SequenceEdit]:
    """Parse a coding HGVS expression into sequence replacements."""
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
            position, _, alternate_base = substitution_match.groups()
            start = int(position) - 1
            sequence_edits.append(SequenceEdit(start, start + 1, alternate_base))
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
    """Apply a coding HGVS edit to a wild-type sequence."""
    mutant_bases = list(wt_nt.upper())
    sequence_edits = parse_nucleotide_edit(hgvs_nt)

    # Applying edits from right to left preserves their original HGVS positions.
    for edit in sorted(
        sequence_edits, key=lambda item: (item.start, item.end), reverse=True
    ):
        mutant_bases[edit.start : edit.end] = edit.replacement

    return "".join(mutant_bases)


def iter_standardized_variants(
    score_set: dict[str, Any],
    score_file: TextIO,
) -> Iterator[VariantRecord]:
    """Convert one MaveDB score CSV into standardized variants."""
    target, organism, wt_nt = get_target(score_set)
    wt_aa = translate_dna(wt_nt)
    reader = csv.DictReader(score_file)

    for source_variant in reader:
        variant_id = source_variant["accession"]
        mutant_nt = apply_nucleotide_edit(wt_nt, source_variant["hgvs_nt"])
        mutant_aa = translate_dna(mutant_nt)

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
            "experimental_score": source_variant["scores.score"],
            "directionality": DIRECTIONALITY,
        }


def standardize_mavedb(
    archive_path: Path = MAVEDB_ARCHIVE_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> int:
    """Standardize the combined E. coli K-12 score sets."""
    archive_dir = archive_path.name.removesuffix(".tar.gz")
    assay_stats = []
    total_variants = 0

    with tarfile.open(archive_path, mode="r:gz") as archive:
        public_dump = load_public_dump(archive, archive_dir)
        score_sets = find_score_sets(public_dump)

        for score_set in score_sets:
            score_set_urn = score_set["urn"]
            file_stem = score_set_urn.replace(":", "-")
            score_member = f"{archive_dir}/csv/{file_stem}.scores.csv"
            score_file = TextIOWrapper(archive.extractfile(score_member))
            output_path = output_dir / f"{file_stem}.csv"

            with score_file:
                num_variants = write_variants(
                    iter_standardized_variants(score_set, score_file),
                    output_path,
                )

            total_variants += num_variants
            assay_stats.append(
                {
                    "assay_id": score_set_urn,
                    "target": get_target(score_set)[0],
                    "num_variants": num_variants,
                }
            )
            print(f"{score_set_urn}: {num_variants:,} variants")

    stats = {
        "source": "MaveDB",
        "source_release": public_dump["asOf"],
        "source_archive": archive_path.name,
        "selection": f"Combined score sets for {TARGET_ORGANISM}",
        "organism": TARGET_ORGANISM,
        "phenotype": "protein folding stability",
        "num_assays": len(assay_stats),
        "num_variants": total_variants,
        "assays": assay_stats,
    }

    with (output_dir / "stats.json").open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2)
        stats_file.write("\n")

    print(f"Total: {total_variants:,} variants")
    return total_variants
