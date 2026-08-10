import csv
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    VariantRecord,
    describe_coding_edit,
    replace_codon,
    translate_dna,
    write_variants,
)

ASSAY_ID = "IF1_ECOLI_Kelsic_2016"
SOURCE_NAME = "NIHMS836960-supplement-1.csv"


def read_source_rows(source_dir: Path) -> tuple[str, list[dict[str, str]]]:
    """Read the MAGE-Seq table and reconstruct the infA WT sequence."""
    with (source_dir / SOURCE_NAME).open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file, skipinitialspace=True))

    wt_codons = {
        int(row["pos"]): row["codon"].upper() for row in rows if row["is_wt"] == "1"
    }
    wt_nt = "".join(wt_codons[position] for position in sorted(wt_codons))
    return wt_nt, rows


def iter_variants(source_dir: Path) -> Iterator[VariantRecord]:
    """Convert scored non-WT infA codons using rich-medium fitness."""
    wt_nt, rows = read_source_rows(source_dir)
    wt_aa = translate_dna(wt_nt)

    for row in rows:
        if row["is_wt"] == "1" or not row["fitness_rich"]:
            continue

        position = int(row["pos"])
        mutant_codon = row["codon"].upper()
        mutant_nt = replace_codon(wt_nt, position, mutant_codon)
        mutant_aa = translate_dna(mutant_nt)

        yield {
            "panel": "evo1",
            "study_id": "kelsic_2016",
            "assay_id": ASSAY_ID,
            "variant_id": f"position_{position}_{mutant_codon}",
            "organism": "Escherichia coli",
            "target": "translation initiation factor IF-1",
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": describe_coding_edit(wt_nt, mutant_nt),
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": (f"{wt_aa[position - 1]}{position}{mutant_aa[position - 1]}"),
            "experimental_score": row["fitness_rich"],
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> int:
    """Write the standardized Kelsic assay."""
    return write_variants(
        iter_variants(source_dir),
        output_dir / f"{ASSAY_ID}.csv",
    )
