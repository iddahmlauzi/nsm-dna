import csv
import io
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from dms.shared import (
    VariantRecord,
    describe_coding_edit,
    translate_dna,
    write_variants,
)

SOURCE_MEMBER = "Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
SINGLE_SUBSTITUTION_PATTERN = re.compile(r"([A-Z])(\d+)([A-Z])")

ASSAYS = {
    "1AOY.pdb": (
        "ARGR_ECOLI_Tsuboyama_2023_1AOY",
        "Escherichia coli",
        "Arginine repressor",
    ),
    "2KRU.pdb": (
        "BCHB_CHLTE_Tsuboyama_2023_2KRU",
        "Chlorobaculum tepidum",
        "Light-independent protochlorophyllide reductase subunit B",
    ),
    "1JIC.pdb": (
        "DN7A_SACS2_Tsuboyama_2023_1JIC",
        "Saccharolobus solfataricus",
        "DNA-binding protein 7a",
    ),
    "2D1U.pdb": (
        "FECA_ECOLI_Tsuboyama_2023_2D1U",
        "Escherichia coli",
        "Fe(3+) dicitrate transport protein FecA",
    ),
    "2LHR.pdb": (
        "ISDH_STAAW_Tsuboyama_2023_2LHR",
        "Staphylococcus aureus",
        "Iron-regulated surface determinant protein H",
    ),
    "1WCL.pdb": (
        "NUSA_ECOLI_Tsuboyama_2023_1WCL",
        "Escherichia coli",
        "Transcription termination/antitermination protein NusA",
    ),
    "2MI6.pdb": (
        "NUSG_MYCTU_Tsuboyama_2023_2MI6",
        "Mycobacterium tuberculosis",
        "Transcription termination/antitermination protein NusG",
    ),
    "1W4G.pdb": (
        "ODP2_GEOSE_Tsuboyama_2023_1W4G",
        "Geobacillus stearothermophilus",
        "Dihydrolipoyllysine-residue acetyltransferase",
    ),
    "1PSE.pdb": (
        "PSAE_PICP2_Tsuboyama_2023_1PSE",
        "Picosynechococcus",
        "Photosystem I reaction center subunit IV",
    ),
    "2LCL.pdb": (
        "RFAH_ECOLI_Tsuboyama_2023_2LCL",
        "Escherichia coli",
        "Transcription antitermination protein RfaH",
    ),
    "1GYZ.pdb": (
        "RL20_AQUAE_Tsuboyama_2023_1GYZ",
        "Aquifex aeolicus",
        "Large ribosomal subunit protein bL20",
    ),
    "1A32.pdb": (
        "RS15_GEOSE_Tsuboyama_2023_1A32",
        "Geobacillus stearothermophilus",
        "Small ribosomal subunit protein uS15",
    ),
    "2JVG.pdb": (
        "SBI_STAAM_Tsuboyama_2023_2JVG",
        "Staphylococcus aureus",
        "Immunoglobulin-binding protein Sbi",
    ),
    "2QFF.pdb": (
        "SCIN_STAAR_Tsuboyama_2023_2QFF",
        "Staphylococcus aureus",
        "Staphylococcal complement inhibitor",
    ),
    "1PV0.pdb": (
        "SDA_BACSU_Tsuboyama_2023_1PV0",
        "Bacillus subtilis",
        "Sporulation inhibitor Sda",
    ),
    "1LP1.pdb": (
        "SPA_STAAU_Tsuboyama_2023_1LP1",
        "Staphylococcus aureus",
        "Immunoglobulin G-binding protein A",
    ),
    "5UBS.pdb": (
        "SPG2_STRSG_Tsuboyama_2023_5UBS",
        "Streptococcus sp. group G",
        "Immunoglobulin G-binding protein G",
    ),
    "2KVT.pdb": (
        "YAIA_ECOLI_Tsuboyama_2023_2KVT",
        "Escherichia coli",
        "Uncharacterized protein YaiA",
    ),
    "2JVD.pdb": (
        "YNZC_BACSU_Tsuboyama_2023_2JVD",
        "Bacillus subtilis",
        "UPF0291 protein YnzC",
    ),
}


def target_coding_sequence(source_row: dict[str, str]) -> str:
    """Slice the assayed domain codons from the full author construct."""
    amino_acids = source_row["aa_seq"]
    offset = source_row["aa_seq_full"].find(amino_acids)

    if offset < 0:
        raise ValueError(f"aa_seq is absent from aa_seq_full for {source_row['name']}")

    start = offset * 3
    end = start + len(amino_acids) * 3
    coding_sequence = source_row["dna_seq"][start:end].upper()

    if translate_dna(coding_sequence) != amino_acids:
        raise ValueError(f"DNA translation mismatch for {source_row['name']}")

    return coding_sequence


def read_benchmark_mutations(source_dir: Path) -> dict[str, dict[str, str]]:
    """Map each ProteinGym mutant protein sequence to its consequence."""
    benchmark_mutations = {}

    for wt_name, (assay_id, _, _) in ASSAYS.items():
        with (source_dir / f"{assay_id}.csv").open(
            encoding="utf-8",
            newline="",
        ) as benchmark_file:
            rows = csv.DictReader(benchmark_file)
            benchmark_mutations[wt_name] = {
                row["mutated_sequence"]: row["mutant"] for row in rows
            }

    return benchmark_mutations


def read_selected_rows(
    source_dir: Path,
) -> tuple[dict[str, str], dict[str, list[dict[str, str]]]]:
    """Stream the author archive once and retain only the 19 assays."""
    wt_sequences: dict[str, str] = {}
    variant_rows = {wt_name: [] for wt_name in ASSAYS}
    benchmark_mutations = read_benchmark_mutations(source_dir)
    found_protein_sequences = {wt_name: set() for wt_name in ASSAYS}
    archive_path = source_dir / "Processed_K50_dG_datasets.zip"

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(SOURCE_MEMBER) as binary_file:
            text_file = io.TextIOWrapper(binary_file, encoding="utf-8")

            for row in csv.DictReader(text_file):
                wt_name = row["WT_name"]

                if wt_name not in ASSAYS:
                    continue

                if row["mut_type"] == "wt" and wt_name not in wt_sequences:
                    wt_sequences[wt_name] = target_coding_sequence(row)
                    continue

                mutant_aa = row["aa_seq"]

                if mutant_aa not in benchmark_mutations[wt_name]:
                    continue

                try:
                    float(row["ddG_ML"])
                except ValueError:
                    continue

                variant_rows[wt_name].append(
                    {
                        "name": row["name"],
                        "aa_change": benchmark_mutations[wt_name][mutant_aa],
                        "experimental_score": row["ddG_ML"],
                        "mutant_nt": target_coding_sequence(row),
                    }
                )
                found_protein_sequences[wt_name].add(mutant_aa)

    missing_wt = ASSAYS.keys() - wt_sequences.keys()

    if missing_wt:
        raise ValueError(f"Missing Tsuboyama WT rows: {sorted(missing_wt)}")

    for wt_name in ASSAYS:
        missing_sequences = (
            benchmark_mutations[wt_name].keys() - found_protein_sequences[wt_name]
        )

        if missing_sequences:
            raise ValueError(
                f"Missing Tsuboyama protein sequences for {wt_name}: "
                f"{sorted(missing_sequences)}"
            )

    return wt_sequences, variant_rows


def iter_assay_variants(
    wt_name: str,
    wt_nt: str,
    source_rows: list[dict[str, str]],
) -> Iterator[VariantRecord]:
    """Convert one Tsuboyama target while preserving mutant codon backgrounds."""
    assay_id, organism, target = ASSAYS[wt_name]
    wt_aa = translate_dna(wt_nt)
    name_counts = Counter(row["name"] for row in source_rows)

    for row in source_rows:
        mutation = row["aa_change"]
        mutant_nt = row["mutant_nt"]
        mutant_aa = translate_dna(mutant_nt)
        nt_edit = describe_coding_edit(wt_nt, mutant_nt)
        variant_id = row["name"]

        if name_counts[variant_id] > 1:
            variant_id = f"{variant_id}|{nt_edit}"

        for substitution in mutation.split(":"):
            match = SINGLE_SUBSTITUTION_PATTERN.fullmatch(substitution)

            if match is None:
                raise ValueError(f"Unsupported substitution: {mutation}")

            wt_residue, position_text, mutant_residue = match.groups()
            position = int(position_text)

            if wt_aa[position - 1] != wt_residue:
                raise ValueError(f"WT residue mismatch for {row['name']}")

            if mutant_aa[position - 1] != mutant_residue:
                raise ValueError(f"Mutant residue mismatch for {row['name']}")

        yield {
            "panel": "evo1",
            "study_id": "tsuboyama_2023",
            "assay_id": assay_id,
            "variant_id": variant_id,
            "organism": organism,
            "target": target,
            "wt_nt": wt_nt,
            "mutant_nt": mutant_nt,
            "nt_edit": nt_edit,
            "wt_aa": wt_aa,
            "mutant_aa": mutant_aa,
            "aa_change": mutation,
            "experimental_score": row["experimental_score"],
            "directionality": 1,
        }


def standardize(source_dir: Path, output_dir: Path) -> dict[str, int]:
    """Write the 19 standardized Tsuboyama assays."""
    wt_sequences, variant_rows = read_selected_rows(source_dir)
    row_counts = {}

    for wt_name, (assay_id, _, _) in ASSAYS.items():
        row_counts[assay_id] = write_variants(
            iter_assay_variants(
                wt_name,
                wt_sequences[wt_name],
                variant_rows[wt_name],
            ),
            output_dir / f"{assay_id}.csv",
        )

    return row_counts
