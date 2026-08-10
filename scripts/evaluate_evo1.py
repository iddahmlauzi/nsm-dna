import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr
from tqdm import tqdm

INPUT_DIR = Path("/workspace/datasets/evo_panel/standardized")
OUTPUT_DIR = Path("/workspace/evo1_reproduction")

MODEL_NAME = "evo-1-8k-base"
MODEL_REVISION = "1.1_fix"

STUDY_NAMES = (
    "adkar_2012",
    "chen_2020",
    "firnberg_2014",
    "jacquier_2013",
    "kelsic_2016",
    "melnikov_2014",
    "rockah_shmuel_2015",
    "tsuboyama_2023",
    "weeks_2023",
)

REPORTED_CORRELATIONS = {
    "adkar_2012": 0.25,
    "chen_2020": 0.57,
    "firnberg_2014": 0.55,
    "jacquier_2013": 0.47,
    "kelsic_2016": 0.60,
    "melnikov_2014": 0.45,
    "rockah_shmuel_2015": 0.32,
    "tsuboyama_2023": 0.42,
    "weeks_2023": 0.50,
}

PREDICTION_COLUMNS = (
    "study_id",
    "assay_id",
    "variant_id",
    "experimental_score",
    "directionality",
    "evo_log_likelihood",
)

CORRELATION_COLUMNS = (
    "study_id",
    "num_variants",
    "spearman",
    "reported_spearman",
    "difference",
)


def parse_args() -> argparse.Namespace:
    """Parse paths and inference settings for the Evo 1 reproduction."""
    parser = argparse.ArgumentParser(description="Reproduce Evo 1 DMS correlations.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--study", action="append", choices=STUDY_NAMES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read rows from a CSV file."""
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def select_assay_files(input_dir: Path, studies: tuple[str, ...]) -> list[Path]:
    """Select all standardized assay files belonging to the requested studies."""
    selected_files = []

    for path in sorted(input_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as input_file:
            first_row = next(csv.DictReader(input_file))

        if first_row["study_id"] in studies:
            selected_files.append(path)

    return selected_files


def source_identifiers(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Return ordered assay and variant identifiers from source rows."""
    return [(row["assay_id"], row["variant_id"]) for row in rows]


def prediction_is_complete(
    source_rows: list[dict[str, str]],
    output_path: Path,
) -> bool:
    """Check whether an existing prediction file exactly covers its source assay."""
    if not output_path.exists():
        return False

    prediction_rows = read_csv_rows(output_path)
    return source_identifiers(prediction_rows) == source_identifiers(source_rows)


def load_evo_model(device: str) -> tuple[Any, Any]:
    """Load the published Evo 1 8k checkpoint and tokenizer."""
    from evo import Evo

    evo = Evo(MODEL_NAME, device=device)
    evo.model.eval()
    return evo.model, evo.tokenizer


def score_sequences(
    sequences: list[str],
    model: Any,
    tokenizer: Any,
    batch_size: int,
    device: str,
    description: str,
) -> list[float]:
    """Compute mean nucleotide log-likelihoods with Evo's official scorer."""
    from evo.scoring import score_sequences as score_evo_sequences

    scores = []
    batch_starts = range(0, len(sequences), batch_size)

    for start in tqdm(batch_starts, desc=description, unit="batch"):
        batch = sequences[start : start + batch_size]
        batch_scores = score_evo_sequences(
            batch,
            model,
            tokenizer,
            reduce_method="mean",
            device=device,
        )
        scores.extend(float(score) for score in batch_scores)

    return scores


def write_predictions(
    source_rows: list[dict[str, str]],
    scores: list[float],
    output_path: Path,
) -> None:
    """Write compact Evo predictions for one assay atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()

        for source_row, score in zip(source_rows, scores, strict=True):
            writer.writerow(
                {
                    "study_id": source_row["study_id"],
                    "assay_id": source_row["assay_id"],
                    "variant_id": source_row["variant_id"],
                    "experimental_score": source_row["experimental_score"],
                    "directionality": source_row["directionality"],
                    "evo_log_likelihood": score,
                }
            )

    temporary_path.replace(output_path)


def calculate_spearman(rows: list[dict[str, str]]) -> float:
    """Correlate Evo likelihood with direction-adjusted experimental fitness."""
    model_scores = [float(row["evo_log_likelihood"]) for row in rows]
    fitness_scores = [
        float(row["experimental_score"]) * int(row["directionality"]) for row in rows
    ]
    return float(spearmanr(model_scores, fitness_scores).statistic)


def study_correlations(
    prediction_paths: list[Path],
) -> dict[str, dict[str, float | int]]:
    """Pool assay predictions by study and calculate nine study-level metrics."""
    rows_by_study: dict[str, list[dict[str, str]]] = defaultdict(list)

    for path in prediction_paths:
        for row in read_csv_rows(path):
            rows_by_study[row["study_id"]].append(row)

    return {
        study: {
            "num_variants": len(rows),
            "spearman": calculate_spearman(rows),
        }
        for study, rows in rows_by_study.items()
    }


def correlation_rows(
    results: dict[str, dict[str, float | int]],
    studies: tuple[str, ...],
) -> list[dict[str, str | int | float]]:
    """Build paper-comparison rows, including the nine-study macro-average."""
    rows = []

    for study in studies:
        result = results[study]
        calculated = float(result["spearman"])
        reported = REPORTED_CORRELATIONS[study]
        rows.append(
            {
                "study_id": study,
                "num_variants": int(result["num_variants"]),
                "spearman": calculated,
                "reported_spearman": reported,
                "difference": calculated - reported,
            }
        )

    if studies == STUDY_NAMES:
        calculated_average = sum(float(results[name]["spearman"]) for name in studies)
        calculated_average /= len(studies)
        rows.append(
            {
                "study_id": "macro_average",
                "num_variants": sum(
                    int(results[name]["num_variants"]) for name in studies
                ),
                "spearman": calculated_average,
                "reported_spearman": 0.46,
                "difference": calculated_average - 0.46,
            }
        )

    return rows


def write_correlations(
    rows: list[dict[str, str | int | float]],
    output_path: Path,
) -> None:
    """Write the study-level comparison with Evo Table S4."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CORRELATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Calculate a file checksum without loading the file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_run_metadata(
    source_paths: list[Path],
    prediction_paths: list[Path],
    output_path: Path,
    batch_size: int,
    device: str,
) -> None:
    """Record the inputs and runtime required to interpret the result."""
    import torch

    input_files = []

    for path in source_paths:
        rows = read_csv_rows(path)
        input_files.append(
            {
                "name": path.name,
                "sha256": sha256(path),
                "rows": len(rows),
                "stop_variants": sum("*" in row["mutant_aa"] for row in rows),
            }
        )

    prediction_rows = sum(len(read_csv_rows(path)) for path in prediction_paths)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "likelihood_reduction": "mean",
        "prepend_eos": True,
        "batch_size": batch_size,
        "device": device,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "evo_model": version("evo-model"),
        "flash_attn": version("flash-attn"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "input_rows": sum(file["rows"] for file in input_files),
        "prediction_rows": prediction_rows,
        "input_files": input_files,
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def evaluate_assays(
    source_paths: list[Path],
    output_dir: Path,
    batch_size: int,
    device: str,
) -> list[Path]:
    """Score incomplete assays and return all selected prediction paths."""
    pending_assays = []
    prediction_paths = []

    for source_path in source_paths:
        source_rows = read_csv_rows(source_path)
        output_path = output_dir / "predictions" / source_path.name
        prediction_paths.append(output_path)

        if prediction_is_complete(source_rows, output_path):
            print(f"Already complete: {source_path.name}")
        else:
            pending_assays.append((source_path, source_rows, output_path))

    if not pending_assays:
        return prediction_paths

    model, tokenizer = load_evo_model(device)

    for source_path, source_rows, output_path in pending_assays:
        scores = score_sequences(
            [row["mutant_nt"] for row in source_rows],
            model,
            tokenizer,
            batch_size,
            device,
            source_path.stem,
        )
        write_predictions(source_rows, scores, output_path)

    return prediction_paths


def main() -> None:
    """Score the selected studies and compare them with Evo Table S4."""
    args = parse_args()

    studies = STUDY_NAMES
    source_paths = sorted(args.input_dir.glob("*.csv"))

    if args.study:
        studies = tuple(args.study)
        source_paths = select_assay_files(args.input_dir, studies)

    prediction_paths = evaluate_assays(
        source_paths,
        args.output_dir,
        args.batch_size,
        args.device,
    )
    results = study_correlations(prediction_paths)
    comparison_rows = correlation_rows(results, studies)
    write_correlations(comparison_rows, args.output_dir / "correlations.csv")
    write_run_metadata(
        source_paths,
        prediction_paths,
        args.output_dir / "run_metadata.json",
        args.batch_size,
        args.device,
    )

    for row in comparison_rows:
        print(
            f"{row['study_id']}: {float(row['spearman']):.4f} "
            f"(reported {float(row['reported_spearman']):.2f})"
        )


if __name__ == "__main__":
    main()
