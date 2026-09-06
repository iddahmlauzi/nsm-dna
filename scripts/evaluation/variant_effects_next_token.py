import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from omegaconf import DictConfig
from scipy.stats import spearmanr
from torch import Tensor
from tqdm import tqdm

from nsm_dna.data import encode_sequence
from nsm_dna.models.next_token import NextTokenModel
from scripts.evaluation.variant_effects_next_scale import (
    PREDICTION_COLUMNS,
    read_assay_windows,
    sha256,
    write_csv,
)


@torch.inference_mode()
def score_next_token_ids(
    model: NextTokenModel,
    input_ids: Int[Tensor, "batch sequence_length"],
) -> Float[Tensor, "batch"]:
    """Return the summed next-nucleotide log probability of each sequence."""
    context_ids = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    with torch.autocast(
        device_type=input_ids.device.type,
        dtype=torch.bfloat16,
        enabled=input_ids.device.type == "cuda",
    ):
        logits = model(context_ids)

    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    return (
        log_probabilities.gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
        .sum(1)
        .double()
    )


def score_sequences(
    sequences: list[str],
    model: NextTokenModel,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    """Score fixed-length DNA sequences in batches."""
    scores = []
    for start in tqdm(range(0, len(sequences), batch_size), unit="batch"):
        batch = sequences[start : start + batch_size]
        input_ids = torch.stack([encode_sequence(sequence) for sequence in batch]).to(
            device
        )
        scores.extend(score_next_token_ids(model, input_ids).cpu().tolist())

    return scores


def evaluate_assay(
    path: Path,
    output_directory: Path,
    model: NextTokenModel,
    batch_size: int,
    device: torch.device,
    prefix_length: int,
    target_length: int,
) -> dict[str, str | int | float]:
    """Score one assay and calculate direction-adjusted Spearman correlation."""
    variants, num_excluded = read_assay_windows(path, prefix_length, target_length)
    unique_sequences = list(
        dict.fromkeys(
            sequence
            for variant in variants
            for windows in (variant.reference_windows, variant.mutant_windows)
            for sequence in windows
        )
    )
    sequence_scores = dict(
        zip(
            unique_sequences,
            score_sequences(unique_sequences, model, batch_size, device),
            strict=True,
        )
    )

    predictions = []
    for variant in variants:
        row = variant.source_row
        reference_score = sum(
            sequence_scores[window] for window in variant.reference_windows
        ) / len(variant.reference_windows)
        mutant_score = sum(
            sequence_scores[window] for window in variant.mutant_windows
        ) / len(variant.mutant_windows)
        predictions.append(
            {
                "study_id": row["study_id"],
                "assay_id": row["assay_id"],
                "variant_id": row.get("variant_id", ""),
                "nt_edit": row["nt_edit"],
                "experimental_score": row["experimental_score"],
                "directionality": row["directionality"],
                "num_reference_windows": len(variant.reference_windows),
                "num_mutant_windows": len(variant.mutant_windows),
                "reference_mean_window_log_probability": reference_score,
                "mutant_mean_window_log_probability": mutant_score,
                "variant_score": mutant_score - reference_score,
            }
        )

    experimental_scores = [
        float(row["experimental_score"]) * int(row["directionality"])
        for row in predictions
    ]
    model_scores = [float(row["variant_score"]) for row in predictions]
    correlation = float(spearmanr(model_scores, experimental_scores).statistic)

    write_csv(
        predictions,
        PREDICTION_COLUMNS,
        output_directory / "predictions" / path.name,
    )
    return {
        "study_id": variants[0].source_row["study_id"],
        "assay_id": variants[0].source_row["assay_id"],
        "num_variants": len(variants),
        "num_excluded": num_excluded,
        "num_reference_windows": sum(
            len(variant.reference_windows) for variant in variants
        ),
        "num_mutant_windows": sum(
            len(variant.mutant_windows) for variant in variants
        ),
        "num_unique_window_sequences": len(unique_sequences),
        "spearman": correlation,
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="next_token_variant_effects",
)
def main(config: DictConfig) -> None:
    """Evaluate one next-token checkpoint on the selected variant assays."""
    device = torch.device(config.device)
    checkpoint_path = Path(config.checkpoint)
    output_directory = Path(config.output_directory)
    assay_paths = [Path(path) for path in config.assay_paths]
    model, checkpoint_step = NextTokenModel.from_checkpoint(
        checkpoint_path,
        device,
        frozen=True,
    )
    window_length = model.max_sequence_length + 1
    prefix_length = window_length // 2
    target_length = window_length - prefix_length

    results = []
    for path in assay_paths:
        print(f"Scoring {path.name}")
        result = evaluate_assay(
            path,
            output_directory,
            model,
            config.batch_size,
            device,
            prefix_length,
            target_length,
        )
        results.append(result)
        print(f"{result['assay_id']}: Spearman {float(result['spearman']):.4f}")

    result_columns = (
        "study_id",
        "assay_id",
        "num_variants",
        "num_excluded",
        "num_reference_windows",
        "num_mutant_windows",
        "num_unique_window_sequences",
        "spearman",
    )
    write_csv(results, result_columns, output_directory / "correlations.csv")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "score": (
            "mutant minus reference mean fixed-window next-nucleotide log "
            "probability over each complete sequence"
        ),
        "prefix_length": prefix_length,
        "target_length": target_length,
        "window_length": window_length,
        "stride": target_length,
        "window_alignment": (
            "fixed stride from the sequence start with one end-aligned window "
            "for a trailing remainder"
        ),
        "batch_size": config.batch_size,
        "device": str(device),
        "input_files": [
            {"path": str(path), "sha256": sha256(path)} for path in assay_paths
        ],
        "results": results,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
