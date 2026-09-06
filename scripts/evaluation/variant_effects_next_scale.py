import csv
import hashlib
import json
from dataclasses import dataclass
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
from nsm_dna.models.next_scale import (
    NSMDNA,
    MultiscaleTokenizer,
    NextScaleTransformer,
)


@dataclass(frozen=True)
class BlockPredictionBatch:
    """Tokenizer outputs used to score one target block."""

    target_ids: Int[Tensor, "batch block_length"]
    prefix: Float[Tensor, "batch prefix_length latent_dim"] | None
    scale_inputs: list[Float[Tensor, "batch scale_length latent_dim"]]
    targets_by_scale: list[Int[Tensor, "batch scale_length"]]


@dataclass(frozen=True)
class VariantWindows:
    """Source metadata and complete window tilings for one variant."""

    source_row: dict[str, str]
    reference_windows: tuple[str, ...]
    mutant_windows: tuple[str, ...]


@torch.no_grad()
def prepare_block_predictions(
    tokenizer: MultiscaleTokenizer,
    input_ids: Int[Tensor, "batch sequence_length"],
) -> list[BlockPredictionBatch]:
    """Encode the first block as context and the second as the scored target."""
    prefix_ids, target_ids = input_ids.split(tokenizer.context_length, dim=1)
    prefix = tokenizer.encode(prefix_ids)
    targets_by_scale = tokenizer.encode_indices(target_ids)
    return [
        BlockPredictionBatch(
            target_ids=target_ids,
            prefix=prefix,
            scale_inputs=tokenizer.indices_to_next_scale_inputs(targets_by_scale),
            targets_by_scale=targets_by_scale,
        )
    ]


PREDICTION_COLUMNS = (
    "study_id",
    "assay_id",
    "variant_id",
    "nt_edit",
    "experimental_score",
    "directionality",
    "num_reference_windows",
    "num_mutant_windows",
    "reference_mean_window_log_probability",
    "mutant_mean_window_log_probability",
    "variant_score",
)


def sequence_windows(
    sequence: str,
    window_length: int,
    stride: int,
) -> tuple[str, ...]:
    """Tile one sequence with fixed-stride windows and cover its final remainder."""
    sequence = sequence.upper()
    if len(sequence) < window_length:
        return ()

    final_start = len(sequence) - window_length
    window_starts = list(range(0, final_start + 1, stride))

    # A final end-aligned window covers the remainder when the sequence length
    # is not an exact multiple of the stride.
    if window_starts[-1] != final_start:
        window_starts.append(final_start)

    return tuple(
        sequence[window_start : window_start + window_length]
        for window_start in window_starts
    )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[NextScaleTransformer, MultiscaleTokenizer, int]:
    """Restore the frozen components of one jointly trained NSM-DNA checkpoint."""
    system, checkpoint_step = NSMDNA.from_checkpoint(
        checkpoint_path,
        device,
        frozen=True,
    )
    return system.transformer, system.tokenizer, checkpoint_step


@torch.inference_mode()
def score_token_ids(
    model: NextScaleTransformer,
    tokenizer: MultiscaleTokenizer,
    input_ids: Int[Tensor, "batch sequence_length"],
) -> tuple[
    Float[Tensor, "batch num_scales"],
    Float[Tensor, "batch"],
]:
    """Return per-scale hierarchy scores and the decoder score."""
    hierarchy_scores = torch.zeros(
        input_ids.shape[0],
        len(tokenizer.scale_lengths) - 1,
        device=input_ids.device,
        dtype=torch.float64,
    )
    decoder_scores = torch.zeros(
        input_ids.shape[0], device=input_ids.device, dtype=torch.float64
    )
    if input_ids.shape[1] == tokenizer.context_length:
        target_ids = input_ids
        targets_by_scale = tokenizer.encode_indices(target_ids)
        predictions = [
            BlockPredictionBatch(
                target_ids=target_ids,
                prefix=None,
                scale_inputs=tokenizer.indices_to_next_scale_inputs(targets_by_scale),
                targets_by_scale=targets_by_scale,
            )
        ]
    else:
        predictions = prepare_block_predictions(tokenizer, input_ids)

    for prediction in predictions:
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            logits = model(prediction.scale_inputs, prefix=prediction.prefix)
            decoder_logits = tokenizer.decode(prediction.targets_by_scale)

        # Likelihood counts every predicted code once; the training loss's scale
        # weights do not enter the sequence score.
        logits_by_scale = torch.split(logits, tokenizer.scale_lengths[1:], dim=1)
        for scale_index, (scale_logits, scale_targets) in enumerate(
            zip(logits_by_scale, prediction.targets_by_scale[1:], strict=True)
        ):
            log_probabilities = F.log_softmax(scale_logits.float(), dim=-1)
            hierarchy_scores[:, scale_index] += (
                log_probabilities.gather(-1, scale_targets.unsqueeze(-1))
                .squeeze(-1)
                .sum(1)
                .double()
            )

        # The decoder supplies the probability of the observed nucleotides given
        # the same complete hierarchy whose probability NSM-DNA assigned above.
        nucleotide_log_probabilities = F.log_softmax(decoder_logits.float(), dim=-1)
        nucleotide_targets = prediction.target_ids
        decoder_scores += (
            nucleotide_log_probabilities.gather(-1, nucleotide_targets.unsqueeze(-1))
            .squeeze(-1)
            .sum(1)
            .double()
        )

    return hierarchy_scores, decoder_scores


def score_sequences(
    sequences: list[str],
    model: NextScaleTransformer,
    tokenizer: MultiscaleTokenizer,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Score fixed-length DNA sequences and retain each score component."""
    predicted_scale_lengths = tokenizer.scale_lengths[1:]
    scores = {f"scale_{length}": [] for length in predicted_scale_lengths}
    scores.update({"hierarchy": [], "decoder": [], "joint": []})

    for start in tqdm(range(0, len(sequences), batch_size), unit="batch"):
        batch = sequences[start : start + batch_size]
        input_ids = torch.stack([encode_sequence(sequence) for sequence in batch]).to(
            device
        )
        hierarchy_by_scale, decoder = score_token_ids(model, tokenizer, input_ids)
        hierarchy = hierarchy_by_scale.sum(1)
        batch_scores = {
            f"scale_{length}": hierarchy_by_scale[:, scale_index]
            for scale_index, length in enumerate(predicted_scale_lengths)
        }
        batch_scores["hierarchy"] = hierarchy
        batch_scores["decoder"] = decoder
        batch_scores["joint"] = hierarchy + decoder
        for name, values in batch_scores.items():
            scores[name].extend(values.cpu().tolist())

    return scores


def read_assay_windows(
    path: Path,
    prefix_length: int,
    target_length: int,
) -> tuple[list[VariantWindows], int]:
    """Read variants and tile each complete reference and mutant independently."""
    selected_rows = []
    num_excluded = 0
    window_length = prefix_length + target_length

    with path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            reference = row["wt_nt"].upper()
            mutant = row["mutant_nt"].upper()
            reference_windows = sequence_windows(
                reference,
                window_length,
                target_length,
            )
            mutant_windows = sequence_windows(
                mutant,
                window_length,
                target_length,
            )
            if not reference_windows or not mutant_windows or reference == mutant:
                num_excluded += 1
                continue

            selected_rows.append(
                VariantWindows(
                    source_row=row,
                    reference_windows=reference_windows,
                    mutant_windows=mutant_windows,
                )
            )

    return selected_rows, num_excluded


def write_csv(
    rows: list[dict[str, str | int | float]],
    columns: tuple[str, ...],
    path: Path,
) -> None:
    """Write one result table atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(path)


def evaluate_assay(
    path: Path,
    output_dir: Path,
    model: NextScaleTransformer,
    tokenizer: MultiscaleTokenizer,
    batch_size: int,
    device: torch.device,
    prefix_length: int,
    target_length: int,
) -> dict[str, str | int | float]:
    """Score one assay and calculate its direction-adjusted Spearman correlation."""
    variants, num_excluded = read_assay_windows(path, prefix_length, target_length)

    # References, mutants, and separate variants often share complete windows.
    # Score each distinct sequence once and reuse it wherever it occurs.
    unique_sequences = list(
        dict.fromkeys(
            sequence
            for variant in variants
            for windows in (variant.reference_windows, variant.mutant_windows)
            for sequence in windows
        )
    )
    scores_by_component = score_sequences(
        unique_sequences,
        model,
        tokenizer,
        batch_size,
        device,
    )
    sequence_scores = {
        name: dict(zip(unique_sequences, scores, strict=True))
        for name, scores in scores_by_component.items()
    }

    predictions = []
    for variant in variants:
        row = variant.source_row
        reference_scores = {
            name: sum(scores[window] for window in variant.reference_windows)
            / len(variant.reference_windows)
            for name, scores in sequence_scores.items()
        }
        mutant_scores = {
            name: sum(scores[window] for window in variant.mutant_windows)
            / len(variant.mutant_windows)
            for name, scores in sequence_scores.items()
        }
        variant_scores = {
            name: mutant_scores[name] - reference_scores[name]
            for name in sequence_scores
        }
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
                "reference_mean_window_log_probability": reference_scores["joint"],
                "mutant_mean_window_log_probability": mutant_scores["joint"],
                "variant_score": variant_scores["joint"],
                **{
                    f"{name}_variant_score": score
                    for name, score in variant_scores.items()
                    if name != "joint"
                },
            }
        )

    experimental_scores = [
        float(row["experimental_score"]) * int(row["directionality"])
        for row in predictions
    ]
    correlations = {}
    for name in scores_by_component:
        score_column = "variant_score" if name == "joint" else f"{name}_variant_score"
        model_scores = [float(row[score_column]) for row in predictions]
        correlations[name] = float(
            spearmanr(model_scores, experimental_scores).statistic
        )
    component_names = [name for name in scores_by_component if name != "joint"]
    write_csv(
        predictions,
        PREDICTION_COLUMNS + tuple(f"{name}_variant_score" for name in component_names),
        output_dir / "predictions" / path.name,
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
        "spearman": correlations["joint"],
        **{f"spearman_{name}": correlations[name] for name in component_names},
    }


def sha256(path: Path) -> str:
    """Calculate a file checksum without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="variant_effects",
)
def main(config: DictConfig) -> None:
    """Run the selected assays and write predictions, correlations, and metadata."""
    device = torch.device(config.device)
    assay_paths = [Path(path) for path in config.assay_paths]
    output_directory = Path(config.output_directory)
    checkpoint_path = Path(config.checkpoint)
    model, tokenizer, checkpoint_step = load_model(
        checkpoint_path,
        device,
    )
    configured_prefix_length = config.prefix_length
    prefix_length = (
        int(model.max_prefix_length)
        if configured_prefix_length is None
        else int(configured_prefix_length)
    )
    if not 0 <= prefix_length <= model.max_prefix_length:
        raise ValueError(
            f"Evaluation prefix length {prefix_length} must be between 0 and "
            f"the model maximum of {model.max_prefix_length}."
        )
    target_length = int(tokenizer.context_length)
    component_names = [f"scale_{length}" for length in tokenizer.scale_lengths[1:]]
    component_names.extend(["hierarchy", "decoder"])

    results = []
    for path in assay_paths:
        print(f"Scoring {path.name}")
        result = evaluate_assay(
            path,
            output_directory,
            model,
            tokenizer,
            config.batch_size,
            device,
            prefix_length,
            target_length,
        )
        results.append(result)
        correlations = [
            f"joint {float(result['spearman']):.4f}",
            *(
                f"{name} {float(result[f'spearman_{name}']):.4f}"
                for name in component_names
            ),
        ]
        print(f"{result['assay_id']}: {', '.join(correlations)}")

    write_csv(
        results,
        (
            "study_id",
            "assay_id",
            "num_variants",
            "num_excluded",
            "num_reference_windows",
            "num_mutant_windows",
            "num_unique_window_sequences",
            "spearman",
            *(f"spearman_{name}" for name in component_names),
        ),
        output_directory / "correlations.csv",
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "score": (
            "mutant minus reference mean fixed-window hierarchy and nucleotide "
            "log probability over each complete sequence"
        ),
        "score_components": ["joint"] + component_names,
        "prefix_length": prefix_length,
        "target_length": target_length,
        "window_length": prefix_length + target_length,
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
    (output_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
