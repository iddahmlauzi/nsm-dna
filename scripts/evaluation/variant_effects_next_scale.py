import csv
import hashlib
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
from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_nsm import prepare_block_predictions

PREDICTION_COLUMNS = (
    "study_id",
    "assay_id",
    "variant_id",
    "nt_edit",
    "experimental_score",
    "directionality",
    "window_start_0_based",
    "reference_log_probability",
    "mutant_log_probability",
    "variant_score",
)


def mutation_target_window(
    reference: str,
    mutant: str,
    prefix_length: int,
    target_length: int,
) -> tuple[str, str, int] | None:
    """Place every changed nucleotide in the target after a full prefix."""
    reference = reference.upper()
    mutant = mutant.upper()
    window_length = prefix_length + target_length
    if len(reference) != len(mutant) or len(reference) < window_length:
        return None

    changed_positions = [
        position
        for position, (reference_base, mutant_base) in enumerate(zip(reference, mutant))
        if reference_base != mutant_base
    ]
    if not changed_positions:
        return None

    first_change = changed_positions[0]
    last_change = changed_positions[-1]
    if first_change < prefix_length:
        return None

    window_start = min(
        first_change - prefix_length,
        len(reference) - window_length,
    )
    window_end = window_start + window_length
    target_start = window_start + prefix_length
    if first_change < target_start or last_change >= window_end:
        return None

    return (
        reference[window_start:window_end],
        mutant[window_start:window_end],
        window_start,
    )


def load_model(
    checkpoint_path: Path,
    tokenizer_checkpoint_path: Path,
    device: torch.device,
) -> tuple[NSM, VQVAE, int]:
    """Restore the frozen tokenizer and one trained NSM-DNA checkpoint."""
    tokenizer = VQVAE.from_checkpoint(
        tokenizer_checkpoint_path,
        device,
        frozen=True,
    )
    model, checkpoint_step = NSM.from_checkpoint(
        checkpoint_path,
        tokenizer,
        device,
        frozen=True,
    )
    return model, tokenizer, checkpoint_step


@torch.inference_mode()
def score_token_ids(
    model: NSM,
    tokenizer: VQVAE,
    input_ids: Int[Tensor, "batch sequence_length"],
) -> tuple[
    Float[Tensor, "batch num_scales"],
    Float[Tensor, "batch"],
]:
    """Return per-scale hierarchy scores and the decoder score."""
    hierarchy_scores = torch.zeros(
        input_ids.shape[0],
        len(tokenizer.scale_lengths),
        device=input_ids.device,
        dtype=torch.float64,
    )
    decoder_scores = torch.zeros(
        input_ids.shape[0], device=input_ids.device, dtype=torch.float64
    )
    for prediction in prepare_block_predictions(tokenizer, input_ids):
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            logits = model(prediction.scale_inputs, prefix=prediction.prefix)
            decoder_logits = tokenizer.decode(prediction.targets_by_scale)

        # Likelihood counts every predicted code once; the training loss's scale
        # weights do not enter the sequence score.
        logits_by_scale = torch.split(logits, tokenizer.scale_lengths, dim=1)
        for scale_index, (scale_logits, scale_targets) in enumerate(
            zip(logits_by_scale, prediction.targets_by_scale)
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
    model: NSM,
    tokenizer: VQVAE,
    batch_size: int,
    device: torch.device,
) -> dict[str, list[float]]:
    """Score fixed-length DNA sequences and retain each score component."""
    scores = {f"scale_{length}": [] for length in tokenizer.scale_lengths}
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
            for scale_index, length in enumerate(tokenizer.scale_lengths)
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
) -> tuple[list[dict[str, str]], int]:
    """Read rows with a full prefix and all edits inside the target."""
    selected_rows = []
    num_excluded = 0

    with path.open(encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            window = mutation_target_window(
                row["wt_nt"],
                row["mutant_nt"],
                prefix_length,
                target_length,
            )
            if window is None:
                num_excluded += 1
                continue

            reference, mutant, window_start = window
            row["reference_window"] = reference
            row["mutant_window"] = mutant
            row["window_start_0_based"] = str(window_start)
            selected_rows.append(row)

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
    model: NSM,
    tokenizer: VQVAE,
    batch_size: int,
    device: torch.device,
    prefix_length: int,
    target_length: int,
) -> dict[str, str | int | float]:
    """Score one assay and calculate its direction-adjusted Spearman correlation."""
    rows, num_excluded = read_assay_windows(path, prefix_length, target_length)

    # Variants at nearby positions often share a reference window. Score each
    # distinct sequence once and reuse its result for every matching row.
    unique_sequences = list(
        dict.fromkeys(
            sequence
            for row in rows
            for sequence in (row["reference_window"], row["mutant_window"])
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
    for row in rows:
        reference = row["reference_window"]
        mutant = row["mutant_window"]
        variant_scores = {
            name: scores[mutant] - scores[reference]
            for name, scores in sequence_scores.items()
        }
        predictions.append(
            {
                "study_id": row["study_id"],
                "assay_id": row["assay_id"],
                "variant_id": row.get("variant_id", ""),
                "nt_edit": row["nt_edit"],
                "experimental_score": row["experimental_score"],
                "directionality": row["directionality"],
                "window_start_0_based": row["window_start_0_based"],
                "reference_log_probability": sequence_scores["joint"][reference],
                "mutant_log_probability": sequence_scores["joint"][mutant],
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
        "study_id": rows[0]["study_id"],
        "assay_id": rows[0]["assay_id"],
        "num_variants": len(rows),
        "num_excluded": num_excluded,
        "num_unique_windows": len(unique_sequences),
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
    checkpoint_path = Path(config.nsm_checkpoint)
    tokenizer_checkpoint_path = Path(config.tokenizer_checkpoint)
    model, tokenizer, checkpoint_step = load_model(
        checkpoint_path,
        tokenizer_checkpoint_path,
        device,
    )
    prefix_length = int(model.max_prefix_length)
    target_length = int(tokenizer.context_length)
    component_names = [f"scale_{length}" for length in tokenizer.scale_lengths]
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
            "num_unique_windows",
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
        "tokenizer_checkpoint": str(tokenizer_checkpoint_path),
        "tokenizer_checkpoint_sha256": sha256(tokenizer_checkpoint_path),
        "score": (
            "mutant minus reference summed hierarchy and nucleotide log probability"
        ),
        "score_components": ["joint"] + component_names,
        "prefix_length": prefix_length,
        "target_length": target_length,
        "window_length": prefix_length + target_length,
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
