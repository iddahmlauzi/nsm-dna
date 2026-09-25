import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_nsm import (
    build_codebook_distance_matrices,
    build_codebook_neighbor_tables,
    prepare_block_predictions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_generated_scale(
    generated_indices: Tensor,
    true_indices: Tensor,
    mode: str,
    neighbor_table: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Replace wrong generated codes while preserving their correctness mask."""
    if mode == "oracle_true":
        return true_indices

    wrong = generated_indices != true_indices
    if mode == "same_errors_nearest_1":
        replacements = neighbor_table[true_indices, 0]
    elif mode == "same_errors_nearest_5":
        neighbor_ranks = torch.randint(
            neighbor_table.shape[1],
            true_indices.shape,
            generator=generator,
        ).to(true_indices.device)
        replacements = neighbor_table[true_indices].gather(
            dim=-1,
            index=neighbor_ranks.unsqueeze(-1),
        ).squeeze(-1)
    elif mode == "same_errors_random":
        codebook_size = neighbor_table.shape[0]
        random_offsets = torch.randint(
            codebook_size - 1,
            true_indices.shape,
            generator=generator,
        ).to(true_indices.device)
        replacements = random_offsets + (random_offsets >= true_indices)
    else:
        raise ValueError(f"Unknown intervention mode: {mode}")

    return torch.where(wrong, replacements, true_indices)


@torch.no_grad()
def rollout_with_intervention(
    model: NSM,
    tokenizer: VQVAE,
    prefix_by_scale: list[Tensor],
    prefix_token_ids: Tensor,
    true_indices_by_scale: list[Tensor],
    intervention_scale_index: int | None,
    intervention_mode: str | None,
    neighbor_table: Tensor | None,
    generator: torch.Generator,
) -> list[Tensor]:
    predicted_indices_by_scale = []
    for scale_index in range(len(tokenizer.scale_lengths)):
        with torch.autocast(
            device_type=prefix_token_ids.device.type,
            dtype=torch.bfloat16,
            enabled=prefix_token_ids.device.type == "cuda",
        ):
            logits = model.predict_scale(
                prefix_by_scale,
                predicted_indices_by_scale,
                prefix_token_ids=prefix_token_ids,
            )
        generated_indices = logits.argmax(dim=-1)
        if scale_index == intervention_scale_index:
            assert intervention_mode is not None
            assert neighbor_table is not None
            generated_indices = replace_generated_scale(
                generated_indices,
                true_indices_by_scale[scale_index],
                intervention_mode,
                neighbor_table,
                generator,
            )
        predicted_indices_by_scale.append(generated_indices)

    return predicted_indices_by_scale


@dataclass
class RolloutMetrics:
    nucleotide_nll_sum: float
    nucleotide_correct: int
    nucleotide_count: int
    code_correct_by_scale: list[int]
    code_count_by_scale: list[int]


def empty_rollout_metrics(num_scales: int) -> RolloutMetrics:
    return RolloutMetrics(
        nucleotide_nll_sum=0.0,
        nucleotide_correct=0,
        nucleotide_count=0,
        code_correct_by_scale=[0] * num_scales,
        code_count_by_scale=[0] * num_scales,
    )


def update_rollout_metrics(
    metrics: RolloutMetrics,
    tokenizer: VQVAE,
    predicted_indices_by_scale: list[Tensor],
    true_indices_by_scale: list[Tensor],
    target_ids: Tensor,
) -> None:
    with torch.inference_mode(), torch.autocast(
        device_type=target_ids.device.type,
        dtype=torch.bfloat16,
        enabled=target_ids.device.type == "cuda",
    ):
        nucleotide_logits = tokenizer.decode_scale(
            predicted_indices_by_scale[-1],
            len(predicted_indices_by_scale) - 1,
        )
    metrics.nucleotide_nll_sum += F.cross_entropy(
        nucleotide_logits.float().flatten(0, 1),
        target_ids.flatten(),
        reduction="sum",
    ).item()
    metrics.nucleotide_correct += (
        nucleotide_logits.argmax(dim=-1) == target_ids
    ).sum().item()
    metrics.nucleotide_count += target_ids.numel()
    for scale_index, (predicted, true) in enumerate(
        zip(predicted_indices_by_scale, true_indices_by_scale, strict=True)
    ):
        metrics.code_correct_by_scale[scale_index] += (
            predicted == true
        ).sum().item()
        metrics.code_count_by_scale[scale_index] += true.numel()


def summarize_rollout_metrics(
    metrics: RolloutMetrics,
    scale_lengths: list[int],
) -> dict[str, object]:
    return {
        "nucleotide_nll_nats": (
            metrics.nucleotide_nll_sum / metrics.nucleotide_count
        ),
        "nucleotide_accuracy": (
            metrics.nucleotide_correct / metrics.nucleotide_count
        ),
        "code_accuracy_by_scale": {
            str(scale_length): correct / count
            for scale_length, correct, count in zip(
                scale_lengths,
                metrics.code_correct_by_scale,
                metrics.code_count_by_scale,
                strict=True,
            )
        },
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="rollout_geometry",
)
def main(config: DictConfig) -> None:
    device = torch.device(config.device)
    checkpoint_path = Path(config.checkpoint)
    tokenizer_checkpoint_path = Path(config.tokenizer_checkpoint)
    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

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
    distance_matrices = build_codebook_distance_matrices(
        [codebook.codebook for codebook in tokenizer.quantizer.codebooks]
    )
    neighbor_tables = build_codebook_neighbor_tables(
        distance_matrices,
        int(config.neighbor_count),
    )
    intervention_scale_lengths = [
        int(scale_length) for scale_length in config.intervention_scale_lengths
    ]
    intervention_scale_indices = {
        scale_length: tokenizer.scale_lengths.index(scale_length)
        for scale_length in intervention_scale_lengths
    }
    intervention_modes = [
        "oracle_true",
        "same_errors_nearest_1",
        "same_errors_nearest_5",
        "same_errors_random",
    ]

    dataset_directory = Path(config.data.subset_directory)
    split = str(config.data.split)
    dataset = load_gtdb_dataset(
        subset_directory=dataset_directory,
        split=split,
        context_length=2 * tokenizer.context_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        collate_fn=collate_dna_sequences,
        num_workers=int(config.data.num_workers),
    )
    baseline_metrics = empty_rollout_metrics(len(tokenizer.scale_lengths))
    intervention_metrics = {
        str(scale_length): {
            mode: empty_rollout_metrics(len(tokenizer.scale_lengths))
            for mode in intervention_modes
        }
        for scale_length in intervention_scale_lengths
    }
    generated_error_counts = {
        str(scale_length): 0 for scale_length in intervention_scale_lengths
    }
    intervention_code_counts = {
        str(scale_length): 0 for scale_length in intervention_scale_lengths
    }
    generated_error_distances = {
        str(scale_length): [] for scale_length in intervention_scale_lengths
    }
    generated_error_neighbor_ranks = {
        str(scale_length): [] for scale_length in intervention_scale_lengths
    }

    for batch_index, batch in enumerate(tqdm(data_loader, unit="batch")):
        if batch_index == int(config.data.max_batches):
            break
        prediction = prepare_block_predictions(
            tokenizer,
            batch["input_ids"].to(device),
        )[0]
        baseline_indices = rollout_with_intervention(
            model,
            tokenizer,
            prediction.prefix_by_scale,
            prediction.prefix_ids,
            prediction.targets_by_scale,
            intervention_scale_index=None,
            intervention_mode=None,
            neighbor_table=None,
            generator=torch.Generator().manual_seed(int(config.seed)),
        )
        update_rollout_metrics(
            baseline_metrics,
            tokenizer,
            baseline_indices,
            prediction.targets_by_scale,
            prediction.target_ids,
        )

        for scale_length, scale_index in intervention_scale_indices.items():
            scale_name = str(scale_length)
            generated_error_counts[scale_name] += (
                baseline_indices[scale_index]
                != prediction.targets_by_scale[scale_index]
            ).sum().item()
            generated_indices = baseline_indices[scale_index]
            true_indices = prediction.targets_by_scale[scale_index]
            wrong = generated_indices != true_indices
            target_distance_rows = distance_matrices[scale_index][true_indices]
            generated_distances = target_distance_rows.gather(
                dim=-1,
                index=generated_indices.unsqueeze(-1),
            ).squeeze(-1)
            error_distances = generated_distances[wrong]
            neighbor_ranks = (
                (target_distance_rows < generated_distances.unsqueeze(-1)).sum(dim=-1)
                + 1
            )[wrong]
            generated_error_distances[scale_name].append(error_distances.cpu())
            generated_error_neighbor_ranks[scale_name].append(
                neighbor_ranks.cpu()
            )
            intervention_code_counts[scale_name] += prediction.targets_by_scale[
                scale_index
            ].numel()
            for mode_index, mode in enumerate(intervention_modes):
                generator = torch.Generator().manual_seed(
                    int(config.seed)
                    + 10_000 * batch_index
                    + 100 * scale_index
                    + mode_index
                )
                predicted_indices = rollout_with_intervention(
                    model,
                    tokenizer,
                    prediction.prefix_by_scale,
                    prediction.prefix_ids,
                    prediction.targets_by_scale,
                    intervention_scale_index=scale_index,
                    intervention_mode=mode,
                    neighbor_table=neighbor_tables[scale_index],
                    generator=generator,
                )
                update_rollout_metrics(
                    intervention_metrics[scale_name][mode],
                    tokenizer,
                    predicted_indices,
                    prediction.targets_by_scale,
                    prediction.target_ids,
                )

    validation_paths = sorted(
        (dataset_directory / split).glob("chunks-*.parquet")
    )
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "tokenizer_checkpoint": str(tokenizer_checkpoint_path),
        "tokenizer_checkpoint_sha256": sha256(tokenizer_checkpoint_path),
        "validation_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in validation_paths
        ],
        "seed": int(config.seed),
        "num_examples": baseline_metrics.nucleotide_count
        // tokenizer.context_length,
        "intervention_scale_lengths": intervention_scale_lengths,
        "neighbor_count": int(config.neighbor_count),
        "intervention_definition": (
            "Preserve the generated scale's exact/wrong position mask; replace "
            "only its wrong codes, then continue greedy rollout."
        ),
        "baseline": summarize_rollout_metrics(
            baseline_metrics,
            tokenizer.scale_lengths,
        ),
        "interventions": {
            scale_name: {
                "generated_error_rate_at_intervention": (
                    generated_error_counts[scale_name]
                    / intervention_code_counts[scale_name]
                ),
                "generated_error_geometry": {
                    "normalized_distance_mean": float(
                        torch.cat(generated_error_distances[scale_name]).mean()
                    ),
                    "within_nearest_alternatives": {
                        "1": float(
                            (
                                torch.cat(
                                    generated_error_neighbor_ranks[scale_name]
                                )
                                <= 2
                            )
                            .float()
                            .mean()
                        ),
                        "5": float(
                            (
                                torch.cat(
                                    generated_error_neighbor_ranks[scale_name]
                                )
                                <= 6
                            )
                            .float()
                            .mean()
                        ),
                        "10": float(
                            (
                                torch.cat(
                                    generated_error_neighbor_ranks[scale_name]
                                )
                                <= 11
                            )
                            .float()
                            .mean()
                        ),
                    },
                },
                "conditions": {
                    mode: summarize_rollout_metrics(
                        metrics,
                        tokenizer.scale_lengths,
                    )
                    for mode, metrics in modes.items()
                },
            }
            for scale_name, modes in intervention_metrics.items()
        },
    }
    output_path = output_directory / "results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print(
        "baseline rollout: "
        f"{results['baseline']['nucleotide_accuracy']:.2%}"
    )
    for scale_name, scale_results in results["interventions"].items():
        print(
            f"scale {scale_name} generated error rate: "
            f"{scale_results['generated_error_rate_at_intervention']:.2%}"
        )
        for mode, metrics in scale_results["conditions"].items():
            print(f"  {mode}: {metrics['nucleotide_accuracy']:.2%}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
