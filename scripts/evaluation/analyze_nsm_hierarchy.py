import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from scripts.training.train_nsm import (
    prepare_block_predictions,
    rollout_hierarchy,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_target_controls(
    input_ids: Tensor,
    prefix_length: int,
    generator: torch.Generator,
) -> dict[str, Tensor]:
    """Keep the real prefix while replacing only the target sequence."""
    prefix_ids = input_ids[:, :prefix_length]
    target_ids = input_ids[:, prefix_length:]
    shuffled_targets = torch.stack(
        [
            target[torch.randperm(target.shape[0], generator=generator)]
            for target in target_ids
        ]
    )
    random_targets = torch.randint(
        low=0,
        high=4,
        size=target_ids.shape,
        generator=generator,
    )
    return {
        "real": input_ids,
        "composition_shuffled": torch.cat([prefix_ids, shuffled_targets], dim=1),
        "uniform_random": torch.cat([prefix_ids, random_targets], dim=1),
    }


@dataclass
class ConditionMetrics:
    nll_sums: list[float]
    entropy_sums: list[float]
    correct_counts: list[int]
    code_counts: list[int]
    num_examples: int = 0


def empty_condition_metrics(num_scales: int) -> ConditionMetrics:
    return ConditionMetrics(
        nll_sums=[0.0] * num_scales,
        entropy_sums=[0.0] * num_scales,
        correct_counts=[0] * num_scales,
        code_counts=[0] * num_scales,
    )


def update_condition_metrics(
    metrics: ConditionMetrics,
    logits_by_scale: list[Tensor],
    targets_by_scale: list[Tensor],
) -> None:
    for scale_index, (logits, targets) in enumerate(
        zip(logits_by_scale, targets_by_scale, strict=True)
    ):
        log_probabilities = F.log_softmax(logits.float(), dim=-1)
        probabilities = log_probabilities.exp()
        metrics.nll_sums[scale_index] += F.nll_loss(
            log_probabilities.flatten(0, 1),
            targets.flatten(),
            reduction="sum",
        ).item()
        metrics.entropy_sums[scale_index] += (
            -(probabilities * log_probabilities).sum(dim=-1).sum().item()
        )
        metrics.correct_counts[scale_index] += (
            (logits.argmax(dim=-1) == targets).sum().item()
        )
        metrics.code_counts[scale_index] += targets.numel()

    metrics.num_examples += targets_by_scale[0].shape[0]


def summarize_condition(
    metrics: ConditionMetrics,
    scale_lengths: list[int],
    codebook_sizes: list[int],
    target_length: int,
) -> dict[str, object]:
    scales = {}
    for scale_length, codebook_size, nll_sum, entropy_sum, correct, count in zip(
        scale_lengths,
        codebook_sizes,
        metrics.nll_sums,
        metrics.entropy_sums,
        metrics.correct_counts,
        metrics.code_counts,
        strict=True,
    ):
        scales[str(scale_length)] = {
            "nll_nats_per_code": nll_sum / count,
            "accuracy": correct / count,
            "predictive_entropy_nats": entropy_sum / count,
            "normalized_predictive_entropy": (
                entropy_sum / count / math.log(codebook_size)
            ),
        }

    total_nll = sum(metrics.nll_sums)
    return {
        "num_examples": metrics.num_examples,
        "nll_nats_per_target": total_nll / metrics.num_examples,
        "bits_per_target_base": total_nll
        / (metrics.num_examples * target_length * math.log(2)),
        "scales": scales,
    }


def empty_reconstruction_metrics() -> dict[str, dict[str, float | int]]:
    return {
        name: {"nll_sum": 0.0, "correct": 0, "count": 0}
        for name in (
            "tokenizer",
            "first_scale_only",
            "teacher_forced_argmax",
            "rollout_argmax",
        )
    }


def update_reconstruction_metrics(
    metrics: dict[str, dict[str, float | int]],
    name: str,
    logits: Tensor,
    targets: Tensor,
) -> None:
    values = metrics[name]
    values["nll_sum"] = float(values["nll_sum"]) + F.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        reduction="sum",
    ).item()
    values["correct"] = int(values["correct"]) + (
        (logits.argmax(dim=-1) == targets).sum().item()
    )
    values["count"] = int(values["count"]) + targets.numel()


def summarize_reconstructions(
    metrics: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "nucleotide_nll_nats": float(values["nll_sum"]) / int(values["count"]),
            "nucleotide_accuracy": int(values["correct"]) / int(values["count"]),
        }
        for name, values in metrics.items()
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="analyze_nsm_hierarchy",
)
def main(config: DictConfig) -> None:
    """Measure hierarchy likelihoods and reconstruction on fixed validation data."""
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
    model_config = OmegaConf.create(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)["config"]
    )
    sequence_length = int(model_config.data.sequence_length)
    prefix_length = sequence_length - tokenizer.context_length
    target_length = tokenizer.context_length

    dataset_directory = Path(config.data.subset_directory)
    split = str(config.data.split)
    dataset = load_gtdb_dataset(
        subset_directory=dataset_directory,
        split=split,
        context_length=sequence_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=int(config.data.batch_size),
        collate_fn=collate_dna_sequences,
        num_workers=int(config.data.num_workers),
    )

    predicted_scale_lengths = tokenizer.scale_lengths
    predicted_codebook_sizes = tokenizer.codebook_sizes
    condition_metrics = {
        name: empty_condition_metrics(len(predicted_scale_lengths))
        for name in ("real", "composition_shuffled", "uniform_random")
    }
    reconstruction_metrics = empty_reconstruction_metrics()
    real_code_counts = [
        torch.zeros(codebook_size, dtype=torch.long)
        for codebook_size in predicted_codebook_sizes
    ]
    rollout_correct_counts = [0] * len(predicted_scale_lengths)
    rollout_code_counts = [0] * len(predicted_scale_lengths)
    generator = torch.Generator().manual_seed(int(config.seed))

    for batch_index, batch in enumerate(tqdm(data_loader, unit="batch")):
        if batch_index == int(config.data.max_batches):
            break
        controls = make_target_controls(
            batch["input_ids"],
            prefix_length,
            generator,
        )

        for condition_name, condition_ids in controls.items():
            prediction = prepare_block_predictions(
                tokenizer,
                condition_ids.to(device),
            )[0]
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits_by_scale = model(
                    prediction.targets_by_scale,
                    prefix=prediction.prefix,
                    prefix_code=prediction.prefix_code,
                )
            update_condition_metrics(
                condition_metrics[condition_name],
                logits_by_scale,
                prediction.targets_by_scale,
            )

            if condition_name != "real":
                continue

            for scale_index, targets in enumerate(prediction.targets_by_scale):
                real_code_counts[scale_index] += torch.bincount(
                    targets.cpu().flatten(),
                    minlength=predicted_codebook_sizes[scale_index],
                )

            true_indices = prediction.targets_by_scale
            teacher_forced_indices = [
                logits.argmax(dim=-1) for logits in logits_by_scale
            ]
            if batch_index < int(config.data.rollout_max_batches):
                rollout_indices = rollout_hierarchy(
                    model,
                    tokenizer,
                    batch_size=prediction.target_ids.shape[0],
                    prefix=prediction.prefix,
                    prefix_code=prediction.prefix_code,
                )
                for scale_index, (rollout_targets, true_targets) in enumerate(
                    zip(
                        rollout_indices,
                        prediction.targets_by_scale,
                        strict=True,
                    )
                ):
                    rollout_correct_counts[scale_index] += (
                        (rollout_targets == true_targets).sum().item()
                    )
                    rollout_code_counts[scale_index] += true_targets.numel()
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                final_scale_index = len(true_indices) - 1
                reconstruction_logits = {
                    "tokenizer": tokenizer.decode_scale(
                        true_indices[-1], final_scale_index
                    ),
                    "first_scale_only": tokenizer.decode_scale(true_indices[0], 0),
                    "teacher_forced_argmax": tokenizer.decode_scale(
                        teacher_forced_indices[-1], final_scale_index
                    ),
                }
                if batch_index < int(config.data.rollout_max_batches):
                    reconstruction_logits["rollout_argmax"] = tokenizer.decode_scale(
                        rollout_indices[-1], final_scale_index
                    )
            for name, logits in reconstruction_logits.items():
                update_reconstruction_metrics(
                    reconstruction_metrics,
                    name,
                    logits,
                    prediction.target_ids,
                )

    conditions = {
        name: summarize_condition(
            metrics,
            predicted_scale_lengths,
            predicted_codebook_sizes,
            target_length,
        )
        for name, metrics in condition_metrics.items()
    }
    baselines = {}
    for scale_length, codebook_size, counts in zip(
        predicted_scale_lengths,
        predicted_codebook_sizes,
        real_code_counts,
        strict=True,
    ):
        probabilities = counts.double() / counts.sum()
        nonzero_probabilities = probabilities[probabilities > 0]
        baselines[str(scale_length)] = {
            "uniform_accuracy": 1 / codebook_size,
            "uniform_nll_nats_per_code": math.log(codebook_size),
            "empirical_majority_accuracy": probabilities.max().item(),
            "empirical_marginal_nll_nats_per_code": (
                -(nonzero_probabilities * nonzero_probabilities.log()).sum().item()
            ),
        }

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
        "max_batches": int(config.data.max_batches),
        "batch_size": int(config.data.batch_size),
        "num_examples": conditions["real"]["num_examples"],
        "prefix_length": prefix_length,
        "target_length": target_length,
        "first_scale_is_supplied": False,
        "baselines": baselines,
        "conditions": conditions,
        "rollout_code_accuracy_by_scale": {
            str(scale_length): correct / count
            for scale_length, correct, count in zip(
                predicted_scale_lengths,
                rollout_correct_counts,
                rollout_code_counts,
                strict=True,
            )
        },
        "reconstruction": summarize_reconstructions(reconstruction_metrics),
    }
    output_path = output_directory / "results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"checkpoint step: {checkpoint_step}")
    for scale_length in predicted_scale_lengths:
        scale_name = str(scale_length)
        real = conditions["real"]["scales"][scale_name]
        shuffled = conditions["composition_shuffled"]["scales"][scale_name]
        baseline = baselines[scale_name]
        print(
            f"scale {scale_length}: accuracy {real['accuracy']:.2%}, "
            f"NLL {real['nll_nats_per_code']:.4f}, "
            f"marginal NLL {baseline['empirical_marginal_nll_nats_per_code']:.4f}, "
            f"shuffled NLL {shuffled['nll_nats_per_code']:.4f}"
        )
    for name, metrics in results["reconstruction"].items():
        print(f"{name} reconstruction: {metrics['nucleotide_accuracy']:.2%}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
