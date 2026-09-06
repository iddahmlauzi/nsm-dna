import json
from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.losses import NSMDNALosses, nsm_dna_losses
from nsm_dna.models.next_scale import (
    NSMDNA,
    NSMDNAOutput,
    TokenizerStabilitySnapshot,
)
from nsm_dna.training import (
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)


def target_ids_from_sequence(
    sequence_ids: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Return the fixed-length target at the end of each training sequence."""
    return sequence_ids[:, -target_length:]


def module_gradient_norm(module: nn.Module) -> float:
    """Calculate the L2 norm of the gradients in one model component."""
    squared_norms = [
        parameter.grad.detach().float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not squared_norms:
        return 0.0
    return torch.stack(squared_norms).sum().sqrt().item()


def tokenizer_stability_snapshot_to_cpu(
    snapshot: TokenizerStabilitySnapshot,
) -> TokenizerStabilitySnapshot:
    """Move one fixed-anchor snapshot off the accelerator between evaluations."""
    return TokenizerStabilitySnapshot(
        encoder_latent=snapshot.encoder_latent.float().cpu(),
        indices_by_scale=[indices.cpu() for indices in snapshot.indices_by_scale],
        codebooks_by_scale=[
            codebook.float().cpu() for codebook in snapshot.codebooks_by_scale
        ],
        clean_consistency_states_by_scale=[
            state.float().cpu()
            for state in snapshot.clean_consistency_states_by_scale
        ],
        rollout_consistency_states_by_scale=[
            state.float().cpu()
            for state in snapshot.rollout_consistency_states_by_scale
        ],
    )


def tokenizer_stability_metrics(
    previous: TokenizerStabilitySnapshot,
    current: TokenizerStabilitySnapshot,
    scale_lengths: list[int],
) -> dict[str, float]:
    """Measure tokenizer motion on the same sequences across evaluations."""
    encoder_latent_drift = torch.mean(
        (current.encoder_latent - previous.encoder_latent).square()
    ).sqrt()
    metrics = {"encoder_latent_drift": encoder_latent_drift.item()}

    code_retentions = []
    active_codebook_drifts = []
    scale_snapshots = zip(
        scale_lengths,
        previous.indices_by_scale,
        current.indices_by_scale,
        previous.codebooks_by_scale,
        current.codebooks_by_scale,
        strict=True,
    )
    for (
        scale_length,
        previous_indices,
        current_indices,
        previous_codebook,
        current_codebook,
    ) in scale_snapshots:
        code_retention = (current_indices == previous_indices).float().mean()
        active_indices = torch.unique(
            torch.cat([previous_indices.flatten(), current_indices.flatten()])
        )
        active_codebook_drift = torch.mean(
            (
                current_codebook[active_indices]
                - previous_codebook[active_indices]
            ).square()
        ).sqrt()

        metrics[f"code_retention_scale_{scale_length}"] = code_retention.item()
        metrics[f"active_codebook_drift_scale_{scale_length}"] = (
            active_codebook_drift.item()
        )
        code_retentions.append(code_retention)
        active_codebook_drifts.append(active_codebook_drift)

    target_state_drifts = []
    rollout_state_errors = []
    relative_target_drifts = []
    for scale_length, previous_state, current_state, rollout_state in zip(
        scale_lengths[1:],
        previous.clean_consistency_states_by_scale,
        current.clean_consistency_states_by_scale,
        current.rollout_consistency_states_by_scale,
        strict=True,
    ):
        target_state_drift = torch.mean(
            (current_state - previous_state).square()
        ).sqrt()
        rollout_state_error = torch.mean(
            (rollout_state - current_state).square()
        ).sqrt()
        relative_target_drift = target_state_drift / rollout_state_error.clamp_min(
            1e-12
        )

        metrics[f"target_state_drift_after_scale_{scale_length}"] = (
            target_state_drift.item()
        )
        metrics[f"rollout_state_error_after_scale_{scale_length}"] = (
            rollout_state_error.item()
        )
        metrics[
            f"target_drift_relative_to_rollout_error_after_scale_{scale_length}"
        ] = relative_target_drift.item()
        target_state_drifts.append(target_state_drift)
        rollout_state_errors.append(rollout_state_error)
        relative_target_drifts.append(relative_target_drift)

    metrics.update(
        {
            "code_retention": torch.stack(code_retentions).mean().item(),
            "active_codebook_drift": torch.stack(
                active_codebook_drifts
            ).mean().item(),
            "target_state_drift": torch.stack(target_state_drifts).mean().item(),
            "rollout_state_error": torch.stack(rollout_state_errors).mean().item(),
            "target_drift_relative_to_rollout_error": torch.stack(
                relative_target_drifts
            ).mean().item(),
        }
    )
    return metrics


def configure_training_phase(
    model: NSMDNA,
    *,
    freeze_tokenizer: bool,
) -> None:
    """Keep a frozen tokenizer deterministic while the transformer trains."""
    if freeze_tokenizer:
        model.tokenizer.requires_grad_(False)
        model.tokenizer.eval()


@torch.no_grad()
def evaluate(
    model: NSMDNA,
    data_loader: DataLoader,
    use_mixed_precision: bool,
    max_batches: int | None = None,
    *,
    partial_reconstruction_loss_weight: float = 0.0,
    teacher_forced_prediction_reconstruction_loss_weight: float = 0.0,
    next_scale_prediction_loss_weight: float = 1.0,
    rollout_state_consistency_loss_weight: float = 0.0,
    entropy_loss_weight: float = 1.0,
    entropy_temperature: float = 1.0,
) -> dict[str, float]:
    """Evaluate joint losses and tokenizer/next-scale diagnostics."""
    was_training = model.training
    tokenizer_was_training = model.tokenizer.training
    model.eval()
    device = next(model.parameters()).device
    num_scales = len(model.tokenizer.scale_lengths)
    num_predicted_scales = num_scales - 1

    # The losses returned by the model are batch means, so they are accumulated
    # by example. Accuracy and codebook statistics are accumulated as raw counts
    # so that short final batches do not receive disproportionate weight.
    loss_sums = torch.zeros(8, dtype=torch.float64, device=device)
    teacher_forced_prediction_loss_sums = torch.zeros(
        num_predicted_scales, dtype=torch.float64, device=device
    )
    rollout_state_consistency_loss_sums = torch.zeros(
        num_predicted_scales, dtype=torch.float64, device=device
    )
    encoder_commitment_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    quantization_loss_sums = torch.zeros(num_scales, dtype=torch.float64, device=device)
    teacher_forced_prediction_correct = torch.zeros(
        num_predicted_scales, dtype=torch.long, device=device
    )
    prediction_counts = torch.zeros(
        num_predicted_scales, dtype=torch.long, device=device
    )
    rollout_prediction_correct = torch.zeros(
        num_predicted_scales, dtype=torch.long, device=device
    )
    teacher_forced_rollout_prediction_agreement = torch.zeros(
        num_predicted_scales, dtype=torch.long, device=device
    )
    cumulative_nucleotide_correct = torch.zeros(
        num_scales, dtype=torch.long, device=device
    )
    rollout_cumulative_nucleotide_correct = torch.zeros(
        num_scales, dtype=torch.long, device=device
    )
    confidence_sums = torch.zeros(num_scales, dtype=torch.float64, device=device)
    confidence_counts = torch.zeros(num_scales, dtype=torch.long, device=device)
    code_counts_by_scale = [
        torch.zeros(codebook_size, dtype=torch.long, device=device)
        for codebook_size in model.tokenizer.codebook_sizes
    ]
    nucleotide_reconstruction_correct = torch.zeros((), dtype=torch.long, device=device)
    teacher_forced_prediction_reconstruction_correct = torch.zeros(
        (), dtype=torch.long, device=device
    )
    rollout_nucleotide_loss_sum = torch.zeros(
        (), dtype=torch.float64, device=device
    )
    rollout_nucleotide_correct = torch.zeros(
        (), dtype=torch.long, device=device
    )
    target_count = torch.zeros((), dtype=torch.long, device=device)
    example_count = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index == max_batches:
            break

        sequence_ids = batch["input_ids"].to(device)
        target_ids = target_ids_from_sequence(
            sequence_ids,
            model.tokenizer.context_length,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            # Corruption is a training intervention, not part of the validation
            # objective. Decoding every cumulative latent is enabled only here
            # because it requires one decoder pass per scale and is diagnostic.
            output = model(
                sequence_ids,
                corruption_probability=0.0,
                return_cumulative_reconstructions=True,
                return_teacher_forced_prediction_reconstruction=(
                    teacher_forced_prediction_reconstruction_loss_weight != 0
                ),
                return_rollout=True,
                return_rollout_reconstructions=True,
            )
            losses = nsm_dna_losses(
                output,
                target_ids,
                partial_reconstruction_loss_weight=(
                    partial_reconstruction_loss_weight
                ),
                teacher_forced_prediction_reconstruction_loss_weight=(
                    teacher_forced_prediction_reconstruction_loss_weight
                ),
                next_scale_prediction_loss_weight=(
                    next_scale_prediction_loss_weight
                ),
                rollout_state_consistency_loss_weight=(
                    rollout_state_consistency_loss_weight
                ),
                entropy_loss_weight=entropy_loss_weight,
                entropy_temperature=entropy_temperature,
            )
            rollout = output.rollout
            assert rollout is not None
            rollout_logits = rollout.final_reconstruction_logits
            assert rollout_logits is not None
            rollout_nucleotide_loss = F.cross_entropy(
                rollout_logits.flatten(0, 1).float(),
                target_ids.flatten(),
            )

        batch_size = sequence_ids.shape[0]
        example_count += batch_size
        loss_sums += (
            torch.stack(
                [
                    losses.total,
                    losses.nucleotide_reconstruction,
                    losses.partial_reconstruction,
                    losses.teacher_forced_prediction_reconstruction,
                    losses.vq,
                    losses.teacher_forced_prediction,
                    losses.rollout_state_consistency,
                    losses.entropy,
                ]
            ).double()
            * batch_size
        )
        teacher_forced_prediction_loss_sums += (
            torch.stack(losses.teacher_forced_prediction_by_scale).double()
            * batch_size
        )
        rollout_state_consistency_loss_sums += (
            torch.stack(losses.rollout_state_consistency_by_scale).double()
            * batch_size
        )
        encoder_commitment_loss_sum += (
            output.quantizer.encoder_commitment_loss.double() * batch_size
        )
        quantization_loss_sums += (
            torch.stack(output.quantizer.quantization_losses_by_scale).double()
            * batch_size
        )
        rollout_nucleotide_loss_sum += (
            rollout_nucleotide_loss.double() * batch_size
        )

        nucleotide_reconstruction_correct += (
            output.reconstruction_logits.argmax(dim=-1) == target_ids
        ).sum()
        if output.teacher_forced_prediction_reconstruction_logits is not None:
            teacher_forced_prediction_reconstruction_correct += (
                output.teacher_forced_prediction_reconstruction_logits.argmax(dim=-1)
                == target_ids
            ).sum()
        rollout_nucleotide_correct += (
            rollout_logits.argmax(dim=-1) == target_ids
        ).sum()
        target_count += target_ids.numel()

        cumulative_logits = output.cumulative_reconstruction_logits_by_scale
        assert cumulative_logits is not None
        rollout_cumulative_logits = (
            rollout.cumulative_reconstruction_logits_by_scale
        )
        assert rollout_cumulative_logits is not None
        # Tokenizer diagnostics cover all scales, including the supplied first scale.
        for scale_index, (
            scale_targets,
            probabilities,
            cumulative_scale_logits,
        ) in enumerate(
            zip(
                output.quantizer.indices_by_scale,
                output.quantizer.assignment_probabilities_by_scale,
                cumulative_logits,
                strict=True,
            )
        ):
            cumulative_nucleotide_correct[scale_index] += (
                cumulative_scale_logits.argmax(dim=-1) == target_ids
            ).sum()
            rollout_cumulative_nucleotide_correct[scale_index] += (
                rollout_cumulative_logits[scale_index].argmax(dim=-1) == target_ids
            ).sum()
            confidence_sums[scale_index] += (
                probabilities.max(dim=-1).values.double().sum()
            )
            confidence_counts[scale_index] += scale_targets.numel()
            code_counts_by_scale[scale_index] += torch.bincount(
                scale_targets.flatten(),
                minlength=model.tokenizer.codebook_sizes[scale_index],
            )

        # Prediction diagnostics begin after the supplied first scale.
        for prediction_index, (
            teacher_forced_logits,
            clean_targets,
            rollout_prediction_logits,
            rollout_indices,
        ) in enumerate(
            zip(
                output.next_scale_logits_by_scale,
                output.quantizer.indices_by_scale[1:],
                rollout.prediction_logits_by_scale,
                rollout.indices_by_scale[1:],
                strict=True,
            )
        ):
            teacher_forced_predictions = teacher_forced_logits.argmax(dim=-1)
            rollout_predictions = rollout_prediction_logits.argmax(dim=-1)
            teacher_forced_prediction_correct[prediction_index] += (
                teacher_forced_predictions == clean_targets
            ).sum()
            rollout_prediction_correct[prediction_index] += (
                rollout_predictions == clean_targets
            ).sum()
            teacher_forced_rollout_prediction_agreement[prediction_index] += (
                teacher_forced_predictions == rollout_indices
            ).sum()
            prediction_counts[prediction_index] += clean_targets.numel()

    model.train(was_training)
    model.tokenizer.train(tokenizer_was_training)
    if example_count == 0:
        raise ValueError("Validation data loader produced no batches.")

    mean_losses = loss_sums / example_count
    metrics = {
        "total_loss": mean_losses[0].item(),
        "nucleotide_reconstruction_loss": mean_losses[1].item(),
        "partial_reconstruction_loss": mean_losses[2].item(),
        "teacher_forced_prediction_reconstruction_loss": mean_losses[3].item(),
        "vq_loss": mean_losses[4].item(),
        "teacher_forced_prediction_loss": mean_losses[5].item(),
        "rollout_state_consistency_loss": mean_losses[6].item(),
        "entropy_loss": mean_losses[7].item(),
        "rollout_nucleotide_loss": (
            rollout_nucleotide_loss_sum / example_count
        ).item(),
        "nucleotide_reconstruction_accuracy": (
            nucleotide_reconstruction_correct / target_count
        ).item(),
        "teacher_forced_prediction_reconstruction_accuracy": (
            teacher_forced_prediction_reconstruction_correct / target_count
        ).item(),
        "teacher_forced_prediction_accuracy": (
            teacher_forced_prediction_correct.sum() / prediction_counts.sum()
        ).item(),
        "rollout_prediction_accuracy": (
            rollout_prediction_correct.sum() / prediction_counts.sum()
        ).item(),
        "rollout_nucleotide_accuracy": (
            rollout_nucleotide_correct / target_count
        ).item(),
        "encoder_commitment_loss": (
            encoder_commitment_loss_sum / example_count
        ).item(),
    }

    for scale_index, scale_length in enumerate(model.tokenizer.scale_lengths):
        scale_quantization_loss = quantization_loss_sums[scale_index] / example_count
        code_counts = code_counts_by_scale[scale_index]
        used_code_counts = code_counts[code_counts > 0].float()
        code_probabilities = used_code_counts / used_code_counts.sum()
        # Perplexity is the effective number of codes used under the observed
        # assignment distribution; usage alone only says whether a code appeared.
        perplexity = torch.exp(-(code_probabilities * code_probabilities.log()).sum())

        if scale_index > 0:
            prediction_index = scale_index - 1
            metrics[f"teacher_forced_prediction_loss_scale_{scale_length}"] = (
                teacher_forced_prediction_loss_sums[prediction_index] / example_count
            ).item()
            metrics[f"teacher_forced_prediction_accuracy_scale_{scale_length}"] = (
                teacher_forced_prediction_correct[prediction_index]
                / prediction_counts[prediction_index]
            ).item()
            metrics[f"rollout_prediction_accuracy_scale_{scale_length}"] = (
                rollout_prediction_correct[prediction_index]
                / prediction_counts[prediction_index]
            ).item()
            metrics[f"teacher_forced_rollout_agreement_scale_{scale_length}"] = (
                teacher_forced_rollout_prediction_agreement[prediction_index]
                / prediction_counts[prediction_index]
            ).item()
            metrics[
                f"rollout_state_consistency_loss_after_scale_{scale_length}"
            ] = (
                rollout_state_consistency_loss_sums[prediction_index]
                / example_count
            ).item()
        metrics[f"quantization_loss_scale_{scale_length}"] = (
            scale_quantization_loss.item()
        )
        metrics[f"cumulative_nucleotide_accuracy_scale_{scale_length}"] = (
            cumulative_nucleotide_correct[scale_index] / target_count
        ).item()
        metrics[f"rollout_cumulative_nucleotide_accuracy_scale_{scale_length}"] = (
            rollout_cumulative_nucleotide_correct[scale_index] / target_count
        ).item()
        metrics[f"code_usage_scale_{scale_length}"] = (
            (code_counts > 0).float().mean().item()
        )
        metrics[f"code_perplexity_scale_{scale_length}"] = perplexity.item()
        metrics[f"soft_confidence_scale_{scale_length}"] = (
            confidence_sums[scale_index] / confidence_counts[scale_index]
        ).item()
    return metrics


def _all_reduce_training_statistics(
    tensors: list[torch.Tensor],
) -> None:
    """Sum training statistics across every DDP rank."""
    if not dist.is_available() or not dist.is_initialized():
        return
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def _add_training_statistics(
    output: NSMDNAOutput,
    losses: NSMDNALosses,
    target_ids: torch.Tensor,
    loss_sums: torch.Tensor,
    teacher_forced_prediction_correct: torch.Tensor,
    rollout_prediction_correct: torch.Tensor,
    prediction_count: torch.Tensor,
    nucleotide_reconstruction_correct: torch.Tensor,
    teacher_forced_prediction_reconstruction_correct: torch.Tensor,
    target_count: torch.Tensor,
    example_count: torch.Tensor,
) -> None:
    """Accumulate detached metrics from one training microbatch."""
    # Training logs use outputs already produced for the loss. Cumulative
    # nucleotide accuracy stays validation-only to avoid extra decoder passes.
    batch_size = target_ids.shape[0]
    loss_sums += (
        torch.stack(
            [
                losses.total,
                losses.nucleotide_reconstruction,
                losses.partial_reconstruction,
                losses.teacher_forced_prediction_reconstruction,
                losses.vq,
                losses.teacher_forced_prediction,
                losses.rollout_state_consistency,
                losses.entropy,
            ]
        ).detach()
        * batch_size
    )
    nucleotide_reconstruction_correct += (
        output.reconstruction_logits.detach().argmax(dim=-1) == target_ids
    ).sum()
    assert output.teacher_forced_prediction_reconstruction_logits is not None
    teacher_forced_prediction_reconstruction_correct += (
        output.teacher_forced_prediction_reconstruction_logits.detach().argmax(dim=-1)
        == target_ids
    ).sum()
    target_count += target_ids.numel()
    example_count += batch_size

    rollout = output.rollout
    for prediction_index, (teacher_forced_logits, clean_targets) in enumerate(
        zip(
            output.next_scale_logits_by_scale,
            output.quantizer.indices_by_scale[1:],
            strict=True,
        )
    ):
        teacher_forced_prediction_correct += (
            teacher_forced_logits.detach().argmax(dim=-1) == clean_targets
        ).sum()
        if rollout is not None:
            rollout_prediction_correct += (
                rollout.prediction_logits_by_scale[prediction_index]
                .detach()
                .argmax(dim=-1)
                == clean_targets
            ).sum()
        prediction_count += clean_targets.numel()


def _validation_metrics_for_wandb(
    validation_metrics: dict[str, float],
    scale_lengths: list[int],
    best_validation_loss: float,
) -> dict[str, float]:
    """Keep the W&B dashboard focused on outcomes and codebook health."""
    wandb_metrics = {
        "validation/objective/total": validation_metrics["total_loss"],
        "validation/objective/rollout_state_consistency": validation_metrics[
            "rollout_state_consistency_loss"
        ],
        "validation/accuracy/nucleotide_reconstruction": validation_metrics[
            "nucleotide_reconstruction_accuracy"
        ],
        "validation/accuracy/teacher_forced_prediction_reconstruction": (
            validation_metrics[
                "teacher_forced_prediction_reconstruction_accuracy"
            ]
        ),
        "validation/accuracy/teacher_forced_prediction": validation_metrics[
            "teacher_forced_prediction_accuracy"
        ],
        "validation/accuracy/rollout_prediction": validation_metrics[
            "rollout_prediction_accuracy"
        ],
        "validation/accuracy/rollout_nucleotide": validation_metrics[
            "rollout_nucleotide_accuracy"
        ],
        "validation/objective/best_total": best_validation_loss,
    }
    stability_metric_names = (
        "code_retention",
        "encoder_latent_drift",
        "active_codebook_drift",
        "target_state_drift",
        "target_drift_relative_to_rollout_error",
    )
    for metric_name in stability_metric_names:
        if metric_name in validation_metrics:
            wandb_metrics[f"validation/stability/{metric_name}"] = (
                validation_metrics[metric_name]
            )

    scale_metric_names = {
        "teacher_forced_prediction_accuracy": (
            "teacher_forced_prediction_accuracy"
        ),
        "rollout_prediction_accuracy": "rollout_prediction_accuracy",
        "teacher_forced_rollout_agreement": (
            "teacher_forced_rollout_agreement"
        ),
        "cumulative_nucleotide_accuracy": "cumulative_nucleotide_accuracy",
        "rollout_cumulative_nucleotide_accuracy": (
            "rollout_cumulative_nucleotide_accuracy"
        ),
        "code_usage": "code_usage",
        "code_perplexity": "code_perplexity",
    }
    for scale_number, scale_length in enumerate(scale_lengths, start=1):
        section = f"scale_{scale_number:02d}_length_{scale_length}"
        for panel_name, metric_name in scale_metric_names.items():
            metric_key = f"{metric_name}_scale_{scale_length}"
            if metric_key in validation_metrics:
                wandb_metrics[f"{section}/{panel_name}"] = validation_metrics[
                    metric_key
                ]

    return wandb_metrics


def append_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    """Preserve complete metric history without creating one W&B graph per value."""
    record = {"step": step, **metrics}
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@hydra.main(version_base=None, config_path="../../configs", config_name="nsm")
def main(config: DictConfig) -> None:
    distributed_environment = initialize_distributed_training()
    torch.manual_seed(config.run.seed + distributed_environment.rank)
    run_directory = Path(HydraConfig.get().runtime.output_dir)
    device = distributed_environment.device
    if config.run.initialize_from is not None and config.run.resume_from is not None:
        raise ValueError("Set only one of run.initialize_from and run.resume_from.")
    sequence_length = int(config.data.sequence_length)
    total_steps = calculate_training_steps(
        config,
        distributed_environment.world_size,
        sequence_length,
    )
    warmup_steps = round(total_steps * config.optimizer.warmup_fraction)
    next_scale_prediction_loss_weight = float(
        config.training.next_scale_prediction_loss_weight
    )
    entropy_loss_weight = float(config.training.entropy_loss_weight)
    if distributed_environment.is_main_process:
        print(
            f"training for {total_steps:,} optimizer steps "
            f"with {warmup_steps:,} warmup steps"
        )
        print(
            "constant objective weights: next-scale prediction "
            f"{next_scale_prediction_loss_weight:g}, entropy "
            f"{entropy_loss_weight:g}"
        )
    wandb_run = None
    if config.wandb.enabled and distributed_environment.is_main_process:
        wandb_run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.name,
            config=OmegaConf.to_container(config, resolve=True),
            dir=run_directory,
        )

    train_dataset = load_gtdb_dataset(
        subset_directory=Path(config.data.subset_directory),
        split=config.data.train_split,
        context_length=sequence_length,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        seed=config.run.seed,
        rank=distributed_environment.rank,
        world_size=distributed_environment.world_size,
    )
    # Keep data-loader randomness reproducible and rank-specific without consuming
    # the global RNG used by parameter initialization, dropout, and corruption.
    train_generator = torch.Generator().manual_seed(
        config.run.seed + distributed_environment.rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        collate_fn=collate_dna_sequences,
        num_workers=config.data.num_workers,
        generator=train_generator,
    )

    validation_loader = None
    validation_anchor_ids = None
    if distributed_environment.is_main_process:
        # Validation runs on rank zero only. model.eval() prevents EMA updates, so
        # evaluation does not require matching quantizer collectives on other ranks.
        validation_dataset = load_gtdb_dataset(
            subset_directory=Path(config.data.subset_directory),
            split=config.data.validation_split,
            context_length=sequence_length,
        )
        validation_generator = torch.Generator().manual_seed(config.run.seed)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.data.validation_batch_size,
            collate_fn=collate_dna_sequences,
            num_workers=config.data.num_workers,
            generator=validation_generator,
        )
        anchor_batch = next(iter(validation_loader))
        validation_anchor_ids = anchor_batch["input_ids"][
            : config.evaluation.stability_anchor_size
        ].clone()

    model = NSMDNA.from_config(config).to(device)
    initialized_checkpoint_step = None
    if config.run.initialize_from is not None:
        checkpoint_path = Path(config.run.initialize_from)
        initialized_checkpoint_step = load_model_checkpoint(
            checkpoint_path,
            model,
            device,
        )
    freeze_tokenizer = bool(config.training.freeze_tokenizer)
    configure_training_phase(
        model,
        freeze_tokenizer=freeze_tokenizer,
    )
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"

    ordinary_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    codebook_parameters = model.tokenizer.quantizer.num_codebook_parameters
    if distributed_environment.is_main_process:
        print(
            "NSM-DNA parameters: "
            f"{(ordinary_parameters + codebook_parameters) / 1e6:.2f}M "
            f"total ({trainable_parameters / 1e6:.2f}M gradient-trained, "
            f"{codebook_parameters / 1e6:.2f}M EMA codebook)"
        )
        if freeze_tokenizer:
            print("frozen-hierarchy phase: training only the next-scale transformer")
        if initialized_checkpoint_step is not None:
            print(
                "initialized model weights from "
                f"{config.run.initialize_from} (source step "
                f"{initialized_checkpoint_step})"
            )
        if wandb_run is not None:
            wandb_run.summary["model/trainable_parameters"] = trainable_parameters
            wandb_run.summary["model/codebook_parameters"] = codebook_parameters
            if initialized_checkpoint_step is not None:
                wandb_run.summary["initialization/source_step"] = (
                    initialized_checkpoint_step
                )

    # Retain the full parameter group so a continuation phase can restore the
    # existing optimizer state. Frozen tokenizer parameters receive no gradients
    # and therefore are not updated; codebooks remain EMA buffers.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = build_learning_rate_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        decay_end_step=total_steps,
        learning_rate=config.optimizer.learning_rate,
        min_learning_rate=config.optimizer.min_learning_rate,
    )

    start_step = 0
    best_validation_loss = float("inf")
    if config.run.resume_from is not None:
        # A unified checkpoint restores both model components and the optimizer/
        # scheduler position, so training resumes at the next optimizer step.
        checkpoint_path = Path(config.run.resume_from)
        start_step, best_validation_loss = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )
        if config.run.reset_best_validation_loss:
            best_validation_loss = float("inf")
        if distributed_environment.is_main_process:
            print(f"resumed from checkpoint: {checkpoint_path} (step {start_step})")
            if config.run.reset_best_validation_loss:
                print("reset best validation loss for the new training phase")

    if distributed_environment.is_distributed:
        # Rank-specific seeds can create different initial EMA buffers. Synchronize
        # them once; subsequent EMA updates all-reduce hard-assignment statistics.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

    training_model: NSMDNA | DistributedDataParallel = model
    if distributed_environment.is_distributed:
        # Quantizer EMA state is synchronized explicitly during its update. Asking
        # DDP to broadcast every buffer on every forward would be redundant and can
        # overwrite the freshly synchronized EMA state.
        if device.type == "cuda":
            training_model = DistributedDataParallel(
                model,
                device_ids=[distributed_environment.local_rank],
                output_device=distributed_environment.local_rank,
                broadcast_buffers=False,
            )
        else:
            training_model = DistributedDataParallel(
                model,
                broadcast_buffers=False,
            )

    previous_stability_snapshot = None
    if distributed_environment.is_main_process:
        assert validation_anchor_ids is not None
        previous_stability_snapshot = tokenizer_stability_snapshot_to_cpu(
            model.tokenizer_stability_snapshot(validation_anchor_ids.to(device))
        )
        print(
            "captured initial tokenizer stability reference from "
            f"{validation_anchor_ids.shape[0]} fixed validation sequences"
        )
    if distributed_environment.is_distributed:
        dist.barrier()

    training_epoch = 0
    train_iterator = iter(train_loader)
    progress_bar = tqdm(
        range(start_step + 1, total_steps + 1),
        desc="Training",
        disable=not distributed_environment.is_main_process,
    )
    gradient_accumulation_steps = config.optimizer.gradient_accumulation_steps
    corruption_probability = config.training.input_code_corruption_probability
    entropy_temperature = config.training.entropy_temperature
    teacher_forced_prediction_reconstruction_loss_weight = (
        config.training.teacher_forced_prediction_reconstruction_loss_weight
    )
    partial_reconstruction_loss_weight = (
        config.training.partial_reconstruction_loss_weight
    )
    rollout_state_consistency_loss_weight = (
        config.training.rollout_state_consistency_loss_weight
    )
    use_rollout_training = rollout_state_consistency_loss_weight != 0

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        should_log = step % config.training.log_interval == 0

        if should_log:
            loss_sums = torch.zeros(8, device=device)
            teacher_forced_prediction_correct = torch.zeros(
                (), dtype=torch.long, device=device
            )
            rollout_prediction_correct = torch.zeros(
                (), dtype=torch.long, device=device
            )
            prediction_count = torch.zeros(
                (), dtype=torch.long, device=device
            )
            nucleotide_reconstruction_correct = torch.zeros(
                (), dtype=torch.long, device=device
            )
            teacher_forced_prediction_reconstruction_correct = torch.zeros(
                (), dtype=torch.long, device=device
            )
            target_count = torch.zeros((), dtype=torch.long, device=device)
            example_count = torch.zeros((), dtype=torch.long, device=device)

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                training_epoch += 1
                train_dataset.set_epoch(training_epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            sequence_ids = batch["input_ids"].to(device)
            target_ids = target_ids_from_sequence(
                sequence_ids,
                model.tokenizer.context_length,
            )
            is_last_micro_step = micro_step == gradient_accumulation_steps - 1
            if distributed_environment.is_distributed and not is_last_micro_step:
                # Delay gradient synchronization until the last microbatch. This
                # affects DDP gradients only; EMA statistics still synchronize in
                # each quantizer forward pass.
                synchronization_context = training_model.no_sync()
            else:
                synchronization_context = nullcontext()

            with synchronization_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_mixed_precision,
                ):
                    output = training_model(
                        sequence_ids,
                        corruption_probability=corruption_probability,
                        return_partial_reconstruction=(
                            partial_reconstruction_loss_weight != 0
                        ),
                        return_teacher_forced_prediction_reconstruction=True,
                        return_rollout=use_rollout_training,
                    )
                    losses = nsm_dna_losses(
                        output,
                        target_ids,
                        partial_reconstruction_loss_weight=(
                            partial_reconstruction_loss_weight
                        ),
                        teacher_forced_prediction_reconstruction_loss_weight=(
                            teacher_forced_prediction_reconstruction_loss_weight
                        ),
                        next_scale_prediction_loss_weight=(
                            next_scale_prediction_loss_weight
                        ),
                        rollout_state_consistency_loss_weight=(
                            rollout_state_consistency_loss_weight
                        ),
                        entropy_loss_weight=entropy_loss_weight,
                        entropy_temperature=entropy_temperature,
                    )
                    accumulated_loss = losses.total / gradient_accumulation_steps
                accumulated_loss.backward()

            if should_log:
                with torch.no_grad():
                    _add_training_statistics(
                        output,
                        losses,
                        target_ids,
                        loss_sums,
                        teacher_forced_prediction_correct,
                        rollout_prediction_correct,
                        prediction_count,
                        nucleotide_reconstruction_correct,
                        teacher_forced_prediction_reconstruction_correct,
                        target_count,
                        example_count,
                    )

        # Record component norms before clipping to expose where unstable gradients
        # originate. The global norm is then clipped across the complete model.
        component_gradient_norms = None
        if should_log:
            component_gradient_norms = {
                "encoder": module_gradient_norm(model.tokenizer.encoder),
                "quantizer": module_gradient_norm(model.tokenizer.quantizer),
                "decoder": module_gradient_norm(model.tokenizer.decoder),
                "next_scale_transformer": module_gradient_norm(model.transformer),
            }
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        if should_log:
            # Loss sums are example-weighted and accuracies are pooled counts, so
            # summing them gives true global metrics across DDP ranks.
            statistics = [
                loss_sums,
                teacher_forced_prediction_correct,
                rollout_prediction_correct,
                prediction_count,
                nucleotide_reconstruction_correct,
                teacher_forced_prediction_reconstruction_correct,
                target_count,
                example_count,
            ]
            _all_reduce_training_statistics(statistics)

            if distributed_environment.is_main_process:
                mean_losses = loss_sums / example_count
                metrics = {
                    "total_loss": mean_losses[0].item(),
                    "nucleotide_reconstruction_loss": mean_losses[1].item(),
                    "partial_reconstruction_loss": mean_losses[2].item(),
                    "teacher_forced_prediction_reconstruction_loss": (
                        mean_losses[3].item()
                    ),
                    "vq_loss": mean_losses[4].item(),
                    "teacher_forced_prediction_loss": mean_losses[5].item(),
                    "rollout_state_consistency_loss": mean_losses[6].item(),
                    "entropy_loss": mean_losses[7].item(),
                    "nucleotide_reconstruction_accuracy": (
                        nucleotide_reconstruction_correct / target_count
                    ).item(),
                    "teacher_forced_prediction_reconstruction_accuracy": (
                        teacher_forced_prediction_reconstruction_correct / target_count
                    ).item(),
                    "teacher_forced_prediction_accuracy": (
                        teacher_forced_prediction_correct
                        / prediction_count
                    ).item(),
                    "gradient_norm": gradient_norm.item(),
                    "learning_rate": learning_rate,
                }
                if use_rollout_training:
                    metrics["rollout_prediction_accuracy"] = (
                        rollout_prediction_correct / prediction_count
                    ).item()
                assert component_gradient_norms is not None
                for component, norm in component_gradient_norms.items():
                    metrics[f"gradient_norm_{component}"] = norm

                append_metrics(
                    run_directory / "training_metrics.jsonl",
                    step,
                    metrics,
                )

                progress_bar.set_postfix(
                    loss=f"{metrics['total_loss']:.4f}",
                    accuracy=(
                        f"{metrics['nucleotide_reconstruction_accuracy']:.2%}"
                    ),
                )
                if wandb_run is not None:
                    dashboard_metrics = {
                        "train/objective/total": metrics["total_loss"],
                        "train/objective/entropy": metrics["entropy_loss"],
                        "train/objective/rollout_state_consistency": metrics[
                            "rollout_state_consistency_loss"
                        ],
                        "train/accuracy/nucleotide_reconstruction": metrics[
                            "nucleotide_reconstruction_accuracy"
                        ],
                        "train/accuracy/teacher_forced_prediction_reconstruction": (
                            metrics[
                                "teacher_forced_prediction_reconstruction_accuracy"
                            ]
                        ),
                        "train/accuracy/teacher_forced_prediction": metrics[
                            "teacher_forced_prediction_accuracy"
                        ],
                        "optimization/gradient_norm": metrics["gradient_norm"],
                        "optimization/learning_rate": metrics["learning_rate"],
                    }
                    if use_rollout_training:
                        dashboard_metrics["train/accuracy/rollout_prediction"] = (
                            metrics["rollout_prediction_accuracy"]
                        )
                    wandb_run.log(dashboard_metrics, step=step)

        if step % config.evaluation.interval == 0:
            if distributed_environment.is_main_process:
                assert validation_loader is not None
                validation_metrics = evaluate(
                    model,
                    validation_loader,
                    use_mixed_precision,
                    max_batches=config.evaluation.max_batches,
                    partial_reconstruction_loss_weight=(
                        partial_reconstruction_loss_weight
                    ),
                    teacher_forced_prediction_reconstruction_loss_weight=(
                        teacher_forced_prediction_reconstruction_loss_weight
                    ),
                    next_scale_prediction_loss_weight=(
                        next_scale_prediction_loss_weight
                    ),
                    rollout_state_consistency_loss_weight=(
                        rollout_state_consistency_loss_weight
                    ),
                    entropy_loss_weight=entropy_loss_weight,
                    entropy_temperature=entropy_temperature,
                )
                assert validation_anchor_ids is not None
                assert previous_stability_snapshot is not None
                current_stability_snapshot = tokenizer_stability_snapshot_to_cpu(
                    model.tokenizer_stability_snapshot(
                        validation_anchor_ids.to(device)
                    )
                )
                validation_metrics.update(
                    tokenizer_stability_metrics(
                        previous_stability_snapshot,
                        current_stability_snapshot,
                        model.tokenizer.scale_lengths,
                    )
                )
                previous_stability_snapshot = current_stability_snapshot
                rollout_accuracy = validation_metrics["rollout_nucleotide_accuracy"]
                tqdm.write(
                    f"step {step} validation: total loss "
                    f"{validation_metrics['total_loss']:.4f}, reconstruction "
                    "accuracy "
                    f"{validation_metrics['nucleotide_reconstruction_accuracy']:.2%}, "
                    "first-scale-conditioned rollout accuracy "
                    f"{rollout_accuracy:.2%}, code retention "
                    f"{validation_metrics['code_retention']:.2%}, target-state drift "
                    f"{validation_metrics['target_state_drift']:.4f}"
                )

                validation_loss = validation_metrics["total_loss"]
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_checkpoint_path = save_training_checkpoint(
                        run_directory,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        best_validation_loss,
                        checkpoint_name="best.pt",
                    )
                    tqdm.write(f"saved best checkpoint: {best_checkpoint_path}")

                append_metrics(
                    run_directory / "validation_metrics.jsonl",
                    step,
                    {
                        **validation_metrics,
                        "best_total_loss": best_validation_loss,
                    },
                )

                if wandb_run is not None:
                    wandb_run.log(
                        _validation_metrics_for_wandb(
                            validation_metrics,
                            model.tokenizer.scale_lengths,
                            best_validation_loss,
                        ),
                        step=step,
                    )

            if distributed_environment.is_distributed:
                dist.barrier()

        is_recovery_step = step % config.checkpoint.recovery_interval == 0
        checkpoints_to_save: list[tuple[str, str | None]] = []
        # latest.pt is an overwriteable recovery point. Numbered milestones are
        # retained so important stages of the run are not lost when latest advances.
        if is_recovery_step:
            checkpoints_to_save.append(("recovery", "latest.pt"))
        if step % config.checkpoint.milestone_interval == 0:
            checkpoints_to_save.append(("milestone", None))

        if checkpoints_to_save:
            if distributed_environment.is_main_process:
                for checkpoint_type, checkpoint_name in checkpoints_to_save:
                    checkpoint_path = save_training_checkpoint(
                        run_directory,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        best_validation_loss,
                        checkpoint_name=checkpoint_name,
                    )
                    tqdm.write(f"saved {checkpoint_type} checkpoint: {checkpoint_path}")

            if distributed_environment.is_distributed:
                dist.barrier()

    if distributed_environment.is_main_process:
        final_checkpoint_path = save_training_checkpoint(
            run_directory,
            model,
            optimizer,
            scheduler,
            config,
            total_steps,
            best_validation_loss,
            checkpoint_name="final.pt",
        )
        tqdm.write(f"saved final checkpoint: {final_checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()

    if distributed_environment.is_distributed:
        dist.barrier()
    cleanup_distributed_training()


if __name__ == "__main__":
    main()
