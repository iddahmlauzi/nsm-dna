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
from nsm_dna.models.next_scale import NSMDNA, NSMDNAOutput
from nsm_dna.training import (
    GenFirstLossSchedule,
    GenerativeLossWeights,
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
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


@torch.no_grad()
def evaluate(
    model: NSMDNA,
    data_loader: DataLoader,
    use_mixed_precision: bool,
    max_batches: int | None = None,
    *,
    next_scale_prediction_loss_weight: float = 1.0,
    entropy_loss_weight: float = 1.0,
    entropy_temperature: float = 1.0,
) -> dict[str, float]:
    """Evaluate all four losses and tokenizer/next-scale diagnostics."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    num_scales = len(model.tokenizer.scale_lengths)

    # The losses returned by the model are batch means, so they are accumulated
    # by example. Accuracy and codebook statistics are accumulated as raw counts
    # so that short final batches do not receive disproportionate weight.
    loss_sums = torch.zeros(5, dtype=torch.float64, device=device)
    next_scale_prediction_loss_sums = torch.zeros(
        num_scales, dtype=torch.float64, device=device
    )
    commitment_loss_sums = torch.zeros(
        num_scales, dtype=torch.float64, device=device
    )
    quantization_loss_sums = torch.zeros(
        num_scales, dtype=torch.float64, device=device
    )
    next_scale_prediction_correct = torch.zeros(
        num_scales, dtype=torch.long, device=device
    )
    next_scale_prediction_counts = torch.zeros(
        num_scales, dtype=torch.long, device=device
    )
    cumulative_nucleotide_correct = torch.zeros(
        num_scales, dtype=torch.long, device=device
    )
    confidence_sums = torch.zeros(num_scales, dtype=torch.float64, device=device)
    confidence_counts = torch.zeros(num_scales, dtype=torch.long, device=device)
    code_counts_by_scale = [
        torch.zeros(codebook_size, dtype=torch.long, device=device)
        for codebook_size in model.tokenizer.codebook_sizes
    ]
    nucleotide_reconstruction_correct = torch.zeros(
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
            )
            losses = nsm_dna_losses(
                output,
                target_ids,
                next_scale_prediction_loss_weight=(
                    next_scale_prediction_loss_weight
                ),
                entropy_loss_weight=entropy_loss_weight,
                entropy_temperature=entropy_temperature,
            )
            # Teacher-forced code accuracy can remain high while errors compound
            # during generation. Run the exact hard inference path so validation
            # directly exposes whether the learned hierarchy is modelable.
            generation = model.generate(
                sequence_ids[:, : -model.tokenizer.context_length]
            )
            rollout_nucleotide_loss = F.cross_entropy(
                generation.nucleotide_logits.flatten(0, 1).float(),
                target_ids.flatten(),
            )

        batch_size = sequence_ids.shape[0]
        example_count += batch_size
        loss_sums += torch.stack(
            [
                losses.total,
                losses.nucleotide_reconstruction,
                losses.vq,
                losses.next_scale_prediction,
                losses.entropy,
            ]
        ).double() * batch_size
        next_scale_prediction_loss_sums += (
            torch.stack(losses.next_scale_prediction_by_scale).double() * batch_size
        )
        commitment_loss_sums += torch.stack(
            output.quantizer.commitment_losses_by_scale
        ).double() * batch_size
        quantization_loss_sums += torch.stack(
            output.quantizer.quantization_losses_by_scale
        ).double() * batch_size
        rollout_nucleotide_loss_sum += (
            rollout_nucleotide_loss.double() * batch_size
        )

        nucleotide_reconstruction_correct += (
            output.reconstruction_logits.argmax(dim=-1) == target_ids
        ).sum()
        rollout_nucleotide_correct += (
            generation.nucleotide_logits.argmax(dim=-1) == target_ids
        ).sum()
        target_count += target_ids.numel()

        cumulative_logits = output.cumulative_reconstruction_logits_by_scale
        assert cumulative_logits is not None
        # Per-scale metrics separate three different failure modes: inability to
        # predict codes, loss of information in cumulative latents, and codebook
        # collapse or overly uncertain assignments.
        for scale_index, (
            scale_logits,
            scale_targets,
            probabilities,
            cumulative_scale_logits,
        ) in enumerate(
            zip(
                output.next_scale_logits_by_scale,
                output.quantizer.indices_by_scale,
                output.quantizer.assignment_probabilities_by_scale,
                cumulative_logits,
                strict=True,
            )
        ):
            next_scale_prediction_correct[scale_index] += (
                scale_logits.argmax(dim=-1) == scale_targets
            ).sum()
            next_scale_prediction_counts[scale_index] += scale_targets.numel()
            cumulative_nucleotide_correct[scale_index] += (
                cumulative_scale_logits.argmax(dim=-1) == target_ids
            ).sum()
            confidence_sums[scale_index] += (
                probabilities.max(dim=-1).values.double().sum()
            )
            confidence_counts[scale_index] += scale_targets.numel()
            code_counts_by_scale[scale_index] += torch.bincount(
                scale_targets.flatten(),
                minlength=model.tokenizer.codebook_sizes[scale_index],
            )

    model.train(was_training)
    if example_count == 0:
        raise ValueError("Validation data loader produced no batches.")

    mean_losses = loss_sums / example_count
    metrics = {
        "total_loss": mean_losses[0].item(),
        "nucleotide_reconstruction_loss": mean_losses[1].item(),
        "vq_loss": mean_losses[2].item(),
        "next_scale_prediction_loss": mean_losses[3].item(),
        "entropy_loss": mean_losses[4].item(),
        "rollout_nucleotide_loss": (
            rollout_nucleotide_loss_sum / example_count
        ).item(),
        "nucleotide_reconstruction_accuracy": (
            nucleotide_reconstruction_correct / target_count
        ).item(),
        "next_scale_prediction_accuracy": (
            next_scale_prediction_correct.sum()
            / next_scale_prediction_counts.sum()
        ).item(),
        "rollout_nucleotide_accuracy": (
            rollout_nucleotide_correct / target_count
        ).item(),
    }

    for scale_index, scale_length in enumerate(model.tokenizer.scale_lengths):
        commitment_loss = commitment_loss_sums[scale_index] / example_count
        scale_quantization_loss = (
            quantization_loss_sums[scale_index] / example_count
        )
        code_counts = code_counts_by_scale[scale_index]
        used_code_counts = code_counts[code_counts > 0].float()
        code_probabilities = used_code_counts / used_code_counts.sum()
        # Perplexity is the effective number of codes used under the observed
        # assignment distribution; usage alone only says whether a code appeared.
        perplexity = torch.exp(
            -(code_probabilities * code_probabilities.log()).sum()
        )

        metrics[f"next_scale_prediction_loss_scale_{scale_length}"] = (
            next_scale_prediction_loss_sums[scale_index] / example_count
        ).item()
        metrics[f"next_scale_prediction_accuracy_scale_{scale_length}"] = (
            next_scale_prediction_correct[scale_index]
            / next_scale_prediction_counts[scale_index]
        ).item()
        metrics[f"commitment_loss_scale_{scale_length}"] = commitment_loss.item()
        metrics[f"quantization_loss_scale_{scale_length}"] = (
            scale_quantization_loss.item()
        )
        metrics[f"vq_loss_scale_{scale_length}"] = (
            commitment_loss + scale_quantization_loss
        ).item()
        metrics[f"cumulative_nucleotide_accuracy_scale_{scale_length}"] = (
            cumulative_nucleotide_correct[scale_index] / target_count
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
    next_scale_prediction_correct: torch.Tensor,
    next_scale_prediction_count: torch.Tensor,
    nucleotide_reconstruction_correct: torch.Tensor,
    target_count: torch.Tensor,
    example_count: torch.Tensor,
) -> None:
    """Accumulate detached metrics from one training microbatch."""
    # Training logs use outputs already produced for the loss. Cumulative
    # nucleotide accuracy stays validation-only to avoid extra decoder passes.
    batch_size = target_ids.shape[0]
    loss_sums += torch.stack(
        [
            losses.total,
            losses.nucleotide_reconstruction,
            losses.vq,
            losses.next_scale_prediction,
            losses.entropy,
        ]
    ).detach() * batch_size
    nucleotide_reconstruction_correct += (
        output.reconstruction_logits.detach().argmax(dim=-1) == target_ids
    ).sum()
    target_count += target_ids.numel()
    example_count += batch_size

    for scale_logits, scale_targets in zip(
        output.next_scale_logits_by_scale,
        output.quantizer.indices_by_scale,
        strict=True,
    ):
        next_scale_prediction_correct += (
            scale_logits.detach().argmax(dim=-1) == scale_targets
        ).sum()
        next_scale_prediction_count += scale_targets.numel()


def _validation_metrics_for_wandb(
    validation_metrics: dict[str, float],
    scale_lengths: list[int],
    best_validation_loss: float,
) -> dict[str, float]:
    """Separate overall validation metrics from scale-specific W&B sections."""
    overall_metric_names = (
        "total_loss",
        "nucleotide_reconstruction_loss",
        "vq_loss",
        "next_scale_prediction_loss",
        "entropy_loss",
        "rollout_nucleotide_loss",
        "nucleotide_reconstruction_accuracy",
        "next_scale_prediction_accuracy",
        "rollout_nucleotide_accuracy",
    )
    wandb_metrics = {
        f"validation/{name}": validation_metrics[name]
        for name in overall_metric_names
    }
    wandb_metrics["validation/best_total_loss"] = best_validation_loss

    scale_metric_names = {
        "prediction_loss": "next_scale_prediction_loss",
        "prediction_accuracy": "next_scale_prediction_accuracy",
        "cumulative_nucleotide_accuracy": "cumulative_nucleotide_accuracy",
        "vq_loss": "vq_loss",
        "code_usage": "code_usage",
        "code_perplexity": "code_perplexity",
        "soft_assignment_confidence": "soft_confidence",
    }
    for scale_number, scale_length in enumerate(scale_lengths, start=1):
        section = f"scale_{scale_number:02d}_length_{scale_length}"
        for panel_name, metric_name in scale_metric_names.items():
            wandb_metrics[f"{section}/{panel_name}"] = validation_metrics[
                f"{metric_name}_scale_{scale_length}"
            ]

    return wandb_metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="nsm")
def main(config: DictConfig) -> None:
    distributed_environment = initialize_distributed_training()
    torch.manual_seed(config.run.seed + distributed_environment.rank)
    run_directory = Path(HydraConfig.get().runtime.output_dir)
    device = distributed_environment.device
    sequence_length = int(config.data.sequence_length)
    total_steps = calculate_training_steps(
        config,
        distributed_environment.world_size,
        sequence_length,
    )
    warmup_steps = round(total_steps * config.optimizer.warmup_fraction)
    loss_schedule_config = config.training.loss_schedule
    generation_first_config = loss_schedule_config.generation_first
    refinement_config = loss_schedule_config.reconstruction_refinement
    loss_schedule = GenFirstLossSchedule(
        total_steps=total_steps,
        generation_first_fraction=loss_schedule_config.generation_first_fraction,
        generation_first_weights=GenerativeLossWeights(
            next_scale_prediction=(
                generation_first_config.next_scale_prediction_loss_weight
            ),
            entropy=generation_first_config.entropy_loss_weight,
        ),
        refinement_weights=GenerativeLossWeights(
            next_scale_prediction=(
                refinement_config.next_scale_prediction_loss_weight
            ),
            entropy=refinement_config.entropy_loss_weight,
        ),
    )
    if distributed_environment.is_main_process:
        print(
            f"training for {total_steps:,} optimizer steps "
            f"with {warmup_steps:,} warmup steps"
        )
        print(
            "GenFirst schedule: "
            f"{loss_schedule.generation_first_steps:,} generation-first steps, "
            f"{total_steps - loss_schedule.generation_first_steps:,} "
            "reconstruction-refinement steps"
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

    model = NSMDNA.from_config(config).to(device)
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    codebook_parameters = model.tokenizer.quantizer.num_codebook_parameters
    if distributed_environment.is_main_process:
        print(
            "NSM-DNA parameters: "
            f"{(trainable_parameters + codebook_parameters) / 1e6:.2f}M "
            f"total ({trainable_parameters / 1e6:.2f}M gradient-trained, "
            f"{codebook_parameters / 1e6:.2f}M EMA codebook)"
        )
        if wandb_run is not None:
            wandb_run.summary["model/trainable_parameters"] = trainable_parameters
            wandb_run.summary["model/codebook_parameters"] = codebook_parameters

    # EMA codebooks are buffers rather than gradient-trained parameters. Passing
    # model.parameters() therefore optimizes every ordinary parameter exactly once
    # while leaving codebook updates to the quantizer.
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
        if distributed_environment.is_main_process:
            print(f"resumed from checkpoint: {checkpoint_path} (step {start_step})")

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

    for step in progress_bar:
        generative_loss_weights = loss_schedule.weights_at_step(step)
        if (
            distributed_environment.is_main_process
            and step == loss_schedule.generation_first_steps + 1
        ):
            tqdm.write(
                "starting reconstruction-refinement phase with next-scale "
                f"weight {generative_loss_weights.next_scale_prediction:g} "
                f"and entropy weight {generative_loss_weights.entropy:g}"
            )
        optimizer.zero_grad(set_to_none=True)
        should_log = step % config.training.log_interval == 0

        if should_log:
            loss_sums = torch.zeros(5, device=device)
            next_scale_prediction_correct = torch.zeros(
                (), dtype=torch.long, device=device
            )
            next_scale_prediction_count = torch.zeros(
                (), dtype=torch.long, device=device
            )
            nucleotide_reconstruction_correct = torch.zeros(
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
                    # Code corruption changes only transformer inputs. Entropy uses
                    # the original target assignments, as do reconstruction and EMA.
                    output = training_model(
                        sequence_ids,
                        corruption_probability=corruption_probability,
                    )
                    losses = nsm_dna_losses(
                        output,
                        target_ids,
                        next_scale_prediction_loss_weight=(
                            generative_loss_weights.next_scale_prediction
                        ),
                        entropy_loss_weight=generative_loss_weights.entropy,
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
                        next_scale_prediction_correct,
                        next_scale_prediction_count,
                        nucleotide_reconstruction_correct,
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
                next_scale_prediction_correct,
                next_scale_prediction_count,
                nucleotide_reconstruction_correct,
                target_count,
                example_count,
            ]
            _all_reduce_training_statistics(statistics)

            if distributed_environment.is_main_process:
                mean_losses = loss_sums / example_count
                metrics = {
                    "train/total_loss": mean_losses[0].item(),
                    "train/nucleotide_reconstruction_loss": mean_losses[1].item(),
                    "train/vq_loss": mean_losses[2].item(),
                    "train/next_scale_prediction_loss": mean_losses[3].item(),
                    "train/entropy_loss": mean_losses[4].item(),
                    "train/nucleotide_reconstruction_accuracy": (
                        nucleotide_reconstruction_correct / target_count
                    ).item(),
                    "train/next_scale_prediction_accuracy": (
                        next_scale_prediction_correct
                        / next_scale_prediction_count
                    ).item(),
                    "optimization/gradient_norm": gradient_norm.item(),
                    "optimization/learning_rate": learning_rate,
                }
                assert component_gradient_norms is not None
                for component, norm in component_gradient_norms.items():
                    metrics[f"gradients/{component}"] = norm

                progress_bar.set_postfix(
                    loss=f"{metrics['train/total_loss']:.4f}",
                    accuracy=(
                        f"{metrics['train/nucleotide_reconstruction_accuracy']:.2%}"
                    ),
                )
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

        if step % config.evaluation.interval == 0:
            if distributed_environment.is_main_process:
                assert validation_loader is not None
                validation_metrics = evaluate(
                    model,
                    validation_loader,
                    use_mixed_precision,
                    max_batches=config.evaluation.max_batches,
                    next_scale_prediction_loss_weight=(
                        generative_loss_weights.next_scale_prediction
                    ),
                    entropy_loss_weight=generative_loss_weights.entropy,
                    entropy_temperature=entropy_temperature,
                )
                tqdm.write(
                    f"step {step} validation: total loss "
                    f"{validation_metrics['total_loss']:.4f}, reconstruction "
                    "accuracy "
                    f"{validation_metrics['nucleotide_reconstruction_accuracy']:.2%}, "
                    "hard rollout accuracy "
                    f"{validation_metrics['rollout_nucleotide_accuracy']:.2%}"
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
                    tqdm.write(
                        f"saved {checkpoint_type} checkpoint: {checkpoint_path}"
                    )

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
