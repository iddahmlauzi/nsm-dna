from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.models.next_scale import NSM, tokenizer_scale_indices
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import (
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
    load_training_checkpoint,
    save_training_checkpoint,
)


@dataclass(frozen=True)
class BlockPredictionBatch:
    """Inputs and targets for predicting one block from its preceding blocks."""

    target_ids: Int[Tensor, "batch block_length"]
    prefix_by_scale: list[Float[Tensor, "batch scale_length vq_dim"]]
    targets_by_scale: list[Int[Tensor, "batch scale_length"]]


@dataclass(frozen=True)
class Losses:
    total: Tensor
    hierarchy_by_scale: Tensor


def build_scale_loss_weights(
    scale_lengths: list[int],
    scale_loss_alpha: float,
    device: torch.device,
) -> Tensor:
    """Return each scale's share of the total next-scale loss.

    Alpha 1 gives every scale equal weight. Alpha 0 gives every code position
    equal weight, so longer scales receive a proportionally larger share.
    """
    lengths = torch.tensor(scale_lengths, dtype=torch.float32, device=device)
    scale_weights = lengths.pow(1.0 - scale_loss_alpha)
    return scale_weights / scale_weights.sum()


@torch.no_grad()
def build_codebook_neighbor_tables(
    codebook_vectors: list[Tensor],
    neighbor_count: int,
) -> list[Tensor]:
    """Return the nearest alternative codes within each selected codebook."""
    neighbor_tables = []
    for vectors in codebook_vectors:
        distances = torch.cdist(vectors.float(), vectors.float())
        distances.fill_diagonal_(torch.inf)
        neighbor_tables.append(
            distances.topk(neighbor_count, largest=False).indices
        )
    return neighbor_tables


@torch.no_grad()
def corrupt_context_indices(
    context_indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    neighbor_tables_by_scale: list[Int[Tensor, "codebook_size neighbor"]],
    corruption_probabilities: list[float],
) -> list[Int[Tensor, "batch scale_length"]]:
    """Replace selected context codes with nearby codes from the same scale."""
    corrupted_context = []
    for context_indices, neighbor_table, probability in zip(
        context_indices_by_scale,
        neighbor_tables_by_scale,
        corruption_probabilities,
        strict=True,
    ):
        if probability == 0:
            corrupted_context.append(context_indices)
            continue

        neighbor_rank = torch.randint(
            neighbor_table.shape[1],
            context_indices.shape,
            device=context_indices.device,
        )
        replacement_indices = neighbor_table[context_indices].gather(
            dim=-1,
            index=neighbor_rank.unsqueeze(-1),
        ).squeeze(-1)
        corruption_mask = torch.rand(
            context_indices.shape,
            device=context_indices.device,
        ) < probability
        corrupted_context.append(
            torch.where(corruption_mask, replacement_indices, context_indices)
        )

    return corrupted_context


def decode_predicted_scale(
    tokenizer: VQVAE,
    scale_logits: Tensor,
    tokenizer_scale_index: int,
) -> Tensor:
    """Decode hard predicted codes while passing gradients through probabilities."""
    probabilities = scale_logits.float().softmax(dim=-1)
    hard_assignments = F.one_hot(
        probabilities.argmax(dim=-1),
        num_classes=probabilities.shape[-1],
    ).to(probabilities.dtype)
    assignments = hard_assignments + probabilities - probabilities.detach()

    codebook = tokenizer.quantizer.codebooks[tokenizer_scale_index].codebook.float()
    scale_latent = assignments @ codebook
    full_latent = tokenizer.quantizer.upsample_to_full_length(
        scale_latent,
        tokenizer_scale_index,
    )
    return tokenizer.decoder(full_latent)


def compute_losses(
    logits_by_scale: list[Tensor],
    targets_by_scale: list[Tensor],
    scale_loss_weights: Tensor,
) -> Losses:
    """Average exact-code classification losses across hierarchy scales."""
    hierarchy_by_scale = torch.stack(
        [
            F.cross_entropy(
                scale_logits.flatten(0, 1),
                scale_targets.flatten(),
            )
            for scale_logits, scale_targets in zip(
                logits_by_scale,
                targets_by_scale,
            )
        ]
    )
    hierarchy = (hierarchy_by_scale * scale_loss_weights).sum()
    return Losses(
        total=hierarchy,
        hierarchy_by_scale=hierarchy_by_scale,
    )


def soft_conditioning_probability(
    step: int,
    total_steps: int,
    num_epochs: int,
    teacher_forcing_epochs: float,
    transition_epochs: float,
) -> float:
    """Transition from teacher forcing to fully predicted soft context."""
    epoch_progress = (step - 1) * num_epochs / total_steps
    transition_progress = (
        epoch_progress - teacher_forcing_epochs
    ) / transition_epochs
    return max(0.0, min(1.0, transition_progress))


@torch.no_grad()
def prepare_block_predictions(
    tokenizer: VQVAE,
    input_ids: Int[Tensor, "batch sequence_length"],
    scale_lengths: list[int],
) -> list[BlockPredictionBatch]:
    """Create one task that predicts the second block from the first block."""
    block_length = tokenizer.context_length
    prefix_ids = input_ids[:, :block_length]
    target_ids = input_ids[:, block_length:]
    tokenizer_prefix = tokenizer.encode_scales(prefix_ids)
    tokenizer_indices = tokenizer.encode_indices(target_ids)
    scale_indices = tokenizer_scale_indices(
        tokenizer.scale_lengths,
        scale_lengths,
    )
    prefix_by_scale = [tokenizer_prefix[index] for index in scale_indices]
    indices_by_scale = [tokenizer_indices[index] for index in scale_indices]

    return [
        BlockPredictionBatch(
            target_ids=target_ids,
            prefix_by_scale=prefix_by_scale,
            targets_by_scale=indices_by_scale,
        )
    ]


@torch.no_grad()
def rollout_hierarchy(
    model: NSM,
    prefix_by_scale: list[Float[Tensor, "batch scale_length vq_dim"]],
) -> list[Int[Tensor, "batch scale_length"]]:
    """Generate hard outputs while carrying soft uncertainty between scales."""
    predicted_indices_by_scale = []
    previous_scale_latent = None

    for scale_index, prefix in enumerate(prefix_by_scale):
        logits = model.predict_scale(
            prefix,
            scale_index,
            previous_scale_latent,
        )
        predicted_indices_by_scale.append(logits.argmax(dim=-1))
        probabilities = logits.float().softmax(dim=-1)
        previous_scale_latent = (
            probabilities @ model.codebook_vectors(scale_index)
        )

    return predicted_indices_by_scale


@torch.no_grad()
def evaluate(
    model: NSM,
    tokenizer: VQVAE,
    data_loader: DataLoader,
    scale_loss_weights: Tensor,
    use_mixed_precision: bool,
    max_batches: int | None = None,
    rollout_max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate teacher-forced predictions and optional greedy rollouts.

    Evaluate the entire data loader when max_batches is None.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    num_scales = len(model.scale_lengths)
    scale_indices = tokenizer_scale_indices(
        tokenizer.scale_lengths,
        model.scale_lengths,
    )
    loss_sum = 0.0
    hierarchy_loss_sums_by_scale = torch.zeros(num_scales, device=device)
    correct_codes_by_scale = torch.zeros(num_scales, device=device)
    num_codes_by_scale = torch.zeros(num_scales, device=device)
    code_counts_by_scale = [
        torch.zeros(codebook_size, dtype=torch.long, device=device)
        for codebook_size in model.codebook_sizes
    ]
    correct_nucleotides = 0
    num_nucleotides = 0
    num_block_predictions = 0
    rollout_nucleotide_loss_sum = 0.0
    rollout_correct_nucleotides = 0
    rollout_num_nucleotides = 0
    num_rollouts = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index == max_batches:
            break

        input_ids = batch["input_ids"].to(device)

        # Evaluate each tokenizer-sized block separately. Later blocks use the
        # encoded real preceding blocks as prefix context.
        for prediction in prepare_block_predictions(
            tokenizer,
            input_ids,
            model.scale_lengths,
        ):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_mixed_precision,
            ):
                # Every scale is predicted in parallel from completed earlier
                # scales, with all scale tasks packed into one transformer pass.
                output = model(
                    prediction.targets_by_scale[:-1],
                    prefix_by_scale=prediction.prefix_by_scale,
                )
                losses = compute_losses(
                    output.hierarchy_logits,
                    prediction.targets_by_scale,
                    scale_loss_weights,
                )

                # Rollout predicts the first scale from the real prefix, then
                # supplies generated target codes.
                if batch_index < rollout_max_batches:
                    rollout_indices = rollout_hierarchy(
                        model,
                        prefix_by_scale=prediction.prefix_by_scale,
                    )
                    rollout_latent = tokenizer.quantizer.indices_to_scale_latent(
                        rollout_indices[-1],
                        scale_indices[-1],
                    )
                    rollout_logits = tokenizer.decoder(rollout_latent)
                    rollout_nucleotide_loss = F.cross_entropy(
                        rollout_logits.flatten(0, 1),
                        prediction.target_ids.flatten(),
                    )

            # Pool exact code matches by scale across all evaluated blocks.
            for scale_index, (scale_logits, scale_targets) in enumerate(
                zip(
                    output.hierarchy_logits,
                    prediction.targets_by_scale,
                    strict=True,
                )
            ):
                predictions_at_scale = scale_logits.argmax(dim=-1)
                correct_at_scale = predictions_at_scale == scale_targets
                correct_codes_by_scale[scale_index] += correct_at_scale.sum().item()
                num_codes_by_scale[scale_index] += scale_targets.numel()
                code_counts_by_scale[scale_index] += torch.bincount(
                    scale_targets.flatten(),
                    minlength=model.codebook_sizes[scale_index],
                )

            loss_sum += losses.total.item()
            hierarchy_loss_sums_by_scale += losses.hierarchy_by_scale
            finest_logits = decode_predicted_scale(
                tokenizer,
                output.hierarchy_logits[-1],
                scale_indices[-1],
            )
            correct_nucleotides += (
                finest_logits.argmax(dim=-1) == prediction.target_ids
            ).sum().item()
            num_nucleotides += prediction.target_ids.numel()
            num_block_predictions += 1

            if batch_index < rollout_max_batches:
                rollout_nucleotide_loss_sum += rollout_nucleotide_loss.item()
                rollout_correct_nucleotides += (
                    (rollout_logits.argmax(dim=-1) == prediction.target_ids)
                    .sum()
                    .item()
                )
                rollout_num_nucleotides += prediction.target_ids.numel()
                num_rollouts += 1

    model.train(was_training)

    metrics = {
        "loss": loss_sum / num_block_predictions,
        "hierarchy_accuracy": (
            correct_codes_by_scale.sum() / num_codes_by_scale.sum()
        ).item(),
        "nucleotide_accuracy": correct_nucleotides / num_nucleotides,
    }

    for scale_index, (scale_length, codebook_size) in enumerate(
        zip(model.scale_lengths, model.codebook_sizes, strict=True)
    ):
        counts = code_counts_by_scale[scale_index].float()
        probabilities = counts[counts > 0] / counts.sum()
        perplexity = torch.exp(-(probabilities * probabilities.log()).sum())
        section = f"scale_{scale_index + 1:02d}_length_{scale_length}"
        metrics[f"{section}/hierarchy_loss"] = (
            hierarchy_loss_sums_by_scale[scale_index] / num_block_predictions
        ).item()
        metrics[f"{section}/prediction_accuracy"] = (
            correct_codes_by_scale[scale_index] / num_codes_by_scale[scale_index]
        ).item()
        metrics[f"{section}/codebook_usage"] = (
            perplexity / codebook_size
        ).item()

    # Rollout loss averages the block-batch losses, while rollout accuracy pools
    # every decoded nucleotide position.
    if num_rollouts > 0:
        metrics["rollout_nucleotide_loss"] = rollout_nucleotide_loss_sum / num_rollouts
        metrics["rollout_nucleotide_accuracy"] = (
            rollout_correct_nucleotides / rollout_num_nucleotides
        )

    return metrics


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
    if distributed_environment.is_main_process:
        print(f"training for {total_steps:,} optimizer steps")

    # The tokenizer checkpoint owns every tokenizer architectural choice. Keeping
    # those settings out of this config prevents stage 2 from silently rebuilding
    # a tokenizer that differs from the one that produced its code targets.
    tokenizer_checkpoint_path = Path(config.tokenizer_checkpoint)
    tokenizer = VQVAE.from_checkpoint(
        tokenizer_checkpoint_path,
        device,
        frozen=True,
    )
    if sequence_length != 2 * tokenizer.context_length:
        raise ValueError(
            "data.sequence_length must contain one prefix block and one target block."
        )

    # Create the experiment logger.
    wandb_run = None
    if config.wandb.enabled and distributed_environment.is_main_process:
        wandb_run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.wandb.name,
            config=OmegaConf.to_container(config, resolve=True),
            dir=run_directory,
        )

    # Create the dataset and data loader.
    train_dataset = load_gtdb_dataset(
        subset_directory=Path(config.data.subset_directory),
        split=config.data.train_split,
        context_length=sequence_length,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        seed=config.run.seed,
        rank=distributed_environment.rank,
        world_size=distributed_environment.world_size,
    )

    # Keep DataLoader iterator seeding separate from the model's random state.
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

    model = NSM.from_config(config, tokenizer).to(device)
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    if distributed_environment.is_main_process:
        print(f"NSM-DNA parameters: {num_parameters / 1e6:.2f}M")
        if wandb_run is not None:
            wandb_run.summary["model/parameters"] = num_parameters

    scale_loss_weights = build_scale_loss_weights(
        model.scale_lengths,
        config.objective.scale_loss_alpha,
        device,
    )
    scale_indices = tokenizer_scale_indices(
        tokenizer.scale_lengths,
        model.scale_lengths,
    )
    codebook_neighbor_tables = build_codebook_neighbor_tables(
        [
            tokenizer.quantizer.codebooks[index].codebook
            for index in scale_indices
        ],
        int(config.training.context_neighbor_count),
    )
    tokenizer_corruption_probabilities = [
        float(probability)
        for probability in config.training.context_corruption_probabilities
    ]
    context_corruption_probabilities = [
        tokenizer_corruption_probabilities[index]
        for index in scale_indices
    ]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta_1, config.optimizer.beta_2),
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = build_learning_rate_scheduler(
        optimizer,
        warmup_steps=config.optimizer.warmup_steps,
        decay_end_step=total_steps,
        learning_rate=config.optimizer.learning_rate,
        min_learning_rate=config.optimizer.min_learning_rate,
    )

    # Resume only the stage-two model and optimizer. The tokenizer is always
    # restored independently from the checkpoint named in tokenizer_checkpoint.
    start_step = 0
    best_validation_loss = float("inf")
    if config.run.resume_from is not None:
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

    training_model: NSM | DistributedDataParallel = model
    if distributed_environment.is_distributed:
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

    # Train the stage-two model while the tokenizer supplies fixed inputs and
    # targets. Only NSM parameters are owned by the optimizer.
    training_epoch = 0
    train_iterator = iter(train_loader)
    progress_bar = tqdm(
        range(start_step + 1, total_steps + 1),
        desc="Training",
        disable=not distributed_environment.is_main_process,
    )
    gradient_accumulation_steps = config.optimizer.gradient_accumulation_steps

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        should_log = step % config.training.log_interval == 0

        if should_log:
            mean_loss = torch.zeros((), device=device)
            correct_codes = torch.zeros((), device=device, dtype=torch.long)
            num_codes = torch.zeros((), device=device, dtype=torch.long)

        conditioning_probability = soft_conditioning_probability(
            step,
            total_steps,
            int(config.training.num_epochs),
            float(config.training.teacher_forcing_epochs),
            float(config.training.soft_conditioning_transition_epochs),
        )

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                training_epoch += 1
                train_dataset.set_epoch(training_epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)
            block_predictions = prepare_block_predictions(
                tokenizer,
                input_ids,
                model.scale_lengths,
            )
            num_predictions_per_step = gradient_accumulation_steps * len(
                block_predictions
            )

            for block_index, prediction in enumerate(block_predictions):
                is_last_prediction = (
                    micro_step == gradient_accumulation_steps - 1
                    and block_index == len(block_predictions) - 1
                )
                if distributed_environment.is_distributed and not is_last_prediction:
                    synchronization_context = training_model.no_sync()
                else:
                    synchronization_context = nullcontext()

                # Average every block prediction in the optimizer step and only
                # synchronize DDP gradients on the final backward pass.
                with synchronization_context:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=use_mixed_precision,
                    ):
                        corrupted_indices_by_scale = corrupt_context_indices(
                            prediction.targets_by_scale,
                            codebook_neighbor_tables,
                            context_corruption_probabilities,
                        )
                        output = training_model(
                            corrupted_indices_by_scale[:-1],
                            prefix_by_scale=prediction.prefix_by_scale,
                            soft_conditioning_probability=conditioning_probability,
                        )
                        losses = compute_losses(
                            output.hierarchy_logits,
                            prediction.targets_by_scale,
                            scale_loss_weights,
                        )
                        accumulated_loss = losses.total / num_predictions_per_step
                    accumulated_loss.backward()

                if should_log:
                    mean_loss += losses.total.detach() / num_predictions_per_step

                    with torch.no_grad():
                        for scale_logits, scale_targets in zip(
                            output.hierarchy_logits,
                            prediction.targets_by_scale,
                            strict=True,
                        ):
                            correct_codes += (
                                scale_logits.argmax(dim=-1) == scale_targets
                            ).sum()
                            num_codes += scale_targets.numel()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()

        # Set the learning rate that will be used by the next optimizer step.
        scheduler.step()

        if should_log:
            if distributed_environment.is_distributed:
                for values in (
                    mean_loss,
                    correct_codes,
                    num_codes,
                ):
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                mean_loss /= distributed_environment.world_size

            if distributed_environment.is_main_process:
                hierarchy_accuracy = (correct_codes / num_codes).item()

                progress_bar.set_postfix(
                    loss=f"{mean_loss.item():.4f}",
                )

                if wandb_run is not None:
                    wandb_metrics = {
                        "train/loss": mean_loss.item(),
                        "train/hierarchy_accuracy": hierarchy_accuracy,
                        "train/soft_conditioning_probability": conditioning_probability,
                        "train/gradient_norm": gradient_norm.item(),
                        "train/learning_rate": learning_rate,
                    }
                    wandb_run.log(
                        wandb_metrics,
                        step=step,
                        commit=step % config.evaluation.interval != 0,
                    )

        if step % config.evaluation.interval == 0:
            if distributed_environment.is_main_process:
                assert validation_loader is not None
                validation_metrics = evaluate(
                    model,
                    tokenizer,
                    validation_loader,
                    scale_loss_weights,
                    use_mixed_precision,
                    max_batches=config.evaluation.max_batches,
                    rollout_max_batches=config.evaluation.rollout_max_batches,
                )
                tqdm.write(
                    f"step {step} validation: loss "
                    f"{validation_metrics['loss']:.4f}, nucleotide accuracy "
                    f"{validation_metrics['nucleotide_accuracy']:.2%}, "
                    f"rollout accuracy "
                    f"{validation_metrics['rollout_nucleotide_accuracy']:.2%}"
                )

                validation_loss = validation_metrics["loss"]
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
                    wandb_metrics = {
                        "validation/loss": validation_metrics["loss"],
                        "validation/hierarchy_accuracy": validation_metrics[
                            "hierarchy_accuracy"
                        ],
                        "validation/nucleotide_accuracy": validation_metrics[
                            "nucleotide_accuracy"
                        ],
                        "rollout/nucleotide_loss": validation_metrics[
                            "rollout_nucleotide_loss"
                        ],
                        "rollout/nucleotide_accuracy": validation_metrics[
                            "rollout_nucleotide_accuracy"
                        ],
                    }
                    for scale_index, scale_length in enumerate(model.scale_lengths):
                        scale_name = (
                            f"scale_{scale_index + 1:02d}_length_{scale_length}"
                        )
                        wandb_metrics[f"{scale_name}/hierarchy_loss"] = (
                            validation_metrics[f"{scale_name}/hierarchy_loss"]
                        )
                        wandb_metrics[f"{scale_name}/prediction_accuracy"] = (
                            validation_metrics[f"{scale_name}/prediction_accuracy"]
                        )
                        wandb_metrics[f"{scale_name}/codebook_usage"] = (
                            validation_metrics[f"{scale_name}/codebook_usage"]
                        )

                    wandb_run.log(wandb_metrics, step=step)

            if distributed_environment.is_distributed:
                dist.barrier()

        if step % config.checkpoint.interval == 0:
            if distributed_environment.is_main_process:
                checkpoint_path = save_training_checkpoint(
                    run_directory,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    best_validation_loss,
                    checkpoint_name="latest.pt",
                )
                tqdm.write(f"saved checkpoint: {checkpoint_path}")

            if distributed_environment.is_distributed:
                dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()

    if distributed_environment.is_distributed:
        dist.barrier()
    cleanup_distributed_training()


if __name__ == "__main__":
    main()
