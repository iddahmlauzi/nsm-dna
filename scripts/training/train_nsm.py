from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import einx
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
from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import (
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
    load_training_checkpoint,
    save_training_checkpoint,
    upload_checkpoint_to_hugging_face,
)


@dataclass(frozen=True)
class BlockPredictionBatch:
    """Inputs and targets for predicting one block from its preceding blocks."""

    target_ids: Int[Tensor, "batch block_length"]
    prefix: Float[Tensor, "batch prefix_length vq_dim"] | None
    scale_inputs: list[Float[Tensor, "batch scale_length vq_dim"]]
    targets_by_scale: list[Int[Tensor, "batch scale_length"]]


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


def compute_next_scale_loss(
    logits: Tensor,
    targets_by_scale: list[Tensor],
    scale_loss_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute cross-entropy at each scale and their configured weighted mean."""
    scale_lengths = [targets.shape[1] for targets in targets_by_scale]
    logits_by_scale = torch.split(logits, scale_lengths, dim=1)
    losses_by_scale = torch.stack(
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
    loss = (losses_by_scale * scale_loss_weights).sum()
    return loss, losses_by_scale


@torch.no_grad()
def prepare_nsm_batch(
    tokenizer: VQVAE,
    input_ids: Int[Tensor, "batch block_length"],
) -> tuple[
    list[Float[Tensor, "batch scale_length vq_dim"]],
    list[Int[Tensor, "batch scale_length"]],
]:
    """Create hierarchy inputs and targets for a batch of target blocks.

    The first scale has no continuous input because the learned BOS predicts it.
    Each remaining input is the preceding scales' cumulative reconstruction,
    resized to the length of the scale that NSM must predict.
    """
    targets_by_scale = tokenizer.encode_indices(input_ids)
    next_scale_inputs = tokenizer.indices_to_next_scale_inputs(targets_by_scale)
    return next_scale_inputs, targets_by_scale


@torch.no_grad()
def prepare_block_predictions(
    tokenizer: VQVAE,
    input_ids: Int[Tensor, "batch sequence_length"],
) -> list[BlockPredictionBatch]:
    """Create prediction tasks for blocks that have preceding DNA context."""
    batch_size = input_ids.shape[0]

    # Split every long sequence into tokenizer-sized blocks while retaining
    # which sequence and block position each piece came from.
    blocks = einx.id(
        "batch (num_blocks block_length) -> batch num_blocks block_length",
        input_ids,
        block_length=tokenizer.context_length,
    )

    # The tokenizer operates on one block at a time. Temporarily combine the
    # batch and block axes so all blocks can be tokenized in one call.
    flattened_blocks = einx.id(
        "batch num_blocks block_length -> (batch num_blocks) block_length",
        blocks,
    )
    flattened_scale_inputs, flattened_targets_by_scale = prepare_nsm_batch(
        tokenizer, flattened_blocks
    )

    # Restore the separate batch and block axes so tensors can later be selected
    # by their position in the original long sequence.
    scale_inputs_by_scale = [
        einx.id(
            "(batch num_blocks) scale_length vq_dim -> "
            "batch num_blocks scale_length vq_dim",
            scale_input,
            batch=batch_size,
        )
        for scale_input in flattened_scale_inputs
    ]
    targets_by_scale = [
        einx.id(
            "(batch num_blocks) scale_length -> batch num_blocks scale_length",
            scale_targets,
            batch=batch_size,
        )
        for scale_targets in flattened_targets_by_scale
    ]

    # Encode each possible prefix block once; later targets concatenate the
    # required leading blocks without recomputing their representations.
    flattened_prefix_latents = tokenizer.encode(
        einx.id(
            "batch num_blocks block_length -> (batch num_blocks) block_length",
            blocks[:, :-1],
        )
    )
    prefix_latents_by_block = einx.id(
        "(batch num_blocks) block_length vq_dim -> "
        "batch num_blocks block_length vq_dim",
        flattened_prefix_latents,
        batch=batch_size,
    )

    block_predictions = []
    for block_index in range(1, blocks.shape[1]):
        # Concatenate every block before the target into one continuous prefix.
        prefix = einx.id(
            "batch num_blocks block_length vq_dim -> "
            "batch (num_blocks block_length) vq_dim",
            prefix_latents_by_block[:, :block_index],
        )

        # Select the real DNA block, teacher-forced scale inputs, and scale
        # targets that all belong to this block position.
        block_predictions.append(
            BlockPredictionBatch(
                target_ids=blocks[:, block_index],
                prefix=prefix,
                scale_inputs=[
                    scale_input[:, block_index]
                    for scale_input in scale_inputs_by_scale
                ],
                targets_by_scale=[
                    scale_targets[:, block_index]
                    for scale_targets in targets_by_scale
                ],
            )
        )

    return block_predictions


@torch.no_grad()
def rollout_scale_predictions(
    model: NSM,
    tokenizer: VQVAE,
    batch_size: int,
    prefix: Float[Tensor, "batch prefix_length vq_dim"] | None,
) -> list[Int[Tensor, "batch scale_length"]]:
    """Greedily predict a hierarchy, feeding every prediction into the next scale."""
    device = next(model.parameters()).device
    scale_inputs = [
        torch.zeros(
            batch_size,
            scale_length,
            tokenizer.embed_dim,
            device=device,
        )
        for scale_length in tokenizer.scale_lengths[1:]
    ]
    predicted_indices_by_scale = []

    for scale_index in range(len(tokenizer.scale_lengths)):
        # Unpredicted scale sections contain zeros. The model's block-diagonal
        # attention keeps them from affecting the section currently predicted.
        logits = model(scale_inputs, prefix=prefix)
        scale_logits = torch.split(logits, tokenizer.scale_lengths, dim=1)[
            scale_index
        ]
        predicted_indices_by_scale.append(scale_logits.argmax(dim=-1))

        if scale_index < len(scale_inputs):
            scale_inputs[scale_index] = tokenizer.indices_to_next_scale_input(
                predicted_indices_by_scale
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
    num_scales = len(tokenizer.scale_lengths)

    loss_sum = 0.0
    correct_codes_by_scale = [0] * num_scales
    num_codes_by_scale = [0] * num_scales
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
        for prediction in prepare_block_predictions(tokenizer, input_ids):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_mixed_precision,
            ):
                # Teacher forcing supplies the correct preceding-scale
                # reconstructions and predicts every scale in one model call.
                logits = model(prediction.scale_inputs, prefix=prediction.prefix)
                loss, _ = compute_next_scale_loss(
                    logits,
                    prediction.targets_by_scale,
                    scale_loss_weights,
                )

                # Rollout builds each next-scale input from NSM's own previous
                # predictions, then decodes the completed hierarchy into DNA.
                if batch_index < rollout_max_batches:
                    rollout_indices = rollout_scale_predictions(
                        model,
                        tokenizer,
                        batch_size=prediction.target_ids.shape[0],
                        prefix=prediction.prefix,
                    )
                    rollout_logits = tokenizer.decode(rollout_indices)
                    rollout_nucleotide_loss = F.cross_entropy(
                        rollout_logits.flatten(0, 1),
                        prediction.target_ids.flatten(),
                    )

            # Pool exact code matches by scale across all evaluated blocks.
            logits_by_scale = torch.split(logits, tokenizer.scale_lengths, dim=1)
            for scale_index, (scale_logits, scale_targets) in enumerate(
                zip(
                    logits_by_scale,
                    prediction.targets_by_scale,
                    strict=True,
                )
            ):
                correct_codes_by_scale[scale_index] += (
                    (scale_logits.argmax(dim=-1) == scale_targets).sum().item()
                )
                num_codes_by_scale[scale_index] += scale_targets.numel()

            loss_sum += loss.item()
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

    total_correct_codes = sum(correct_codes_by_scale)
    total_codes = sum(num_codes_by_scale)

    # Overall accuracy is position-weighted, so longer scales contribute more
    # code predictions than shorter scales.
    metrics = {
        "loss": loss_sum / num_block_predictions,
        "accuracy": total_correct_codes / total_codes,
    }

    for scale_index, scale_length in enumerate(tokenizer.scale_lengths):
        metrics[f"accuracy_scale_{scale_length}"] = (
            correct_codes_by_scale[scale_index] / num_codes_by_scale[scale_index]
        )

    # Rollout loss averages the block-batch losses, while rollout accuracy pools
    # every decoded nucleotide position.
    if num_rollouts > 0:
        metrics["rollout_nucleotide_loss"] = (
            rollout_nucleotide_loss_sum / num_rollouts
        )
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
    if sequence_length % tokenizer.context_length != 0:
        raise ValueError(
            "data.sequence_length must be divisible by the tokenizer context length."
        )

    # One shared output head requires every tokenizer scale to use the same
    # number of codes, as in the final long-context NSM configuration.
    if len(set(tokenizer.codebook_sizes)) != 1:
        raise ValueError("NSM requires equal codebook sizes across all scales.")

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

    model = NSM(
        vq_embed_dim=tokenizer.embed_dim,
        model_dim=config.model.model_dim,
        scale_lengths=tokenizer.scale_lengths,
        codebook_size=tokenizer.codebook_sizes[0],
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        bias=config.model.bias,
        use_qk_norm=config.model.use_qk_norm,
        rope_base=config.model.rope_base,
        head_num_blocks=config.model.head_num_blocks,
        head_hidden_multiplier=config.model.head_hidden_multiplier,
        input_refinement_kernel_size=config.model.input_refinement_kernel_size,
        max_prefix_length=sequence_length - tokenizer.context_length,
    ).to(device)
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    if distributed_environment.is_main_process:
        print(f"NSM-DNA parameters: {num_parameters / 1e6:.2f}M")
        if wandb_run is not None:
            wandb_run.summary["model/parameters"] = num_parameters

    scale_loss_weights = build_scale_loss_weights(
        tokenizer.scale_lengths,
        config.optimizer.scale_loss_alpha,
        device,
    )

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

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                training_epoch += 1
                train_dataset.set_epoch(training_epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)
            block_predictions = prepare_block_predictions(tokenizer, input_ids)
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
                        logits = training_model(
                            prediction.scale_inputs,
                            prefix=prediction.prefix,
                        )
                        loss, _ = compute_next_scale_loss(
                            logits,
                            prediction.targets_by_scale,
                            scale_loss_weights,
                        )
                        accumulated_loss = loss / num_predictions_per_step
                    accumulated_loss.backward()

                if should_log:
                    mean_loss += loss.detach() / num_predictions_per_step

                    with torch.no_grad():
                        targets = torch.cat(
                            prediction.targets_by_scale,
                            dim=1,
                        )
                        correct_codes += (logits.argmax(dim=-1) == targets).sum()
                        num_codes += targets.numel()

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
                accuracy = (correct_codes / num_codes).item()

                progress_bar.set_postfix(
                    loss=f"{mean_loss.item():.4f}",
                    accuracy=f"{accuracy:.2%}",
                )

                if wandb_run is not None:
                    wandb_metrics = {
                        "train/loss": mean_loss.item(),
                        "train/accuracy": accuracy,
                        "train/gradient_norm": gradient_norm.item(),
                        "train/learning_rate": learning_rate,
                    }
                    wandb_run.log(wandb_metrics, step=step)

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
                    f"{validation_metrics['loss']:.4f}, accuracy "
                    f"{validation_metrics['accuracy']:.2%}, rollout accuracy "
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
                    upload_checkpoint_to_hugging_face(
                        best_checkpoint_path,
                        config.checkpoint.huggingface,
                    )

                if wandb_run is not None:
                    wandb_metrics = {
                        "validation/loss": validation_metrics["loss"],
                        "validation/accuracy": validation_metrics["accuracy"],
                        "rollout/nucleotide_loss": validation_metrics[
                            "rollout_nucleotide_loss"
                        ],
                        "rollout/nucleotide_accuracy": validation_metrics[
                            "rollout_nucleotide_accuracy"
                        ],
                    }
                    for scale_index, scale_length in enumerate(tokenizer.scale_lengths):
                        scale_name = (
                            f"scale_{scale_index + 1:02d}_length_{scale_length}"
                        )
                        wandb_metrics[f"{scale_name}/validation_accuracy"] = (
                            validation_metrics[f"accuracy_scale_{scale_length}"]
                        )

                    wandb_run.log(wandb_metrics, step=step)

            if distributed_environment.is_distributed:
                dist.barrier()

        checkpoints_to_save: list[tuple[str, str | None]] = []
        if step % config.checkpoint.recovery_interval == 0:
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
                    upload_checkpoint_to_hugging_face(
                        checkpoint_path,
                        config.checkpoint.huggingface,
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
        upload_checkpoint_to_hugging_face(
            final_checkpoint_path,
            config.checkpoint.huggingface,
        )

    if wandb_run is not None:
        wandb_run.finish()

    if distributed_environment.is_distributed:
        dist.barrier()
    cleanup_distributed_training()


if __name__ == "__main__":
    main()
