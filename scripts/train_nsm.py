import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import einx
import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from huggingface_hub import HfApi
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.models.nsm import NSM
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.optimization import build_learning_rate_scheduler
from nsm_dna.training import (
    cleanup_distributed_training,
    initialize_distributed_training,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class BlockPredictionBatch:
    """Inputs and targets for predicting one block from its preceding blocks."""

    prefix: Float[Tensor, "batch prefix_length vq_dim"] | None
    scale_inputs: list[Float[Tensor, "batch scale_length vq_dim"]]
    targets_by_scale: list[Int[Tensor, "batch scale_length"]]


def calculate_training_steps(config: DictConfig, world_size: int) -> int:
    """Use max_steps when set; otherwise derive the run length from num_epochs."""
    if config.training.max_steps is not None:
        return int(config.training.max_steps)

    stats_path = Path(config.data.subset_directory) / "subset_stats.json"
    with stats_path.open() as handle:
        split_stats = json.load(handle)["splits"][config.data.train_split]
    num_training_bases = split_stats["bases"]

    bases_per_step = (
        config.data.sequence_length
        * config.data.train_batch_size
        * world_size
        * config.optimizer.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(num_training_bases / bases_per_step)
    return int(config.training.num_epochs) * steps_per_epoch


def load_tokenizer(checkpoint_path: Path, device: torch.device) -> VQVAE:
    """Rebuild the stage-one VQ-VAE from its checkpoint and freeze it."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    tokenizer_config = OmegaConf.create(checkpoint["config"]).model

    tokenizer = VQVAE(
        vocab_size=tokenizer_config.vocab_size,
        context_length=tokenizer_config.context_length,
        embed_dim=tokenizer_config.embed_dim,
        num_heads=tokenizer_config.num_heads,
        scale_lengths=list(tokenizer_config.scale_lengths),
        codebook_sizes=list(tokenizer_config.codebook_sizes),
        encoder_dropout=tokenizer_config.encoder_dropout,
        decoder_dropout=tokenizer_config.decoder_dropout,
        bias=tokenizer_config.bias,
        pre_quant_num_groups=tokenizer_config.pre_quant_num_groups,
        commitment_cost=tokenizer_config.commitment_cost,
        decay=tokenizer_config.decay,
        eps=tokenizer_config.eps,
        refinement_ratio=tokenizer_config.refinement_ratio,
        refinement_kernel_size=tokenizer_config.refinement_kernel_size,
    )
    tokenizer.load_state_dict(checkpoint["model"])
    tokenizer = tokenizer.to(device)

    # Evaluation mode prevents dropout and EMA codebook updates. Disabling
    # gradients makes the boundary between tokenizer and NSM training explicit.
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


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
    next_scale_inputs = tokenizer.quantizer.indices_to_next_scale_inputs(
        targets_by_scale
    )
    return next_scale_inputs, targets_by_scale


@torch.no_grad()
def prepare_block_predictions(
    tokenizer: VQVAE,
    input_ids: Int[Tensor, "batch sequence_length"],
) -> list[BlockPredictionBatch]:
    """Prepare each block as a target conditioned on all preceding blocks."""
    batch_size = input_ids.shape[0]
    blocks = einx.id(
        "batch (num_blocks block_length) -> batch num_blocks block_length",
        input_ids,
        block_length=tokenizer.context_length,
    )

    flattened_blocks = einx.id(
        "batch num_blocks block_length -> (batch num_blocks) block_length",
        blocks,
    )
    scale_inputs_by_scale, targets_by_scale = prepare_nsm_batch(
        tokenizer, flattened_blocks
    )
    scale_inputs_by_scale = [
        einx.id(
            "(batch num_blocks) scale_length vq_dim -> "
            "batch num_blocks scale_length vq_dim",
            x,
            batch=batch_size,
        )
        for x in scale_inputs_by_scale
    ]
    targets_by_scale = [
        einx.id(
            "(batch num_blocks) scale_length -> batch num_blocks scale_length",
            targets,
            batch=batch_size,
        )
        for targets in targets_by_scale
    ]

    # Encode each possible prefix block once; later targets concatenate the
    # required leading blocks without recomputing their representations.
    prefix_latents_by_block = tokenizer.encode(
        einx.id(
            "batch num_blocks block_length -> (batch num_blocks) block_length",
            blocks[:, :-1],
        )
    )
    prefix_latents_by_block = einx.id(
        "(batch num_blocks) block_length vq_dim -> "
        "batch num_blocks block_length vq_dim",
        prefix_latents_by_block,
        batch=batch_size,
    )

    block_predictions = []
    for block_index in range(blocks.shape[1]):
        prefix = None
        if block_index > 0:
            prefix = einx.id(
                "batch num_blocks block_length vq_dim -> "
                "batch (num_blocks block_length) vq_dim",
                prefix_latents_by_block[:, :block_index],
            )

        block_predictions.append(
            BlockPredictionBatch(
                prefix=prefix,
                scale_inputs=[x[:, block_index] for x in scale_inputs_by_scale],
                targets_by_scale=[x[:, block_index] for x in targets_by_scale],
            )
        )

    return block_predictions


@torch.no_grad()
def evaluate(
    model: NSM,
    tokenizer: VQVAE,
    data_loader: DataLoader,
    scale_loss_weights: Tensor,
    use_mixed_precision: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate every block prediction without changing the tokenizer.

    Evaluate the entire data loader when max_batches is None.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    num_scales = len(tokenizer.scale_lengths)

    loss_sum = 0.0
    loss_sums_by_scale = [0.0] * num_scales
    correct_codes_by_scale = [0] * num_scales
    num_codes_by_scale = [0] * num_scales
    num_block_predictions = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index == max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        for prediction in prepare_block_predictions(tokenizer, input_ids):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_mixed_precision,
            ):
                logits = model(prediction.scale_inputs, prefix=prediction.prefix)
                loss, losses_by_scale = compute_next_scale_loss(
                    logits,
                    prediction.targets_by_scale,
                    scale_loss_weights,
                )

            logits_by_scale = torch.split(logits, tokenizer.scale_lengths, dim=1)
            for scale_index, (scale_logits, scale_targets) in enumerate(
                zip(logits_by_scale, prediction.targets_by_scale)
            ):
                loss_sums_by_scale[scale_index] += losses_by_scale[scale_index].item()
                correct_codes_by_scale[scale_index] += (
                    (scale_logits.argmax(dim=-1) == scale_targets).sum().item()
                )
                num_codes_by_scale[scale_index] += scale_targets.numel()

            loss_sum += loss.item()
            num_block_predictions += 1

    if was_training:
        model.train()

    total_correct_codes = sum(correct_codes_by_scale)
    total_codes = sum(num_codes_by_scale)
    metrics = {
        "loss": loss_sum / num_block_predictions,
        "accuracy": total_correct_codes / total_codes,
    }

    for scale_index, scale_length in enumerate(tokenizer.scale_lengths):
        metrics[f"loss_scale_{scale_length}"] = (
            loss_sums_by_scale[scale_index] / num_block_predictions
        )
        metrics[f"accuracy_scale_{scale_length}"] = (
            correct_codes_by_scale[scale_index] / num_codes_by_scale[scale_index]
        )

    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="nsm")
def main(config: DictConfig) -> None:
    distributed_environment = initialize_distributed_training()
    torch.manual_seed(config.run.seed + distributed_environment.rank)
    run_directory = Path(HydraConfig.get().runtime.output_dir)
    device = distributed_environment.device
    total_steps = calculate_training_steps(config, distributed_environment.world_size)
    if distributed_environment.is_main_process:
        print(f"training for {total_steps:,} optimizer steps")

    hugging_face_api = HfApi() if config.checkpoint.huggingface.enabled else None

    def upload_checkpoint(checkpoint_path: Path) -> None:
        if hugging_face_api is None:
            return

        hugging_face_api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=(
                f"{config.checkpoint.huggingface.repository_directory}/"
                f"{checkpoint_path.name}"
            ),
            repo_id=config.checkpoint.huggingface.repository_id,
            repo_type="model",
        )

    # The tokenizer checkpoint owns every tokenizer architectural choice. Keeping
    # those settings out of this config prevents stage 2 from silently rebuilding
    # a tokenizer that differs from the one that produced its code targets.
    tokenizer_checkpoint_path = Path(config.tokenizer_checkpoint)
    tokenizer = load_tokenizer(tokenizer_checkpoint_path, device)
    sequence_length = int(config.data.sequence_length)
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
    if distributed_environment.is_main_process:
        scale_weight_summary = ", ".join(
            f"{scale_length}: {weight.item():.2%}"
            for scale_length, weight in zip(
                tokenizer.scale_lengths,
                scale_loss_weights,
            )
        )
        print(f"loss share by scale length: {scale_weight_summary}")

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
        start_step, best_validation_loss = load_checkpoint(
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
    num_blocks = sequence_length // tokenizer.context_length
    num_scales = len(tokenizer.scale_lengths)

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        should_log = step % config.training.log_interval == 0

        if should_log:
            loss_sum = 0.0
            loss_sums_by_block_and_scale = torch.zeros(
                (num_blocks, num_scales), device=device
            )
            correct_codes_by_block_and_scale = torch.zeros_like(
                loss_sums_by_block_and_scale, dtype=torch.long
            )
            num_codes_by_block_and_scale = torch.zeros_like(
                correct_codes_by_block_and_scale
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
            block_predictions = prepare_block_predictions(tokenizer, input_ids)
            num_accumulated_predictions = gradient_accumulation_steps * len(
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
                        loss, losses_by_scale = compute_next_scale_loss(
                            logits,
                            prediction.targets_by_scale,
                            scale_loss_weights,
                        )
                        accumulated_loss = loss / num_accumulated_predictions
                    accumulated_loss.backward()

                if should_log:
                    loss_sum += loss.item() / num_accumulated_predictions
                    loss_sums_by_block_and_scale[block_index] += (
                        losses_by_scale.detach() / gradient_accumulation_steps
                    )

                    with torch.no_grad():
                        logits_by_scale = torch.split(
                            logits, tokenizer.scale_lengths, dim=1
                        )
                        for scale_index, (scale_logits, scale_targets) in enumerate(
                            zip(logits_by_scale, prediction.targets_by_scale)
                        ):
                            correct_codes_by_block_and_scale[
                                block_index, scale_index
                            ] += (scale_logits.argmax(dim=-1) == scale_targets).sum()
                            num_codes_by_block_and_scale[block_index, scale_index] += (
                                scale_targets.numel()
                            )

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()

        # Set the learning rate that will be used by the next optimizer step.
        scheduler.step()

        if should_log:
            loss_value = torch.tensor(loss_sum, device=device)
            if distributed_environment.is_distributed:
                for values in (
                    loss_value,
                    loss_sums_by_block_and_scale,
                    correct_codes_by_block_and_scale,
                    num_codes_by_block_and_scale,
                ):
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                loss_value /= distributed_environment.world_size
                loss_sums_by_block_and_scale /= distributed_environment.world_size

            if distributed_environment.is_main_process:
                accuracy = (
                    correct_codes_by_block_and_scale.sum()
                    / num_codes_by_block_and_scale.sum()
                ).item()

                progress_bar.set_postfix(
                    loss=f"{loss_value.item():.4f}",
                    accuracy=f"{accuracy:.2%}",
                )

                if wandb_run is not None:
                    wandb_metrics = {
                        "train/loss": loss_value.item(),
                        "train/accuracy": accuracy,
                        "train/gradient_norm": gradient_norm.item(),
                        "train/learning_rate": learning_rate,
                    }
                    accuracies_by_block_and_scale = (
                        correct_codes_by_block_and_scale / num_codes_by_block_and_scale
                    )
                    for scale_index, scale_length in enumerate(tokenizer.scale_lengths):
                        scale = f"scale_{scale_index + 1:02d}_length_{scale_length}"
                        wandb_metrics[f"{scale}/train_loss"] = (
                            loss_sums_by_block_and_scale[:, scale_index].mean().item()
                        )
                        wandb_metrics[f"{scale}/train_accuracy"] = (
                            correct_codes_by_block_and_scale[:, scale_index].sum()
                            / num_codes_by_block_and_scale[:, scale_index].sum()
                        ).item()

                        for block_index in range(num_blocks):
                            prefix_length = block_index * tokenizer.context_length
                            block = f"block_{block_index + 1:02d}_prefix_{prefix_length:03d}"
                            wandb_metrics[f"{block}/{scale}_train_loss"] = (
                                loss_sums_by_block_and_scale[
                                    block_index, scale_index
                                ].item()
                            )
                            wandb_metrics[f"{block}/{scale}_train_accuracy"] = (
                                accuracies_by_block_and_scale[
                                    block_index, scale_index
                                ].item()
                            )

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
                )
                tqdm.write(
                    f"step {step} validation: loss "
                    f"{validation_metrics['loss']:.4f}, accuracy "
                    f"{validation_metrics['accuracy']:.2%}"
                )

                validation_loss = validation_metrics["loss"]
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_checkpoint_path = save_checkpoint(
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
                    upload_checkpoint(best_checkpoint_path)

                scale_accuracies = ", ".join(
                    f"{scale_length}: "
                    f"{validation_metrics[f'accuracy_scale_{scale_length}']:.2%}"
                    for scale_length in tokenizer.scale_lengths
                )
                tqdm.write(
                    f"step {step} validation accuracy by scale: {scale_accuracies}"
                )

                if wandb_run is not None:
                    wandb_metrics = {
                        "validation/loss": validation_metrics["loss"],
                        "validation/accuracy": validation_metrics["accuracy"],
                        "validation/best_loss": best_validation_loss,
                    }
                    for scale_index, scale_length in enumerate(tokenizer.scale_lengths):
                        section = f"scale_{scale_index + 1:02d}_length_{scale_length}"
                        wandb_metrics[f"{section}/validation_loss"] = (
                            validation_metrics[f"loss_scale_{scale_length}"]
                        )
                        wandb_metrics[f"{section}/validation_accuracy"] = (
                            validation_metrics[f"accuracy_scale_{scale_length}"]
                        )

                    wandb_run.log(wandb_metrics, step=step)

            if distributed_environment.is_distributed:
                dist.barrier()

        save_recovery_checkpoint = step % config.checkpoint.recovery_interval == 0
        save_milestone_checkpoint = step % config.checkpoint.milestone_interval == 0
        if save_recovery_checkpoint or save_milestone_checkpoint:
            if distributed_environment.is_main_process:
                if save_recovery_checkpoint:
                    latest_checkpoint_path = save_checkpoint(
                        run_directory,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        best_validation_loss,
                        checkpoint_name="latest.pt",
                    )
                    tqdm.write(f"saved recovery checkpoint: {latest_checkpoint_path}")
                    upload_checkpoint(latest_checkpoint_path)

                if save_milestone_checkpoint:
                    milestone_checkpoint_path = save_checkpoint(
                        run_directory,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        best_validation_loss,
                    )
                    tqdm.write(
                        f"saved milestone checkpoint: {milestone_checkpoint_path}"
                    )
                    upload_checkpoint(milestone_checkpoint_path)

            if distributed_environment.is_distributed:
                dist.barrier()

    if distributed_environment.is_main_process:
        final_checkpoint_path = save_checkpoint(
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
        upload_checkpoint(final_checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()

    if distributed_environment.is_distributed:
        dist.barrier()
    cleanup_distributed_training()


if __name__ == "__main__":
    main()
