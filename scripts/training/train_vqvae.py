from contextlib import nullcontext
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.models.vqvae import VQVAE
from nsm_dna.training import (
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
    load_training_checkpoint,
    save_training_checkpoint,
)


@torch.no_grad()
def evaluate(
    model: VQVAE,
    data_loader: DataLoader,
    use_mixed_precision: bool,
    partial_reconstruction_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate reconstruction at each independent scale without updating codebooks.

    Evaluate the entire data loader when max_batches is None.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    full_reconstruction_loss_sum = 0.0
    correct_tokens = 0
    num_tokens = 0
    num_batches = 0

    # Decoder quality from each scale on its own.
    reconstruction_loss_sums_by_scale = [0.0] * len(model.scale_lengths)
    correct_tokens_by_scale = [0] * len(model.scale_lengths)

    # Distance between each scale's expanded latent and the encoder latent.
    latent_mse_sums_by_scale = [0.0] * len(model.scale_lengths)

    scale_latent_squared_sums_by_scale = [0.0] * len(model.scale_lengths)

    # Assignment frequencies used to calculate effective codebook size.
    code_counts_by_scale = [
        torch.zeros(codebook_size, dtype=torch.long)
        for codebook_size in model.codebook_sizes
    ]

    # Encoder magnitude, which position-wise LayerNorm should keep stable.
    encoder_latent_squared_sum = 0.0
    num_latent_values = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index == max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            logits, _, indices_by_scale = model(input_ids)
            full_reconstruction_loss = F.cross_entropy(
                logits.flatten(0, 1),
                input_ids.flatten(),
            )

            encoder_latent = model.encode(input_ids)
            scale_latents = model.quantizer.indices_to_scale_latents(
                indices_by_scale
            )

            encoder_latent_squared_sum += encoder_latent.float().square().sum().item()
            num_latent_values += encoder_latent.numel()
            for scale_index, scale_latent in enumerate(scale_latents):
                scale_logits = model.decoder(scale_latent)
                scale_reconstruction_loss = F.cross_entropy(
                    scale_logits.flatten(0, 1),
                    input_ids.flatten(),
                )
                reconstruction_loss_sums_by_scale[scale_index] += (
                    scale_reconstruction_loss.item()
                )
                correct_tokens_by_scale[scale_index] += (
                    (scale_logits.argmax(dim=-1) == input_ids).sum().item()
                )
                latent_mse_sums_by_scale[scale_index] += F.mse_loss(
                    scale_latent.float(),
                    encoder_latent.float(),
                ).item()
                scale_latent_squared_sums_by_scale[scale_index] += (
                    scale_latent.float().square().sum().item()
                )

        for scale_index, scale_indices in enumerate(indices_by_scale):
            code_counts_by_scale[scale_index] += torch.bincount(
                scale_indices.flatten().cpu(),
                minlength=model.codebook_sizes[scale_index],
            )

        full_reconstruction_loss_sum += full_reconstruction_loss.item()
        correct_tokens += (logits.argmax(dim=-1) == input_ids).sum().item()
        num_tokens += input_ids.numel()
        num_batches += 1

    if was_training:
        model.train()

    full_reconstruction_loss = full_reconstruction_loss_sum / num_batches
    partial_reconstruction_loss = sum(
        reconstruction_loss_sums_by_scale[:-1]
    ) / ((len(model.scale_lengths) - 1) * num_batches)
    metrics = {
        "full_reconstruction_loss": full_reconstruction_loss,
        "partial_reconstruction_loss": partial_reconstruction_loss,
        "total_loss": (
            full_reconstruction_loss
            + partial_reconstruction_weight * partial_reconstruction_loss
        ),
        "accuracy": correct_tokens / num_tokens,
        "encoder_latent_rms": (encoder_latent_squared_sum / num_latent_values) ** 0.5,
    }

    scale_metrics = zip(
        model.scale_lengths,
        reconstruction_loss_sums_by_scale,
        correct_tokens_by_scale,
    )
    for scale_index, (scale_length, loss_sum, scale_correct_tokens) in enumerate(
        scale_metrics
    ):
        metrics[f"reconstruction_loss_scale_{scale_length}"] = (
            loss_sum / num_batches
        )
        metrics[f"accuracy_scale_{scale_length}"] = (
            scale_correct_tokens / num_tokens
        )
        metrics[f"latent_mse_scale_{scale_length}"] = (
            latent_mse_sums_by_scale[scale_index] / num_batches
        )
        scale_latent_squared_sum = scale_latent_squared_sums_by_scale[scale_index]
        metrics[f"scale_latent_rms_scale_{scale_length}"] = (
            scale_latent_squared_sum / num_latent_values
        ) ** 0.5

        code_counts = code_counts_by_scale[scale_index].float()
        code_probabilities = code_counts[code_counts > 0] / code_counts.sum()

        # Perplexity is the effective number of codes used and remains informative
        # after the ever-used utilization metric reaches 100%.
        metrics[f"codebook_perplexity_scale_{scale_length}"] = torch.exp(
            -(code_probabilities * code_probabilities.log()).sum()
        ).item()
        codebook_vectors = model.quantizer.codebooks[scale_index].codebook
        metrics[f"codebook_rms_scale_{scale_length}"] = (
            codebook_vectors.float().square().mean().sqrt().item()
        )

    return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="vqvae")
def main(config: DictConfig) -> None:
    distributed_environment = initialize_distributed_training()
    torch.manual_seed(config.run.seed + distributed_environment.rank)
    run_directory = Path(HydraConfig.get().runtime.output_dir)
    total_steps = calculate_training_steps(
        config,
        distributed_environment.world_size,
        int(config.model.context_length),
    )
    if distributed_environment.is_main_process:
        print(f"training for {total_steps:,} optimizer steps")

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
        context_length=config.model.context_length,
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
            context_length=config.model.context_length,
        )
        validation_generator = torch.Generator().manual_seed(config.run.seed)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.data.validation_batch_size,
            collate_fn=collate_dna_sequences,
            num_workers=config.data.num_workers,
            generator=validation_generator,
        )

    # Create the model.
    model = VQVAE(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        latent_length=config.model.latent_length,
        embed_dim=config.model.embed_dim,
        quantization_dim=config.model.quantization_dim,
        num_heads=config.model.num_heads,
        scale_lengths=list(config.model.scale_lengths),
        codebook_sizes=list(config.model.codebook_sizes),
        decoder_num_layers=config.model.decoder_num_layers,
        use_qk_norm=config.model.use_qk_norm,
        bias=config.model.bias,
        rope_base=config.model.rope_base,
        decay=config.model.decay,
        eps=config.model.eps,
    )

    device = distributed_environment.device
    model = model.to(device)
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"

    # Report gradient-trained network parameters and EMA-trained codebooks.
    network_parameters = sum(parameter.numel() for parameter in model.parameters())
    codebook_parameters = model.quantizer.num_codebook_parameters
    total_parameters = network_parameters + codebook_parameters
    if distributed_environment.is_main_process:
        print(
            f"VQ-VAE parameters: {total_parameters / 1e6:.2f}M total "
            f"({network_parameters / 1e6:.2f}M network, "
            f"{codebook_parameters / 1e6:.2f}M codebook)"
        )
        if wandb_run is not None:
            wandb_run.summary["model/network_parameters"] = network_parameters
            wandb_run.summary["model/codebook_parameters"] = codebook_parameters
            wandb_run.summary["model/total_parameters"] = total_parameters

    # Create the optimizer.
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

    # Resume from a checkpoint when one is provided.
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

    # DDP synchronizes gradients. EMA codebook statistics are synchronized
    # separately inside the quantizer, so they do not need per-forward broadcasts.
    # Only the randomly selected partial scale uses its downsampling path each step.
    training_model: VQVAE | DistributedDataParallel = model
    if distributed_environment.is_distributed:
        if device.type == "cuda":
            training_model = DistributedDataParallel(
                model,
                device_ids=[distributed_environment.local_rank],
                output_device=distributed_environment.local_rank,
                find_unused_parameters=True,
            )
        else:
            training_model = DistributedDataParallel(
                model,
                find_unused_parameters=True,
            )

    # Train the model.
    training_epoch = 0
    train_iterator = iter(train_loader)
    progress_bar = tqdm(
        range(start_step + 1, total_steps + 1),
        desc="Training",
        disable=not distributed_environment.is_main_process,
    )
    gradient_accumulation_steps = config.optimizer.gradient_accumulation_steps
    partial_reconstruction_weight = config.model.partial_reconstruction_weight

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        full_reconstruction_loss_sum = 0.0
        partial_reconstruction_loss_sum = 0.0

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                training_epoch += 1
                train_dataset.set_epoch(training_epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)

            is_last_micro_step = micro_step == gradient_accumulation_steps - 1
            if distributed_environment.is_distributed and not is_last_micro_step:
                synchronization_context = training_model.no_sync()
            else:
                synchronization_context = nullcontext()

            # Accumulate gradients locally, synchronizing DDP only on the final
            # micro-step. Dividing the loss preserves the gradient's average scale.
            with synchronization_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_mixed_precision,
                ):
                    full_logits, partial_logits, _ = training_model(
                        input_ids,
                        include_partial_reconstruction=True,
                    )
                    assert partial_logits is not None
                    full_reconstruction_loss = F.cross_entropy(
                        full_logits.flatten(0, 1),
                        input_ids.flatten(),
                    )
                    partial_reconstruction_loss = F.cross_entropy(
                        partial_logits.flatten(0, 1),
                        input_ids.flatten(),
                    )
                    loss = (
                        full_reconstruction_loss
                        + partial_reconstruction_weight
                        * partial_reconstruction_loss
                    )
                    accumulated_loss = loss / gradient_accumulation_steps
                accumulated_loss.backward()

            full_reconstruction_loss_sum += (
                full_reconstruction_loss.item() / gradient_accumulation_steps
            )
            partial_reconstruction_loss_sum += (
                partial_reconstruction_loss.item() / gradient_accumulation_steps
            )

        # Limit unusually large parameter updates before the optimizer step.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()

        # Set the learning rate that will be used by the next optimizer step.
        scheduler.step()

        if step % config.training.log_interval == 0:
            loss_sums = torch.tensor(
                [
                    full_reconstruction_loss_sum,
                    partial_reconstruction_loss_sum,
                ],
                device=device,
            )
            if distributed_environment.is_distributed:
                dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
                loss_sums /= distributed_environment.world_size

            if distributed_environment.is_main_process:
                (
                    full_reconstruction_loss_value,
                    partial_reconstruction_loss_value,
                ) = loss_sums.tolist()
                total_loss_value = (
                    full_reconstruction_loss_value
                    + partial_reconstruction_weight
                    * partial_reconstruction_loss_value
                )
                global_utilization = model.global_utilization.item()
                progress_bar.set_postfix(
                    full_reconstruction_loss=(
                        f"{full_reconstruction_loss_value:.4f}"
                    ),
                    partial_reconstruction_loss=(
                        f"{partial_reconstruction_loss_value:.4f}"
                    ),
                    total_loss=f"{total_loss_value:.4f}",
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/full_reconstruction_loss": (
                                full_reconstruction_loss_value
                            ),
                            "train/partial_reconstruction_loss": (
                                partial_reconstruction_loss_value
                            ),
                            "train/total_loss": total_loss_value,
                            "train/gradient_norm": gradient_norm.item(),
                            "train/learning_rate": learning_rate,
                            "codebook/global_utilization": global_utilization,
                        },
                        step=step,
                    )

        if step % config.evaluation.interval == 0:
            if distributed_environment.is_main_process:
                assert validation_loader is not None
                validation_metrics = evaluate(
                    model,
                    validation_loader,
                    use_mixed_precision,
                    partial_reconstruction_weight,
                    max_batches=config.evaluation.max_batches,
                )
                tqdm.write(
                    f"step {step} validation: "
                    f"full reconstruction loss "
                    f"{validation_metrics['full_reconstruction_loss']:.4f}, "
                    f"partial reconstruction loss "
                    f"{validation_metrics['partial_reconstruction_loss']:.4f}, "
                    f"total loss {validation_metrics['total_loss']:.4f}, "
                    f"accuracy {validation_metrics['accuracy']:.2%}"
                )

                if (
                    validation_metrics["full_reconstruction_loss"]
                    < best_validation_loss
                ):
                    best_validation_loss = validation_metrics[
                        "full_reconstruction_loss"
                    ]
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

                scale_utilizations = {
                    scale_length: utilization.item()
                    for scale_length, utilization in zip(
                        config.model.scale_lengths,
                        model.utilization_by_scale,
                    )
                }
                utilization_by_scale = ", ".join(
                    f"{scale_length}: {utilization:.2%}"
                    for scale_length, utilization in scale_utilizations.items()
                )
                tqdm.write(
                    f"step {step} codebook utilization by scale: {utilization_by_scale}"
                )

                scale_accuracies = ", ".join(
                    f"{scale_length}: "
                    f"{validation_metrics[f'accuracy_scale_{scale_length}']:.2%}"
                    for scale_length in config.model.scale_lengths
                )
                tqdm.write(
                    f"step {step} validation accuracy by scale: "
                    f"{scale_accuracies}"
                )

                if wandb_run is not None:
                    wandb_metrics = {
                        "validation/full_reconstruction_loss": validation_metrics[
                            "full_reconstruction_loss"
                        ],
                        "validation/partial_reconstruction_loss": validation_metrics[
                            "partial_reconstruction_loss"
                        ],
                        "validation/total_loss": validation_metrics["total_loss"],
                        "validation/accuracy": validation_metrics["accuracy"],
                        "validation/encoder_latent_rms": validation_metrics[
                            "encoder_latent_rms"
                        ],
                        "validation/best_reconstruction_loss": best_validation_loss,
                    }
                    scale_metric_names = {
                        "reconstruction_loss": "reconstruction_loss",
                        "accuracy": "accuracy",
                        "latent_mse": "latent_mse",
                        "scale_latent_rms": "scale_latent_rms",
                        "codebook_perplexity": "codebook_perplexity",
                        "codebook_rms": "codebook_rms",
                    }

                    for scale_number, scale_length in enumerate(
                        config.model.scale_lengths,
                        start=1,
                    ):
                        section = f"scale_{scale_number:02d}_length_{scale_length}"
                        for panel_name, metric_name in scale_metric_names.items():
                            wandb_metrics[f"{section}/{panel_name}"] = (
                                validation_metrics[
                                    f"{metric_name}_scale_{scale_length}"
                                ]
                            )

                        if scale_length in scale_utilizations:
                            wandb_metrics[f"{section}/utilization"] = (
                                scale_utilizations[scale_length]
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
