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
    sequence_length: int | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate full and cumulative reconstruction without updating codebooks.

    Evaluate the entire data loader when max_batches is None.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    reconstruction_loss_sum = 0.0
    vq_loss_sum = 0.0
    correct_tokens = 0
    num_tokens = 0
    num_batches = 0

    # Decoder quality after adding each successive scale.
    sequence_length = sequence_length or model.context_length
    scale_lengths = model.scale_lengths_for_input_length(sequence_length)
    reconstruction_loss_sums_by_scale = [0.0] * len(scale_lengths)
    correct_tokens_by_scale = [0] * len(scale_lengths)

    # Distance between each cumulative quantized latent and the encoder latent.
    latent_mse_sums_by_scale = [0.0] * len(scale_lengths)

    # Squared values used to measure the magnitude added by each scale.
    contribution_squared_sums_by_scale = [0.0] * len(scale_lengths)

    # Assignment frequencies used to calculate effective codebook size.
    code_counts_by_scale = [
        torch.zeros(codebook_size, dtype=torch.long)
        for codebook_size in model.codebook_sizes
    ]

    # Encoder magnitude, which GroupNorm should keep stable.
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
            logits, _, vq_loss, indices_by_scale = model(input_ids)
            reconstruction_loss = F.cross_entropy(
                logits.flatten(0, 1),
                input_ids.flatten(),
            )

            encoder_latent = model.encode(input_ids)
            cumulative_latents = model.quantizer.indices_to_cumulative_latents(
                indices_by_scale
            )
            previous_latent = torch.zeros_like(cumulative_latents[0])

            encoder_latent_squared_sum += encoder_latent.float().square().sum().item()
            num_latent_values += encoder_latent.numel()
            for scale_index, cumulative_latent in enumerate(cumulative_latents):
                scale_logits = model.decoder(cumulative_latent)
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
                    cumulative_latent.float(),
                    encoder_latent.float(),
                ).item()

                scale_contribution = cumulative_latent - previous_latent
                contribution_squared_sums_by_scale[scale_index] += (
                    scale_contribution.float().square().sum().item()
                )
                previous_latent = cumulative_latent

        for scale_index, scale_indices in enumerate(indices_by_scale):
            code_counts_by_scale[scale_index] += torch.bincount(
                scale_indices.flatten().cpu(),
                minlength=model.codebook_sizes[scale_index],
            )

        reconstruction_loss_sum += reconstruction_loss.item()
        vq_loss_sum += vq_loss.item()
        correct_tokens += (logits.argmax(dim=-1) == input_ids).sum().item()
        num_tokens += input_ids.numel()
        num_batches += 1

    if was_training:
        model.train()

    reconstruction_loss = reconstruction_loss_sum / num_batches
    vq_loss = vq_loss_sum / num_batches
    metrics = {
        "reconstruction_loss": reconstruction_loss,
        "vq_loss": vq_loss,
        "total_loss": reconstruction_loss + vq_loss,
        "accuracy": correct_tokens / num_tokens,
        "encoder_latent_rms": (encoder_latent_squared_sum / num_latent_values) ** 0.5,
    }

    scale_metrics = zip(
        scale_lengths,
        reconstruction_loss_sums_by_scale,
        correct_tokens_by_scale,
    )
    for scale_index, (scale_length, loss_sum, scale_correct_tokens) in enumerate(
        scale_metrics
    ):
        metrics[f"cumulative_reconstruction_loss_scale_{scale_length}"] = (
            loss_sum / num_batches
        )
        metrics[f"cumulative_accuracy_scale_{scale_length}"] = (
            scale_correct_tokens / num_tokens
        )
        metrics[f"cumulative_latent_mse_scale_{scale_length}"] = (
            latent_mse_sums_by_scale[scale_index] / num_batches
        )
        contribution_squared_sum = contribution_squared_sums_by_scale[scale_index]
        metrics[f"contribution_rms_scale_{scale_length}"] = (
            contribution_squared_sum / num_latent_values
        ) ** 0.5

        code_counts = code_counts_by_scale[scale_index].float()
        code_probabilities = code_counts[code_counts > 0] / code_counts.sum()

        # Perplexity is the effective number of codes used and remains informative
        # after the cumulative ever-used utilization metric reaches 100%.
        metrics[f"codebook_perplexity_scale_{scale_length}"] = torch.exp(
            -(code_probabilities * code_probabilities.log()).sum()
        ).item()
        metrics[f"codebook_rms_scale_{scale_length}"] = (
            model.quantizer.codebooks[scale_index]
            .codebook.float()
            .square()
            .mean()
            .sqrt()
            .item()
        )

    return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="vqvae")
def main(config: DictConfig) -> None:
    distributed_environment = initialize_distributed_training()
    torch.manual_seed(config.run.seed + distributed_environment.rank)
    run_directory = Path(HydraConfig.get().runtime.output_dir)
    sequence_lengths = [int(length) for length in config.data.sequence_lengths]
    train_batch_sizes = [
        int(batch_size) for batch_size in config.data.train_batch_sizes
    ]
    validation_batch_sizes = [
        int(batch_size) for batch_size in config.data.validation_batch_sizes
    ]
    if not (
        len(sequence_lengths)
        == len(train_batch_sizes)
        == len(validation_batch_sizes)
    ):
        raise ValueError(
            "Each sequence length must have one training and validation batch size."
        )
    if int(config.model.context_length) not in sequence_lengths:
        raise ValueError("The reference context length must be a training length.")
    reference_index = sequence_lengths.index(int(config.model.context_length))
    total_steps = calculate_training_steps(
        config,
        distributed_environment.world_size,
        int(config.model.context_length),
        batch_size=train_batch_sizes[reference_index],
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
    train_datasets = {}
    train_loaders = {}
    for sequence_length, batch_size in zip(sequence_lengths, train_batch_sizes):
        train_dataset = load_gtdb_dataset(
            subset_directory=Path(config.data.subset_directory),
            split=config.data.train_split,
            context_length=sequence_length,
            shuffle_buffer_size=config.data.shuffle_buffer_size,
            seed=config.run.seed + sequence_length,
            rank=distributed_environment.rank,
            world_size=distributed_environment.world_size,
        )
        train_generator = torch.Generator().manual_seed(
            config.run.seed + distributed_environment.rank + sequence_length
        )
        train_datasets[sequence_length] = train_dataset
        train_loaders[sequence_length] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_dna_sequences,
            num_workers=config.data.num_workers,
            generator=train_generator,
        )

    validation_loaders = None
    if distributed_environment.is_main_process:
        validation_loaders = {}
        for sequence_length, batch_size in zip(
            sequence_lengths,
            validation_batch_sizes,
        ):
            validation_dataset = load_gtdb_dataset(
                subset_directory=Path(config.data.subset_directory),
                split=config.data.validation_split,
                context_length=sequence_length,
            )
            validation_generator = torch.Generator().manual_seed(
                config.run.seed + sequence_length
            )
            validation_loaders[sequence_length] = DataLoader(
                validation_dataset,
                batch_size=batch_size,
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
        max_context_length=config.model.max_context_length,
        encoder_dropout=config.model.encoder_dropout,
        decoder_dropout=config.model.decoder_dropout,
        decoder_num_layers=config.model.decoder_num_layers,
        bias=config.model.bias,
        rope_base=config.model.rope_base,
        pre_quant_num_groups=config.model.pre_quant_num_groups,
        commitment_cost=config.model.commitment_cost,
        decay=config.model.decay,
        eps=config.model.eps,
        refinement_ratio=config.model.refinement_ratio,
        refinement_kernel_size=config.model.refinement_kernel_size,
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
    encoder_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            encoder_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters},
            {
                "params": encoder_parameters,
                "lr": config.optimizer.encoder_learning_rate,
            },
        ],
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
    training_model: VQVAE | DistributedDataParallel = model
    if distributed_environment.is_distributed:
        if device.type == "cuda":
            training_model = DistributedDataParallel(
                model,
                device_ids=[distributed_environment.local_rank],
                output_device=distributed_environment.local_rank,
            )
        else:
            training_model = DistributedDataParallel(model)

    # Train the model.
    training_epochs = {sequence_length: 0 for sequence_length in sequence_lengths}
    train_iterators = {
        sequence_length: iter(train_loaders[sequence_length])
        for sequence_length in sequence_lengths
    }
    progress_bar = tqdm(
        range(start_step + 1, total_steps + 1),
        desc="Training",
        disable=not distributed_environment.is_main_process,
    )
    gradient_accumulation_steps = config.optimizer.gradient_accumulation_steps
    partial_quantizer_weight = config.model.partial_reconstruction_quantizer_weight
    partial_decoder_weight = config.model.partial_reconstruction_decoder_weight
    partial_latent_gradient_scale = (
        partial_quantizer_weight / partial_decoder_weight
    )
    length_generator = torch.Generator().manual_seed(config.run.seed)
    if start_step > 0:
        torch.randint(
            len(sequence_lengths),
            (start_step,),
            generator=length_generator,
        )

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        reconstruction_loss_sum = 0.0
        partial_reconstruction_loss_sum = 0.0
        vq_loss_sum = 0.0
        length_index = torch.randint(
            len(sequence_lengths),
            (),
            generator=length_generator,
        ).item()
        sequence_length = sequence_lengths[length_index]

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterators[sequence_length])
            except StopIteration:
                training_epochs[sequence_length] += 1
                train_datasets[sequence_length].set_epoch(
                    training_epochs[sequence_length]
                )
                train_iterators[sequence_length] = iter(
                    train_loaders[sequence_length]
                )
                batch = next(train_iterators[sequence_length])

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
                    logits, partial_logits, vq_loss, _ = training_model(
                        input_ids,
                        include_partial_reconstruction=True,
                        partial_latent_gradient_scale=partial_latent_gradient_scale,
                    )
                    assert partial_logits is not None
                    reconstruction_loss = F.cross_entropy(
                        logits.flatten(0, 1),
                        input_ids.flatten(),
                    )
                    partial_reconstruction_loss = F.cross_entropy(
                        partial_logits.flatten(0, 1),
                        input_ids.flatten(),
                    )
                    loss = (
                        reconstruction_loss
                        + partial_decoder_weight * partial_reconstruction_loss
                        + vq_loss
                    )
                    accumulated_loss = loss / gradient_accumulation_steps
                accumulated_loss.backward()

            reconstruction_loss_sum += (
                reconstruction_loss.item() / gradient_accumulation_steps
            )
            partial_reconstruction_loss_sum += (
                partial_reconstruction_loss.item() / gradient_accumulation_steps
            )
            vq_loss_sum += vq_loss.item() / gradient_accumulation_steps

        # Limit unusually large parameter updates before the optimizer step.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        encoder_learning_rate = optimizer.param_groups[1]["lr"]
        optimizer.step()

        # Set the learning rate that will be used by the next optimizer step.
        scheduler.step()

        if step % config.training.log_interval == 0:
            loss_sums = torch.tensor(
                [
                    reconstruction_loss_sum,
                    partial_reconstruction_loss_sum,
                    vq_loss_sum,
                ],
                device=device,
            )
            if distributed_environment.is_distributed:
                dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
                loss_sums /= distributed_environment.world_size

            if distributed_environment.is_main_process:
                (
                    reconstruction_loss_value,
                    partial_reconstruction_loss_value,
                    vq_loss_value,
                ) = loss_sums.tolist()
                total_loss_value = (
                    reconstruction_loss_value
                    + partial_decoder_weight * partial_reconstruction_loss_value
                    + vq_loss_value
                )
                global_utilization = model.global_utilization.item()
                progress_bar.set_postfix(
                    reconstruction_loss=f"{reconstruction_loss_value:.4f}",
                    partial_reconstruction_loss=(
                        f"{partial_reconstruction_loss_value:.4f}"
                    ),
                    vq_loss=f"{vq_loss_value:.4f}",
                    total_loss=f"{total_loss_value:.4f}",
                )

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/reconstruction_loss": reconstruction_loss_value,
                            "train/partial_reconstruction_loss": (
                                partial_reconstruction_loss_value
                            ),
                            "train/vq_loss": vq_loss_value,
                            "train/total_loss": total_loss_value,
                            "train/gradient_norm": gradient_norm.item(),
                            "train/learning_rate": learning_rate,
                            "train/encoder_learning_rate": encoder_learning_rate,
                            "train/sequence_length": sequence_length,
                            "codebook/global_utilization": global_utilization,
                        },
                        step=step,
                    )

        if step % config.evaluation.interval == 0:
            if distributed_environment.is_main_process:
                assert validation_loaders is not None
                validation_metrics_by_length = {
                    sequence_length: evaluate(
                        model,
                        validation_loaders[sequence_length],
                        use_mixed_precision,
                        sequence_length=sequence_length,
                        max_batches=config.evaluation.max_batches,
                    )
                    for sequence_length in sequence_lengths
                }
                for sequence_length, validation_metrics in (
                    validation_metrics_by_length.items()
                ):
                    tqdm.write(
                        f"step {step} validation length {sequence_length}: "
                        f"reconstruction loss "
                        f"{validation_metrics['reconstruction_loss']:.4f}, "
                        f"VQ loss {validation_metrics['vq_loss']:.4f}, "
                        f"total loss {validation_metrics['total_loss']:.4f}, "
                        f"accuracy {validation_metrics['accuracy']:.2%}"
                    )

                mean_validation_reconstruction_loss = sum(
                    metrics["reconstruction_loss"]
                    for metrics in validation_metrics_by_length.values()
                ) / len(validation_metrics_by_length)
                if mean_validation_reconstruction_loss < best_validation_loss:
                    best_validation_loss = mean_validation_reconstruction_loss
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

                scale_utilizations = [
                    utilization.item() for utilization in model.utilization_by_scale
                ]
                utilization_by_scale = ", ".join(
                    f"codebook {scale_index}: {utilization:.2%}"
                    for scale_index, utilization in enumerate(
                        scale_utilizations,
                        start=1,
                    )
                )
                tqdm.write(
                    f"step {step} codebook utilization by scale: {utilization_by_scale}"
                )

                for sequence_length, validation_metrics in (
                    validation_metrics_by_length.items()
                ):
                    runtime_scale_lengths = model.scale_lengths_for_input_length(
                        sequence_length
                    )
                    cumulative_accuracies = ", ".join(
                        f"{scale_length}: "
                        f"{validation_metrics[f'cumulative_accuracy_scale_{scale_length}']:.2%}"
                        for scale_length in runtime_scale_lengths
                    )
                    tqdm.write(
                        f"step {step} length {sequence_length} cumulative "
                        f"validation accuracy by scale: {cumulative_accuracies}"
                    )

                if wandb_run is not None:
                    wandb_metrics = {
                        "validation/mean_reconstruction_loss": (
                            mean_validation_reconstruction_loss
                        ),
                        "validation/best_mean_reconstruction_loss": (
                            best_validation_loss
                        ),
                    }
                    scale_metric_names = {
                        "reconstruction_loss": "cumulative_reconstruction_loss",
                        "accuracy": "cumulative_accuracy",
                        "latent_mse": "cumulative_latent_mse",
                        "contribution_rms": "contribution_rms",
                        "codebook_perplexity": "codebook_perplexity",
                        "codebook_rms": "codebook_rms",
                    }

                    for sequence_length, validation_metrics in (
                        validation_metrics_by_length.items()
                    ):
                        length_section = f"validation_length_{sequence_length}"
                        for metric_name in (
                            "reconstruction_loss",
                            "vq_loss",
                            "total_loss",
                            "accuracy",
                            "encoder_latent_rms",
                        ):
                            wandb_metrics[f"{length_section}/{metric_name}"] = (
                                validation_metrics[metric_name]
                            )

                        runtime_scale_lengths = model.scale_lengths_for_input_length(
                            sequence_length
                        )
                        for scale_number, scale_length in enumerate(
                            runtime_scale_lengths,
                            start=1,
                        ):
                            scale_section = (
                                f"{length_section}/scale_{scale_number:02d}_"
                                f"length_{scale_length}"
                            )
                            for panel_name, metric_name in scale_metric_names.items():
                                wandb_metrics[f"{scale_section}/{panel_name}"] = (
                                    validation_metrics[
                                        f"{metric_name}_scale_{scale_length}"
                                    ]
                                )

                    for scale_number, utilization in enumerate(
                        scale_utilizations,
                        start=1,
                    ):
                        wandb_metrics[
                            f"codebook_{scale_number:02d}/utilization"
                        ] = utilization

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
