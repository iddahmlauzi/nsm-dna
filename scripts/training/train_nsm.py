from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from nsm_dna.data import collate_dna_sequences, load_gtdb_dataset
from nsm_dna.models.end_to_end import NSMDNA, NSMDNAOutput
from nsm_dna.training import (
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
    load_training_checkpoint,
    save_training_checkpoint,
)
from nsm_dna.triplet_analysis import assign_triplets, code_table_rows


@dataclass
class Losses:
    total: Tensor
    reconstruction: Tensor
    partial_reconstruction: Tensor
    hierarchy: Tensor
    nucleotide: Tensor


def build_scale_loss_weights(
    scale_lengths: list[int],
    scale_loss_alpha: float,
    device: torch.device,
) -> Tensor:
    """Weight scales equally at alpha 1 and positions equally at alpha 0."""
    lengths = torch.tensor(scale_lengths, dtype=torch.float32, device=device)
    weights = lengths.pow(1.0 - scale_loss_alpha)
    return weights / weights.sum()


def compute_losses(
    output: NSMDNAOutput,
    target_ids: Tensor,
    scale_loss_weights: Tensor,
    partial_reconstruction_weight: float,
) -> Losses:
    reconstruction = F.cross_entropy(
        output.reconstruction_logits.flatten(0, 1),
        target_ids.flatten(),
    )
    partial_reconstruction = (
        F.cross_entropy(
            output.partial_reconstruction_logits.flatten(0, 1),
            target_ids.flatten(),
        )
        if output.partial_reconstruction_logits is not None
        else reconstruction.new_zeros(())
    )
    hierarchy_by_scale = torch.stack(
        [
            F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            for logits, targets in zip(
                output.hierarchy_logits,
                output.target_indices,
                strict=True,
            )
        ]
    )
    hierarchy = (hierarchy_by_scale * scale_loss_weights).sum()
    nucleotide = F.cross_entropy(
        output.nucleotide_logits.flatten(0, 1),
        target_ids.flatten(),
    )
    total = (
        reconstruction
        + partial_reconstruction_weight * partial_reconstruction
        + hierarchy
        + nucleotide
    )
    return Losses(
        total=total,
        reconstruction=reconstruction,
        partial_reconstruction=partial_reconstruction,
        hierarchy=hierarchy,
        nucleotide=nucleotide,
    )


def split_blocks(input_ids: Tensor, block_length: int) -> tuple[Tensor, Tensor]:
    """Split each sequence into one prefix block and one target block."""
    return input_ids[:, :block_length], input_ids[:, block_length:]


@torch.no_grad()
def evaluate(
    model: NSMDNA,
    data_loader: DataLoader,
    scale_loss_weights: Tensor,
    partial_reconstruction_weight: float,
    use_mixed_precision: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loss_sums = torch.zeros(5, device=device)
    correct_reconstructions = 0
    correct_nucleotides = 0
    num_nucleotides = 0
    correct_codes_by_scale = [0] * len(model.tokenizer.scale_lengths)
    num_codes_by_scale = [0] * len(model.tokenizer.scale_lengths)
    code_counts_by_scale = [
        torch.zeros(codebook_size, dtype=torch.long, device=device)
        for codebook_size in model.tokenizer.codebook_sizes
    ]
    num_batches = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index == max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        prefix_ids, target_ids = split_blocks(
            input_ids,
            model.tokenizer.context_length,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            output = model(
                prefix_ids,
                target_ids,
                include_partial_reconstruction=False,
            )
            losses = compute_losses(
                output,
                target_ids,
                scale_loss_weights,
                partial_reconstruction_weight,
            )
            if len(output.target_indices) > 1:
                partial_losses = [
                    F.cross_entropy(
                        logits.flatten(0, 1),
                        target_ids.flatten(),
                    )
                    for logits in model.tokenizer.decode_scales(
                        output.target_indices[:-1]
                    )
                ]
                losses.partial_reconstruction = torch.stack(partial_losses).mean()
                losses.total += (
                    partial_reconstruction_weight
                    * losses.partial_reconstruction
                )

        loss_sums += torch.stack(
            [
                losses.total,
                losses.reconstruction,
                losses.partial_reconstruction,
                losses.hierarchy,
                losses.nucleotide,
            ]
        )
        correct_reconstructions += (
            output.reconstruction_logits.argmax(dim=-1) == target_ids
        ).sum().item()
        correct_nucleotides += (
            output.nucleotide_logits.argmax(dim=-1) == target_ids
        ).sum().item()
        scale_predictions = zip(
            output.hierarchy_logits,
            output.target_indices,
            strict=True,
        )
        for scale_index, (logits, targets) in enumerate(scale_predictions):
            correct_codes_by_scale[scale_index] += (
                logits.argmax(dim=-1) == targets
            ).sum().item()
            num_codes_by_scale[scale_index] += targets.numel()
            code_counts_by_scale[scale_index] += torch.bincount(
                targets.flatten(),
                minlength=model.tokenizer.codebook_sizes[scale_index],
            )
        num_nucleotides += target_ids.numel()
        num_batches += 1

    model.train(was_training)
    mean_losses = loss_sums / num_batches
    metrics = {
        "loss": mean_losses[0].item(),
        "reconstruction_loss": mean_losses[1].item(),
        "partial_reconstruction_loss": mean_losses[2].item(),
        "hierarchy_loss": mean_losses[3].item(),
        "nucleotide_loss": mean_losses[4].item(),
        "reconstruction_accuracy": correct_reconstructions / num_nucleotides,
        "hierarchy_accuracy": (
            sum(correct_codes_by_scale) / sum(num_codes_by_scale)
        ),
        "nucleotide_accuracy": correct_nucleotides / num_nucleotides,
    }
    for scale_number, (scale_length, codebook_size) in enumerate(
        zip(
            model.tokenizer.scale_lengths,
            model.tokenizer.codebook_sizes,
            strict=True,
        ),
        start=1,
    ):
        counts = code_counts_by_scale[scale_number - 1].float()
        probabilities = counts[counts > 0] / counts.sum()
        perplexity = torch.exp(-(probabilities * probabilities.log()).sum()).item()
        section = f"scale_{scale_number:02d}_length_{scale_length}"
        # Normalize the effective number of used codes so every scale is on a
        # comparable zero-to-one range despite different codebook sizes.
        metrics[f"{section}/codebook_usage"] = perplexity / codebook_size
        metrics[f"{section}/prediction_accuracy"] = (
            correct_codes_by_scale[scale_number - 1]
            / num_codes_by_scale[scale_number - 1]
        )
    return metrics


@hydra.main(version_base=None, config_path="../../configs", config_name="nsm")
def main(config: DictConfig) -> None:
    distributed = initialize_distributed_training()
    torch.manual_seed(config.run.seed + distributed.rank)
    device = distributed.device
    run_directory = Path(HydraConfig.get().runtime.output_dir)
    sequence_length = int(config.data.sequence_length)
    total_steps = calculate_training_steps(
        config,
        distributed.world_size,
        sequence_length,
    )

    model = NSMDNA.from_config(config).to(device)
    block_length = model.tokenizer.context_length
    if sequence_length != 2 * block_length:
        raise ValueError("data.sequence_length must contain two tokenizer blocks.")

    if distributed.is_main_process:
        num_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"training for {total_steps:,} optimizer steps")
        print(f"NSM-DNA parameters: {num_parameters / 1e6:.2f}M")

    wandb_run = None
    if config.wandb.enabled and distributed.is_main_process:
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
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.train_batch_size,
        collate_fn=collate_dna_sequences,
        num_workers=config.data.num_workers,
        generator=torch.Generator().manual_seed(config.run.seed + distributed.rank),
    )

    validation_loader = None
    if distributed.is_main_process:
        validation_dataset = load_gtdb_dataset(
            subset_directory=Path(config.data.subset_directory),
            split=config.data.validation_split,
            context_length=sequence_length,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.data.validation_batch_size,
            collate_fn=collate_dna_sequences,
            num_workers=config.data.num_workers,
            generator=torch.Generator().manual_seed(config.run.seed),
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
        if distributed.is_main_process:
            print(f"resumed from checkpoint: {checkpoint_path} (step {start_step})")

    training_model: NSMDNA | DistributedDataParallel = model
    if distributed.is_distributed:
        ddp_options = {"broadcast_buffers": False}
        if device.type == "cuda":
            ddp_options.update(
                device_ids=[distributed.local_rank],
                output_device=distributed.local_rank,
            )
        training_model = DistributedDataParallel(model, **ddp_options)

    scale_loss_weights = build_scale_loss_weights(
        model.tokenizer.scale_lengths,
        config.objective.scale_loss_alpha,
        device,
    )
    partial_reconstruction_weight = float(
        config.objective.partial_reconstruction_weight
    )
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"
    gradient_accumulation_steps = int(config.optimizer.gradient_accumulation_steps)
    training_epoch = 0
    train_iterator = iter(train_loader)
    progress_bar = tqdm(
        range(start_step + 1, total_steps + 1),
        desc="Training",
        disable=not distributed.is_main_process,
    )

    for step in progress_bar:
        optimizer.zero_grad(set_to_none=True)
        should_log = step % config.training.log_interval == 0
        loss_sums = torch.zeros(5, device=device)
        nucleotide_counts = torch.zeros(3, dtype=torch.long, device=device)

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                training_epoch += 1
                train_dataset.set_epoch(training_epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)
            prefix_ids, target_ids = split_blocks(input_ids, block_length)
            is_last_micro_step = micro_step == gradient_accumulation_steps - 1
            synchronization = (
                nullcontext()
                if not distributed.is_distributed or is_last_micro_step
                else training_model.no_sync()
            )

            with synchronization:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_mixed_precision,
                ):
                    output = training_model(prefix_ids, target_ids)
                    losses = compute_losses(
                        output,
                        target_ids,
                        scale_loss_weights,
                        partial_reconstruction_weight,
                    )
                    accumulated_loss = losses.total / gradient_accumulation_steps
                accumulated_loss.backward()

            if should_log:
                loss_sums += torch.stack(
                    [
                        losses.total.detach(),
                        losses.reconstruction.detach(),
                        losses.partial_reconstruction.detach(),
                        losses.hierarchy.detach(),
                        losses.nucleotide.detach(),
                    ]
                ) / gradient_accumulation_steps
                with torch.no_grad():
                    nucleotide_counts[0] += (
                        output.reconstruction_logits.argmax(dim=-1) == target_ids
                    ).sum()
                    nucleotide_counts[1] += (
                        output.nucleotide_logits.argmax(dim=-1) == target_ids
                    ).sum()
                    nucleotide_counts[2] += target_ids.numel()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        if should_log:
            if distributed.is_distributed:
                dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
                dist.all_reduce(nucleotide_counts, op=dist.ReduceOp.SUM)
                loss_sums /= distributed.world_size
            if distributed.is_main_process:
                progress_bar.set_postfix(loss=f"{loss_sums[0].item():.4f}")
                if wandb_run is not None:
                    names = (
                        "loss",
                        "reconstruction_loss",
                        "partial_reconstruction_loss",
                        "hierarchy_loss",
                        "nucleotide_loss",
                    )
                    metrics = {
                        f"train/{name}": value.item()
                        for name, value in zip(names, loss_sums, strict=True)
                    }
                    metrics["train/gradient_norm"] = gradient_norm.item()
                    metrics["train/learning_rate"] = learning_rate
                    metrics["train/reconstruction_accuracy"] = (
                        nucleotide_counts[0] / nucleotide_counts[2]
                    ).item()
                    metrics["train/nucleotide_accuracy"] = (
                        nucleotide_counts[1] / nucleotide_counts[2]
                    ).item()
                    wandb_run.log(metrics, step=step)

        if step % config.evaluation.interval == 0:
            if distributed.is_main_process:
                assert validation_loader is not None
                metrics = evaluate(
                    model,
                    validation_loader,
                    scale_loss_weights,
                    partial_reconstruction_weight,
                    use_mixed_precision,
                    max_batches=config.evaluation.max_batches,
                )
                tqdm.write(
                    f"step {step} validation: loss {metrics['loss']:.4f}, "
                    f"nucleotide accuracy {metrics['nucleotide_accuracy']:.2%}"
                )
                if metrics["loss"] < best_validation_loss:
                    best_validation_loss = metrics["loss"]
                    path = save_training_checkpoint(
                        run_directory,
                        model,
                        optimizer,
                        scheduler,
                        config,
                        step,
                        best_validation_loss,
                        checkpoint_name="best.pt",
                    )
                    tqdm.write(f"saved best checkpoint: {path}")
                if wandb_run is not None:
                    wandb_metrics = {
                        (
                            name
                            if name.startswith("scale_")
                            else f"validation/{name}"
                        ): value
                        for name, value in metrics.items()
                    }
                    triplet_assignments = assign_triplets(
                        model.tokenizer,
                        device,
                    )
                    wandb_metrics["tokenizer/triplet_assignments"] = wandb.Table(
                        columns=["code", "triplets", "amino acids"],
                        data=code_table_rows(
                            triplet_assignments,
                            model.tokenizer.codebook_sizes[-1],
                        ),
                    )
                    wandb_run.log(wandb_metrics, step=step)
            if distributed.is_distributed:
                dist.barrier()

        if step % config.checkpoint.interval == 0:
            if distributed.is_main_process:
                path = save_training_checkpoint(
                    run_directory,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    best_validation_loss,
                    checkpoint_name="latest.pt",
                )
                tqdm.write(f"saved checkpoint: {path}")
            if distributed.is_distributed:
                dist.barrier()

    if distributed.is_main_process:
        path = save_training_checkpoint(
            run_directory,
            model,
            optimizer,
            scheduler,
            config,
            total_steps,
            best_validation_loss,
            checkpoint_name="final.pt",
        )
        tqdm.write(f"saved final checkpoint: {path}")
        if wandb_run is not None:
            wandb_run.finish()

    if distributed.is_distributed:
        dist.barrier()
    cleanup_distributed_training()


if __name__ == "__main__":
    main()
