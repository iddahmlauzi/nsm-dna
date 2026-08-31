from contextlib import nullcontext
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
from nsm_dna.models.next_token import NextTokenModel
from nsm_dna.training import (
    build_learning_rate_scheduler,
    calculate_training_steps,
    cleanup_distributed_training,
    initialize_distributed_training,
    load_training_checkpoint,
    save_training_checkpoint,
    upload_checkpoint_to_hugging_face,
)


def prepare_next_token_batch(
    input_ids: Int[Tensor, "batch sequence_length"],
) -> tuple[
    Int[Tensor, "batch context_length"],
    Int[Tensor, "batch context_length"],
]:
    """Align each nucleotide context position with the nucleotide after it."""
    return input_ids[:, :-1], input_ids[:, 1:]


def compute_next_token_loss(
    logits: Float[Tensor, "batch context_length vocab_size"],
    targets: Int[Tensor, "batch context_length"],
) -> Tensor:
    """Average cross-entropy over every predicted nucleotide."""
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def evaluate(
    model: NextTokenModel,
    data_loader: DataLoader,
    use_mixed_precision: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate next-token loss and accuracy over validation sequences."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    loss_sum = 0.0
    num_correct = 0
    num_tokens = 0

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index == max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        context_ids, targets = prepare_next_token_batch(input_ids)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_mixed_precision,
        ):
            logits = model(context_ids)
            batch_loss_sum = F.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )

        loss_sum += batch_loss_sum.item()
        num_correct += (logits.argmax(dim=-1) == targets).sum().item()
        num_tokens += targets.numel()

    model.train(was_training)
    return {
        "loss": loss_sum / num_tokens,
        "accuracy": num_correct / num_tokens,
    }


def save_and_upload_checkpoint(
    run_directory: Path,
    model: NextTokenModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: DictConfig,
    step: int,
    best_validation_loss: float,
    checkpoint_name: str | None = None,
) -> Path:
    """Save a checkpoint locally and immediately copy it to Hugging Face."""
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
    upload_checkpoint_to_hugging_face(
        checkpoint_path,
        config.checkpoint.huggingface,
    )
    return checkpoint_path


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="next_token",
)
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

    model = NextTokenModel.from_config(config).to(device)
    use_mixed_precision = config.mixed_precision.enabled and device.type == "cuda"

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    if distributed_environment.is_main_process:
        print(f"next-token parameters: {num_parameters / 1e6:.2f}M")
        if wandb_run is not None:
            wandb_run.summary["model/parameters"] = num_parameters

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
    best_checkpoint_path: Path | None = None
    best_checkpoint_needs_upload = False
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

    training_model: NextTokenModel | DistributedDataParallel = model
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
            num_correct = torch.zeros((), device=device, dtype=torch.long)
            num_tokens = torch.zeros((), device=device, dtype=torch.long)

        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iterator)
            except StopIteration:
                training_epoch += 1
                train_dataset.set_epoch(training_epoch)
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            input_ids = batch["input_ids"].to(device)
            context_ids, targets = prepare_next_token_batch(input_ids)
            is_last_micro_step = micro_step == gradient_accumulation_steps - 1
            if distributed_environment.is_distributed and not is_last_micro_step:
                synchronization_context = training_model.no_sync()
            else:
                synchronization_context = nullcontext()

            with synchronization_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_mixed_precision,
                ):
                    logits = training_model(context_ids)
                    loss = compute_next_token_loss(logits, targets)
                    accumulated_loss = loss / gradient_accumulation_steps
                accumulated_loss.backward()

            if should_log:
                mean_loss += loss.detach() / gradient_accumulation_steps
                with torch.no_grad():
                    num_correct += (logits.argmax(dim=-1) == targets).sum()
                    num_tokens += targets.numel()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.optimizer.max_gradient_norm,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        if should_log:
            if distributed_environment.is_distributed:
                for values in (mean_loss, num_correct, num_tokens):
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                mean_loss /= distributed_environment.world_size

            if distributed_environment.is_main_process:
                accuracy = (num_correct / num_tokens).item()
                progress_bar.set_postfix(
                    loss=f"{mean_loss.item():.4f}",
                    accuracy=f"{accuracy:.2%}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": mean_loss.item(),
                            "train/accuracy": accuracy,
                            "train/gradient_norm": gradient_norm.item(),
                            "train/learning_rate": learning_rate,
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
                    best_checkpoint_needs_upload = True
                    tqdm.write(f"saved best checkpoint: {best_checkpoint_path}")

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "validation/loss": validation_metrics["loss"],
                            "validation/accuracy": validation_metrics["accuracy"],
                            "validation/best_loss": best_validation_loss,
                        },
                        step=step,
                    )

            if distributed_environment.is_distributed:
                dist.barrier()

        is_recovery_step = step % config.checkpoint.recovery_interval == 0
        checkpoints_to_save: list[tuple[str, str | None]] = []
        if is_recovery_step:
            checkpoints_to_save.append(("recovery", "latest.pt"))
        if step % config.checkpoint.milestone_interval == 0:
            checkpoints_to_save.append(("milestone", None))

        if checkpoints_to_save:
            if distributed_environment.is_main_process:
                for checkpoint_type, checkpoint_name in checkpoints_to_save:
                    checkpoint_path = save_and_upload_checkpoint(
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

                if is_recovery_step and best_checkpoint_needs_upload:
                    assert best_checkpoint_path is not None
                    upload_checkpoint_to_hugging_face(
                        best_checkpoint_path,
                        config.checkpoint.huggingface,
                    )
                    best_checkpoint_needs_upload = False
                    tqdm.write(
                        f"uploaded best checkpoint: {best_checkpoint_path}"
                    )

            if distributed_environment.is_distributed:
                dist.barrier()

    if distributed_environment.is_main_process:
        final_checkpoint_path = save_and_upload_checkpoint(
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
        if best_checkpoint_needs_upload:
            assert best_checkpoint_path is not None
            upload_checkpoint_to_hugging_face(
                best_checkpoint_path,
                config.checkpoint.huggingface,
            )
            tqdm.write(f"uploaded best checkpoint: {best_checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()

    if distributed_environment.is_distributed:
        dist.barrier()
    cleanup_distributed_training()


if __name__ == "__main__":
    main()
