import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


@dataclass(frozen=True)
class DistributedEnvironment:
    """Process identity and device for single-process or distributed training."""

    device: torch.device
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class GenerativeLossWeights:
    """Weights for the objectives that shape how modelable the latent space is."""

    next_scale_prediction: float
    entropy: float


@dataclass(frozen=True)
class GenFirstLossSchedule:
    """Use strong generative pressure before reconstruction refinement."""

    total_steps: int
    generation_first_fraction: float
    generation_first_weights: GenerativeLossWeights
    refinement_weights: GenerativeLossWeights

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive.")
        if not 0 < self.generation_first_fraction < 1:
            raise ValueError("generation_first_fraction must be between zero and one.")
        weights = (
            self.generation_first_weights.next_scale_prediction,
            self.generation_first_weights.entropy,
            self.refinement_weights.next_scale_prediction,
            self.refinement_weights.entropy,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("GenFirst loss weights must be nonnegative.")

    @property
    def generation_first_steps(self) -> int:
        return round(self.total_steps * self.generation_first_fraction)

    def weights_at_step(self, step: int) -> GenerativeLossWeights:
        """Return the piecewise-constant weights for one optimizer step."""
        if step <= self.generation_first_steps:
            return self.generation_first_weights
        return self.refinement_weights

    def phase_at_step(self, step: int) -> str:
        if step <= self.generation_first_steps:
            return "generation_first"
        return "reconstruction_refinement"


def calculate_training_steps(
    config: DictConfig,
    world_size: int,
    sequence_length: int,
) -> int:
    """Use max_steps when set; otherwise derive the run length from num_epochs."""
    if config.training.max_steps is not None:
        return int(config.training.max_steps)

    stats_path = Path(config.data.subset_directory) / "subset_stats.json"
    with stats_path.open() as handle:
        subset_stats = json.load(handle)
    split_stats = subset_stats["splits"][config.data.train_split]
    num_training_bases = (
        split_stats["chunks"] * subset_stats["selection"]["chunk_length"]
    )

    bases_per_step = (
        sequence_length
        * config.data.train_batch_size
        * world_size
        * config.optimizer.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(num_training_bases / bases_per_step)
    return int(config.training.num_epochs) * steps_per_epoch


def build_learning_rate_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    decay_end_step: int,
    learning_rate: float,
    min_learning_rate: float,
) -> LambdaLR:
    """Create a linear-warmup, cosine-decay learning-rate schedule.

    The learning rate increases to `learning_rate` over the warmup, follows a
    cosine curve down to `min_learning_rate`, and remains there after
    `decay_end_step`.
    """
    min_learning_rate_factor = min_learning_rate / learning_rate

    def learning_rate_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)

        if step >= decay_end_step:
            return min_learning_rate_factor

        decay_progress = (step - warmup_steps) / max(
            1,
            decay_end_step - warmup_steps,
        )
        cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
        return min_learning_rate_factor + cosine_factor * (1 - min_learning_rate_factor)

    return LambdaLR(optimizer, learning_rate_factor)


def initialize_distributed_training() -> DistributedEnvironment:
    """Initialize DDP when launched by torchrun and select this process's device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size == 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedEnvironment(device, rank=0, local_rank=0, world_size=1)

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return DistributedEnvironment(
        device=device,
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=world_size,
    )


def cleanup_distributed_training() -> None:
    """Destroy the distributed process group when one was initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_training_checkpoint(
    run_directory: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: DictConfig,
    step: int,
    best_validation_loss: float,
    checkpoint_name: str | None = None,
) -> Path:
    """Save the state needed to resume a training run."""
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    checkpoint_name = checkpoint_name or f"step-{step}.pt"
    checkpoint_path = checkpoint_directory / checkpoint_name

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
            "best_validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


def upload_checkpoint_to_hugging_face(
    checkpoint_path: Path,
    config: DictConfig,
) -> None:
    """Upload a saved checkpoint when Hugging Face syncing is enabled."""
    if not config.enabled:
        return

    HfApi().upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo=f"{config.repository_directory}/{checkpoint_path.name}",
        repo_id=config.repository_id,
        repo_type="model",
    )


def load_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Restore training state and return the step and best validation loss."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["step"], checkpoint.get("best_validation_loss", float("inf"))


def load_model_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
) -> int:
    """Load model weights without carrying optimizer or scheduler state forward."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    return int(checkpoint["step"])
