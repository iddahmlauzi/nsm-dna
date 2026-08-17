import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


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


def save_checkpoint(
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


def load_checkpoint(
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
