import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from nsm_dna.data import encode_sequence
from nsm_dna.models.next_scale import NSM
from nsm_dna.models.vqvae import VQVAE
from scripts.evaluation.lambda_probe_next_token import (
    LambdaSplit,
    fit_linear_probe,
    fit_three_layer_probe,
    read_lambda_split,
    sha256,
    write_predictions,
)


class NSMWindowEncoder(nn.Module):
    """Mean-pool final NSM states for one prefix-target DNA window."""

    def __init__(self, model: NSM, tokenizer: VQVAE) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        block_length = self.tokenizer.context_length
        prefix_ids = input_ids[:, :block_length]
        target_ids = input_ids[:, block_length:]

        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            prefix = self.tokenizer.encode(prefix_ids)
            targets_by_scale = self.tokenizer.encode_indices(target_ids)
            scale_inputs = self.tokenizer.indices_to_next_scale_inputs(
                targets_by_scale
            )
            hidden_states = self.model.encode(scale_inputs, prefix=prefix)

        return hidden_states.float().mean(dim=1)


def segment_window_starts(
    sequence_length: int,
    window_length: int,
    stride: int,
) -> list[int]:
    """Advance one target block at a time and include the segment's end."""
    starts = list(range(0, sequence_length - window_length + 1, stride))
    final_start = sequence_length - window_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def extract_segment_embeddings(
    encoder: nn.Module,
    model_dim: int,
    window_length: int,
    stride: int,
    sequences: list[str],
    batch_size: int,
    device: torch.device,
    *,
    description: str,
) -> np.ndarray:
    """Mean-pool native-window NSM embeddings for each 2 kb segment."""
    embeddings = np.empty((len(sequences), model_dim), dtype=np.float32)
    window_starts = segment_window_starts(
        len(sequences[0]),
        window_length,
        stride,
    )

    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc=description,
        unit="batch",
    ):
        batch_sequences = sequences[start : start + batch_size]
        batch_ids = torch.stack(
            [encode_sequence(sequence) for sequence in batch_sequences]
        )
        window_ids = torch.stack(
            [
                batch_ids[:, window_start : window_start + window_length]
                for window_start in window_starts
            ],
            dim=1,
        )
        num_sequences, num_windows, _ = window_ids.shape
        window_embeddings = encoder(
            window_ids.reshape(num_sequences * num_windows, window_length).to(device)
        )
        segment_embeddings = window_embeddings.reshape(
            num_sequences,
            num_windows,
            model_dim,
        ).mean(dim=1)
        embeddings[start : start + num_sequences] = segment_embeddings.cpu().numpy()

    return embeddings


def evaluate_probes(
    name: str,
    embeddings: dict[str, np.ndarray],
    splits: dict[str, LambdaSplit],
    config: DictConfig,
    output_directory: Path,
    device: torch.device,
) -> dict[str, object]:
    """Fit the official linear and 3-layer LAMBDA probes."""
    linear_metrics, linear_predictions, linear_probabilities = fit_linear_probe(
        embeddings["train"],
        splits["train"].labels,
        embeddings["test"],
        splits["test"].labels,
        int(config.probe.seed),
    )
    nn_metrics, nn_predictions, nn_probabilities, completed_epochs = (
        fit_three_layer_probe(
            embeddings["train"],
            splits["train"].labels,
            embeddings["validation"],
            splits["validation"].labels,
            embeddings["test"],
            splits["test"].labels,
            config.probe,
            device,
        )
    )

    write_predictions(
        output_directory / "predictions" / f"{name}_linear.csv",
        splits["test"],
        linear_predictions,
        linear_probabilities,
    )
    write_predictions(
        output_directory / "predictions" / f"{name}_three_layer.csv",
        splits["test"],
        nn_predictions,
        nn_probabilities,
    )
    return {
        "linear_probe": linear_metrics,
        "three_layer_probe": {
            **nn_metrics,
            "completed_epochs": completed_epochs,
        },
    }


def evaluate_representation(
    name: str,
    encoder: nn.Module,
    embedding_dim: int,
    window_length: int,
    stride: int,
    splits: dict[str, LambdaSplit],
    config: DictConfig,
    output_directory: Path,
    device: torch.device,
) -> dict[str, object]:
    """Extract one NSM representation and run both LAMBDA probes."""
    embeddings = {
        split_name: extract_segment_embeddings(
            encoder,
            embedding_dim,
            window_length,
            stride,
            split.sequences,
            int(config.embedding_batch_size),
            device,
            description=f"{name} {split_name}",
        )
        for split_name, split in splits.items()
    }
    return evaluate_probes(
        name,
        embeddings,
        splits,
        config,
        output_directory,
        device,
    )


def build_parallel_encoder(
    model: NSM,
    tokenizer: VQVAE,
    device_ids: list[int],
) -> nn.Module:
    """Use every configured GPU for NSM embedding extraction."""
    encoder = NSMWindowEncoder(model, tokenizer)
    if len(device_ids) == 1:
        return encoder
    return nn.DataParallel(
        encoder,
        device_ids=device_ids,
        output_device=device_ids[0],
    )


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="lambda_probe_next_scale",
)
def main(config: DictConfig) -> None:
    """Compare trained and random NSM-DNA final embeddings on LAMBDA."""
    device_ids = [int(device_id) for device_id in config.device_ids]
    device = torch.device(f"cuda:{device_ids[0]}")
    dataset_directory = Path(config.dataset_directory)
    checkpoint_path = Path(config.checkpoint)
    tokenizer_checkpoint_path = Path(config.tokenizer_checkpoint)
    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    split_paths = {
        "train": dataset_directory / "train.csv",
        "validation": dataset_directory / "dev.csv",
        "test": dataset_directory / "test.csv",
    }
    splits = {
        name: read_lambda_split(path, int(config.expected_sequence_length))
        for name, path in split_paths.items()
    }

    tokenizer = VQVAE.from_checkpoint(
        tokenizer_checkpoint_path,
        device,
        frozen=True,
    )
    trained_model, checkpoint_step = NSM.from_checkpoint(
        checkpoint_path,
        tokenizer,
        device,
        frozen=True,
    )
    model_config = OmegaConf.create(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)["config"]
    )
    window_length = int(model_config.data.sequence_length)
    stride = tokenizer.context_length
    parameter_count = sum(
        parameter.numel() for parameter in trained_model.parameters()
    )

    trained_results = evaluate_representation(
        "trained",
        build_parallel_encoder(trained_model, tokenizer, device_ids),
        trained_model.model_dim,
        window_length,
        stride,
        splits,
        config,
        output_directory,
        device,
    )
    del trained_model
    torch.cuda.empty_cache()

    random_seed = int(config.random_model_seed)
    torch.manual_seed(random_seed)
    random_model = NSM.from_config(model_config, tokenizer).to(device)
    random_model.eval()
    random_model.requires_grad_(False)
    random_results = evaluate_representation(
        f"random_seed_{random_seed}",
        build_parallel_encoder(random_model, tokenizer, device_ids),
        random_model.model_dim,
        window_length,
        stride,
        splits,
        config,
        output_directory,
        device,
    )

    results = {
        "trained": trained_results,
        f"random_seed_{random_seed}": random_results,
        "delta_mcc": {
            "linear_probe": (
                trained_results["linear_probe"]["mcc"]
                - random_results["linear_probe"]["mcc"]
            ),
            "three_layer_probe": (
                trained_results["three_layer_probe"]["mcc"]
                - random_results["three_layer_probe"]["mcc"]
            ),
        },
    }
    window_starts = segment_window_starts(
        int(config.expected_sequence_length),
        window_length,
        stride,
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "parameter_count": parameter_count,
        "tokenizer_checkpoint": str(tokenizer_checkpoint_path),
        "tokenizer_checkpoint_sha256": sha256(tokenizer_checkpoint_path),
        "dataset": "LAMBDA binary_segments_2k",
        "dataset_files": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "num_sequences": len(splits[name].sequences),
                "num_excluded_ambiguous": splits[name].num_excluded_ambiguous,
            }
            for name, path in split_paths.items()
        },
        "representation": (
            "mean of all final normalized NSM transformer states for each "
            "teacher-forced 128-base prefix and 128-base target window, then "
            "mean across windows"
        ),
        "window_length": window_length,
        "window_stride": stride,
        "window_starts": window_starts,
        "embedding_batch_size": int(config.embedding_batch_size),
        "device_ids": device_ids,
        "random_model_seed": random_seed,
        "probe_config": OmegaConf.to_container(config.probe, resolve=True),
        "results": results,
    }
    (output_directory / "results.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
