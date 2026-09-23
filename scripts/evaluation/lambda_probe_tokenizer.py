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
from nsm_dna.models.vqvae import VQVAE
from scripts.evaluation.lambda_probe_next_scale import (
    evaluate_probes,
)
from scripts.evaluation.lambda_probe_next_token import (
    LambdaSplit,
    read_lambda_split,
    sha256,
)


class TokenizerWindowEncoder(nn.Module):
    """Mean-pool decoder states immediately before nucleotide prediction."""

    def __init__(self, tokenizer: VQVAE) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            latent = self.tokenizer.encode(input_ids)
            quantized_latent, _, _ = self.tokenizer.quantizer(latent)
            hidden_states = self.tokenizer.decoder.encode(quantized_latent)

        return hidden_states.float().mean(dim=1)


def frame_window_starts(
    sequence_length: int,
    window_length: int,
    frame_offset: int,
) -> list[int]:
    """Return non-overlapping windows aligned to one triplet reading frame."""
    return list(
        range(frame_offset, sequence_length - window_length + 1, window_length)
    )


@torch.inference_mode()
def extract_frame_embeddings(
    encoder: nn.Module,
    embed_dim: int,
    window_length: int,
    frame_offset: int,
    sequences: list[str],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Pool decoder states over windows aligned to one reading frame."""
    embeddings = np.empty((len(sequences), embed_dim), dtype=np.float32)
    window_starts = frame_window_starts(
        len(sequences[0]),
        window_length,
        frame_offset,
    )

    for start in tqdm(
        range(0, len(sequences), batch_size),
        desc=f"frame {frame_offset + 1}",
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
        embeddings[start : start + num_sequences] = (
            window_embeddings.reshape(num_sequences, num_windows, embed_dim)
            .mean(dim=1)
            .cpu()
            .numpy()
        )

    return embeddings


def build_parallel_encoder(
    tokenizer: VQVAE,
    device_ids: list[int],
) -> nn.Module:
    """Use every configured GPU for tokenizer embedding extraction."""
    encoder = TokenizerWindowEncoder(tokenizer)
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
    config_name="lambda_probe_tokenizer",
)
def main(config: DictConfig) -> None:
    """Evaluate trained VQ-VAE representations on LAMBDA 2 kb segments."""
    device_ids = [int(device_id) for device_id in config.device_ids]
    device = torch.device(f"cuda:{device_ids[0]}")
    dataset_directory = Path(config.dataset_directory)
    checkpoint_path = Path(config.tokenizer_checkpoint)
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

    tokenizer = VQVAE.from_checkpoint(checkpoint_path, device, frozen=True)
    encoder = build_parallel_encoder(tokenizer, device_ids)
    results = {}
    for frame_offset in range(3):
        embeddings = {
            split_name: extract_frame_embeddings(
                encoder,
                tokenizer.embed_dim,
                tokenizer.context_length,
                frame_offset,
                split.sequences,
                int(config.embedding_batch_size),
                device,
            )
            for split_name, split in splits.items()
        }
        frame_name = f"frame_{frame_offset + 1}"
        results[frame_name] = evaluate_probes(
            f"trained_tokenizer_decoder_{frame_name}",
            embeddings,
            splits,
            config,
            output_directory,
            device,
        )

    window_starts_by_frame = {
        f"frame_{frame_offset + 1}": frame_window_starts(
            int(config.expected_sequence_length),
            tokenizer.context_length,
            frame_offset,
        )
        for frame_offset in range(3)
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer_checkpoint": str(checkpoint_path),
        "tokenizer_checkpoint_sha256": sha256(checkpoint_path),
        "tokenizer_parameter_count": sum(
            parameter.numel() for parameter in tokenizer.parameters()
        ),
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
            "mean of final normalized decoder hidden states immediately before "
            "nucleotide projection, then mean across non-overlapping windows"
        ),
        "frame_offsets": [0, 1, 2],
        "window_length": tokenizer.context_length,
        "window_stride": tokenizer.context_length,
        "window_starts_by_frame": window_starts_by_frame,
        "embedding_batch_size": int(config.embedding_batch_size),
        "device_ids": device_ids,
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
