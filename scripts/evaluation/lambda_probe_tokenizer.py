import json
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from nsm_dna.models.vqvae import VQVAE
from scripts.evaluation.lambda_probe_next_scale import (
    evaluate_probes,
    extract_segment_embeddings,
    segment_window_starts,
)
from scripts.evaluation.lambda_probe_next_token import (
    LambdaSplit,
    read_lambda_split,
    sha256,
)


class TokenizerWindowEncoder(nn.Module):
    """Mean-pool the tokenizer's exact dinucleotide vectors."""

    def __init__(self, tokenizer: VQVAE) -> None:
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            finest_vectors = self.tokenizer.encode(input_ids)

        return finest_vectors.float().mean(dim=1)


def evaluate_tokenizer_representations(
    encoder: nn.Module,
    embed_dim: int,
    window_length: int,
    splits: dict[str, LambdaSplit],
    config: DictConfig,
    output_directory: Path,
    device: torch.device,
) -> dict[str, dict[str, object]]:
    """Evaluate the tokenizer's exact dinucleotide representation."""
    embeddings = {
        split_name: extract_segment_embeddings(
            encoder,
            embed_dim,
            window_length,
            window_length,
            split.sequences,
            int(config.embedding_batch_size),
            device,
            description=f"tokenizer {split_name}",
        )
        for split_name, split in splits.items()
    }
    return {
        "finest_dinucleotide": evaluate_probes(
            "trained_tokenizer_finest_dinucleotide",
            embeddings,
            splits,
            config,
            output_directory,
            device,
        )
    }


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
    results = evaluate_tokenizer_representations(
        build_parallel_encoder(tokenizer, device_ids),
        tokenizer.quantization_dim,
        tokenizer.context_length,
        splits,
        config,
        output_directory,
        device,
    )

    window_starts = segment_window_starts(
        int(config.expected_sequence_length),
        tokenizer.context_length,
        tokenizer.context_length,
    )
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
        "representations": {
            "finest_dinucleotide": (
                "mean of exact scale-128 dinucleotide vectors for each "
                "128-base window, then mean across windows"
            ),
        },
        "window_length": tokenizer.context_length,
        "window_stride": tokenizer.context_length,
        "window_starts": window_starts,
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
