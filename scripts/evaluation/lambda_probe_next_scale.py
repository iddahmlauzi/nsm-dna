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
from nsm_dna.models.next_scale import NSMDNA, NextScaleTransformer
from nsm_dna.models.next_scale import MultiscaleTokenizer
from scripts.evaluation.lambda_probe_next_token import (
    LambdaSplit,
    fit_linear_probe,
    fit_three_layer_probe,
    read_lambda_split,
    sha256,
    write_predictions,
)


class NSMWindowEncoder(nn.Module):
    """Preserve the prefix and each scale as separate pooled features."""

    def __init__(
        self, model: NextScaleTransformer, tokenizer: MultiscaleTokenizer
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        target_length = self.tokenizer.context_length
        prefix_length = input_ids.shape[1] - target_length
        if prefix_length < 0:
            raise ValueError(
                f"Expected at least {target_length} sequence positions, "
                f"received {input_ids.shape[1]}."
            )

        target_ids = input_ids[:, prefix_length:]

        with torch.autocast(
            device_type=input_ids.device.type,
            dtype=torch.bfloat16,
            enabled=input_ids.device.type == "cuda",
        ):
            prefix = (
                self.tokenizer.encode(input_ids[:, :prefix_length])
                if prefix_length > 0
                else None
            )
            targets_by_scale = self.tokenizer.encode_indices(target_ids)
            scale_inputs = self.tokenizer.indices_to_next_scale_inputs(targets_by_scale)
            hidden_states = self.model.encode(scale_inputs, prefix=prefix)

        prefix_context_length = (
            self.model.prefix_context_length(prefix_length)
            if hasattr(self.model, "prefix_context_length")
            else prefix_length
        )
        hierarchy_hidden_states = hidden_states[:, prefix_context_length:]
        hidden_states_by_scale = torch.split(
            hierarchy_hidden_states,
            getattr(
                self.model,
                "predicted_scale_lengths",
                self.tokenizer.scale_lengths[1:],
            ),
            dim=1,
        )
        pooled_by_scale = torch.cat(
            [
                scale_hidden_states.mean(dim=1)
                for scale_hidden_states in hidden_states_by_scale
            ],
            dim=1,
        )
        if prefix is None:
            return pooled_by_scale.float()

        pooled_prefix = hidden_states[:, :prefix_length].mean(dim=1)
        pooled_sections = [pooled_prefix]
        if getattr(self.model, "use_prefix_memory", False):
            pooled_memory = hidden_states[
                :, prefix_length:prefix_context_length
            ].mean(dim=1)
            pooled_sections.append(pooled_memory)
        pooled_sections.append(pooled_by_scale)
        return torch.cat(pooled_sections, dim=1).float()


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
    """Average window representations across each 2 kb segment."""
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
    """Fit the requested LAMBDA probes."""
    linear_metrics, linear_predictions, linear_probabilities = fit_linear_probe(
        embeddings["train"],
        splits["train"].labels,
        embeddings["test"],
        splits["test"].labels,
        int(config.probe.seed),
    )
    write_predictions(
        output_directory / "predictions" / f"{name}_linear.csv",
        splits["test"],
        linear_predictions,
        linear_probabilities,
    )
    results: dict[str, object] = {"linear_probe": linear_metrics}

    if bool(config.evaluate_three_layer_probe):
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
            output_directory / "predictions" / f"{name}_three_layer.csv",
            splits["test"],
            nn_predictions,
            nn_probabilities,
        )
        results["three_layer_probe"] = {
            **nn_metrics,
            "completed_epochs": completed_epochs,
        }

    return results


def evaluate_representations(
    name: str,
    encoder: nn.Module,
    model_dim: int,
    scale_lengths: list[int],
    window_length: int,
    stride: int,
    splits: dict[str, LambdaSplit],
    config: DictConfig,
    output_directory: Path,
    device: torch.device,
    *,
    include_prefix: bool,
    include_memory: bool = False,
) -> dict[str, dict[str, object]]:
    """Extract all pooled NSM sections once and probe them separately."""
    split_names = ["train", "test"]
    if bool(config.evaluate_three_layer_probe):
        split_names.insert(1, "validation")
    combined_embeddings = {
        split_name: extract_segment_embeddings(
            encoder,
            (
                len(scale_lengths)
                + int(include_prefix)
                + int(include_memory)
            )
            * model_dim,
            window_length,
            stride,
            splits[split_name].sequences,
            int(config.embedding_batch_size),
            device,
            description=f"{name} {split_name}",
        )
        for split_name in split_names
    }

    representations = {}
    if include_prefix:
        representations["prefix"] = {
            split_name: embeddings[:, :model_dim]
            for split_name, embeddings in combined_embeddings.items()
        }
    memory_offset = int(include_prefix)
    if include_memory:
        section_start = memory_offset * model_dim
        section_end = section_start + model_dim
        representations["memory"] = {
            split_name: embeddings[:, section_start:section_end]
            for split_name, embeddings in combined_embeddings.items()
        }
    scale_offset = int(include_prefix) + int(include_memory)
    for scale_index, scale_length in enumerate(scale_lengths):
        section_start = (scale_index + scale_offset) * model_dim
        section_end = section_start + model_dim
        representations[f"scale_length_{scale_length}"] = {
            split_name: embeddings[:, section_start:section_end]
            for split_name, embeddings in combined_embeddings.items()
        }

    configured_names = getattr(config, "representation_names", None)
    if configured_names is not None:
        representations = {
            representation_name: representations[representation_name]
            for representation_name in configured_names
        }

    return {
        representation_name: evaluate_probes(
            f"{name}_{representation_name}",
            embeddings,
            splits,
            config,
            output_directory,
            device,
        )
        for representation_name, embeddings in representations.items()
    }


def build_parallel_encoder(
    model: NextScaleTransformer,
    tokenizer: MultiscaleTokenizer,
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
    """Evaluate trained NSM-DNA embeddings and an optional random baseline."""
    device_ids = [int(device_id) for device_id in config.device_ids]
    device = torch.device(f"cuda:{device_ids[0]}")
    dataset_directory = Path(config.dataset_directory)
    checkpoint_path = Path(config.checkpoint)
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

    trained_system, checkpoint_step = NSMDNA.from_checkpoint(
        checkpoint_path,
        device,
        frozen=True,
    )
    tokenizer = trained_system.tokenizer
    trained_model = trained_system.transformer
    model_config = OmegaConf.create(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)["config"]
    )
    window_length = int(model_config.data.sequence_length)
    stride = tokenizer.context_length
    include_prefix = trained_model.max_prefix_length > 0
    include_memory = trained_model.use_prefix_memory and include_prefix
    parameter_count = sum(parameter.numel() for parameter in trained_model.parameters())

    trained_results = evaluate_representations(
        "trained",
        build_parallel_encoder(trained_model, tokenizer, device_ids),
        trained_model.model_dim,
        trained_model.predicted_scale_lengths,
        window_length,
        stride,
        splits,
        config,
        output_directory,
        device,
        include_prefix=include_prefix,
        include_memory=include_memory,
    )
    results = {"trained": trained_results}
    random_seed = None
    if bool(config.evaluate_random_model):
        del trained_model, trained_system
        torch.cuda.empty_cache()

        random_seed = int(config.random_model_seed)
        torch.manual_seed(random_seed)
        random_model = NSMDNA.from_config(model_config).transformer.to(device)
        random_model.eval()
        random_model.requires_grad_(False)
        random_name = f"random_seed_{random_seed}"
        random_results = evaluate_representations(
            random_name,
            build_parallel_encoder(random_model, tokenizer, device_ids),
            random_model.model_dim,
            random_model.predicted_scale_lengths,
            window_length,
            stride,
            splits,
            config,
            output_directory,
            device,
            include_prefix=include_prefix,
            include_memory=include_memory,
        )
        results[random_name] = random_results
        results["delta_mcc"] = {
            representation_name: {
                probe_name: (
                    trained_results[representation_name][probe_name]["mcc"]
                    - random_results[representation_name][probe_name]["mcc"]
                )
                for probe_name in trained_results[representation_name]
            }
            for representation_name in trained_results
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
            **(
                {
                    "prefix": (
                        "final normalized NSM prefix hidden states, mean-pooled "
                        "within each window and then across windows"
                    )
                }
                if include_prefix
                else {}
            ),
            **(
                {
                    "memory": (
                        "final normalized learned memory-token state after "
                        "reading the prefix, mean-pooled across windows"
                    )
                }
                if include_memory
                else {}
            ),
            **{
                f"scale_length_{scale_length}": (
                    (
                        "final normalized prefix-facing scale-1 BOS hidden state "
                        "within each window, then mean-pooled across windows"
                    )
                    if trained_model.predicts_first_scale
                    and scale_length == tokenizer.scale_lengths[0]
                    else (
                        "final normalized NSM hidden states for the teacher-forced "
                        f"length-{scale_length} scale, mean-pooled within each "
                        "window and then across windows"
                    )
                )
                for scale_length in trained_model.predicted_scale_lengths
            },
        },
        "window_length": window_length,
        "window_stride": stride,
        "window_starts": window_starts,
        "embedding_batch_size": int(config.embedding_batch_size),
        "device_ids": device_ids,
        "evaluate_three_layer_probe": bool(config.evaluate_three_layer_probe),
        "evaluate_random_model": bool(config.evaluate_random_model),
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
