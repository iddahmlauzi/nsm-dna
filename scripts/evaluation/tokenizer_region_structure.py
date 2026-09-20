"""Measure regional code preferences and parent-to-child predictability."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from nsm_dna.data import encode_sequence
from nsm_dna.models.vqvae import VQVAE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_regions(paths: list[Path], region_length: int) -> list[dict]:
    """Read full-length chunks while preserving their genome and position IDs."""
    regions = []
    columns = ["chunk_id", "gtdb_accession", "record_id", "chunk_start", "sequence"]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=64, columns=columns
        ):
            for region in batch.to_pylist():
                if len(region["sequence"]) == region_length:
                    regions.append(region)
    return regions


@torch.inference_mode()
def encode_regions(
    model: VQVAE,
    regions: list[dict],
    region_length: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Return one [region, window, scale_position] index array per scale."""
    window_length = model.context_length
    windows_per_region = region_length // window_length
    windows = torch.stack(
        [
            encode_sequence(region["sequence"][start : start + window_length])
            for region in regions
            for start in range(0, region_length, window_length)
        ]
    )
    indices_by_scale: list[list[np.ndarray]] = [[] for _ in model.scale_lengths]
    for start in tqdm(range(0, len(windows), batch_size), unit="batch"):
        batch = windows[start : start + batch_size].to(device)
        for scale_index, indices in enumerate(model.encode_indices(batch)):
            indices_by_scale[scale_index].append(indices.cpu().numpy())
    return [
        np.concatenate(parts).reshape(len(regions), windows_per_region, scale_length)
        for scale_length, parts in zip(
            model.scale_lengths, indices_by_scale, strict=True
        )
    ]


def code_profiles(
    codes: np.ndarray, codebook_size: int
) -> tuple[np.ndarray, list[dict], dict]:
    """Summarize how concentrated each region's scale codes are."""
    flattened = codes.reshape(codes.shape[0], -1)
    counts = np.stack(
        [np.bincount(row, minlength=codebook_size) for row in flattened]
    )
    sorted_counts = np.sort(counts, axis=1)
    positions_per_region = flattened.shape[1]
    probabilities = counts / positions_per_region
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(
            probabilities > 0, probabilities * np.log(probabilities), 0
        ).sum(axis=1)

    profiles = []
    for region_index, region_counts in enumerate(counts):
        top_ids = np.argsort(region_counts)[::-1][: max(4, codebook_size // 8)]
        profiles.append(
            {
                "distinct_codes": int((region_counts > 0).sum()),
                "top_eighth_coverage": float(
                    sorted_counts[region_index, -max(1, codebook_size // 8) :].sum()
                    / positions_per_region
                ),
                "top_quarter_coverage": float(
                    sorted_counts[region_index, -max(1, codebook_size // 4) :].sum()
                    / positions_per_region
                ),
                "effective_codes": float(np.exp(entropy[region_index])),
                "adjacent_match_fraction": float(
                    np.mean(flattened[region_index, 1:] == flattened[region_index, :-1])
                ),
                "top_codes": [
                    [int(code), int(region_counts[code])]
                    for code in top_ids
                    if region_counts[code] > 0
                ],
            }
        )

    summary = {
        "codebook_size": codebook_size,
        "positions_per_region": positions_per_region,
        "codes_seen_globally": int((counts.sum(axis=0) > 0).sum()),
        "median_distinct_codes": float(
            np.median([p["distinct_codes"] for p in profiles])
        ),
        "median_top_eighth_coverage": float(
            np.median([p["top_eighth_coverage"] for p in profiles])
        ),
        "median_top_quarter_coverage": float(
            np.median([p["top_quarter_coverage"] for p in profiles])
        ),
        "median_effective_codes": float(
            np.median([p["effective_codes"] for p in profiles])
        ),
        "mean_adjacent_match_fraction": float(
            np.mean([p["adjacent_match_fraction"] for p in profiles])
        ),
    }
    return counts, profiles, summary


def longest_run(codes: np.ndarray) -> int:
    """Count the longest uninterrupted sequence of identical scale-1 codes."""
    changes = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    return int(np.diff(np.r_[0, changes, len(codes)]).max())


def score_children(
    parent_codes: np.ndarray,
    child_codes: np.ndarray,
    parent_size: int,
    child_size: int,
    train_regions: np.ndarray,
    test_regions: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Test whether an observed parent improves held-out child prediction."""
    parent = parent_codes.reshape(parent_codes.shape[0], -1)
    children = child_codes.reshape(child_codes.shape[0], -1, 2)
    counts = np.zeros((2, parent_size, child_size), dtype=np.int64)
    for side in range(2):
        np.add.at(
            counts[side],
            (parent[train_regions].ravel(), children[train_regions, :, side].ravel()),
            1,
        )

    marginal_counts = counts.sum(axis=1)
    marginal = (marginal_counts + 1) / (
        marginal_counts.sum(axis=1, keepdims=True) + child_size
    )
    prior_strength = 10.0
    conditional = (counts + prior_strength * marginal[:, None, :]) / (
        counts.sum(axis=2, keepdims=True) + prior_strength
    )
    test_parents = parent[test_regions].ravel()
    by_side = {}
    for side, side_name in enumerate(("left", "right")):
        targets = children[test_regions, :, side].ravel()
        marginal_nll = -np.log(marginal[side, targets]).mean()
        conditional_nll = -np.log(conditional[side, test_parents, targets]).mean()
        by_side[side_name] = {
            "num_children": len(targets),
            "marginal_top1_accuracy": float(
                np.mean(marginal[side].argmax() == targets)
            ),
            "parent_top1_accuracy": float(
                np.mean(conditional[side, test_parents].argmax(axis=1) == targets)
            ),
            "marginal_nll_nats": float(marginal_nll),
            "parent_nll_nats": float(conditional_nll),
            "bits_gained_from_parent": float(
                (marginal_nll - conditional_nll) / math.log(2)
            ),
        }

    summary = {
        "train_regions": len(train_regions),
        "test_regions": len(test_regions),
        "parent_codes_seen_in_train": int((counts.sum(axis=(0, 2)) > 0).sum()),
        "prior_strength": prior_strength,
        "left": by_side["left"],
        "right": by_side["right"],
        "mean_bits_gained_from_parent": float(
            (
                by_side["left"]["bits_gained_from_parent"]
                + by_side["right"]["bits_gained_from_parent"]
            )
            / 2
        ),
    }
    return summary, counts


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation",
    config_name="tokenizer_region_structure",
)
def main(config: DictConfig) -> None:
    checkpoint_path = Path(config.checkpoint)
    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_step = int(checkpoint["step"])
    device = torch.device(config.device)
    model = VQVAE.from_checkpoint(checkpoint_path, device, frozen=True)

    subset_directory = Path(config.data.subset_directory)
    split = str(config.data.split)
    parquet_paths = sorted((subset_directory / split).glob("chunks-*.parquet"))
    region_length = int(config.data.region_length)
    regions = load_regions(parquet_paths, region_length)
    rng = np.random.default_rng(int(config.seed))
    if config.data.max_regions is not None:
        selected = np.sort(
            rng.choice(len(regions), int(config.data.max_regions), replace=False)
        )
        regions = [regions[index] for index in selected]
    codes_by_scale = encode_regions(
        model, regions, region_length, int(config.data.batch_size), device
    )

    windows_per_region = region_length // model.context_length
    window_permutation = rng.permutation(len(regions) * windows_per_region)
    region_profiles = [
        {
            "chunk_id": region["chunk_id"],
            "gtdb_accession": region["gtdb_accession"],
            "record_id": region["record_id"],
            "chunk_start": region["chunk_start"],
            "scales": {},
        }
        for region in regions
    ]
    scale_summaries = {}
    arrays = {}
    for scale_length, codebook_size, codes in zip(
        model.scale_lengths, model.codebook_sizes, codes_by_scale, strict=True
    ):
        real_counts, real_profiles, real_summary = code_profiles(codes, codebook_size)
        mixed_codes = codes.reshape(-1, scale_length)[window_permutation].reshape(
            codes.shape
        )
        mixed_counts, _, mixed_summary = code_profiles(mixed_codes, codebook_size)
        scale_summaries[str(scale_length)] = {
            "real": real_summary,
            "mixed": mixed_summary,
        }
        arrays[f"region_counts_scale_{scale_length}"] = real_counts
        arrays[f"mixed_counts_scale_{scale_length}"] = mixed_counts
        for region_index, profile in enumerate(real_profiles):
            region_profiles[region_index]["scales"][str(scale_length)] = profile

    for region_index, codes in enumerate(codes_by_scale[0]):
        region_profiles[region_index]["scale_1_longest_run"] = longest_run(
            codes.ravel()
        )

    genomes = sorted({region["gtdb_accession"] for region in regions})
    heldout_genomes = set(rng.permutation(genomes)[: max(1, len(genomes) // 4)])
    test_regions = np.array(
        [
            i
            for i, region in enumerate(regions)
            if region["gtdb_accession"] in heldout_genomes
        ]
    )
    train_regions = np.array(
        [
            i
            for i, region in enumerate(regions)
            if region["gtdb_accession"] not in heldout_genomes
        ]
    )
    transitions = {}
    for scale_index, (parent_length, child_length) in enumerate(
        zip(model.scale_lengths, model.scale_lengths[1:])
    ):
        transition, counts = score_children(
            codes_by_scale[scale_index],
            codes_by_scale[scale_index + 1],
            model.codebook_sizes[scale_index],
            model.codebook_sizes[scale_index + 1],
            train_regions,
            test_regions,
        )
        key = f"{parent_length}_to_{child_length}"
        transitions[key] = transition
        arrays[f"parent_child_counts_{key}"] = counts

    suffix = f"step-{checkpoint_step}"
    profile_path = output_directory / f"regions-{suffix}.jsonl"
    with profile_path.open("w", encoding="utf-8") as output_file:
        for profile in region_profiles:
            output_file.write(json.dumps(profile) + "\n")
    counts_path = output_directory / f"counts-{suffix}.npz"
    np.savez_compressed(counts_path, **arrays)
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "input_files": [
            {"path": str(path), "sha256": sha256(path)} for path in parquet_paths
        ],
        "seed": int(config.seed),
        "split": split,
        "region_length": region_length,
        "window_length": model.context_length,
        "num_regions": len(regions),
        "num_genomes": len(genomes),
        "control": (
            "The same 256-bp windows are randomly regrouped into synthetic "
            "8192-bp regions."
        ),
        "scale_summaries": scale_summaries,
        "parent_to_child": transitions,
        "region_profiles": str(profile_path),
        "count_arrays": str(counts_path),
    }
    result_path = output_directory / f"results-{suffix}.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    for scale_length in model.scale_lengths:
        real = scale_summaries[str(scale_length)]["real"]
        mixed = scale_summaries[str(scale_length)]["mixed"]
        print(
            f"scale {scale_length}: top eighth covers "
            f"{real['median_top_eighth_coverage']:.1%} real vs "
            f"{mixed['median_top_eighth_coverage']:.1%} mixed"
        )
    for name, transition in transitions.items():
        print(
            f"{name}: parent gives "
            f"{transition['mean_bits_gained_from_parent']:.3f} bits/child"
        )
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
