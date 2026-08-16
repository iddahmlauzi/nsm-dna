from pathlib import Path

import numpy as np
import torch
from datasets import IterableDataset, load_dataset

BASE_TO_TOKEN_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}

_ASCII_TO_TOKEN_ID = np.full(256, -1, dtype=np.int64)
for base, token_id in BASE_TO_TOKEN_ID.items():
    _ASCII_TO_TOKEN_ID[ord(base)] = token_id


def load_gtdb_dataset(
    subset_directory: Path,
    split: str,
    context_length: int,
    *,
    shuffle_buffer_size: int = 10_000,
    seed: int = 0,
) -> IterableDataset:
    """Stream one GTDB split as fixed-length training sequences."""
    dataset = load_dataset(
        "parquet",
        data_files={
            split: str(subset_directory / split / "chunks-*.parquet"),
        },
        split=split,
        streaming=True,
        columns=["sequence"],
    )
    dataset = dataset.map(
        split_sequences,
        batched=True,
        fn_kwargs={"context_length": context_length},
    )

    if split == "train":
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )

    return dataset


def split_sequences(
    examples: dict[str, list[str]],
    context_length: int,
) -> dict[str, list[str]]:
    """Divide longer stored sequences into training-length examples."""
    sequences = []

    for sequence in examples["sequence"]:
        sequences.extend(
            sequence[start : start + context_length]
            for start in range(0, len(sequence), context_length)
        )

    return {"sequence": sequences}


def encode_sequence(sequence: str) -> torch.Tensor:
    """Convert an A/C/G/T sequence to token IDs."""
    ascii_ids = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    token_ids = _ASCII_TO_TOKEN_ID[ascii_ids]

    if (token_ids < 0).any():
        raise ValueError("Sequence contains an unsupported base.")

    return torch.from_numpy(token_ids)


def collate_dna_sequences(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    """Encode a batch of equal-length DNA sequences."""
    input_ids = torch.stack(
        [encode_sequence(example["sequence"]) for example in examples]
    )
    return {"input_ids": input_ids}
