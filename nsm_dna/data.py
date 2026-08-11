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

PAD_TOKEN_ID = 4
MAX_SEQUENCE_LENGTH = 8_192

_ASCII_TO_TOKEN_ID = np.full(256, -1, dtype=np.int64)
for base, token_id in BASE_TO_TOKEN_ID.items():
    _ASCII_TO_TOKEN_ID[ord(base)] = token_id


def load_gtdb_dataset(data_directory: Path) -> IterableDataset:
    """Load GTDB sequences as a streaming dataset."""
    dataset = load_dataset(
        "parquet",
        data_files=str(data_directory / "chunks-*.parquet"),
        streaming=True,
        columns=["sequence"],
    )
    return dataset["train"]


def encode_sequence(sequence: str) -> torch.Tensor:
    """Convert an A/C/G/T sequence to token IDs."""
    ascii_ids = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    token_ids = _ASCII_TO_TOKEN_ID[ascii_ids]

    if (token_ids < 0).any():
        raise ValueError("Sequence contains an unsupported base.")

    return torch.from_numpy(token_ids)


def collate_dna_sequences(examples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    """Encode and right-pad DNA sequences for training."""
    input_ids = torch.full(
        (len(examples), MAX_SEQUENCE_LENGTH),
        PAD_TOKEN_ID,
        dtype=torch.long,
    )

    for i, example in enumerate(examples):
        token_ids = encode_sequence(example["sequence"])
        input_ids[i, :len(token_ids)] = token_ids

    valid_mask = input_ids != PAD_TOKEN_ID
    return {
        "input_ids": input_ids,
        "valid_mask": valid_mask,
    }
