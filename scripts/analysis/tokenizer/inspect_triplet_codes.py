"""Show how a triplet tokenizer assigns the 64 DNA triplets."""

import argparse
import itertools
from pathlib import Path

import torch

from nsm_dna.data import encode_sequence
from nsm_dna.models.vqvae import VQVAE


def assign_triplets(model: VQVAE, device: torch.device) -> dict[str, int]:
    """Return the code selected for each possible A/C/G/T triplet."""
    if model.context_length != 3 * model.latent_length:
        raise ValueError("This analysis requires three bases per latent position.")

    triplets = ["".join(bases) for bases in itertools.product("ACGT", repeat=3)]
    repeats = model.context_length // 3
    input_ids = torch.stack(
        [encode_sequence(triplet * repeats) for triplet in triplets]
    ).to(device)

    indices = model.encode_indices(input_ids)[-1]
    if not torch.all(indices == indices[:, :1]):
        raise RuntimeError("A repeated triplet received different codes by position.")

    return {
        triplet: int(code)
        for triplet, code in zip(triplets, indices[:, 0].cpu(), strict=True)
    }


def format_report(assignments: dict[str, int], codebook_size: int) -> str:
    triplets_by_code = {code: [] for code in range(codebook_size)}
    for triplet, code in assignments.items():
        triplets_by_code[code].append(triplet)

    lines = ["triplet\tcode"]
    lines.extend(f"{triplet}\t{code}" for triplet, code in assignments.items())
    lines.extend(["", "code\ttriplets"])
    lines.extend(
        f"{code}\t{','.join(triplets)}"
        for code, triplets in triplets_by_code.items()
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = VQVAE.from_checkpoint(args.checkpoint, device, frozen=True)
    assignments = assign_triplets(model, device)
    print(format_report(assignments, model.codebook_sizes[-1]))


if __name__ == "__main__":
    main()
