"""Show how a triplet tokenizer assigns the 64 DNA triplets."""

import argparse
from pathlib import Path

import torch

from nsm_dna.models.vqvae import VQVAE
from nsm_dna.triplet_analysis import assign_triplets, code_table_rows


def format_report(assignments: dict[str, int], codebook_size: int) -> str:
    lines = ["triplet\tcode"]
    lines.extend(f"{triplet}\t{code}" for triplet, code in assignments.items())
    lines.extend(["", "code\ttriplets\tamino acids"])
    lines.extend(
        f"{code}\t{triplets}\t{amino_acids}"
        for code, triplets, amino_acids in code_table_rows(
            assignments,
            codebook_size,
        )
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
