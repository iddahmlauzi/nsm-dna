import itertools

import torch

from nsm_dna.data import encode_sequence
from nsm_dna.data.variant_effects.shared import CODON_TABLE
from nsm_dna.models.vqvae import VQVAE


AMINO_ACID_NAMES = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "*": "Stop",
}


def assign_triplets(model: VQVAE, device: torch.device) -> dict[str, int]:
    """Return the finest-scale code selected for each DNA triplet."""
    if model.context_length != 3 * model.latent_length:
        raise ValueError("This analysis requires three bases per latent position.")

    triplets = ["".join(bases) for bases in itertools.product("ACGT", repeat=3)]
    repeats = model.context_length // 3
    input_ids = torch.stack(
        [encode_sequence(triplet * repeats) for triplet in triplets]
    ).to(device)

    was_training = model.training
    model.eval()
    indices = model.encode_indices(input_ids)[-1]
    model.train(was_training)
    if not torch.all(indices == indices[:, :1]):
        raise RuntimeError("A repeated triplet received different codes by position.")

    return {
        triplet: int(code)
        for triplet, code in zip(triplets, indices[:, 0].cpu(), strict=True)
    }


def code_table_rows(
    assignments: dict[str, int],
    codebook_size: int,
) -> list[list[str | int]]:
    """Group triplets and their translated amino acids by code."""
    triplets_by_code = {code: [] for code in range(codebook_size)}
    for triplet, code in assignments.items():
        triplets_by_code[code].append(triplet)

    return [
        [
            code,
            ", ".join(triplets),
            ", ".join(
                AMINO_ACID_NAMES[CODON_TABLE[triplet]] for triplet in triplets
            ),
        ]
        for code, triplets in triplets_by_code.items()
    ]
