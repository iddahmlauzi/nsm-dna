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


def assign_sixmers(
    model: VQVAE,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, int]:
    """Return the length-64 code selected for each ordered triplet pair."""
    if model.context_length != 3 * model.latent_length:
        raise ValueError("This analysis requires three bases per latent position.")

    scale_length = model.latent_length // 2
    try:
        scale_index = model.scale_lengths.index(scale_length)
    except ValueError as error:
        raise ValueError("This analysis requires a scale spanning two triplets.") from error

    sixmers = ["".join(bases) for bases in itertools.product("ACGT", repeat=6)]
    repeats = model.context_length // 6
    assignments = {}

    was_training = model.training
    model.eval()
    for start in range(0, len(sixmers), batch_size):
        batch = sixmers[start : start + batch_size]
        input_ids = torch.stack(
            [encode_sequence(sixmer * repeats) for sixmer in batch]
        ).to(device)
        indices = model.encode_indices(input_ids)[scale_index]
        if not torch.all(indices == indices[:, :1]):
            raise RuntimeError("A repeated 6-mer received different codes by position.")
        assignments.update(
            {
                sixmer: int(code)
                for sixmer, code in zip(
                    batch,
                    indices[:, 0].cpu(),
                    strict=True,
                )
            }
        )
    model.train(was_training)
    return assignments


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


def sixmer_code_table_rows(
    assignments: dict[str, int],
    codebook_size: int,
) -> list[list[str | int]]:
    """Group ordered triplet pairs and their amino-acid pairs by code."""
    sixmers_by_code = {code: [] for code in range(codebook_size)}
    for sixmer, code in assignments.items():
        sixmers_by_code[code].append(sixmer)

    return [
        [
            code,
            ", ".join(sixmers),
            ", ".join(
                "/".join(
                    (
                        AMINO_ACID_NAMES[CODON_TABLE[sixmer[:3]]],
                        AMINO_ACID_NAMES[CODON_TABLE[sixmer[3:]]],
                    )
                )
                for sixmer in sixmers
            ),
        ]
        for code, sixmers in sixmers_by_code.items()
    ]
