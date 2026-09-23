from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .next_scale import NSM
from .vqvae import VQVAE


@dataclass
class NSMDNAOutput:
    reconstruction_logits: Float[Tensor, "batch target_length vocab_size"]
    partial_reconstruction_logits: Float[
        Tensor, "batch target_length vocab_size"
    ] | None
    hierarchy_logits: list[Float[Tensor, "batch scale_length codebook_size"]]
    nucleotide_logits: Float[Tensor, "batch target_length vocab_size"]
    target_indices: list[Int[Tensor, "batch scale_length"]]


class NSMDNA(nn.Module):
    """Train the tokenizer and next-scale model as one differentiable system."""

    def __init__(self, tokenizer: VQVAE, nsm: NSM) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.nsm = nsm

    @classmethod
    def from_config(cls, config: DictConfig) -> "NSMDNA":
        tokenizer = VQVAE.from_config(config.tokenizer)
        return cls(tokenizer, NSM.from_config(config, tokenizer))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        *,
        frozen: bool = False,
    ) -> tuple["NSMDNA", int]:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = cls.from_config(OmegaConf.create(checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)
        if frozen:
            model.eval()
            model.requires_grad_(False)
        return model, int(checkpoint["step"])

    def forward(
        self,
        prefix_ids: Int[Tensor, "batch target_length"],
        target_ids: Int[Tensor, "batch target_length"],
        *,
        include_partial_reconstruction: bool = True,
    ) -> NSMDNAOutput:
        prefix = self.tokenizer.encode(prefix_ids)
        target_latent = self.tokenizer.encode(target_ids)
        (
            quantized_latent,
            partial_quantized_latent,
            target_indices,
            assignments_by_scale,
        ) = self.tokenizer.quantizer(
            target_latent,
            include_partial_reconstruction=include_partial_reconstruction,
        )
        vectors_by_scale = [
            self.tokenizer.quantizer.assignments_to_vectors(
                assignments,
                scale_index,
            )
            for scale_index, assignments in enumerate(assignments_by_scale)
        ]

        partial_reconstruction_logits = None
        if partial_quantized_latent is not None:
            partial_reconstruction_logits = self.tokenizer.decoder(
                partial_quantized_latent
            )

        return NSMDNAOutput(
            reconstruction_logits=self.tokenizer.decoder(quantized_latent),
            partial_reconstruction_logits=partial_reconstruction_logits,
            hierarchy_logits=self.nsm(vectors_by_scale[:-1], prefix=prefix),
            nucleotide_logits=self.nsm.predict_nucleotides(
                vectors_by_scale[-1],
                prefix=prefix,
            ),
            target_indices=target_indices,
        )
