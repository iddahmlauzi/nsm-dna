from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from torch import Tensor

from .autoencoder import Decoder
from .quantization import MultiscaleVectorQuantizer


class VQVAEHierarchyOutput(NamedTuple):
    """Outputs used to train the discrete pairwise hierarchy."""

    logits: Tensor
    indices_by_scale: list[Tensor]
    commitment_losses_by_scale: Tensor
    child_logits_by_scale: list[tuple[Tensor, Tensor]]


class _PairPredictionHead(nn.Module):
    """Predict the ordered left and right children of one parent code."""

    def __init__(self, input_dim: int, child_size: int, bias: bool) -> None:
        super().__init__()
        self.left = nn.Linear(input_dim, child_size, bias=bias)
        self.right = nn.Linear(input_dim, child_size, bias=bias)

    def forward(self, parent: Tensor) -> tuple[Tensor, Tensor]:
        return self.left(parent), self.right(parent)


class VQVAE(nn.Module):
    """Pairwise hierarchy with exact dinucleotides at the finest scale."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        latent_length: int,
        embed_dim: int,
        quantization_dim: int,
        num_heads: int,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        *,
        decoder_num_layers: int = 1,
        use_qk_norm: bool = False,
        bias: bool = False,
        rope_base: float = 10000.0,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.latent_length = latent_length
        self.embed_dim = embed_dim
        self.quantization_dim = quantization_dim
        self.num_heads = num_heads
        self.decoder_num_layers = decoder_num_layers
        self.use_qk_norm = use_qk_norm
        self.rope_base = rope_base
        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)
        if context_length != 2 * latent_length or codebook_sizes[-1] != 16:
            raise ValueError(
                "The final scale requires one code for each dinucleotide."
            )

        self.quantizer = MultiscaleVectorQuantizer(
            self.scale_lengths,
            self.codebook_sizes,
            self.quantization_dim,
            latent_length=self.latent_length,
            decay=decay,
            eps=eps,
        )
        self.child_predictors = nn.ModuleList(
            [
                _PairPredictionHead(
                    self.quantization_dim,
                    child_size,
                    bias,
                )
                for child_size in self.codebook_sizes[1:]
            ]
        )
        self.decoder = Decoder(
            self.vocab_size,
            self.context_length,
            self.latent_length,
            self.embed_dim,
            self.quantization_dim,
            self.num_heads,
            num_layers=self.decoder_num_layers,
            use_qk_norm=self.use_qk_norm,
            bias=bias,
            rope_base=self.rope_base,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device,
        *,
        frozen: bool = False,
    ) -> "VQVAE":
        """Rebuild a VQ-VAE from its saved configuration and weights."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        config = OmegaConf.create(checkpoint["config"]).model

        model = cls(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            latent_length=config.latent_length,
            embed_dim=config.embed_dim,
            quantization_dim=config.quantization_dim,
            num_heads=config.num_heads,
            decoder_num_layers=config.decoder_num_layers,
            use_qk_norm=config.use_qk_norm,
            scale_lengths=list(config.scale_lengths),
            codebook_sizes=list(config.codebook_sizes),
            bias=config.bias,
            rope_base=config.rope_base,
            decay=config.decay,
            eps=config.eps,
        )
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        if frozen:
            model.eval()
            model.requires_grad_(False)

        return model

    def encode(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch latent_length quantization_dim"]:
        """Look up the trainable vector for each exact dinucleotide ID."""
        finest_indices = self._finest_scale_indices(token_ids)
        return self.quantizer.codebooks[-1].codebook[finest_indices]

    def _finest_scale_indices(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Int[Tensor, "batch latent_length"]:
        left_tokens = token_ids[:, 0::2]
        right_tokens = token_ids[:, 1::2]
        return left_tokens * self.vocab_size + right_tokens

    @torch.no_grad()
    def encode_scales(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Float[Tensor, "batch scale_length quantization_dim"]]:
        """Encode DNA into the pre-quantization state at every scale."""
        if self.training:
            raise RuntimeError("Call model.eval() before encoding sequences.")

        return self.quantizer._downsample_to_scales(
            self._finest_scale_indices(token_ids),
        )

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> VQVAEHierarchyOutput:
        quantization = self.quantizer.quantize(
            self._finest_scale_indices(token_ids),
        )
        logits = self.decoder(quantization.quantized_latent)
        child_logits_by_scale = [
            predictor(quantized)
            for predictor, quantized in zip(
                self.child_predictors,
                quantization.quantized_latents_by_scale[:-1],
                strict=True,
            )
        ]
        return VQVAEHierarchyOutput(
            logits=logits,
            indices_by_scale=quantization.indices_by_scale,
            commitment_losses_by_scale=(
                quantization.commitment_losses_by_scale
            ),
            child_logits_by_scale=child_logits_by_scale,
        )

    @torch.no_grad()
    def encode_indices(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Int[Tensor, "batch scale_length"]]:
        """Encode a target block into code indices at every hierarchy scale."""
        if self.training:
            raise RuntimeError("Call model.eval() before encoding sequences.")

        return self.quantizer.quantize(
            self._finest_scale_indices(token_ids)
        ).indices_by_scale

    @torch.no_grad()
    def decode_scale(
        self,
        scale_indices: Int[Tensor, "batch scale_length"],
        scale_index: int,
    ) -> Float[Tensor, "batch length vocab_size"]:
        """Decode one scale's absolute codes into nucleotide logits."""
        scale_latent = self.quantizer.indices_to_scale_latent(
            scale_indices, scale_index
        )
        return self.decoder(scale_latent)

    @torch.no_grad()
    def decode_scales(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length vocab_size"]]:
        """Decode each scale independently."""
        return [
            self.decode_scale(scale_indices, scale_index)
            for scale_index, scale_indices in enumerate(indices_by_scale)
        ]

    @property
    def utilization_by_scale(self) -> list[Float[Tensor, ""]]:
        """Fraction of each scale's codes that have been used."""
        return self.quantizer.utilization_by_scale

    @property
    def global_utilization(self) -> Float[Tensor, ""]:
        """Fraction of codebook entries used across all scales."""
        return self.quantizer.global_utilization
