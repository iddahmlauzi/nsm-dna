from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from torch import Tensor

from .autoencoder import Decoder, Encoder
from .quantization import MultiscaleResidualVectorQuantizer


class VQVAE(nn.Module):
    """VQ-VAE with a multiscale residual quantization bottleneck."""

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
        commitment_cost: float = 0.25,
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

        self.encoder = Encoder(
            self.vocab_size,
            self.context_length,
            self.latent_length,
            self.embed_dim,
            self.quantization_dim,
            bias=bias,
        )

        self.quantizer = MultiscaleResidualVectorQuantizer(
            self.scale_lengths,
            self.codebook_sizes,
            self.quantization_dim,
            latent_length=self.latent_length,
            commitment_cost=commitment_cost,
            decay=decay,
            eps=eps,
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
            commitment_cost=config.commitment_cost,
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
        """Encode DNA into the normalized continuous latent space.

        VQ-VAE training quantizes this latent before reconstruction. NSM-DNA
        uses the same latent directly when a completed block is prefix context.
        """
        return self.encoder(token_ids)

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
        *,
        include_partial_reconstruction: bool = False,
    ) -> tuple[
        Float[Tensor, "batch length vocab_size"],
        Float[Tensor, "batch length vocab_size"] | None,
        Float[Tensor, ""],
        list[Int[Tensor, "batch scale_length"]],
    ]:
        latent = self.encode(token_ids)
        (
            quantized_latent,
            partial_quantized_latent,
            commitment_loss,
            indices_by_scale,
        ) = self.quantizer(
            latent,
            include_partial_reconstruction=include_partial_reconstruction,
        )
        logits = self.decoder(quantized_latent)

        partial_logits = None
        if partial_quantized_latent is not None:
            partial_logits = self.decoder(partial_quantized_latent)

        return logits, partial_logits, commitment_loss, indices_by_scale

    @torch.no_grad()
    def encode_indices(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Int[Tensor, "batch scale_length"]]:
        """Encode target blocks into discrete codebook indices at every scale.

        These indices provide both teacher-forced hierarchy inputs and prediction
        targets for stage-two NSM-DNA training.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before encoding sequences.")

        latent = self.encode(token_ids)
        _, _, _, indices_by_scale = self.quantizer(latent)
        return indices_by_scale

    @torch.no_grad()
    def indices_to_next_scale_inputs(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch scale_length quantization_dim"]]:
        """Construct teacher-forced inputs for each scale after the first."""
        return self.quantizer.indices_to_next_scale_inputs(indices_by_scale)

    @torch.no_grad()
    def indices_to_next_scale_input(
        self,
        preceding_indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch next_scale_length quantization_dim"]:
        """Construct the next input from autoregressively predicted indices."""
        return self.quantizer.indices_to_next_scale_input(
            preceding_indices_by_scale
        )

    @torch.no_grad()
    def decode(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch length vocab_size"]:
        """Decode a complete hierarchy of codebook indices into nucleotide logits."""
        quantized_latent = self.quantizer.indices_to_cumulative_latents(
            indices_by_scale
        )[-1]
        return self.decoder(quantized_latent)

    @torch.no_grad()
    def decode_cumulative(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length vocab_size"]]:
        """Decode the reconstruction after each additional quantization scale."""
        cumulative_latents = self.quantizer.indices_to_cumulative_latents(
            indices_by_scale
        )
        return [self.decoder(latent) for latent in cumulative_latents]

    @property
    def utilization_by_scale(self) -> list[Float[Tensor, ""]]:
        """Fraction of each scale's codes that have been used."""
        return self.quantizer.utilization_by_scale

    @property
    def global_utilization(self) -> Float[Tensor, ""]:
        """Fraction of codebook entries used across all scales."""
        return self.quantizer.global_utilization
