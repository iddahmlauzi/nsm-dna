from pathlib import Path

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from torch import Tensor

from .autoencoder import Decoder, Encoder
from .quantization import ResidualVectorQuantizer


class VQVAE(nn.Module):
    """VQ-VAE with a quantized mean-and-detail latent hierarchy."""

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
        third_base_scale: float = 1.0,
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
        self.third_base_scale = third_base_scale
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
            third_base_scale=self.third_base_scale,
            bias=bias,
        )

        self.quantizer = ResidualVectorQuantizer(
            self.scale_lengths,
            self.codebook_sizes,
            self.quantization_dim,
            latent_length=self.latent_length,
            decay=decay,
            eps=eps,
        )
        self.code_lengths = self.quantizer.code_lengths
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
            third_base_scale=config.third_base_scale,
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
        """Encode DNA into the normalized continuous latent."""
        return self.encoder(token_ids)

    def encode_scales(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Float[Tensor, "batch scale_length quantization_dim"]]:
        """Return deterministic continuous block means at every resolution."""
        return self.quantizer.mean_pool_to_scales(self.encode(token_ids).float())

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
        *,
        include_partial_reconstruction: bool = False,
    ) -> tuple[
        Float[Tensor, "batch length vocab_size"],
        Float[Tensor, "batch length vocab_size"] | None,
        list[Int[Tensor, "batch scale_length"]],
    ]:
        latent = self.encode(token_ids)
        quantized_latent, partial_quantized_latent, indices_by_scale = self.quantizer(
            latent,
            include_partial_reconstruction=include_partial_reconstruction,
        )
        logits = self.decoder(quantized_latent)

        partial_logits = None
        if partial_quantized_latent is not None:
            partial_logits = self.decoder(partial_quantized_latent)

        return logits, partial_logits, indices_by_scale

    @torch.no_grad()
    def encode_indices(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Int[Tensor, "batch scale_length"]]:
        """Encode a target block into its mean and successive detail codes."""
        if self.training:
            raise RuntimeError("Call model.eval() before encoding sequences.")

        _, _, indices_by_scale = self.quantizer(self.encode(token_ids))
        return indices_by_scale

    @torch.no_grad()
    def decode_scales(
        self,
        indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> list[Float[Tensor, "batch length vocab_size"]]:
        """Decode the cumulative reconstruction after each hierarchy scale."""
        return [
            self.decoder(latent)
            for latent in self.quantizer.indices_to_cumulative_latents(
                indices_by_scale
            )
        ]

    @property
    def utilization_by_scale(self) -> list[Float[Tensor, ""]]:
        """Fraction of each scale's codes that have been used."""
        return self.quantizer.utilization_by_scale

    @property
    def global_utilization(self) -> Float[Tensor, ""]:
        """Fraction of codebook entries used across all scales."""
        return self.quantizer.global_utilization
