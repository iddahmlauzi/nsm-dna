from pathlib import Path

import einx
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from torch import Tensor

from .autoencoder import Decoder, Encoder
from .quantization import MultiscaleResidualVectorQuantizer


class _ScaleGradient(torch.autograd.Function):
    """Keep a tensor's forward value while scaling its backward gradient."""

    @staticmethod
    def forward(ctx, tensor: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient * ctx.scale, None


class VQVAE(nn.Module):
    """VQ-VAE with a multiscale residual quantization bottleneck."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        num_heads: int,
        scale_lengths: list[int],
        codebook_sizes: list[int],
        *,
        # Encoder and decoder
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.1,
        bias: bool = False,
        rope_base: float = 10000.0,
        pre_quant_num_groups: int | None = None,
        # Codebook updates and quantization loss
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        # Per-scale post-quantization refinement
        refinement_ratio: float = 0.5,
        refinement_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if not scale_lengths or scale_lengths[-1] != context_length:
            raise ValueError("The final scale length must equal the context length.")
        if pre_quant_num_groups is not None and (
            pre_quant_num_groups <= 0 or embed_dim % pre_quant_num_groups != 0
        ):
            raise ValueError("pre_quant_num_groups must evenly divide embed_dim.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rope_base = rope_base
        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)

        self.encoder = Encoder(
            self.vocab_size,
            self.embed_dim,
            dropout=encoder_dropout,
        )

        # Normalize the encoder output before comparing it with codebook vectors.
        # Quantization uses squared Euclidean distance, and the commitment loss is
        # ||z - e||^2, where z is an encoder vector and e is its selected codebook
        # vector. Both therefore depend on the numerical magnitude of these vectors.
        #
        # Reconstruction loss only measures the decoder's predictions. It does not
        # require a particular magnitude for z because the decoder can adjust its
        # downstream transformations to produce similar predictions from larger
        # encoder values. The encoder magnitude can therefore drift upward even when
        # reconstruction is improving, forcing the codebook to follow and increasing
        # the VQ loss. NCM observed that loss rise from about 3 to 31 over 5,000 steps.
        #
        # GroupNorm keeps the encoder values at a fixed statistical scale. Its normal
        # affine transform would apply a learned per-channel scale after normalization,
        # allowing the model to increase their magnitude again, so affine is disabled.
        self.pre_quant_norm = (
            nn.GroupNorm(pre_quant_num_groups, self.embed_dim, affine=False)
            if pre_quant_num_groups is not None
            else None
        )

        self.quantizer = MultiscaleResidualVectorQuantizer(
            self.scale_lengths,
            self.codebook_sizes,
            self.embed_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            eps=eps,
            refinement_ratio=refinement_ratio,
            refinement_kernel_size=refinement_kernel_size,
        )
        self.decoder = Decoder(
            self.vocab_size,
            self.context_length,
            self.embed_dim,
            self.num_heads,
            dropout=decoder_dropout,
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
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            scale_lengths=list(config.scale_lengths),
            codebook_sizes=list(config.codebook_sizes),
            encoder_dropout=config.encoder_dropout,
            decoder_dropout=config.decoder_dropout,
            bias=config.bias,
            rope_base=getattr(config, "rope_base", 10000.0),
            pre_quant_num_groups=config.pre_quant_num_groups,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
            eps=config.eps,
            refinement_ratio=config.refinement_ratio,
            refinement_kernel_size=config.refinement_kernel_size,
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
    ) -> Float[Tensor, "batch length embed_dim"]:
        """Encode DNA into the normalized continuous latent space.

        VQ-VAE training quantizes this latent before reconstruction. NSM-DNA
        uses the same latent directly when a completed block is prefix context.
        """
        latent = self.encoder(token_ids)

        if self.pre_quant_norm is not None:
            latent = einx.id("b l d -> b d l", latent)
            latent = self.pre_quant_norm(latent)
            latent = einx.id("b d l -> b l d", latent)

        return latent

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
        *,
        include_partial_reconstruction: bool = False,
        partial_latent_gradient_scale: float = 1.0,
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
            vq_loss,
            indices_by_scale,
        ) = self.quantizer(
            latent,
            include_partial_reconstruction=include_partial_reconstruction,
        )
        logits = self.decoder(quantized_latent)

        partial_logits = None
        if partial_quantized_latent is not None:
            # The partial loss uses one decoder pass, but its gradient can have
            # different strengths on the quantizer path and decoder parameters.
            # Scaling at the decoder input affects only the gradient flowing back
            # into the quantizer; the loss coefficient controls the decoder.
            partial_quantized_latent = _ScaleGradient.apply(
                partial_quantized_latent,
                partial_latent_gradient_scale,
            )
            partial_logits = self.decoder(partial_quantized_latent)

        return logits, partial_logits, vq_loss, indices_by_scale

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
    ) -> list[Float[Tensor, "batch scale_length embed_dim"]]:
        """Construct teacher-forced inputs for each scale after the first."""
        return self.quantizer.indices_to_next_scale_inputs(indices_by_scale)

    @torch.no_grad()
    def indices_to_next_scale_input(
        self,
        preceding_indices_by_scale: list[Int[Tensor, "batch scale_length"]],
    ) -> Float[Tensor, "batch next_scale_length embed_dim"]:
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
        if self.training:
            raise RuntimeError("Call model.eval() before decoding sequences.")

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
