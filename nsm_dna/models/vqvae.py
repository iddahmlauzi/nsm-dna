import einx
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .autoencoder import Decoder, Encoder
from .quantization import MultiscaleResidualVectorQuantizer


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
        dropout: float = 0.1,
        bias: bool = False,
        pre_quant_num_groups: int | None = None,

        # Codebook updates and quantization loss
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,

        # Per-scale post-quantization refinement
        refinement_ratio: float = 0.5,
        refinement_kernel_size: int = 3,

        # Fine-scale dropout
        fine_dropout_prob: float = 0.5,
        min_scales_to_keep: int = 1,
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
        self.scale_lengths = list(scale_lengths)
        self.codebook_sizes = list(codebook_sizes)

        self.encoder = Encoder(
            self.vocab_size,
            self.context_length,
            self.embed_dim,
            dropout=dropout,
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
            fine_dropout_prob=fine_dropout_prob,
            min_scales_to_keep=min_scales_to_keep,
        )
        self.decoder = Decoder(
            self.vocab_size,
            self.embed_dim,
            self.num_heads,
            dropout=dropout,
            bias=bias,
        )

    def _encode_pre_quant(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        """Return the continuous encoder output used for quantization."""
        latent = self.encoder(token_ids)

        if self.pre_quant_norm is not None:
            latent = einx.id("b l d -> b d l", latent)
            latent = self.pre_quant_norm(latent)
            latent = einx.id("b d l -> b l d", latent)

        return latent

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> tuple[
        Float[Tensor, "batch length vocab_size"],
        Float[Tensor, ""],
        list[Int[Tensor, "batch scale_length"]],
    ]:
        latent = self._encode_pre_quant(token_ids)
        quantized_latent, vq_loss, indices_by_scale = self.quantizer(latent)
        logits = self.decoder(quantized_latent)

        return logits, vq_loss, indices_by_scale

    @torch.no_grad()
    def encode(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Int[Tensor, "batch scale_length"]]:
        """Convert token sequences into discrete codebook indices at every scale.

        These indices are the targets used to train NSM-DNA. Evaluation mode is
        required so encoding does not update the EMA codebooks or apply fine-scale
        dropout.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before encoding sequences.")

        latent = self._encode_pre_quant(token_ids)
        _, _, indices_by_scale = self.quantizer(latent)
        return indices_by_scale

    @property
    def utilization_by_scale(self) -> list[Float[Tensor, ""]]:
        """Fraction of each scale's codes that have been used."""
        return self.quantizer.utilization_by_scale

    @property
    def global_utilization(self) -> Float[Tensor, ""]:
        """Fraction of codebook entries used across all scales."""
        return self.quantizer.global_utilization
