"""Multiscale residual-quantization tokenizer used by NSM-DNA."""

import einx
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from ..common import LayerNorm, TransformerBlock, precompute_rope_cosine_and_sine
from .quantization import MultiscaleResidualVectorQuantizer, QuantizerOutput


class Encoder(nn.Module):
    """Embed nucleotides independently without absolute position information."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        return self.drop(self.token_embedding(token_ids))


class Decoder(nn.Module):
    """Decode latent representations into nucleotide logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        positions = torch.arange(context_length)
        head_dim = embed_dim // num_heads
        rope_cosine, rope_sine = precompute_rope_cosine_and_sine(
            positions,
            head_dim,
            rope_base,
        )
        self.register_buffer("rope_cosine", rope_cosine, persistent=False)
        self.register_buffer("rope_sine", rope_sine, persistent=False)

        self.block = TransformerBlock(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.final_norm = LayerNorm(embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=bias)

    def forward(
        self,
        latent: Float[Tensor, "batch length embed_dim"],
    ) -> Float[Tensor, "batch length vocab_size"]:
        x = self.block(
            latent,
            rotary_embeddings=(self.rope_cosine, self.rope_sine),
            is_causal=False,
        )
        x = self.final_norm(x)
        return self.out_proj(x)


class _ScaleGradient(torch.autograd.Function):
    """Keep a tensor's forward value while scaling its backward gradient."""

    @staticmethod
    def forward(ctx, tensor: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, None]:
        return gradient * ctx.scale, None


class MultiscaleTokenizer(nn.Module):
    """DNA tokenizer with a multiscale residual quantization bottleneck."""

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
        temperature: float = 1.0,
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
            temperature=temperature,
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

    def _normalize(
        self,
        latent: Float[Tensor, "batch length embed_dim"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        """Normalize one block without mixing its statistics with another block."""
        if self.pre_quant_norm is None or latent.shape[1] == 0:
            return latent

        latent = einx.id("b l d -> b d l", latent)
        latent = self.pre_quant_norm(latent)
        return einx.id("b d l -> b l d", latent)

    def encode(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        """Encode DNA into the normalized continuous latent space.

        NSM-DNA quantizes target latents but uses completed prefix latents
        directly as transformer context.
        """
        return self._normalize(self.encoder(token_ids))

    def encode_pair(
        self,
        sequence_ids: Int[Tensor, "batch sequence_length"],
    ) -> tuple[
        Float[Tensor, "batch prefix_length embed_dim"],
        Float[Tensor, "batch context_length embed_dim"],
    ]:
        """Embed a variable prefix and fixed-length target in one encoder call."""
        if sequence_ids.shape[1] < self.context_length:
            raise ValueError(
                f"Expected at least {self.context_length} target positions."
            )

        latent = self.encoder(sequence_ids)
        prefix = latent[:, : -self.context_length]
        target = latent[:, -self.context_length :]
        return self._normalize(prefix), self._normalize(target)

    def quantize(
        self,
        target_latent: Float[Tensor, "batch context_length embed_dim"],
        *,
        corruption_probability: float = 0.0,
    ) -> QuantizerOutput:
        """Quantize one target block and prepare differentiable hierarchy inputs."""
        return self.quantizer(
            target_latent,
            corruption_probability=corruption_probability,
        )

    def decode_latent(
        self,
        latent: Float[Tensor, "batch context_length embed_dim"],
    ) -> Float[Tensor, "batch context_length vocab_size"]:
        """Decode one full-length cumulative latent into nucleotide logits."""
        return self.decoder(latent)

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
        quantizer_output = self.quantizer(
            latent,
            include_partial_reconstruction=include_partial_reconstruction,
        )
        logits = self.decoder(quantizer_output.final_latent)

        partial_logits = None
        if quantizer_output.partial_latent is not None:
            # The partial loss uses one decoder pass, but its gradient can have
            # different strengths on the quantizer path and decoder parameters.
            # Scaling at the decoder input affects only the gradient flowing back
            # into the quantizer; the loss coefficient controls the decoder.
            partial_quantized_latent = _ScaleGradient.apply(
                quantizer_output.partial_latent,
                partial_latent_gradient_scale,
            )
            partial_logits = self.decoder(partial_quantized_latent)

        return (
            logits,
            partial_logits,
            quantizer_output.vq_loss,
            quantizer_output.indices_by_scale,
        )

    @torch.no_grad()
    def encode_indices(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> list[Int[Tensor, "batch scale_length"]]:
        """Encode target blocks into discrete codebook indices at every scale.

        These indices identify the hard codes selected at every hierarchy scale.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before encoding sequences.")

        latent = self.encode(token_ids)
        return self.quantizer(latent).indices_by_scale

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
        return self.quantizer.indices_to_next_scale_input(preceding_indices_by_scale)

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
