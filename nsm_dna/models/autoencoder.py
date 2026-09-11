import einx
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .common import (
    LayerNorm,
    TransformerBlock,
    precompute_rope_cosine_and_sine,
)
from .sampling import (
    make_cascaded_downsampler,
    make_cascaded_upsampler,
)


def _sampling_factor(context_length: int, latent_length: int) -> int:
    """Return the total stride needed to reduce context_length to latent_length."""
    if latent_length <= 0 or context_length % latent_length != 0:
        raise ValueError("context_length must be divisible by latent_length.")
    return context_length // latent_length


class Encoder(nn.Module):
    """Embed nucleotides and downsample them into a continuous latent."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        latent_length: int,
        embed_dim: int,
        quantization_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.downsampler = make_cascaded_downsampler(
            embed_dim,
            total_stride=_sampling_factor(context_length, latent_length),
            base_stride=2,
            bias=bias,
        )
        self.quantization_projection = nn.Linear(
            embed_dim,
            quantization_dim,
            bias=bias,
        )

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch latent_length quantization_dim"]:
        x = self.drop(self.token_embedding(token_ids))
        x = einx.id("b l d -> b d l", x)
        x = self.downsampler(x)
        x = einx.id("b d l -> b l d", x)
        return self.quantization_projection(x)


class Decoder(nn.Module):
    """Decode latent representations into nucleotide logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        latent_length: int,
        embed_dim: int,
        quantization_dim: int,
        num_heads: int,
        max_context_length: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        bias: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        max_context_length = max_context_length or context_length
        if max_context_length < context_length:
            raise ValueError("max_context_length cannot be shorter than context_length.")

        positions = torch.arange(max_context_length)
        head_dim = embed_dim // num_heads
        rope_cosine, rope_sine = precompute_rope_cosine_and_sine(
            positions,
            head_dim,
            rope_base,
        )
        self.register_buffer("rope_cosine", rope_cosine, persistent=False)
        self.register_buffer("rope_sine", rope_sine, persistent=False)

        self.latent_projection = nn.Linear(
            quantization_dim,
            embed_dim,
            bias=bias,
        )
        self.block = TransformerBlock(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.additional_blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
            )
            for _ in range(num_layers - 1)
        )
        self.upsampler = make_cascaded_upsampler(
            embed_dim,
            total_stride=_sampling_factor(context_length, latent_length),
            base_stride=2,
            bias=bias,
        )
        self.final_norm = LayerNorm(embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=bias)

    def forward(
        self,
        latent: Float[Tensor, "batch latent_length quantization_dim"],
    ) -> Float[Tensor, "batch context_length vocab_size"]:
        x = self.latent_projection(latent)
        x = einx.id("b l d -> b d l", x)
        x = self.upsampler(x)
        x = einx.id("b d l -> b l d", x)
        output_length = x.shape[1]
        if output_length > self.rope_cosine.shape[0]:
            raise ValueError(
                f"Decoder output length {output_length} exceeds the configured "
                f"maximum of {self.rope_cosine.shape[0]}."
            )
        rotary_embeddings = (
            self.rope_cosine[:output_length],
            self.rope_sine[:output_length],
        )

        x = self.block(
            x,
            rotary_embeddings=rotary_embeddings,
            is_causal=False,
        )
        for block in self.additional_blocks:
            x = block(
                x,
                rotary_embeddings=rotary_embeddings,
                is_causal=False,
            )
        x = self.final_norm(x)
        return self.out_proj(x)
