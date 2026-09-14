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
    ChannelsFirstLayerNorm,
    make_learned_downsampler,
    make_learned_upsampler,
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
        num_heads: int = 1,
        num_layers: int = 0,
        use_qk_norm: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.context_blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(num_layers)
        )
        self.downsampler = make_learned_downsampler(
            embed_dim,
            stride=_sampling_factor(context_length, latent_length),
            bias=bias,
        )
        self.downsampling_norm = ChannelsFirstLayerNorm(embed_dim)
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
        for block in self.context_blocks:
            x = block(x, is_causal=False)
        x = einx.id("b l d -> b d l", x)
        # Create the shorter continuous latent that the multiscale quantizer models;
        # the quantization scales describe this latent rather than the full input.
        x = self.downsampler(x)
        x = self.downsampling_norm(x)
        x = einx.id("b d l -> b l d", x)
        # Reduce each encoder vector from embed_dim to quantization_dim before
        # finding its nearest code. With many independently varying dimensions and
        # only a fixed number of codebook vectors, even the nearest code may be a
        # poor match. Fewer dimensions mean fewer combinations for the codebook to
        # represent, making a closer match more likely.
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
        num_layers: int = 1,
        use_qk_norm: bool = False,
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

        self.latent_projection = nn.Linear(
            quantization_dim,
            embed_dim,
            bias=bias,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(num_layers)
        )
        self.upsampler = make_learned_upsampler(
            embed_dim,
            stride=_sampling_factor(context_length, latent_length),
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

        for block in self.blocks:
            x = block(
                x,
                rotary_embeddings=(self.rope_cosine, self.rope_sine),
                is_causal=False,
            )
        x = self.final_norm(x)
        return self.out_proj(x)
