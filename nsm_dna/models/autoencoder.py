import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .common import (
    LayerNorm,
    TransformerBlock,
    precompute_rope_cosine_and_sine,
)


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
