import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .common import LayerNorm, TransformerBlock


class Encoder(nn.Module):
    """A simple encoder that only performs an embedding lookup."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        positions = torch.arange(token_ids.size(1), device=token_ids.device)

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)

        return self.drop(token_embeddings + position_embeddings)


class Decoder(nn.Module):
    """Decode latent representations into nucleotide logits."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

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
        x = self.block(latent, is_causal=False)
        x = self.final_norm(x)
        return self.out_proj(x)
