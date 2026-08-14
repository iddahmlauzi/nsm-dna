import torch
import torch.nn as nn

from .common import LayerNorm, TransformerBlock


class Encoder(nn.Module):
    """A simple encoder that only performs an embedding lookup."""

    def __init__(
        self,
        in_vocab_size,
        context_length,
        embed_dim,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(in_vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, token_ids):
        positions = torch.arange(token_ids.size(1), device=token_ids.device)

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(positions)

        return self.drop(token_embeddings + position_embeddings)


class Decoder(nn.Module):
    """Decode latent representations into nucleotide logits."""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=False,
    ):
        super().__init__()

        self.block = TransformerBlock(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.final_norm = LayerNorm(embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, latent, valid_mask=None):
        x = self.block(latent, valid_mask=valid_mask, is_causal=False)
        x = self.final_norm(x)
        return self.out_proj(x)
