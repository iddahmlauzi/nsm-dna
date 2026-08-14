import einx
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            eps=1e-6,
        )


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Compute the query, key, and value projections in one linear layer.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, valid_mask=None, *, is_causal=False):
        qkv = self.qkv_proj(x)
        qkv = einx.id("b l (qkv h d) -> qkv b h l d", qkv, qkv=3, h=self.num_heads)
        q, k, v = qkv.unbind(dim=0)

        attention_mask = None
        if valid_mask is not None:
            attention_mask = einx.id("b l -> b 1 1 l", valid_mask)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = einx.id("b h l d -> b l (h d)", y)
        return self.output_dropout(self.out_proj(y))


class MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, embed_dim, dropout=0.1, bias=False):
        super().__init__()

        hidden_dim = int(round(8 * embed_dim / 3 / 8) * 8)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.output_dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False):
        super().__init__()

        self.attn_norm = LayerNorm(embed_dim, bias=bias)
        self.attn = SelfAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
        )
        self.mlp_norm = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim, dropout=dropout, bias=bias)

    def forward(self, x, valid_mask=None, *, is_causal=False):
        x = x + self.attn(
            self.attn_norm(x),
            valid_mask=valid_mask,
            is_causal=is_causal,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x
