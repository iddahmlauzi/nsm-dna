import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


RotaryEmbeddings = tuple[
    Float[Tensor, "length head_dim"],
    Float[Tensor, "length head_dim"],
]


def precompute_rope_cosine_and_sine(
    positions: Int[Tensor, "length"],
    head_dim: int,
    base: float = 10000.0,
) -> RotaryEmbeddings:
    """Precompute the rotary values for the requested positions."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension.")

    half_head_dim = head_dim // 2
    frequency_indices = torch.arange(
        half_head_dim,
        dtype=torch.float32,
        device=positions.device,
    )
    inverse_frequencies = base ** (-frequency_indices / half_head_dim)
    angles = torch.outer(positions.float(), inverse_frequencies)
    angles = torch.cat([angles, angles], dim=-1)

    return angles.cos(), angles.sin()


def apply_rope(
    x: Float[Tensor, "batch num_heads length head_dim"],
    cosine: Float[Tensor, "length head_dim"],
    sine: Float[Tensor, "length head_dim"],
) -> Float[Tensor, "batch num_heads length head_dim"]:
    """Apply GPT-NeoX-style rotary position embeddings."""
    first_half, second_half = x.chunk(2, dim=-1)
    rotated = torch.cat([-second_half, first_half], dim=-1)

    cosine = cosine.to(dtype=x.dtype)
    sine = sine.to(dtype=x.dtype)
    return x * cosine + rotated * sine


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""

    def __init__(self, embed_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim)) if bias else None

    def forward(
        self,
        x: Float[Tensor, "batch length embed_dim"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            eps=1e-6,
        )


class RMSNorm(nn.Module):
    """Normalize a tensor by its root mean square without centering it."""

    def __init__(self, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(
        self,
        x: Float[Tensor, "batch num_heads length head_dim"],
    ) -> Float[Tensor, "batch num_heads length head_dim"]:
        input_dtype = x.dtype
        # Compute the RMS in float32 for stability under mixed precision.
        x = x.float()
        inverse_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        x = (x * inverse_rms).to(input_dtype)
        return x * self.weight.to(input_dtype)


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_qk_norm = use_qk_norm

        # Compute the query, key, and value projections in one linear layer.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = dropout
        self.output_dropout = nn.Dropout(dropout)

        if self.use_qk_norm:
            head_dim = self.embed_dim // self.num_heads
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        x: Float[Tensor, "batch length embed_dim"],
        *,
        attention_mask: Bool[Tensor, "mask_batch mask_heads length length"] | None = None,
        rotary_embeddings: RotaryEmbeddings | None = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "batch length embed_dim"]:
        qkv = self.qkv_proj(x)
        qkv = einx.id(
            "b l (qkv h d) -> qkv b h l d",
            qkv,
            qkv=3,
            h=self.num_heads,
        )
        q, k, v = qkv.unbind(dim=0)

        # QK norm: RMSNorm each query and key over head_dim, but do not
        # normalize V. Apply it before RoPE so rotation acts on unit-RMS vectors.
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_embeddings is not None:
            cosine, sine = rotary_embeddings
            q = apply_rope(q, cosine, sine)
            k = apply_rope(k, cosine, sine)

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

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        hidden_dim = int(round(8 * embed_dim / 3 / 8) * 8)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Float[Tensor, "batch length embed_dim"],
    ) -> Float[Tensor, "batch length embed_dim"]:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.output_dropout(self.down_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()

        self.attn_norm = LayerNorm(embed_dim, bias=bias)
        self.attn = SelfAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            use_qk_norm=use_qk_norm,
        )
        self.mlp_norm = LayerNorm(embed_dim, bias=bias)
        self.mlp = MLP(embed_dim, dropout=dropout, bias=bias)

    def forward(
        self,
        x: Float[Tensor, "batch length embed_dim"],
        *,
        attention_mask: Bool[Tensor, "mask_batch mask_heads length length"] | None = None,
        rotary_embeddings: RotaryEmbeddings | None = None,
        is_causal: bool = False,
    ) -> Float[Tensor, "batch length embed_dim"]:
        x = x + self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            rotary_embeddings=rotary_embeddings,
            is_causal=is_causal,
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x
