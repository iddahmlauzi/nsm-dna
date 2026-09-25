import einx
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .common import (
    LayerNorm,
    TransformerBlock,
    precompute_rope_cosine_and_sine,
)


def _sampling_factor(context_length: int, latent_length: int) -> int:
    """Return the size of each non-overlapping nucleotide group."""
    if latent_length <= 0 or context_length % latent_length != 0:
        raise ValueError("context_length must be divisible by latent_length.")
    return context_length // latent_length


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
        bias: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        sampling_factor = _sampling_factor(context_length, latent_length)

        positions = torch.arange(context_length)
        head_dim = embed_dim // num_heads
        rope_cosine, rope_sine = precompute_rope_cosine_and_sine(
            positions,
            head_dim,
            rope_base,
        )
        self.register_buffer("rope_cosine", rope_cosine, persistent=False)
        self.register_buffer("rope_sine", rope_sine, persistent=False)

        # Expand each quantized vector back over its nucleotide group while also
        # projecting from quantization_dim to the decoder's hidden dimension.
        self.upsampler = nn.ConvTranspose1d(
            in_channels=quantization_dim,
            out_channels=embed_dim,
            kernel_size=sampling_factor,
            stride=sampling_factor,
            bias=bias,
        )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim,
                num_heads,
                dropout=0.0,
                bias=bias,
                use_qk_norm=use_qk_norm,
            )
            for _ in range(num_layers)
        )
        self.final_norm = LayerNorm(embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=bias)

    def forward(
        self,
        x: Float[Tensor, "batch latent_length quantization_dim"],
    ) -> Float[Tensor, "batch context_length vocab_size"]:
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
