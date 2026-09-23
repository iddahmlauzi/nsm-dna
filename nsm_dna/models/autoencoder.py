import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
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


class Encoder(nn.Module):
    """Embed nucleotides and downsample them into a continuous latent."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        latent_length: int,
        embed_dim: int,
        quantization_dim: int,
        third_base_scale: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        sampling_factor = _sampling_factor(context_length, latent_length)
        if not 0.0 <= third_base_scale <= 1.0:
            raise ValueError("third_base_scale must be between zero and one.")

        self.sampling_factor = sampling_factor
        self.third_base_scale = third_base_scale

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Combine each non-overlapping group of nucleotides into one latent
        # position, reducing context_length positions to latent_length positions.
        # It also reduces embed_dim to quantization_dim. With fewer independently
        # varying values in each vector, a fixed number of codebook vectors can
        # provide closer matches during nearest-code lookup.
        self.downsampler = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=quantization_dim,
            kernel_size=sampling_factor,
            stride=sampling_factor,
            bias=bias,
        )
        self.core_norm = nn.LayerNorm(
            quantization_dim,
            elementwise_affine=False,
        )
        self.refinement_norm = nn.LayerNorm(
            quantization_dim,
            elementwise_affine=False,
        )

        # Euclidean codebook distances grow with the magnitude of the latent
        # vectors. Normalize each vector so that magnitude cannot drift during
        # training and make nearest-code matching progressively harder.
        self.norm = nn.LayerNorm(quantization_dim, elementwise_affine=False)

    def forward(
        self,
        token_ids: Int[Tensor, "batch length"],
    ) -> Float[Tensor, "batch latent_length quantization_dim"]:
        x = self.token_embedding(token_ids)
        x = einx.id("b l d -> b d l", x)

        if self.sampling_factor == 3:
            weights = self.downsampler.weight
            core = F.conv1d(x, weights[:, :, :2], stride=3)
            refinement = F.conv1d(x[:, :, 2:], weights[:, :, 2:], stride=3)
            core = self.core_norm(einx.id("b d l -> b l d", core))
            refinement = self.refinement_norm(einx.id("b d l -> b l d", refinement))
            x = core + self.third_base_scale * refinement
            if self.downsampler.bias is not None:
                x = x + self.downsampler.bias
        else:
            x = self.downsampler(x)
            x = einx.id("b d l -> b l d", x)

        return self.norm(x)


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

    def encode(
        self,
        x: Float[Tensor, "batch latent_length quantization_dim"],
    ) -> Float[Tensor, "batch context_length embed_dim"]:
        """Return final normalized states before nucleotide prediction."""
        x = einx.id("b l d -> b d l", x)
        x = self.upsampler(x)
        x = einx.id("b d l -> b l d", x)

        for block in self.blocks:
            x = block(
                x,
                rotary_embeddings=(self.rope_cosine, self.rope_sine),
                is_causal=False,
            )
        return self.final_norm(x)

    def forward(
        self,
        x: Float[Tensor, "batch latent_length quantization_dim"],
    ) -> Float[Tensor, "batch context_length vocab_size"]:
        return self.out_proj(self.encode(x))
